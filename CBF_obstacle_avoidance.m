%% Four circular obstacles + circle-covered rectangle + implicit PDE step + explicit local DISCRETE CBF-QP correction

clear; clc; close all;
global TAU_SIGN_STATE QP_WARM SIM_STATS
TAU_SIGN_STATE = [];
QP_WARM = [];
SIM_STATS = struct('control_updates',0,'quadprog_calls',0, ...
                   'qp_fallback_calls',0);

overall_tic = tic;

%% ===================== Parameters =====================
N = 30;                         
k = 7;                          
h = 1/N;
alpha = linspace(0,1,N+1).';

time_scale = 2.0;                
T_end = 30.0*time_scale;         
dt_requested = 0.01;            

% Make the final time exactly reachable by a constant step.
nSteps = round(T_end/dt_requested);
dt = T_end/nSteps;
t = (0:nSteps).'*dt;

% QP weights
rho_v   = 0.2;
rho_c   = 0.2;
rho_tau = 10;
k_c     = 2.0;

use_active_set_warm_start = true;

use_predictive_obstacle_activation = true;

%% ===================== Desired formation =====================
x_shape = 16*cos(2*pi*alpha);
y_shape = 8*sin(4*pi*alpha);
u_shape = x_shape + 1i*y_shape;

w = trap_weights(N+1);
one_vec = ones(N+1,1);
wbar = w/(w.'*one_vec);

c_shape = wbar.'*u_shape;
phi_d   = u_shape - one_vec*c_shape;

%% ===================== Desired center trajectory =====================
f_fun = @(tt) 2*(tt/time_scale) + 10i*sin(tt/time_scale);
fdot_fun = @(tt) 2/time_scale + (10/time_scale)*1i*cos(tt/time_scale);

c_fun     = @(tt) c_shape + f_fun(tt);
cdot_fun  = @(tt) fdot_fun(tt);
u_ref_fun = @(tt) one_vec*c_fun(tt) + phi_d;

%% ===================== Initial condition =====================
x_init = 8*cos(2*pi*alpha);
y_init = 8*sin(2*pi*alpha);
u_init = x_init + 1i*y_init;

%% ===================== OBSTACLES: FOUR CIRCLES + ONE RECTANGLE =====================
% ----- Standalone circular obstacles -----
% Existing two circles:
%   Circle 1: (0,-5),   R = 2.0
%   Circle 2: (-10,10),  R = 1.5
%
% Added two circles in the later part of the moving-formation trajectory:
%   Circle 3: (30,-10),  R = 2.0
%   Circle 4: (40, 5),  R = 1.8
%
% Both added circles use the same 0.6 safety margin.
circle_obs_c_vec       = [  0 - 5i; ...
                           -10 + 10i; ...
                            30 - 10i; ...
                            40 + 5i];
circle_obs_R_vec       = [2.0; 1.5; 2.0; 1.8];
circle_safe_margin_vec = [0.6; 0.6; 0.6; 0.6];
circle_R_safe_vec      = circle_obs_R_vec + circle_safe_margin_vec;
nCircleObs             = numel(circle_obs_c_vec);

% ----- Rectangular obstacle -----
% Required physical rectangle:
%       x in [20,30],  y in [10,15].
%
% The rectangle is covered by a 5 x 2 grid of 10 overlapping circles.
% Centers are placed at the centers of ten 2-by-2.5 rectangular cells:
%       x = 21,23,25,27,29,
%       y = 11.25,13.75.
% The farthest point of each cell from its circle center is
%       sqrt(1^2 + 1.25^2) = 1.60078,
% hence rect_cover_R = 1.65 guarantees complete coverage of [20,30]x[10,15].
rect_x_min = 20;
rect_x_max = 30;
rect_y_min = 10;
rect_y_max = 15;

rect_x_centers = 21:2:29;                  
rect_y_centers = [11.25,13.75];            
[RECT_X,RECT_Y] = meshgrid(rect_x_centers,rect_y_centers);
rect_obs_c_vec = RECT_X(:) + 1i*RECT_Y(:); 
rect_n_cover = numel(rect_obs_c_vec);

rect_cover_R = 1.65;                       
rect_cover_safe_margin = 0.50;             
rect_R_safe = rect_cover_R + rect_cover_safe_margin;
rect_obs_R_vec = rect_cover_R*ones(rect_n_cover,1);
rect_R_safe_vec = rect_R_safe*ones(rect_n_cover,1);

% Combined primitive circles used for obstacle safety-function evaluation.
all_obs_c_vec = [circle_obs_c_vec; rect_obs_c_vec];
all_R_safe_vec = [circle_R_safe_vec; rect_R_safe_vec];
nObstaclePrimitives = numel(all_obs_c_vec);

% Obstacle discrete-CBF parameters
gamma = 8.0;
h_act = 1.5;
eta_obs = 1-exp(-gamma*dt);

% Tangential bypass constraint.
% Standalone circles each have their own tangent direction.  The ten rectangle
% cover circles share ONE composite tangent per agent to avoid contradictory
% tangential commands from overlapping cover circles.
tau_sign_eps = 2;
tau_lb0      = 1.0;
nTangentGroups = nCircleObs + 1;            

v_max = 5.0;
speed_limit_start_time = 0.0;
speed_limit_n_dirs = 16;     

%% ===================== Pairwise inter-agent discrete CBF =====================
pair_safe_dist = 1.5;
gamma_pair     = 5.0;
pair_margin    = 0.0;
eta_pair       = 1-exp(-gamma_pair*dt);

exclude_closed_endpoint = true;

%% ===================== PRECOMPUTE constant PDE/QP data =====================
PRE = build_precomputed_data( ...
    N,k,h,phi_d,wbar,rho_v,rho_c,rho_tau, ...
    exclude_closed_endpoint,use_active_set_warm_start);


A_imp = speye(PRE.nU) - dt*PRE.L;
try
    A_imp_dec = decomposition(A_imp,'lu');
    use_decomposition = true;
    factorization_text = 'LU decomposition';
catch
    A_imp_dec = [];
    use_decomposition = false;
    factorization_text = 'MATLAB backslash';
end

PRE.dt       = dt;
PRE.eta_obs  = eta_obs;
PRE.eta_pair = eta_pair;

%% ===================== Initial safety check =====================
h_obs_init = min_obstacle_barrier_primitives( ...
    u_init,all_obs_c_vec,all_R_safe_vec);
h_pair_init = min_pairwise_barrier_fast( ...
    u_init,pair_safe_dist,PRE.pair_i,PRE.pair_j);

fprintf('\n============================================================\n');
fprintf('Implicit-PDE + explicit-local DISCRETE-CBF simulation starts\n');
fprintf('Agents                     : %d\n',PRE.nU);
fprintf('All constrained pairs      : %d\n',PRE.nPairs);
fprintf('Standalone circle obstacles : %d\n',nCircleObs);
for q = 1:nCircleObs
    fprintf('Circle %d center             : %.2f %+.2fi\n', ...
        q,real(circle_obs_c_vec(q)),imag(circle_obs_c_vec(q)));
    fprintf('Circle %d physical radius    : %.2f\n',q,circle_obs_R_vec(q));
    fprintf('Circle %d safe radius        : %.2f\n',q,circle_R_safe_vec(q));
    fprintf('Circle %d activation radius  : %.6f\n', ...
        q,sqrt(circle_R_safe_vec(q)^2+h_act));
end
fprintf('Rectangle x-range          : [%.2f, %.2f]\n',rect_x_min,rect_x_max);
fprintf('Rectangle y-range          : [%.2f, %.2f]\n',rect_y_min,rect_y_max);
fprintf('Rectangle cover circles    : %d (5 columns x 2 rows)\n',rect_n_cover);
fprintf('Rectangle cover radius     : %.4f\n',rect_cover_R);
fprintf('Rectangle cover safe radius: %.4f\n',rect_R_safe);
fprintf('Rectangle activation radius: %.6f\n',sqrt(rect_R_safe^2+h_act));
fprintf('Obstacle activation h_act  : %.4f\n',h_act);
fprintf('eta_obs                    : %.8f\n',eta_obs);
fprintf('eta_pair                   : %.8f\n',eta_pair);
fprintf('Initial min obstacle h     : %.6f\n',h_obs_init);
fprintf('Initial min pairwise h     : %.6f\n',h_pair_init);
fprintf('Requested dt               : %.6f s\n',dt_requested);
fprintf('Actual constant dt         : %.6f s\n',dt);
fprintf('Number of time steps       : %d\n',nSteps);
fprintf('Implicit PDE solve          : %s\n',factorization_text);
fprintf('CBF correction              : explicit local v_i at each agent\n');
if use_predictive_obstacle_activation
    fprintf('Obstacle activation        : current OR nominal-next-step predictive\n');
else
    fprintf('Obstacle activation        : current-state only\n');
end
fprintf('Pairwise activation        : ALL admissible pairs at every step\n');
if use_active_set_warm_start
    fprintf('QP primary algorithm        : active-set with warm start\n');
else
    fprintf('QP primary algorithm        : interior-point-convex\n');
end
fprintf('Requested final time       : %.2f s\n',T_end);
fprintf('============================================================\n\n');

if h_obs_init < 0
    warning('Initial formation violates the obstacle safety set: min h_obs < 0.');
end
if h_pair_init < 0
    warning('Initial formation violates the pairwise safety set: min h_pair < 0.');
end

%% ===================== Split-step simulation =====================
reset_tau_sign_state(PRE.nU,nTangentGroups);
QP_WARM = [];

U = complex(zeros(nSteps+1,PRE.nU));
U(1,:) = u_init.';

h_obs_step  = zeros(nSteps+1,1);
h_pair_step = zeros(nSteps+1,1);
h_obs_step(1)  = h_obs_init;
h_pair_step(1) = h_pair_init;
warned_obs_violation = false;
warned_pair_violation = false;

sim_tic = tic;
next_pct = 5;
fprintf('Simulation progress:   0%% | step 0/%d | elapsed = %.1f s\n', ...
        nSteps,toc(sim_tic));

for n = 1:nSteps
    t_n = t(n);
    u_n = U(n,:).';

    % 1) implicit PDE-only base step -----
    % (I-dt*L)u_base^{n+1} = u^n - dt*L*phi_d.
    rhs_pde = u_n - dt*PRE.L_phi;
    if use_decomposition
        u_base_next = A_imp_dec \ rhs_pde;
    else
        u_base_next = A_imp \ rhs_pde;
    end

    g_base = (u_base_next-u_n)/dt;

    % 2) one QP using DISCRETE constraints consistent with q_n -----
    c_n    = c_fun(t_n);
    cdot_n = cdot_fun(t_n);

    v_n = solve_circles_rectangle_split_discrete_cbf_qp( ...
        u_n,c_n,cdot_n,g_base,PRE, ...
        circle_obs_c_vec,circle_R_safe_vec, ...
        rect_obs_c_vec,rect_R_safe_vec,h_act,k_c, ...
        tau_sign_eps,tau_lb0, ...
        pair_safe_dist,pair_margin, ...
        use_predictive_obstacle_activation, ...
        t_n,v_max,speed_limit_start_time,speed_limit_n_dirs);

    SIM_STATS.control_updates = SIM_STATS.control_updates + 1;

    % 3) explicit LOCAL CBF-QP correction -----
    % u^{n+1} = u_base^{n+1} + dt*v_n.
    u_np1 = u_base_next + dt*v_n;

    if any(~isfinite(u_np1))
        error('Non-finite state detected at time step %d (t = %.6f s).', ...
              n,t(n+1));
    end

    U(n+1,:) = u_np1.';

    
    h_obs_step(n+1) = min_obstacle_barrier_primitives( ...
        u_np1,all_obs_c_vec,all_R_safe_vec);
    h_pair_step(n+1) = min_pairwise_barrier_fast( ...
        u_np1,pair_safe_dist,PRE.pair_i,PRE.pair_j);

 
    if h_obs_step(n+1) < -1e-8 && ~warned_obs_violation
        warning(['Obstacle safety first became negative at t = %.6f s: h_min = %.6e. ', ...
                 'The discrete CBF is consistent with the step; inspect activation ', ...
                 'or reduce dt if this persists.'],t(n+1),h_obs_step(n+1));
        warned_obs_violation = true;
    end
    if h_pair_step(n+1) < -1e-8 && ~warned_pair_violation
        warning(['Pairwise safety first became negative at t = %.6f s: h_pair_min = %.6e. ', ...
                 'All admissible pairs are constrained; inspect QP tolerance/feasibility or reduce dt if this persists.'], ...
                 t(n+1),h_pair_step(n+1));
        warned_pair_violation = true;
    end

    % ----- command-window progress -----
    pct = floor(100*n/nSteps);
    while pct >= next_pct && next_pct <= 95
        fprintf('Simulation progress: %3d%% | step %d/%d | t = %.2f s | elapsed = %.1f s\n', ...
                next_pct,n,nSteps,t(n+1),toc(sim_tic));
        next_pct = next_pct + 5;
    end
end

sim_elapsed = toc(sim_tic);
fprintf('Simulation progress: 100%% | step %d/%d | elapsed = %.1f s\n', ...
        nSteps,nSteps,sim_elapsed);

fprintf('\n============= SPLIT PDE + LOCAL DISCRETE-CBF SUMMARY =============\n');
fprintf('final simulation time      = %.6f s\n',t(end));
fprintf('time steps                 = %d\n',nSteps);
fprintf('control updates            = %d\n',SIM_STATS.control_updates);
fprintf('quadprog calls             = %d\n',SIM_STATS.quadprog_calls);
fprintf('robust QP fallback calls   = %d\n',SIM_STATS.qp_fallback_calls);
fprintf('simulation wall time       = %.2f s\n',sim_elapsed);
fprintf('==========================================================\n\n');

%% ==========================================
Nt = numel(t);
err_inf    = zeros(Nt,1);
err_l2     = zeros(Nt,1);
err_center = zeros(Nt,1);
h_obs_min  = zeros(Nt,1);
h_pair_min = zeros(Nt,1);

for m = 1:Nt
    u_now = U(m,:).';
    u_ref_now = u_ref_fun(t(m));
    e = u_now-u_ref_now;

    err_inf(m)    = max(abs(e));
    err_l2(m)     = sqrt(sum(wbar.*abs(e).^2));
    err_center(m) = abs(wbar.'*e);

    h_obs_min(m) = min_obstacle_barrier_primitives( ...
        u_now,all_obs_c_vec,all_R_safe_vec);
    h_pair_min(m) = min_pairwise_barrier_fast( ...
        u_now,pair_safe_dist,PRE.pair_i,PRE.pair_j);
end

min_h_obs  = min(h_obs_min);
min_h_pair = min(h_pair_min);

fprintf('Post-processing completed.\n');
fprintf('Minimum sampled obstacle safety function = %.6e\n',min_h_obs);
fprintf('Minimum sampled pairwise safety function = %.6e\n\n',min_h_pair);

if min_h_obs < -1e-8
    warning(['Obstacle safety is still violated. Since the QP now uses the ', ...
             'split-step discrete dynamics, first inspect whether an agent ', ...
             'crossed from outside h_act to h<0 in one step; then reduce dt or ', ...
             'increase h_act.']);
end
if min_h_pair < -1e-8
    warning(['Pairwise safety is still violated even though all admissible pairs are ', ...
             'constrained. Inspect the first violating step, QP feasibility/tolerance, ', ...
             'or reduce dt.']);
end

%% ===================== Output 1: tracking error =====================
figure('Color','w');
plot(t,err_inf,'b-','LineWidth',1.8); hold on; grid on;
plot(t,err_l2,'r--','LineWidth',1.8);
plot(t,err_center,'k-.','LineWidth',1.5);
xlabel('t (s)');
ylabel('tracking error');
title('Tracking error');
legend('max_i |u_i-u_i^{ref}|', ...
       'discrete L_2 error', ...
       'weighted center error', ...
       'Location','best');

%% ===================== Output 2: agent-obstacle safety function =====================
figure('Color','w');
plot(t,h_obs_min,'b-','LineWidth',1.8); hold on; grid on;
yline(0,'r--','LineWidth',1.5);
xlabel('t (s)');
ylabel('min_{i,q} h_{i,q}(t)');
title('Agent-obstacle safety function');
legend('min over agents and obstacle cover circles', ...
       'safety boundary h=0', ...
       'Location','best');

%% ===================== Output 3: agent-agent safety function =====================
figure('Color','w');
plot(t,h_pair_min,'b-','LineWidth',1.8); hold on; grid on;
yline(0,'r--','LineWidth',1.5);
xlabel('t (s)');
ylabel('min_{i<j} h_{ij}(t)');
title('Agent-agent safety function');
legend('min_{i<j} (|u_i-u_j|^2-d_{safe}^2)', ...
       'safety boundary h_{ij}=0', ...
       'Location','best');

%% ===================== Additional diagnostic: maximum actual agent speed =====================
Q_actual = diff(U,1,1)/dt;                  
max_agent_speed = max(abs(Q_actual),[],2);  

figure('Color','w');
plot(t(1:end-1),max_agent_speed,'b-','LineWidth',1.8); hold on; grid on;
yline(v_max,'r--','LineWidth',1.5);
xline(speed_limit_start_time,'k:','LineWidth',1.2);
xlabel('t (s)');
ylabel('max_i |q_i(t)|');
title('Maximum actual agent speed');
legend('max_i |q_i(t)|','v_{max}','speed limit on','Location','best');

[max_speed_global,idx_max_speed] = max(max_agent_speed);
fprintf('Maximum actual agent speed over the simulation = %.6f\n',max_speed_global);
fprintf('Time of maximum actual agent speed             = %.6f s\n', ...
        t(idx_max_speed));
post_mask = t(1:end-1) >= speed_limit_start_time;
if any(post_mask)
    max_speed_after_limit = max(max_agent_speed(post_mask));
    fprintf('Maximum speed after speed-limit activation     = %.6f\n\n', ...
            max_speed_after_limit);
end

% ===================== Output 4: obstacle-avoidance video =====================
th = linspace(0,2*pi,400);


obs_all_x = [];
obs_all_y = [];
for q = 1:nCircleObs
    obs_curve_q  = circle_obs_c_vec(q) + circle_obs_R_vec(q)*(cos(th)+1i*sin(th));
    safe_curve_q = circle_obs_c_vec(q) + circle_R_safe_vec(q)*(cos(th)+1i*sin(th));
    obs_all_x = [obs_all_x; real(obs_curve_q(:)); real(safe_curve_q(:))]; 
    obs_all_y = [obs_all_y; imag(obs_curve_q(:)); imag(safe_curve_q(:))]; 
end
for q = 1:rect_n_cover
    cover_curve_q = rect_obs_c_vec(q) + rect_obs_R_vec(q)*(cos(th)+1i*sin(th));
    safe_curve_q  = rect_obs_c_vec(q) + rect_R_safe_vec(q)*(cos(th)+1i*sin(th));
    obs_all_x = [obs_all_x; real(cover_curve_q(:)); real(safe_curve_q(:))]; 
    obs_all_y = [obs_all_y; imag(cover_curve_q(:)); imag(safe_curve_q(:))]; 
end
obs_all_x = [obs_all_x; rect_x_min; rect_x_max];
obs_all_y = [obs_all_y; rect_y_min; rect_y_max];

ref_all = complex(zeros(PRE.nU,Nt));
for kk = 1:Nt
    ref_all(:,kk) = u_ref_fun(t(kk));
end

x_all = [real(U(:));real(ref_all(:));obs_all_x];
y_all = [imag(U(:));imag(ref_all(:));obs_all_y];
plot_pad = 1.0;
x_lim = [min(x_all)-plot_pad,max(x_all)+plot_pad];
y_lim = [min(y_all)-plot_pad,max(y_all)+plot_pad];

video_fps = 15;
video_duration = t(end)-t(1);
n_video_frames = max(2,round(video_duration*video_fps));
video_time = linspace(t(1),t(end),n_video_frames).';

U_video = interp1(t,U,video_time,'linear');

video_name = 'four_circles_rectangle_CBF_QP_time2_vmax15_t60.mp4';
fig_video = figure('Color','w','Position',[100,80,1000,650]);
set(fig_video,'Resize','off');

vw = VideoWriter(video_name,'MPEG-4');
vw.FrameRate = video_fps;
vw.Quality = 85;
open(vw);

video_tic = tic;
fprintf('Writing video:   0%% | %d frames | elapsed = 0.0 s\n',n_video_frames);
next_video_pct = 10;

try
    for kv = 1:n_video_frames
        clf(fig_video);

        t_now = video_time(kv);
        u_now = U_video(kv,:).';
        u_ref_now = u_ref_fun(t_now);

        draw_circles_rectangle_scene( ...
            u_now,u_ref_now, ...
            circle_obs_c_vec,circle_obs_R_vec,circle_R_safe_vec, ...
            rect_x_min,rect_x_max,rect_y_min,rect_y_max, ...
            rect_obs_c_vec,rect_obs_R_vec,rect_R_safe_vec, ...
            th,x_lim,y_lim);
        title(sprintf('Obstacle avoidance,  t = %.2f s',t_now));

        drawnow;
        frame = getframe(fig_video);
        writeVideo(vw,frame);

        pct = floor(100*kv/n_video_frames);
        if pct >= next_video_pct || kv == n_video_frames
            fprintf('Writing video: %3d%% | frame %d/%d | elapsed = %.1f s\n', ...
                    min(pct,100),kv,n_video_frames,toc(video_tic));
            next_video_pct = next_video_pct + 10;
        end
    end
    close(vw);
catch ME
    try
        close(vw);
    catch
    end
    rethrow(ME);
end

fprintf('\nVideo saved to: %s\n',video_name);

%% ===================== Output 5: NINE transient snapshots =====================
snapshot_times = [0, 1.0, 2.5, 5, 10, 15, 25, 45, 60];

U_snapshot = interp1(t,U,snapshot_times,'linear');

snapshot_dir = 'transient_snapshots';
if ~exist(snapshot_dir,'dir')
    mkdir(snapshot_dir);
end

fprintf('\nSaving 9 transient snapshots...\n');

for ks = 1:numel(snapshot_times)
    fig_snap = figure('Color','w','Position',[100,80,1000,650]);

    t_snap = snapshot_times(ks);
    u_snap = U_snapshot(ks,:).';
    u_ref_snap = u_ref_fun(t_snap);

    draw_circles_rectangle_scene( ...
        u_snap,u_ref_snap, ...
        circle_obs_c_vec,circle_obs_R_vec,circle_R_safe_vec, ...
        rect_x_min,rect_x_max,rect_y_min,rect_y_max, ...
        rect_obs_c_vec,rect_obs_R_vec,rect_R_safe_vec, ...
        th,x_lim,y_lim);

    title(sprintf('t = %.2f s',t_snap));

    snap_name = fullfile(snapshot_dir, ...
        sprintf('snapshot_%02d_t_%05.2f_s.png',ks,t_snap));

   
    print(fig_snap,snap_name,'-dpng','-r300');
    close(fig_snap);

    fprintf('  Snapshot %d/9 saved: t = %.2f s\n',ks,t_snap);
end

%% ===================== Output 6: obstacle-layout-only figure =====================
% This figure shows ONLY obstacle geometry:
%   1) four standalone circles:
%        - physical radius: solid line
%        - safety radius: dashed line
%   2) the physical rectangle boundary
%   3) all 10 rectangle cover circles:
%        - physical cover radius: solid line
%        - cover-circle safety radius: dashed line
%
% No agents and no reference formation are shown.

fig_obs_only = figure('Color','w','Position',[100,80,1100,650]);
hold on; grid on; axis equal;

% ----- Four standalone circular obstacles -----
for q = 1:nCircleObs
    physical_curve = circle_obs_c_vec(q) + ...
        circle_obs_R_vec(q)*(cos(th)+1i*sin(th));
    safety_curve = circle_obs_c_vec(q) + ...
        circle_R_safe_vec(q)*(cos(th)+1i*sin(th));

    if q == 1
        plot(real(physical_curve),imag(physical_curve),'m-', ...
             'LineWidth',2.2,'DisplayName','Circle physical boundary');
        plot(real(safety_curve),imag(safety_curve),'m--', ...
             'LineWidth',1.5,'DisplayName','Circle safety boundary');
    else
        plot(real(physical_curve),imag(physical_curve),'m-', ...
             'LineWidth',2.2,'HandleVisibility','off');
        plot(real(safety_curve),imag(safety_curve),'m--', ...
             'LineWidth',1.5,'HandleVisibility','off');
    end
end

% ----- Physical rectangular obstacle boundary -----
rect_x_only = [rect_x_min,rect_x_max,rect_x_max,rect_x_min,rect_x_min];
rect_y_only = [rect_y_min,rect_y_min,rect_y_max,rect_y_max,rect_y_min];

plot(rect_x_only,rect_y_only,'-', ...
     'Color',[0.85 0.33 0.10],'LineWidth',2.5, ...
     'DisplayName','Physical rectangle');

% ----- Ten circles used to cover the rectangle -----
for r = 1:rect_n_cover
    cover_curve = rect_obs_c_vec(r) + ...
        rect_obs_R_vec(r)*(cos(th)+1i*sin(th));
    cover_safe_curve = rect_obs_c_vec(r) + ...
        rect_R_safe_vec(r)*(cos(th)+1i*sin(th));

    if r == 1
        plot(real(cover_curve),imag(cover_curve),'-', ...
             'Color',[0.93 0.55 0.18],'LineWidth',1.2, ...
             'DisplayName','Rectangle cover circles');
        plot(real(cover_safe_curve),imag(cover_safe_curve),'--', ...
             'Color',[0.93 0.55 0.18],'LineWidth',1.2, ...
             'DisplayName','Rectangle cover safety boundaries');
    else
        plot(real(cover_curve),imag(cover_curve),'-', ...
             'Color',[0.93 0.55 0.18],'LineWidth',1.2, ...
             'HandleVisibility','off');
        plot(real(cover_safe_curve),imag(cover_safe_curve),'--', ...
             'Color',[0.93 0.55 0.18],'LineWidth',1.2, ...
             'HandleVisibility','off');
    end
end

% Fixed obstacle-only plotting range from all physical and safety boundaries.
obs_only_x = [];
obs_only_y = [];

for q = 1:nCircleObs
    physical_curve = circle_obs_c_vec(q) + ...
        circle_obs_R_vec(q)*(cos(th)+1i*sin(th));
    safety_curve = circle_obs_c_vec(q) + ...
        circle_R_safe_vec(q)*(cos(th)+1i*sin(th));

    obs_only_x = [obs_only_x; real(physical_curve(:)); real(safety_curve(:))]; 
    obs_only_y = [obs_only_y; imag(physical_curve(:)); imag(safety_curve(:))]; 
end

for r = 1:rect_n_cover
    cover_curve = rect_obs_c_vec(r) + ...
        rect_obs_R_vec(r)*(cos(th)+1i*sin(th));
    cover_safe_curve = rect_obs_c_vec(r) + ...
        rect_R_safe_vec(r)*(cos(th)+1i*sin(th));

    obs_only_x = [obs_only_x; real(cover_curve(:)); real(cover_safe_curve(:))]; 
    obs_only_y = [obs_only_y; imag(cover_curve(:)); imag(cover_safe_curve(:))]; 
end

obs_only_x = [obs_only_x; rect_x_only(:)];
obs_only_y = [obs_only_y; rect_y_only(:)];

obs_pad = 2.0;
xlim([min(obs_only_x)-obs_pad,max(obs_only_x)+obs_pad]);
ylim([min(obs_only_y)-obs_pad,max(obs_only_y)+obs_pad]);

xlabel('x');
ylabel('y');
title('Obstacle layout: physical and safety boundaries');
legend('Location','best');

obstacle_layout_name = 'obstacle_layout_physical_and_safe.png';
print(fig_obs_only,obstacle_layout_name,'-dpng','-r300');

fprintf('\nObstacle-only layout saved to: %s\n',obstacle_layout_name);
fprintf('Transient snapshots saved in folder: %s\n',snapshot_dir);

fprintf('Total code runtime: %.2f s\n',toc(overall_tic));
fprintf('============================================================\n');

%% ===================== Local functions =====================
function PRE = build_precomputed_data( ...
    N,k,h,phi_d,wbar,rho_v,rho_c,rho_tau, ...
    exclude_closed_endpoint,use_active_set_warm_start)

    nU = N+1;
    nVel = 2*nU;
    nSlack = nU;
    nZ = nVel+nSlack;

    PRE = struct();
    PRE.nU = nU;
    PRE.nVel = nVel;
    PRE.nSlack = nSlack;
    PRE.nZ = nZ;
    PRE.one_vec = ones(nU,1);
    PRE.wbar = wbar;
    PRE.zero_slack = zeros(nSlack,1);
    PRE.use_active_set_warm_start = use_active_set_warm_start;

    % ----- Constant Neumann semidiscrete PDE matrix -----
    L0 = spalloc(nU,nU,3*nU);
    L0(1,1) = -2;
    L0(1,2) =  2;
    for i = 2:N
        L0(i,i-1) =  1;
        L0(i,i)   = -2;
        L0(i,i+1) =  1;
    end
    L0(nU,nU-1) =  2;
    L0(nU,nU)   = -2;

    PRE.L = (k/h^2)*L0;
    PRE.L_phi = PRE.L*phi_d;

    % ----- Constant QP objective matrices -----
    W = diag(wbar);
    Wv = kron(W,eye(2));

    C_center = zeros(2,nVel);
    C_center(1,1:2:end) = wbar.';
    C_center(2,2:2:end) = wbar.';

    H_v = rho_v*Wv + rho_c*(C_center.'*C_center);
    H = blkdiag(H_v,rho_tau*W);
    PRE.H = full(0.5*(H+H.'));

    Rep = kron(ones(nU,1),eye(2));
    PRE.Rep = Rep;
    PRE.P_vnom = rho_v*(Wv*Rep) + rho_c*C_center.';
    PRE.lb = [-inf(nVel,1);zeros(nSlack,1)];

    % ----- Constant list of constrained pairs -----
    pair_i = zeros(nU*(nU-1)/2,1);
    pair_j = zeros(nU*(nU-1)/2,1);
    kk = 0;
    for i = 1:nU-1
        for j = i+1:nU
            if exclude_closed_endpoint && i == 1 && j == nU
                continue;
            end
            kk = kk+1;
            pair_i(kk) = i;
            pair_j(kk) = j;
        end
    end
    pair_i = pair_i(1:kk);
    pair_j = pair_j(1:kk);

    PRE.pair_i = pair_i;
    PRE.pair_j = pair_j;
    PRE.nPairs = kk;
    PRE.pair_ref_dir = phi_d(pair_i)-phi_d(pair_j);

    PRE.qp_options_active = optimoptions('quadprog', ...
        'Algorithm','active-set','Display','off');
    PRE.qp_options_robust = optimoptions('quadprog', ...
        'Algorithm','interior-point-convex','Display','off');
end

function w = trap_weights(nU)
    if nU == 1
        w = 1;
    else
        w = [1;2*ones(nU-2,1);1];
    end
end

function reset_tau_sign_state(nU,nGroups)
    global TAU_SIGN_STATE
    TAU_SIGN_STATE = struct();
    TAU_SIGN_STATE.nU = nU;
    TAU_SIGN_STATE.nGroups = nGroups;
    TAU_SIGN_STATE.sigma = zeros(nU,nGroups);
end

function [tau,sigma,s_vel] = tangent_from_tau0_update( ...
    tau0,preferred_step_vel,group_id,tau_sign_eps)

    global TAU_SIGN_STATE

    nU = length(tau0);
    nz = abs(tau0) > 1e-12;

    s_vel = zeros(nU,1);
    s_vel(nz) = real(conj(tau0(nz)).*preferred_step_vel(nz));

    if isempty(TAU_SIGN_STATE) || ~isfield(TAU_SIGN_STATE,'sigma') || ...
            size(TAU_SIGN_STATE.sigma,1) ~= nU || ...
            size(TAU_SIGN_STATE.sigma,2) < group_id
        sigma_prev = zeros(nU,1);
    else
        sigma_prev = TAU_SIGN_STATE.sigma(:,group_id);
    end

    sigma = sigma_prev;
    sigma(s_vel >  tau_sign_eps) =  1;
    sigma(s_vel < -tau_sign_eps) = -1;

    dead = nz & abs(s_vel) <= tau_sign_eps;
    sigma(dead & sigma_prev == 0) = 1;
    sigma(~nz) = 0;

    TAU_SIGN_STATE.sigma(:,group_id) = sigma;
    tau = sigma.*tau0;
end

function v = solve_circles_rectangle_split_discrete_cbf_qp( ...
    u,c_now,cdot_now,g_base,PRE, ...
    circle_obs_c_vec,circle_R_safe_vec, ...
    rect_obs_c_vec,rect_R_safe_vec,h_act,k_c, ...
    tau_sign_eps,tau_lb0, ...
    pair_safe_dist,pair_margin, ...
    use_predictive_obstacle_activation, ...
    t_now,v_max,speed_limit_start_time,speed_limit_n_dirs)

    global QP_WARM SIM_STATS

    nU = PRE.nU;
    nVel = PRE.nVel;
    nZ = PRE.nZ;
    dt = PRE.dt;
    nCircleObs = numel(circle_obs_c_vec);
    nRect = numel(rect_obs_c_vec);
    rect_group_id = nCircleObs + 1;

    %% ---------- Current nominal center velocity ----------
    e_c = PRE.wbar.'*u-c_now;
    v_nom = cdot_now-k_c*e_c;
    c_nom = [real(v_nom);imag(v_nom)];
    f_obj = [-PRE.P_vnom*c_nom;PRE.zero_slack];

    
    preferred_step_vel = g_base + PRE.one_vec*v_nom;

    if use_predictive_obstacle_activation
        u_nom_next = u + dt*preferred_step_vel;
    else
        u_nom_next = [];
    end

    %% ---------- Standalone circles: geometry / activation / tangents ----------
    d_circle = complex(zeros(nU,nCircleObs));
    dist_circle = zeros(nU,nCircleObs);
    h_circle = zeros(nU,nCircleObs);
    active_circle = false(nU,nCircleObs);
    tau_circle = complex(zeros(nU,nCircleObs));
    n_circle_active_total = 0;

    for q = 1:nCircleObs
        d = u-circle_obs_c_vec(q);
        dist = abs(d);
        hq = dist.^2-circle_R_safe_vec(q)^2;

        active = (hq <= h_act) & (dist > 1e-12);
        if use_predictive_obstacle_activation
            h_nom = abs(u_nom_next-circle_obs_c_vec(q)).^2-circle_R_safe_vec(q)^2;
            active = active | ((h_nom <= h_act) & (dist > 1e-12));
        end

        tau0 = complex(zeros(nU,1));
        nz = dist > 1e-12;
        tau0(nz) = 1i*d(nz)./dist(nz);
        tau_q = tangent_from_tau0_update( ...
            tau0,preferred_step_vel,q,tau_sign_eps);

        d_circle(:,q) = d;
        dist_circle(:,q) = dist;
        h_circle(:,q) = hq;
        active_circle(:,q) = active;
        tau_circle(:,q) = tau_q;
        n_circle_active_total = n_circle_active_total + nnz(active);
    end

    %% ---------- Rectangle cover circles: hard normals + one composite tangent ----------
    d_rect = complex(zeros(nU,nRect));
    dist_rect = zeros(nU,nRect);
    h_rect = zeros(nU,nRect);
    active_rect = false(nU,nRect);

    for r = 1:nRect
        d = u-rect_obs_c_vec(r);
        dist = abs(d);
        hr = dist.^2-rect_R_safe_vec(r)^2;

        active = (hr <= h_act) & (dist > 1e-12);
        if use_predictive_obstacle_activation
            h_nom = abs(u_nom_next-rect_obs_c_vec(r)).^2-rect_R_safe_vec(r)^2;
            active = active | ((h_nom <= h_act) & (dist > 1e-12));
        end

        d_rect(:,r) = d;
        dist_rect(:,r) = dist;
        h_rect(:,r) = hr;
        active_rect(:,r) = active;
    end

    n_rect_normal_active = nnz(active_rect);
    rect_agent_active = any(active_rect,2);
    rect_active_agents = find(rect_agent_active);
    n_rect_tangent_active = numel(rect_active_agents);

    
    nbar_rect = complex(zeros(nU,1));
    for i = rect_active_agents.'
        rr = find(active_rect(i,:));
        nloc = complex(zeros(numel(rr),1));
        hloc = h_rect(i,rr).';
        for kk = 1:numel(rr)
            r = rr(kk);
            if dist_rect(i,r) > 1e-12
                nloc(kk) = d_rect(i,r)/dist_rect(i,r);
            else
                nloc(kk) = 1+0i;
            end
        end

        hmin = min(hloc);
        weights = 1./(hloc-hmin+1e-3);
        nsum = sum(weights.*nloc);
        if abs(nsum) <= 1e-12
            [~,kkmin] = min(hloc);
            nbar_rect(i) = nloc(kkmin);
        else
            nbar_rect(i) = nsum/abs(nsum);
        end
    end

    tau0_rect = 1i*nbar_rect;
    tau_rect = tangent_from_tau0_update( ...
        tau0_rect,preferred_step_vel,rect_group_id,tau_sign_eps);

    %% ---------- ALL-PAIR pairwise discrete CBF ----------
    d_pair = u(PRE.pair_i)-u(PRE.pair_j);
    dist_pair = abs(d_pair);
    pair_idx = (1:PRE.nPairs).';
    n_pair_active = PRE.nPairs;

    %% ---------- HARD actual-speed limit |q_i| <= v_max ----------
    speed_limit_active = (t_now >= speed_limit_start_time);
    if speed_limit_active
        n_speed_rows = nU*speed_limit_n_dirs;
    else
        n_speed_rows = 0;
    end

   
    nRows = 2*n_circle_active_total + n_rect_normal_active + ...
            n_rect_tangent_active + n_pair_active + n_speed_rows;

    SIM_STATS.quadprog_calls = SIM_STATS.quadprog_calls + 1;
    A = zeros(nRows,nZ);
    b = zeros(nRows,1);
    row = 0;

    %% ---------- HARD discrete CBFs: standalone circular obstacles ----------
    for q = 1:nCircleObs
        idx = find(active_circle(:,q));
        for aa = 1:numel(idx)
            i = idx(aa);
            row = row+1;

            n_hat = d_circle(i,q)/dist_circle(i,q);
            rhs_n = -PRE.eta_obs*h_circle(i,q)/(2*dist_circle(i,q)*dt) ...
                    - real(conj(n_hat)*g_base(i));

            A(row,2*i-1) = -real(n_hat);
            A(row,2*i)   = -imag(n_hat);
            b(row) = -rhs_n;
        end
    end

    %% ---------- SOFT tangential constraints: standalone circles ----------
    for q = 1:nCircleObs
        idx = find(active_circle(:,q));
        for aa = 1:numel(idx)
            i = idx(aa);
            row = row+1;

            tau_i = tau_circle(i,q);
            rhs_t = tau_lb0-real(conj(tau_i)*g_base(i));

            A(row,2*i-1) = -real(tau_i);
            A(row,2*i)   = -imag(tau_i);
            A(row,nVel+i) = -1;
            b(row) = -rhs_t;
        end
    end

    %% ---------- HARD discrete CBFs: rectangle cover circles ----------
    for r = 1:nRect
        idx = find(active_rect(:,r));
        for aa = 1:numel(idx)
            i = idx(aa);
            row = row+1;

            n_hat = d_rect(i,r)/dist_rect(i,r);
            rhs_n = -PRE.eta_obs*h_rect(i,r)/(2*dist_rect(i,r)*dt) ...
                    - real(conj(n_hat)*g_base(i));

            A(row,2*i-1) = -real(n_hat);
            A(row,2*i)   = -imag(n_hat);
            b(row) = -rhs_n;
        end
    end

    %% ---------- ONE SOFT composite tangential constraint for the rectangle ----------
    for aa = 1:n_rect_tangent_active
        i = rect_active_agents(aa);
        row = row+1;

        tau_i = tau_rect(i);
        rhs_t = tau_lb0-real(conj(tau_i)*g_base(i));

        A(row,2*i-1) = -real(tau_i);
        A(row,2*i)   = -imag(tau_i);
        A(row,nVel+i) = -1;
        b(row) = -rhs_t;
    end

    %% ---------- HARD ALL-PAIR pairwise DISCRETE CBF ----------
    for aa = 1:n_pair_active
        qpair = pair_idx(aa);
        i = PRE.pair_i(qpair);
        j = PRE.pair_j(qpair);
        row = row+1;

        dist_ij = dist_pair(qpair);
        h_ij = dist_ij^2-pair_safe_dist^2;

        if dist_ij <= 1e-12
            ref_dir = PRE.pair_ref_dir(qpair);
            if abs(ref_dir) > 1e-12
                n_ij = ref_dir/abs(ref_dir);
            else
                n_ij = 1+0i;
            end
            dist_eff = pair_safe_dist;
        else
            n_ij = d_pair(qpair)/dist_ij;
            dist_eff = dist_ij;
        end

        g_rel = g_base(i)-g_base(j);
        rhs_pair = -PRE.eta_pair*h_ij/(2*dist_eff*dt) ...
                   + pair_margin ...
                   - real(conj(n_ij)*g_rel);

        A(row,2*i-1) = -real(n_ij);
        A(row,2*i)   = -imag(n_ij);
        A(row,2*j-1) =  real(n_ij);
        A(row,2*j)   =  imag(n_ij);
        b(row) = -rhs_pair;
    end

    %% ---------- HARD actual-agent-speed constraints ----------
    if speed_limit_active
        Mdir = speed_limit_n_dirs;
        speed_rhs = v_max*cos(pi/Mdir);
        theta_speed = 2*pi*(0:Mdir-1)/Mdir;

        for i = 1:nU
            for kk = 1:Mdir
                row = row+1;
                cth = cos(theta_speed(kk));
                sth = sin(theta_speed(kk));

                A(row,2*i-1) = cth;
                A(row,2*i)   = sth;
                b(row) = speed_rhs - ...
                    (cth*real(g_base(i)) + sth*imag(g_base(i)));
            end
        end
    end

    if row ~= nRows
        error('Internal QP row count mismatch: assembled %d rows, expected %d.',row,nRows);
    end

    %% ---------- Solve one QP ----------
    if isempty(QP_WARM) || numel(QP_WARM) ~= nZ || any(~isfinite(QP_WARM))
        QP_WARM = [PRE.Rep*c_nom;zeros(PRE.nSlack,1)];
    end

    z = [];
    exitflag = -Inf;

    if PRE.use_active_set_warm_start
        [z,~,exitflag] = quadprog( ...
            PRE.H,f_obj,A,b,[],[],PRE.lb,[],QP_WARM,PRE.qp_options_active);
    end

    if ~PRE.use_active_set_warm_start || isempty(z) || exitflag <= 0
        SIM_STATS.qp_fallback_calls = SIM_STATS.qp_fallback_calls + 1;
        [z,~,exitflag] = quadprog( ...
            PRE.H,f_obj,A,b,[],[],PRE.lb,[],[],PRE.qp_options_robust);
    end

    if isempty(z) || exitflag <= 0
        error(['DISCRETE CBF-QP failed or became infeasible. ', ...
               'Obstacle-normal and all-pair constraints are hard constraints.']);
    end

    max_violation = max(A*z-b);
    if max_violation > 1e-7
        error('QP inequality violation is too large: %.3e',max_violation);
    end

    QP_WARM = z;
    v = z(1:2:nVel) + 1i*z(2:2:nVel);
end

function hmin = min_obstacle_barrier_primitives(u,obs_c_vec,R_safe_vec)
    hmin = inf;
    for q = 1:numel(obs_c_vec)
        hmin = min(hmin,min(abs(u-obs_c_vec(q)).^2-R_safe_vec(q)^2));
    end
end

function hmin = min_pairwise_barrier_fast(u,pair_safe_dist,pair_i,pair_j)
    if isempty(pair_i)
        hmin = NaN;
        return;
    end

    d_pair = u(pair_i)-u(pair_j);
    hmin = min(abs(d_pair).^2-pair_safe_dist^2);
end

function draw_circles_rectangle_scene( ...
    u_now,u_ref_now, ...
    circle_obs_c_vec,circle_obs_R_vec,circle_R_safe_vec, ...
    rect_x_min,rect_x_max,rect_y_min,rect_y_max, ...
    rect_obs_c_vec,rect_obs_R_vec,rect_R_safe_vec, ...
    th,x_lim,y_lim)

    hold on; grid on; axis equal;

    plot(real(u_ref_now),imag(u_ref_now),'r--', ...
         'LineWidth',1.8,'DisplayName','Reference formation');

    
    u_plot = u_now(1:end-1);
    plot(real(u_plot),imag(u_plot),'bo', ...
         'LineStyle','none','LineWidth',1.2,'MarkerSize',5, ...
         'DisplayName','Current agents');

    
    for q = 1:numel(circle_obs_c_vec)
        obs_curve = circle_obs_c_vec(q)+circle_obs_R_vec(q)*(cos(th)+1i*sin(th));
        safe_curve = circle_obs_c_vec(q)+circle_R_safe_vec(q)*(cos(th)+1i*sin(th));
        if q == 1
            plot(real(obs_curve),imag(obs_curve),'m-', ...
                 'LineWidth',2.0,'DisplayName','Circular obstacles');
            plot(real(safe_curve),imag(safe_curve),'m--', ...
                 'LineWidth',1.3,'DisplayName','Circle safety boundaries');
        else
            plot(real(obs_curve),imag(obs_curve),'m-', ...
                 'LineWidth',2.0,'HandleVisibility','off');
            plot(real(safe_curve),imag(safe_curve),'m--', ...
                 'LineWidth',1.3,'HandleVisibility','off');
        end
    end

    
    rect_x = [rect_x_min, rect_x_max, rect_x_max, rect_x_min, rect_x_min];
    rect_y = [rect_y_min, rect_y_min, rect_y_max, rect_y_max, rect_y_min];
    plot(rect_x,rect_y,'-', ...
         'Color',[0.85 0.33 0.10],'LineWidth',2.2, ...
         'DisplayName','Rectangular obstacle');

    
    for r = 1:numel(rect_obs_c_vec)
        cover_curve = rect_obs_c_vec(r)+rect_obs_R_vec(r)*(cos(th)+1i*sin(th));
        safe_curve = rect_obs_c_vec(r)+rect_R_safe_vec(r)*(cos(th)+1i*sin(th));
        if r == 1
            plot(real(cover_curve),imag(cover_curve),'-', ...
                 'Color',[0.93 0.55 0.18],'LineWidth',0.8, ...
                 'DisplayName','Rectangle cover circles');
            plot(real(safe_curve),imag(safe_curve),'--', ...
                 'Color',[0.93 0.55 0.18],'LineWidth',0.8, ...
                 'DisplayName','Rectangle cover safety boundaries');
        else
            plot(real(cover_curve),imag(cover_curve),'-', ...
                 'Color',[0.93 0.55 0.18],'LineWidth',0.8, ...
                 'HandleVisibility','off');
            plot(real(safe_curve),imag(safe_curve),'--', ...
                 'Color',[0.93 0.55 0.18],'LineWidth',0.8, ...
                 'HandleVisibility','off');
        end
    end

    xlim(x_lim);
    ylim(y_lim);
    xlabel('x');
    ylabel('y');
end
