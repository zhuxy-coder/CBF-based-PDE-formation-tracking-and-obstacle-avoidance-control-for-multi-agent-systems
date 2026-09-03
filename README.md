# CBF-based-PDE-formation-tracking-and-obstacle-avoidance-control-for-multi-agent-systems

# PDE-CBF Multi-Agent Formation Control

This repository contains the MATLAB simulation code for PDE-based multi-agent formation tracking and safe obstacle avoidance using a Control Barrier Function Quadratic Programming (CBF-QP) safety filter.

## Main Features

- Parabolic PDE-based formation control for large-scale multi-agent systems
- Dynamic formation-center trajectory tracking
- CBF-QP-based obstacle avoidance
- Tangential guidance for reducing agent stalling near obstacles
- Collision avoidance between agents
- Circular and composite rectangular obstacles
- Tracking-error and safety-function evaluation
- 9 transient formation snapshots and obstacle-avoidance video generation

## Requirements

- MATLAB
- Optimization Toolbox (`quadprog`)

## Usage

Run the main MATLAB script:

```matlab
CBF_obstacle.m
