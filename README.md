# Waypoint Transformer: Reinforcement Learning via Supervised Learning with Intermediate Targets

This repository is adapted based on the RvS repository, as provided by Emmons et al. (2021). The link to the original repository is: https://github.com/scottemmons/rvs.

# Installation

The code depends on MuJoCo 2.1.0 (for mujoco-py) and MuJoCo 2.1.1 (for dm-control). Here are [instructions for installing MuJoCo 2.1.0](https://github.com/openai/mujoco-py/tree/fb4babe73b1ef18b4bea4c6f36f6307e06335a2f#install-mujoco)
and [instructions for installing MuJoCo 2.1.1](https://github.com/deepmind/dm_control/tree/84fc2faa95ca2b354f3274bb3f3e0d29df7fb337#requirements-and-installation).

# Reproducing Experiments

The `experiments` directory contains a launch script for each task, consistent with the RvS repository.

To train a waypoint network on a particular environment, run:
```bash
bash experiments/launch_waypoint_nets.sh
```

Based on the trained waypoint network, choose a particular checkpoint to use for WT. For example, suppose we wanted to run the experiment for Kitchen Complete across 5 seeds:

```bash
GOAL_NETWORK_CKPT=kitchen_complete_goalwaypt_net/files/checkpoints/gcsl-kitchen-complete-v0-epoch=028-val_loss=3.9324e-04.ckpt bash experiments/launch_kitchen_rvs_g_complete.sh
```
