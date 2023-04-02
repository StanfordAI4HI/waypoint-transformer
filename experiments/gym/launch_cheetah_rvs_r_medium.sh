#!/usr/bin/env bash

config="experiments/config/d4rl/gym_rvs_r_medium.cfg"
# declare -a envs=("halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-expert-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
declare -a envs=("halfcheetah-medium-v2")
use_gpu=true
seeds=5

for env in "${envs[@]}"; do
  for seed in $(seq 1 $((seeds))); do
      python src/rvs/train.py --configs "$config" --env_name "$env" --seed "$seed" --use_gpu
  done
done
wait
