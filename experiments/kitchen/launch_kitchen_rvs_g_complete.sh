#!/usr/bin/env bash

config="experiments/config/d4rl/kitchen_rvs_g.cfg"
# declare -a envs=("kitchen-complete-v0" "kitchen-mixed-v0" "kitchen-partial-v0")
declare -a envs=("kitchen-complete-v0")
seeds=5
use_gpu=true

for env in "${envs[@]}"; do
  for seed in $(seq 1 $((seeds))); do
      python src/rvs/train.py --configs "$config" --env_name "$env" --seed "$seed" --use_gpu
  done
done
wait
