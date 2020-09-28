#!/bin/bash

python src/main_sac.py \
--experiment_name=test \
--env=Point2PointEnv-v5 \
--alg=sac \
--policy=Controller \
--port=6030 \
--static_horizon=1 \
--include_current_state=true \
--include_target_state=false \
--include_distance=true \
--include_time=true \
--include_current_excitations=false \
--include_perceptual_width=false \
--include_acceleration=false \
--include_time=false \
--include_velocity=false \
--include_target_width=false \
--wait_action=0.05 \
--incremental_actions=false \
--zero_excitations_on_reset=true \
--reward=terminal \
--reset_step=10 \
--eval_episode=10 \
--use_wandb=false \
--goal_reward=5 \
--goal_threshold=2 \
--eval_interval=100 \
--save_interval=100 \
--test=false \
--init_artisynth=false \

#--load_path=/home/praneethsv/Pictures/artisynth-rl/results/Point2PointEnv-v5/p2p-1d-sac-pytorch-iros-baseline_no_eb/trained/agent \


