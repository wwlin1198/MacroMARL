#!/bin/bash


# Function to handle the SIGINT signal (Ctrl+C)
cleanup() {
    echo "Caught SIGINT, stopping all child processes..."
    kill $(jobs -p)
    exit 1
}

# Set the trap to call the cleanup function on SIGINT
trap cleanup SIGINT

# Box Pushing 6x6
for ((i=0; i<20; i++))
do
    pg_based_main.py --save_dir='ma_iac_bp6' \
                    --alg='MacIAC' \
                    --env_id='BP-MA-v0' \
                    --n_agent=2 \
                    --env_terminate_step=100 \
                    --big_box_reward=300 \
                    --a_lr=0.0005 \
                    --c_lr=0.003 \
                    --train_freq=32 \
                    --n_env=32 \
                    --c_target_update_freq=32 \
                    --n_step_TD=0 \
                    --grad_clip_norm=0 \
                    --eps_start=1.0 \
                    --eps_end=0.01 \
                    --eps_stable_at=4_000 \
                    --total_epi=40_000 \
                    --grid_dim 6 6 \
                    --gamma=0.98 \
                    --eval_policy \
                    --sample_epi \
                    --run_id=$i &
done

# Box Pushing 8x8
for ((i=0; i<20; i++))
do
    pg_based_main.py --save_dir='ma_iac_bp8' \
                    --alg='MacIAC' \
                    --env_id='BP-MA-v0' \
                    --n_agent=2 \
                    --env_terminate_step=100 \
                    --big_box_reward=300 \
                    --a_lr=0.0005 \
                    --c_lr=0.003 \
                    --train_freq=48 \
                    --n_env=48 \
                    --c_target_update_freq=48 \
                    --n_step_TD=3 \
                    --grad_clip_norm=0 \
                    --eps_start=1.0 \
                    --eps_end=0.01 \
                    --eps_stable_at=4_000 \
                    --total_epi=40_000 \
                    --grid_dim 8 8 \
                    --gamma=0.98 \
                    --eval_policy \
                    --sample_epi \
                    --run_id=$i &
done

wait