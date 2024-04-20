#!/bin/bash


# Function to handle the SIGINT signal (Ctrl+C)
cleanup() {
    echo "Caught SIGINT, stopping all child processes..."
    kill $(jobs -p)
    exit 1
}

# Set the trap to call the cleanup function on SIGINT
trap cleanup SIGINT


# Box Pushing 8x8
for ((i=0; i<1; i++))
do
    pg_based_main.py --save_dir='ma_ippo_bp6' \
                    --alg='MacIPPO' \
                    --env_id='BP-MA-v0' \
                    --n_agent=2 \
                    --env_terminate_step=100 \
                    --big_box_reward=300 \
                    --ippo_clip_value=0.1 \
                    --ippo_epochs=1 \
                    --a_lr=0.0004 \
                    --c_lr=0.004 \
                    --train_freq=48 \
                    --n_env=48 \
                    --c_target_update_freq=64 \
                    --n_step_TD=3 \
                    --grad_clip_norm=0 \
                    --eps_start=1.0 \
                    --eps_end=0.01 \
                    --eps_stable_at=4_000 \
                    --total_epi=40_000 \
                    --grid_dim 10 10 \
                    --gamma=0.98 \
                    --eval_policy \
                    --sample_epi \
                    --run_id=$i &
done

wait