#!/bin/bash

# Function to handle the SIGINT signal (Ctrl+C)
cleanup() {
    echo "Caught SIGINT, stopping all child processes..."
    kill $(jobs -p)
    exit 1
}

# Set the trap to call the cleanup function on SIGINT
trap cleanup SIGINT

# Overcooked-A
for ((i=0; i<20; i++))
do
    pg_based_main.py  --save_dir='mac_iac_overcooked_A' \
                                --alg='MacIAC' \
                                --env_id='Overcooked-MA-v1' \
                                --n_agent=3 \
                                --env_terminate_step=200 \
                                --a_lr=0.0003 \
                                --c_lr=0.003 \
                                --train_freq=4 \
                                --n_env=4 \
                                --c_target_update_freq=8 \
                                --n_step_TD=5 \
                                --grad_clip_norm=0 \
                                --eps_start=1.0 \
                                --eps_end=0.05 \
                                --eps_stable_at=20_000 \
                                --total_epi=100_000 \
                                --grid_dim 7 7 \
                                --gamma=0.99 \
                                --eval_policy \
                                --sample_epi \
                                --run_id=$i \
                                --task=6 \
                                --map_type=A \
                                --step_penalty=-0.1 & 
done

# Overcooked-B
for ((i=0; i<20; i++))
do
    pg_based_main.py  --save_dir='mac_iac_overcooked_B' \
                                --alg='MacIAC' \
                                --env_id='Overcooked-MA-v1' \
                                --n_agent=3 \
                                --env_terminate_step=200 \
                                --a_lr=0.0003 \
                                --c_lr=0.003 \
                                --train_freq=4 \
                                --n_env=4 \
                                --c_target_update_freq=8 \
                                --n_step_TD=5 \
                                --grad_clip_norm=0 \
                                --eps_start=1.0 \
                                --eps_end=0.05 \
                                --eps_stable_at=20_000 \
                                --total_epi=120_000 \
                                --grid_dim 7 7 \
                                --gamma=0.99 \
                                --eval_policy \
                                --sample_epi \
                                --run_id=$i \
                                --task=6 \
                                --map_type=B \
                                --step_penalty=-0.1 &
done

# Overcooked-C
for ((i=0; i<20; i++))
do
    pg_based_main.py  --save_dir='mac_iac_overcooked_C' \
                                --alg='MacIAC' \
                                --env_id='Overcooked-MA-v1' \
                                --n_agent=3 \
                                --env_terminate_step=200 \
                                --a_lr=0.0003 \
                                --c_lr=0.003 \
                                --train_freq=8 \
                                --n_env=8 \
                                --c_target_update_freq=32 \
                                --n_step_TD=5 \
                                --grad_clip_norm=0 \
                                --eps_start=1.0 \
                                --eps_end=0.05 \
                                --eps_stable_at=20_000 \
                                --total_epi=100_000 \
                                --grid_dim 7 7 \
                                --gamma=0.99 \
                                --eval_policy \
                                --sample_epi \
                                --run_id=$i \
                                --task=6 \
                                --map_type=C \
                                --step_penalty=-0.1 &
done

wait