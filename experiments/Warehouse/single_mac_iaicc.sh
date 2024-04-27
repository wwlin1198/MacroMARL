#!/bin/bash

# Function to handle the SIGINT signal (Ctrl+C)
cleanup() {
    echo "Caught SIGINT, stopping all child processes..."
    kill $(jobs -p)
    exit 1
}

# Set the trap to call the cleanup function on SIGINT
trap cleanup SIGINT


# Warehouse-B
for ((i=0; i<2; i++))
do
    pg_based_main.py --save_dir='ma_iaicc_warehouse_B' \
                    --alg='MacIAICC' \
                    --run_id=$i \
                    --env_id='OSD-D-v7' \
                    --n_agent=3 \
                    --l_mode=0 \
                    --env_terminate_step=200 \
                    --a_lr=0.0005 \
                    --c_lr=0.0005 \
                    --train_freq=4 \
                    --n_env=4 \
                    --c_target_update_freq=32 \
                    --n_step_TD=5 \
                    --grad_clip_norm=0 \
                    --eps_start=1.00 \
                    --eps_end=0.05 \
                    --eps_stable_at=10_000 \
                    --total_epi=40_000 \
                    --gamma=1.0 \
                    --a_rnn_layer_size=32 \
                    --c_rnn_layer_size=64 \
                    --h0_speed_ps 18 15 15 15 \
                    --h1_speed_ps 48 18 15 15 \
                    --d_pen=-20.0 \
                    --tb_m_speed=0.8 \
                    --sample_epi \
                    --eval_policy &
done


wait