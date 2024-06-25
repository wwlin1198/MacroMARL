#!/bin/bash

# Function to handle the SIGINT signal (Ctrl+C)
cleanup() {
    echo "Caught SIGINT, stopping all child processes..."
    kill $(jobs -p)
    exit 1
}

# Set the trap to call the cleanup function on SIGINT
trap cleanup SIGINT

# Warehouse-E
for ((i=0; i<3; i++))
do
    pg_based_main.py --save_dir='ma_iaicc_warehouse_E' \
                    --alg='MacIAICC' \
                    --run_id=$i \
                    --env_id='OSD-F-v0' \
                    --n_agent=4 \
                    --l_mode=0 \
                    --env_terminate_step=300 \
                    --a_lr=0.0003 \
                    --c_lr=0.003 \
                    --train_freq=8 \
                    --n_env=8 \
                    --c_target_update_freq=64 \
                    --n_step_TD=5 \
                    --grad_clip_norm=0 \
                    --eps_start=1.0 \
                    --eps_end=0.01 \
                    --eps_stable_at=10_000 \
                    --total_epi=50_000 \
                    --gamma=1.0 \
                    --a_rnn_layer_size=32 \
                    --c_rnn_layer_size=64 \
                    --h0_speed_ps 40 40 40 40 \
                    --h1_speed_ps 40 40 40 40 \
                    --h2_speed_ps 40 40 40 40 \
                    --h3_speed_ps 40 40 40 40 \
                    --d_pen=-20.0 \
                    --tb_m_speed=0.8 \
                    --sample_epi \
                    --eval_policy &
done

wait