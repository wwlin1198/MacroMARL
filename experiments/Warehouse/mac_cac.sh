#!/bin/bash

# Function to handle the SIGINT signal (Ctrl+C)
cleanup() {
    echo "Caught SIGINT, stopping all child processes..."
    kill $(jobs -p)
    exit 1
}

# Set the trap to call the cleanup function on SIGINT
trap cleanup SIGINT

# Warehouse-A
for ((i=0; i<20; i++))
do
    pg_based_main.py --save_dir='ma_cac_warehouse_A' \
                    --alg='MacCAC' \
                    --run_id=$i \
                    --env_id='OSD-D-v7' \
                    --n_agent=3 \
                    --env_terminate_step=200 \
                    --a_lr=0.0003 \
                    --c_lr=0.003 \
                    --train_freq=4 \
                    --n_env=4 \
                    --c_target_update_freq=32 \
                    --n_step_TD=5 \
                    --grad_clip_norm=0 \
                    --eps_start=1.0 \
                    --eps_end=0.05 \
                    --eps_stable_at=10_000 \
                    --total_epi=40_000 \
                    --gamma=1.0 \
                    --a_rnn_layer_size=64 \
                    --c_rnn_layer_size=64 \
                    --h0_speed_ps 27 20 20 20 \
                    --h1_speed_ps 27 20 20 20 \
                    --d_pen=-20.0 \
                    --tb_m_speed=0.8 \
                    --sample_epi \
                    --eval_policy &
done

# Warehouse-B
for ((i=0; i<20; i++))
do
    pg_based_main.py --save_dir='ma_cac_warehouse_B' \
                    --alg='MacCAC' \
                    --run_id=$i \
                    --env_id='OSD-D-v7' \
                    --n_agent=3 \
                    --env_terminate_step=200 \
                    --a_lr=0.0003 \
                    --c_lr=0.003 \
                    --train_freq=4 \
                    --n_env=4 \
                    --c_target_update_freq=16 \
                    --n_step_TD=5 \
                    --grad_clip_norm=0 \
                    --eps_start=1.0 \
                    --eps_end=0.05 \
                    --eps_stable_at=10_000 \
                    --total_epi=40_000 \
                    --gamma=1.0 \
                    --a_rnn_layer_size=64 \
                    --c_rnn_layer_size=64 \
                    --h0_speed_ps 18 15 15 15 \
                    --h1_speed_ps 48 18 15 15 \
                    --d_pen=-20.0 \
                    --tb_m_speed=0.8 \
                    --sample_epi \
                    --eval_policy &
done

# Warehouse-C
for ((i=0; i<20; i++))
do
    pg_based_main.py --save_dir='ma_cac_warehouse_C' \
                    --alg='MacCAC' \
                    --run_id=$i \
                    --env_id='OSD-T-v0' \
                    --n_agent=4 \
                    --env_terminate_step=250 \
                    --a_lr=0.0005 \
                    --c_lr=0.001 \
                    --train_freq=8 \
                    --n_env=8 \
                    --c_target_update_freq=64 \
                    --n_step_TD=5 \
                    --grad_clip_norm=0 \
                    --eps_start=1.0 \
                    --eps_end=0.01 \
                    --eps_stable_at=10_000 \
                    --total_epi=80_000 \
                    --gamma=1.0 \
                    --a_rnn_layer_size=64 \
                    --c_rnn_layer_size=64 \
                    --h0_speed_ps 40 40 40 40 \
                    --h1_speed_ps 40 40 40 40 \
                    --h2_speed_ps 40 40 40 40 \
                    --d_pen=-20.0 \
                    --tb_m_speed=0.8 \
                    --sample_epi \
                    --eval_policy & 
done

# Warehouse-D
for ((i=0; i<20; i++))
do
    pg_based_main.py --save_dir='ma_cac_warehouse_D' \
                    --alg='MacCAC' \
                    --run_id=$i \
                    --env_id='OSD-T-v1' \
                    --n_agent=4 \
                    --env_terminate_step=250 \
                    --a_lr=0.0003 \
                    --c_lr=0.003 \
                    --train_freq=8 \
                    --n_env=8 \
                    --c_target_update_freq=64 \
                    --n_step_TD=3 \
                    --grad_clip_norm=0 \
                    --eps_start=1.0 \
                    --eps_end=0.01 \
                    --eps_stable_at=10_000 \
                    --total_epi=80_000 \
                    --gamma=1.0 \
                    --a_rnn_layer_size=64 \
                    --c_rnn_layer_size=64 \
                    --h0_speed_ps 38 38 38 38 \
                    --h1_speed_ps 38 38 38 38 \
                    --h2_speed_ps 27 27 27 27 \
                    --d_pen=-20.0 \
                    --tb_m_speed=0.8 \
                    --sample_epi \
                    --eval_policy & 
done

# Warehouse-E
for ((i=0; i<20; i++))
do
    pg_based_main.py --save_dir='ma_cac_warehouse_E' \
                    --alg='MacCAC' \
                    --run_id=$i \
                    --env_id='OSD-F-v0' \
                    --n_agent=4 \
                    --env_terminate_step=300 \
                    --a_lr=0.0003 \
                    --c_lr=0.003 \
                    --train_freq=8 \
                    --n_env=8 \
                    --c_target_update_freq=64 \
                    --n_step_TD=3 \
                    --grad_clip_norm=0 \
                    --eps_start=1.0 \
                    --eps_end=0.01 \
                    --eps_stable_at=10_000 \
                    --total_epi=100_000 \
                    --gamma=1.0 \
                    --a_rnn_layer_size=64 \
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

# Ablation
for ((i=0; i<20; i++))
do
    pg_based_main.py --save_dir='ma_cac_warehouse_A_ablation' \
                    --alg='MacCAC' \
                    --run_id=$i \
                    --env_id='OSD-D-v7' \
                    --n_agent=3 \
                    --env_terminate_step=200 \
                    --a_lr=0.0005 \
                    --c_lr=0.001 \
                    --train_freq=8 \
                    --n_env=8 \
                    --c_target_update_freq=64 \
                    --n_step_TD=5 \
                    --grad_clip_norm=0 \
                    --eps_start=1.0 \
                    --eps_end=0.05 \
                    --eps_stable_at=10_000 \
                    --total_epi=40_000 \
                    --gamma=1.0 \
                    --a_rnn_layer_size=64 \
                    --c_rnn_layer_size=64 \
                    --h0_speed_ps 18 18 18 18 \
                    --h1_speed_ps 18 18 18 18 \
                    --d_pen=-20.0 \
                    --tb_m_speed=0.8 \
                    --sample_epi \
                    --eval_policy &
done

wait