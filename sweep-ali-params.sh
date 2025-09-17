#!/bin/bash

DIM=$1
DATASET=$2
DEVICE=$3

python train_ali.py --config configs/single_cell/${DIM}D/${DATASET}-${DIM}D.yaml wandb_name=ablate_run_1 \
    seed_list='[42,]' device=${DEVICE} correct_coeff=1e-9 land_gamma=0.0005 land_t_gamma=0.0005 && \

python train_ali.py --config configs/single_cell/${DIM}D/${DATASET}-${DIM}D.yaml wandb_name=ablate_run_2 \
    seed_list='[42,]' device=${DEVICE} correct_coeff=1e-9 land_gamma=0.0001 land_t_gamma=0.0001 && \

python train_ali.py --config configs/single_cell/${DIM}D/${DATASET}-${DIM}D.yaml wandb_name=ablate_run_3 \
    seed_list='[42,]' device=${DEVICE} correct_coeff=5e-10 land_gamma=0.0005 land_t_gamma=0.0005 && \

python train_ali.py --config configs/single_cell/${DIM}D/${DATASET}-${DIM}D.yaml wandb_name=ablate_run_4 \
    seed_list='[42,]' device=${DEVICE} correct_coeff=5e-10 land_gamma=0.0001 land_t_gamma=0.0001
