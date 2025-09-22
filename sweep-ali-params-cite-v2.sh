#!/bin/bash

DIM=50
DATASET=cite
DEVICE=mps

#python train_ali.py --config configs/single_cell/${DIM}D/${DATASET}-${DIM}D.yaml wandb_name=linear-l2 \
#    seed_list='[42,]' device=${DEVICE} correct_coeff=1e-10 land_gamma=0.0001 land_t_gamma=0.0001 reg_term_type=linear metric=l2 && \

python train_ali.py --config configs/single_cell/${DIM}D/${DATASET}-${DIM}D.yaml wandb_name=piecewise-l2 \
    seed_list='[42,]' device=${DEVICE} correct_coeff=1e-10 land_gamma=0.0001 land_t_gamma=0.0001 reg_term_type=piecewise metric=l2 && \

python train_ali.py --config configs/single_cell/${DIM}D/${DATASET}-${DIM}D.yaml wandb_name=2nd-der-l2 \
    seed_list='[42,]' device=${DEVICE} correct_coeff=1e-10 land_gamma=0.0001 land_t_gamma=0.0001 reg_term_type=2nd_derivative metric=l2
