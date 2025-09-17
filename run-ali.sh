#!/bin/bash

python train_ali.py --config configs/single_cell/50D/multi-50D.yaml   && \
python train_ali.py --config configs/single_cell/50D/cite-50D.yaml    && \
python train_ali.py --config configs/single_cell/100D/multi-100D.yaml && \
python train_ali.py --config configs/single_cell/100D/cite-100D.yaml