#!/bin/bash

echo "=> Generate data.."

python3 data/generate_data.py \
  --dataset_name mnist \
  --n_clients 10 \
  --iid \
  --frac 1.0 \
  --save_dir data/mnist/ \
  --seed 1234

echo "=> Train.."

python3 "train.py" \
  --experiment "mnist" \
  --n_rounds 20 \
  --local_steps 1 \
  --local_optimizer sgd \
  --local_lr 0.001 \
  --server_optimizer sgd \
  --server_lr 0.1 \
  --bz 128 \
  --device "cpu" \
  --log_freq 1 \
  --verbose 1 \
  --logs_dir "logs/mnist/" \
  --seed 12 \
  --aggregator_type "centralized"
