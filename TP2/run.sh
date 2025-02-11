#!/bin/bash

# Array of local steps to test
local_steps_array=(1 5 10 50 100)

for local_steps in "${local_steps_array[@]}"
do
    echo "=> Training with ${local_steps} local steps..."
    
    python train.py \
      --experiment "mnist" \
      --n_rounds 50 \
      --local_steps ${local_steps} \
      --local_optimizer sgd \
      --local_lr 0.001 \
      --server_optimizer sgd \
      --server_lr 0.1 \
      --bz 128 \
      --device "cpu" \
      --log_freq 1 \
      --verbose 1 \
      --logs_dir "logs/mnist_local_steps_${local_steps}/" \
      --seed 12 \
      --sampling_rate 0.2 \
      --sample_with_replacement
done

# Plot results and save to results.txt
python3 plot_results.py --experiment "mnist" --local_steps 1 5 10 50 100 > \\tsclient\UniCA-msc-ds_ai-main-1\semester3\CORE AI TRACK\federated_learning\Codes\TP2\results.txt
