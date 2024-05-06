#!/bin/bash

set -e

seed=42
method="d2dmoe"
dataset_name="emotion"
model_name_or_path="mpiorczynski/relu-bert-base-uncased"
models_dir="checkpoints"
dense_model_dir="${models_dir}/${dataset_name}/${method}/dense"
moe_model_dir="${models_dir}/${dataset_name}/${method}/moe"


python scripts/bert/finetune.py \
    --dataset_name $dataset_name \
    --model_name_or_path $model_name_or_path \
    --output_dir $dense_model_dir \
    --method $method \
    --max_seq_length 128 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 64 \
    --sparsity_enforcement_weight 1e-3 \
    --do_train \
    --do_eval \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --report_to wandb \
    --run_name "${dataset_name}-${model_name_or_path}-sparse-finetuning" \
    --seed $seed


python scripts/bert/dense2moe.py \
    --model_name_or_path $dense_model_dir \
    --dataset_name $dataset_name \
    --method $method \
    --num_experts 128 \
    --expert_split \
    --router_width 128 \
    --num_train_epochs 10 \
    --learning_rate 0.001 \
    --per_device_train_batch_size 64 \
    --tau_to_eval '0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0' \
    --do_train \
    --do_eval \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --report_to wandb \
    --run_name "${dataset_name}-${model_name_or_path}-d2dmoe" \
    --seed $seed
