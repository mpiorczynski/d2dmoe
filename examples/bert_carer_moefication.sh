#!/bin/bash

set -e

seed=42
method="moefication"
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
    --do_train \
    --do_eval \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --report_to wandb \
    --run_name "${dataset_name}-${model_name_or_path}-finetuning" \
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
    --k_to_eval '1, 2, 4, 8, 16, 32, 64, 96, 128' \
    --do_train \
    --do_eval \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --report_to wandb \
    --run_name "${dataset_name}-${model_name_or_path}-moefication" \
    --seed $seed
