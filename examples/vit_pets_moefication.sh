#!/bin/bash

set -e

seed=42
method="moefication"
dataset_name="oxford_iiit_pet"
model_name_or_path="mpiorczynski/relu-vit-base-patch16-224"
models_dir="checkpoints"
dense_model_dir="${models_dir}/${dataset_name}/${method}/dense"
moe_model_dir="${models_dir}/${dataset_name}/${method}/moe"


python scripts/vit/finetune.py \
    --dataset_name $dataset_name \
    --model_name_or_path $model_name_or_path \
    --output_dir $dense_model_dir \
    --method $method \
    --num_train_epochs 25 \
    --learning_rate 1e-2 \
    --per_device_train_batch_size 128 \
    --optim "sgd" \
    --optim_args 'momentum=0.9, weight_decay=0.0' \
    --lr_scheduler_type "cosine" \
    --warmup_steps 100 \
    --bf16 \
    --do_train \
    --do_eval \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --report_to wandb \
    --run_name "${dataset_name}-${model_name_or_path}-finetuning" \
    --seed $seed


python scripts/vit/dense2moe.py \
    --model_name_or_path $dense_model_dir \
    --dataset_name $dataset_name \
    --method $method \
    --expert_split \
    --num_experts 128 \
    --router_width 128 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 0.001 \
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