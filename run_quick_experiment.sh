#!/bin/bash
# 快速实验：200步，单GPU，小Archive

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python main_sparsity_aware.py \
  --model1_path models/Qwen2.5-0.5B-Instruct \
  --model2_path models/Qwen2.5-Coder-0.5B-Instruct \
  --pop_size 5 \
  --total_forward_passes 200 \
  --omega 1.0 \
  --beta 0.0 \
  --pruning_sparsity 0.0 \
  --output_dir results/quick_test \
  --archive_backend gpu \
  --runs 1
