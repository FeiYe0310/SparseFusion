#!/bin/bash
# 本地运行BFCL实验 - 纯本地配置，无需代理和HuggingFace

cd /fs-computility/pdz-grp1/yefei.p/SparseFusion

python natural_niches_sparsity_aware_fn.py \
  --model1_path /fs-computility/pdz-grp1/yefei.p/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --model2_path /fs-computility/pdz-grp1/yefei.p/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --pop_size 5 \
  --total_forward_passes 3000 \
  --runs 1 \
  --omega 0.7 \
  --beta 0.3 \
  --pruning_sparsity 0.3 \
  --eval_subset_size 30 \
  --use_bfcl_eval \
  --bfcl_data_path /fs-computility/pdz-grp1/yefei.p/SparseFusion/bfcl/data/bfcl_test_200.json \
  --gsm8k_weight 0.5 \
  --bfcl_weight 0.5 \
  --output_dir /fs-computility/pdz-grp1/yefei.p/SparseFusion/results_bfcl_local \
  --log_sparsity_stats

