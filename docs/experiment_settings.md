### 实验设置（更新于 2025-10-26）

本页记录当前建议的4组新实验（各两条：Baseline 与 Sparse），以及可直接运行的命令与公共环境设置。状态默认为“待启动”。

#### 全局环境（单机8卡、JAX Sharding、关闭分词器合并）
```bash
export GPUS_PER_NODE=8
export USE_SINGLE_PROCESS_SHARDING=1
export DISABLE_TOKENIZER_MERGE=1
export ARCHIVE_BACKEND=gpu
```

#### 通用参数
- **数据与模型**：
  - `--model1_path` `/mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct`
  - `--model2_path` `/mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct`
  - `--use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf`
- **Few-shot 与模板**：
  - `--gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train`
  - `--mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train`
- **评估子集**：
  - `--eval_subset_size 15`（统一为15）
  - 每10步会在测试集做一次周期评估（文件按JSONL追加保存）
- **权重**：`--gsm8k_weight 0.5 --mbpp_weight 0.5`
- **训练步数**：`--runs 1 --total_forward_passes 5000`
- **静默输出**：未设置 `VERBOSE_EVAL`（默认关闭详细打印）

---

### E1（Baseline，pop_size=16）— 待启动
- 特点：不计入稀疏度得分（`beta=0.0`），无动态稀疏
- 关键参数：`--omega 1.0 --beta 0.0 --alpha 1.0 --tau 1.0 --epsilon 1e-10`
- 命令：
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 16 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 1.0 --beta 0.0 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

### E2（Sparse，pop_size=16）— 待启动
- 特点：启用动态稀疏，轻微稀疏权重
- 关键参数：
  - 分数权重：`--omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10`
  - 动态稀疏：`--use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.60 --sparsity_t0 100 --sparsity_t_mult 2`
- 命令：
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 16 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.60 --sparsity_t0 100 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

### E3（Baseline，pop_size=12）— 待启动
- 特点：不计入稀疏度得分（`beta=0.0`），无动态稀疏
- 关键参数：同 E1
- 命令：
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 12 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 1.0 --beta 0.0 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

### E4（Sparse，pop_size=12）— 待启动
- 特点：启用动态稀疏，轻微稀疏权重
- 关键参数：同 E2
- 命令：
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 12 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.60 --sparsity_t0 100 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

---

#### 备注
- 评估：训练时每步评估基于 `--eval_subset_size 15` 的子集；每10步在测试集做周期评估（JSONL 追加写），用于对照；路径见 `results/fitness_logs/`。
- 安静模式：未设置 `VERBOSE_EVAL=1`，不打印采样与样例。
- 如需切换到 DDP：把 `USE_SINGLE_PROCESS_SHARDING=0` 即可。

### 新增四个实验（Sparse-only，2025-10-26）— 待启动

以下4条均为“启用稀疏度评分与动态稀疏”的配置（不含Baseline）。统一设置：`--eval_subset_size 15`、`--runs 1 --total_forward_passes 5000`、`--gsm8k_weight 0.5 --mbpp_weight 0.5`，并启用 Qwen 3-shot 模板（GSM8K/MBPP 均为 `few_shot_k=3`，`split=train`）。

- S1（pop_size=16，beta=0.10，温和稀疏）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 16 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.60 --sparsity_t0 100 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

- S2（pop_size=16，beta=0.10，适中稀疏；加大上限/更慢启动）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 16 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.20 --sparsity_max 0.70 --sparsity_t0 150 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

- S3（pop_size=12，beta=0.10，温和稀疏）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 12 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.60 --sparsity_t0 100 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

- S4（pop_size=12，beta=0.10，适中稀疏；更慢周期）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 12 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.60 --sparsity_t0 200 --sparsity_t_mult 3 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

### 新增四个实验（Sparse-only，beta=0.20/0.15，2025-10-26）— 待启动

- 统一设置：`--eval_subset_size 15`、`--runs 1 --total_forward_passes 5000`、`--gsm8k_weight 0.5 --mbpp_weight 0.5`、Qwen 3-shot 模板启用。
- 全局环境建议：`GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu`

- S5（pop_size=16，beta=0.20，适中偏强稀疏）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 16 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.8 --beta 0.2 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.70 --sparsity_t0 150 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

- S6（pop_size=16，beta=0.15，中等稀疏）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 16 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.85 --beta 0.15 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.65 --sparsity_t0 120 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

- S7（pop_size=12，beta=0.20，适中偏强稀疏）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 12 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.8 --beta 0.2 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.70 --sparsity_t0 150 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

- S8（pop_size=12，beta=0.15，中等稀疏）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 12 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.85 --beta 0.15 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.65 --sparsity_t0 120 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

本页记录当前建议的4组新实验（各两条：Baseline 与 Sparse），以及可直接运行的命令与公共环境设置。状态默认为“待启动”。

#### 全局环境（单机8卡、JAX Sharding、关闭分词器合并）
```bash
export GPUS_PER_NODE=8
export USE_SINGLE_PROCESS_SHARDING=1
export DISABLE_TOKENIZER_MERGE=1
export ARCHIVE_BACKEND=gpu
```

#### 通用参数
- **数据与模型**：
  - `--model1_path` `/mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct`
  - `--model2_path` `/mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct`
  - `--use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf`
- **Few-shot 与模板**：
  - `--gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train`
  - `--mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train`
- **评估子集**：
  - `--eval_subset_size 15`（统一为15）
  - 每10步会在测试集做一次周期评估（文件按JSONL追加保存）
- **权重**：`--gsm8k_weight 0.5 --mbpp_weight 0.5`
- **训练步数**：`--runs 1 --total_forward_passes 5000`
- **静默输出**：未设置 `VERBOSE_EVAL`（默认关闭详细打印）

---

### E1（Baseline，pop_size=16）— 待启动
- 特点：不计入稀疏度得分（`beta=0.0`），无动态稀疏
- 关键参数：`--omega 1.0 --beta 0.0 --alpha 1.0 --tau 1.0 --epsilon 1e-10`
- 命令：
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 16 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 1.0 --beta 0.0 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

### E2（Sparse，pop_size=16）— 待启动
- 特点：启用动态稀疏，轻微稀疏权重
- 关键参数：
  - 分数权重：`--omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10`
  - 动态稀疏：`--use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.60 --sparsity_t0 100 --sparsity_t_mult 2`
- 命令：
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 16 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.60 --sparsity_t0 100 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

### E3（Baseline，pop_size=12）— 待启动
- 特点：不计入稀疏度得分（`beta=0.0`），无动态稀疏
- 关键参数：同 E1
- 命令：
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 12 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 1.0 --beta 0.0 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

### E4（Sparse，pop_size=12）— 待启动
- 特点：启用动态稀疏，轻微稀疏权重
- 关键参数：同 E2
- 命令：
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 12 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.60 --sparsity_t0 100 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

---

#### 备注
- 评估：训练时每步评估基于 `--eval_subset_size 15` 的子集；每10步在测试集做周期评估（JSONL 追加写），用于对照；路径见 `results/fitness_logs/`。
- 安静模式：未设置 `VERBOSE_EVAL=1`，不打印采样与样例。
- 如需切换到 DDP：把 `USE_SINGLE_PROCESS_SHARDING=0` 即可。

### 新增四个实验（Sparse-only，2025-10-26）— 待启动

以下4条均为“启用稀疏度评分与动态稀疏”的配置（不含Baseline）。统一设置：`--eval_subset_size 15`、`--runs 1 --total_forward_passes 5000`、`--gsm8k_weight 0.5 --mbpp_weight 0.5`，并启用 Qwen 3-shot 模板（GSM8K/MBPP 均为 `few_shot_k=3`，`split=train`）。

- S1（pop_size=16，beta=0.10，温和稀疏）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 16 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.60 --sparsity_t0 100 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

- S2（pop_size=16，beta=0.10，适中稀疏；加大上限/更慢启动）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 16 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.20 --sparsity_max 0.70 --sparsity_t0 150 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

- S3（pop_size=12，beta=0.10，温和稀疏）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 12 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.60 --sparsity_t0 100 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

- S4（pop_size=12，beta=0.10，适中稀疏；更慢周期）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 12 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.9 --beta 0.1 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.60 --sparsity_t0 200 --sparsity_t_mult 3 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

### 新增四个实验（Sparse-only，beta=0.20/0.15，2025-10-26）— 待启动

- 统一设置：`--eval_subset_size 15`、`--runs 1 --total_forward_passes 5000`、`--gsm8k_weight 0.5 --mbpp_weight 0.5`、Qwen 3-shot 模板启用。
- 全局环境建议：`GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu`

- S5（pop_size=16，beta=0.20，适中偏强稀疏）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 16 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.8 --beta 0.2 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.70 --sparsity_t0 150 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

- S6（pop_size=16，beta=0.15，中等稀疏）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 16 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.85 --beta 0.15 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.65 --sparsity_t0 120 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

- S7（pop_size=12，beta=0.20，适中偏强稀疏）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 12 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.8 --beta 0.2 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.70 --sparsity_t0 150 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```

- S8（pop_size=12，beta=0.15，中等稀疏）— 待启动
```bash
export GPUS_PER_NODE=8 USE_SINGLE_PROCESS_SHARDING=1 DISABLE_TOKENIZER_MERGE=1 ARCHIVE_BACKEND=gpu
bash /mnt/shared-storage-user/yefei/SparseFusion/scripts/experiments/run_gsm8k_mbpp_multi.sh \
  --pop_size 12 --runs 1 --total_forward_passes 5000 \
  --model1_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct \
  --model2_path /mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-0.5B-Instruct \
  --eval_subset_size 15 \
  --use_mbpp_eval --mbpp_data_path /mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf \
  --gsm8k_weight 0.5 --mbpp_weight 0.5 \
  --omega 0.85 --beta 0.15 --alpha 1.0 --tau 1.0 --epsilon 1e-10 \
  --use_dynamic_sparsity --sparsity_min 0.10 --sparsity_max 0.65 --sparsity_t0 120 --sparsity_t_mult 2 \
  --gsm8k_qwen_chat --gsm8k_few_shot_k 3 --gsm8k_few_shot_split train \
  --mbpp_qwen_chat --mbpp_few_shot_k 3 --mbpp_few_shot_split train
```


