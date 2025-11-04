#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU launcher for GSM8K + MBPP evaluations with Sparsity-Aware Natural Niches
# - Supports torchrun DDP and single-process JAX sharding
# - Single run per invocation; configure via env or inline vars
# - Allows subset evaluation per-iteration (eval_subset_size)

# ====== Environment & Paths ======
ROOTPATH=${ROOTPATH:-/mnt/shared-storage-user/yefei}
cd "$ROOTPATH/SparseFusion"

# 必要环境（按你要求仅内置这两项；可通过外部覆盖）
export USE_SINGLE_PROCESS_SHARDING=${USE_SINGLE_PROCESS_SHARDING:-1}
export DISABLE_TOKENIZER_MERGE=${DISABLE_TOKENIZER_MERGE:-1}

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-$ROOTPATH/cache}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$ROOTPATH/cache}
export TORCH_EXTENSION_DIR=${TORCH_EXTENSION_DIR:-$ROOTPATH/cache}

export PATH="$ROOTPATH/miniconda3/bin:$PATH"
export PATH="$ROOTPATH/miniconda3/envs/sparsefusion/bin:$PATH"

# ====== Hardware Config ======
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
USE_SINGLE_PROCESS_SHARDING=${USE_SINGLE_PROCESS_SHARDING:-0}
ARCHIVE_BACKEND=${ARCHIVE_BACKEND:-gpu}

# ====== Task & Model Config ======
MODEL_PATH=${MODEL_PATH:-/mnt/shared-storage-user/yefei/SparseFusion/models/Qwen2.5-Coder-0.5B-Instruct}
MBPP_PATH=${MBPP_PATH:-/mnt/shared-storage-user/yefei/SparseFusion/datasets/mbpp_hf}

# Evaluation subset per iteration (speedup). Use 15 per your request.
EVAL_SUBSET_SIZE=${EVAL_SUBSET_SIZE:-15}

# Few-shot (Qwen chat) toggles
USE_GSM8K_QWEN=${USE_GSM8K_QWEN:-1}
USE_MBPP_QWEN=${USE_MBPP_QWEN:-1}
FEW_SHOT_K=${FEW_SHOT_K:-3}
FEW_SHOT_SPLIT=${FEW_SHOT_SPLIT:-train}

# Weights
GSM8K_WEIGHT=${GSM8K_WEIGHT:-0.5}
MBPP_WEIGHT=${MBPP_WEIGHT:-0.5}

# Evolution scale
RUNS=${RUNS:-1}
TOTAL_FP=${TOTAL_FP:-5000}

# Common args passed to main
COMMON_ARGS=(
  --runs "$RUNS"
  --total_forward_passes "$TOTAL_FP"
  --model1_path "$MODEL_PATH"
  --model2_path "$MODEL_PATH"
  --eval_subset_size "$EVAL_SUBSET_SIZE"
  --use_mbpp_eval --mbpp_data_path "$MBPP_PATH"
  --gsm8k_weight "$GSM8K_WEIGHT" --mbpp_weight "$MBPP_WEIGHT"
)

if (( USE_GSM8K_QWEN )); then
  COMMON_ARGS+=(--gsm8k_qwen_chat --gsm8k_few_shot_k "$FEW_SHOT_K" --gsm8k_few_shot_split "$FEW_SHOT_SPLIT")
fi
if (( USE_MBPP_QWEN )); then
  COMMON_ARGS+=(--mbpp_qwen_chat --mbpp_few_shot_k "$FEW_SHOT_K" --mbpp_few_shot_split "$FEW_SHOT_SPLIT")
fi

# ====== Sparsity / Fitness params (external override via env) ======
# Alpha/Beta/Others
if [[ -n "${ALPHA:-}" ]]; then COMMON_ARGS+=(--alpha "$ALPHA"); fi
if [[ -n "${BETA:-}" ]]; then COMMON_ARGS+=(--beta "$BETA"); fi
if [[ -n "${OMEGA:-}" ]]; then COMMON_ARGS+=(--omega "$OMEGA"); fi
if [[ -n "${TAU:-}" ]]; then COMMON_ARGS+=(--tau "$TAU"); fi
if [[ -n "${EPSILON:-}" ]]; then COMMON_ARGS+=(--epsilon "$EPSILON"); fi

# Pruning
if [[ -n "${PRUNING_SPARSITY:-}" ]]; then COMMON_ARGS+=(--pruning_sparsity "$PRUNING_SPARSITY"); fi
if [[ -n "${PRUNING_METHOD:-}" ]]; then COMMON_ARGS+=(--pruning_method "$PRUNING_METHOD"); fi

# Dynamic sparsity schedule
if [[ -n "${USE_DYNAMIC_SPARSITY:-}" ]]; then COMMON_ARGS+=(--use_dynamic_sparsity); fi
if [[ -n "${SPARSITY_MIN:-}" ]]; then COMMON_ARGS+=(--sparsity_min "$SPARSITY_MIN"); fi
if [[ -n "${SPARSITY_MAX:-}" ]]; then COMMON_ARGS+=(--sparsity_max "$SPARSITY_MAX"); fi
if [[ -n "${SPARSITY_T0:-}" ]]; then COMMON_ARGS+=(--sparsity_t0 "$SPARSITY_T0"); fi
if [[ -n "${SPARSITY_T_MULT:-}" ]]; then COMMON_ARGS+=(--sparsity_t_mult "$SPARSITY_T_MULT"); fi

echo "[Launcher] GPUS_PER_NODE=${GPUS_PER_NODE} | ARCHIVE_BACKEND=${ARCHIVE_BACKEND} | JAX_SHARD=${USE_SINGLE_PROCESS_SHARDING}" >&2

if (( GPUS_PER_NODE > 1 )); then
  if [[ "$USE_SINGLE_PROCESS_SHARDING" != "0" ]]; then
    echo "[Mode] Single-process with JAX sharding across ${GPUS_PER_NODE} GPUs" >&2
    JAX_PLATFORM_NAME=cpu \
    python natural_niches_sparsity_aware_fn.py \
      --archive_backend "$ARCHIVE_BACKEND" \
      "${COMMON_ARGS[@]}" "$@"
  else
    echo "[Mode] torchrun DDP with ${GPUS_PER_NODE} processes" >&2
    export JAX_PLATFORM_NAME=cpu
    torchrun --standalone --nproc_per_node="$GPUS_PER_NODE" \
      natural_niches_sparsity_aware_fn.py \
      --distributed \
      --archive_backend "$ARCHIVE_BACKEND" \
      "${COMMON_ARGS[@]}" "$@"
  fi
else
  echo "[Mode] Single GPU" >&2
  python natural_niches_sparsity_aware_fn.py \
    --archive_backend "$ARCHIVE_BACKEND" \
    "${COMMON_ARGS[@]}" "$@"
fi


