#!/usr/bin/env bash
set -euo pipefail

# Concise launcher prints and sane defaults
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export USE_SINGLE_PROCESS_SHARDING=${USE_SINGLE_PROCESS_SHARDING:-1}
export ARCHIVE_BACKEND=${ARCHIVE_BACKEND:-gpu}
export DISABLE_TOKENIZER_MERGE=${DISABLE_TOKENIZER_MERGE:-1}

echo "[Launcher] GPUS_PER_NODE=${GPUS_PER_NODE} | ARCHIVE_BACKEND=${ARCHIVE_BACKEND} | JAX_SHARD=${USE_SINGLE_PROCESS_SHARDING}"
if [[ "${USE_SINGLE_PROCESS_SHARDING}" == "1" ]]; then
  echo "[Mode] Single-process with JAX sharding across ${GPUS_PER_NODE} GPUs"
else
  echo "[Mode] torchrun DDP with ${GPUS_PER_NODE} processes"
fi

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MAIN_PY="${PROJECT_ROOT}/main_sparsity_aware.py"

# Defaults (can be overridden via extra args)
POP_SIZE=${POP_SIZE:-12}
RUNS=${RUNS:-1}
TOTAL_FWD=${TOTAL_FWD:-5000}
EVAL_SUBSET_SIZE=${EVAL_SUBSET_SIZE:-20}

# Default scoring weights: disable GSM8K/MBPP; use mult4/bool only
GSM8K_W=${GSM8K_W:-0.0}
MBPP_W=${MBPP_W:-0.0}
MULT4_W=${MULT4_W:-0.5}
BOOL_W=${BOOL_W:-0.5}

# Qwen chat few-shot defaults
MULT4_FS_K=${MULT4_FS_K:-3}
BOOL_FS_K=${BOOL_FS_K:-3}

# Build command
CMD=(
  python "${MAIN_PY}"
  --pop_size "${POP_SIZE}"
  --runs "${RUNS}" --total_forward_passes "${TOTAL_FWD}"
  --omega 1.0 --beta 0.0 --alpha 1.0 --tau 1.0 --epsilon 1e-10
  --eval_subset_size "${EVAL_SUBSET_SIZE}"
  --gsm8k_weight "${GSM8K_W}" --mbpp_weight "${MBPP_W}"
  --use_mult4_eval --mult4_weight "${MULT4_W}" --mult4_qwen_chat --mult4_few_shot_k "${MULT4_FS_K}"
  --use_bool_eval --bool_weight "${BOOL_W}" --bool_qwen_chat --bool_few_shot_k "${BOOL_FS_K}"
)

# Pass through any extra CLI args (e.g., --model1_path/--model2_path, sparsity schedule, logging flags, etc.)
CMD+=("$@")

# Execute (single-process by default; leave DDP to caller if needed)
"${CMD[@]}"


