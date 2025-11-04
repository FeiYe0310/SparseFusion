#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 🧪 BFCL快速测试 - 小规模验证版本（1-2小时完成）
# ============================================================================

echo "════════════════════════════════════════════════════════════════════════"
echo "🧪 BFCL Quick Test - Fast Validation Run"
echo "════════════════════════════════════════════════════════════════════════"

# ============================================================================
# 1. 代理配置
# ============================================================================
export https_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128
export http_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128

# ============================================================================
# 2. 环境配置
# ============================================================================
ROOTPATH=${ROOTPATH:-/mnt/shared-storage-user/yefei}
cd "$ROOTPATH/SparseFusion"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$ROOTPATH/cache
export WANDB_CACHE_DIR=$ROOTPATH/cache
export TORCH_EXTENSION_DIR=$ROOTPATH/cache
export PATH="$ROOTPATH/miniconda3/bin:$PATH"
export PATH="$ROOTPATH/miniconda3/envs/sparsefusion/bin:$PATH"

# ============================================================================
# 3. 快速测试参数（小规模）
# ============================================================================
GPUS_PER_NODE=1
ARCHIVE_BACKEND="gpu"

MODEL1_PATH="models/Qwen2.5-0.5B-Instruct"
MODEL2_PATH="models/Qwen2.5-0.5B-Instruct"

# 快速测试配置
POP_SIZE=2                         # 小种群
TOTAL_FORWARD_PASSES=100           # 短迭代（快速验证）
RUNS=1

OMEGA=0.7
BETA=0.3
PRUNING_SPARSITY=0.3

EVAL_SUBSET_SIZE=30
BATCH_SIZE=32

USE_BFCL_EVAL=true
BFCL_DATA_PATH="data/bfcl/data/bfcl_test_200.json"
GSM8K_WEIGHT=0.5
BFCL_WEIGHT=0.5

OUTPUT_DIR="results_bfcl_quicktest"

echo ""
echo "📊 快速测试配置:"
echo "  Pop size: $POP_SIZE (小)"
echo "  Iterations: $TOTAL_FORWARD_PASSES (短)"
echo "  Eval samples: $EVAL_SUBSET_SIZE per task"
echo "  预计时间: ~1-2小时"
echo ""
echo "🚀 开始快速测试..."
echo ""

# ============================================================================
# 4. 运行
# ============================================================================
START_TIME=$(date +%s)

python main_natural_niches_sparsity_aware_fn.py \
  --archive_backend "${ARCHIVE_BACKEND}" \
  --model1_path "$MODEL1_PATH" \
  --model2_path "$MODEL2_PATH" \
  --pop_size "$POP_SIZE" \
  --total_forward_passes "$TOTAL_FORWARD_PASSES" \
  --runs "$RUNS" \
  --omega "$OMEGA" \
  --beta "$BETA" \
  --pruning_sparsity "$PRUNING_SPARSITY" \
  --eval_subset_size "$EVAL_SUBSET_SIZE" \
  --use_bfcl_eval \
  --bfcl_data_path "$BFCL_DATA_PATH" \
  --gsm8k_weight "$GSM8K_WEIGHT" \
  --bfcl_weight "$BFCL_WEIGHT" \
  --output_dir "$OUTPUT_DIR" \
  --log_sparsity_stats

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "════════════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ 快速测试完成！运行时间: $((ELAPSED / 60))分钟"
  echo "结果: $OUTPUT_DIR/"
else
  echo "❌ 测试失败 (exit code: $EXIT_CODE)"
fi
echo "════════════════════════════════════════════════════════════════════════"

