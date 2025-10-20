#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Checkpoint Comparison Visualization Script
# 
# 用途：比较两个checkpoint并生成可视化图表
# ============================================================================

# 设置Python环境
ROOTPATH=${ROOTPATH:-/mnt/shared-storage-user/yefei}
export PATH="$ROOTPATH/miniconda3/bin:$PATH"
export PATH="$ROOTPATH/miniconda3/envs/sparsefusion/bin:$PATH"

# 默认checkpoint路径
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/mnt/shared-storage-user/yefei/SparseFusion/results/checkpoints/checkpoint_run1_step3000_20251017_021740.pkl}"
SPARSITY_CHECKPOINT="${SPARSITY_CHECKPOINT:-/mnt/shared-storage-user/yefei/SparseFusion/results/sparsity_aware_w0.80_b0.20_t1.00_prune_wanda_0.30.pkl}"
OUTPUT_PLOT="${OUTPUT_PLOT:-comparison_baseline_vs_sparsity.png}"

echo "=========================================="
echo "  Checkpoint Comparison Visualization"
echo "=========================================="
echo "Baseline checkpoint:  $BASELINE_CHECKPOINT"
echo "Sparsity checkpoint:  $SPARSITY_CHECKPOINT"
echo "Output plot:          $OUTPUT_PLOT"
echo "=========================================="
echo ""

# 检查文件是否存在
if [[ ! -f "$BASELINE_CHECKPOINT" ]]; then
    echo "❌ ERROR: Baseline checkpoint not found: $BASELINE_CHECKPOINT"
    exit 1
fi

if [[ ! -f "$SPARSITY_CHECKPOINT" ]]; then
    echo "❌ ERROR: Sparsity checkpoint not found: $SPARSITY_CHECKPOINT"
    exit 1
fi

# 运行绘图脚本
cd "$ROOTPATH/SparseFusion"

python plot_checkpoint_comparison.py \
    --baseline "$BASELINE_CHECKPOINT" \
    --sparsity "$SPARSITY_CHECKPOINT" \
    --output "$OUTPUT_PLOT"

echo ""
echo "✅ Visualization complete!"
echo "📊 Plot saved to: $OUTPUT_PLOT"

