#!/bin/bash

# Sparsity-Aware Natural Niches Pipeline 测试脚本

echo "========================================="
echo " Sparsity-Aware Pipeline 功能测试"
echo "========================================="
echo ""

# 设置目录
cd /fs-computility/pdz-grp1/yefei.p/SparseFusion

# 测试输出目录
TEST_OUTPUT_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_OUTPUT_DIR"

echo "📁 测试结果将保存到: $TEST_OUTPUT_DIR"
echo ""

# ============================================
# 测试 1: 基本功能测试（仅稀疏度评分，无剪枝）
# ============================================
echo "========================================="
echo "测试 1: 仅稀疏度评分（无主动剪枝）"
echo "========================================="

python natural_niches_sparsity_aware_fn.py \
    --debug_models \
    --pop_size 4 \
    --total_forward_passes 10 \
    --runs 1 \
    --omega 0.5 \
    --beta 0.5 \
    --tau 1.0 \
    --pruning_sparsity 0.0 \
    --output_dir "$TEST_OUTPUT_DIR/test1_scoring_only" \
    --log_sparsity_stats

if [ $? -eq 0 ]; then
    echo "✅ 测试 1 通过"
else
    echo "❌ 测试 1 失败"
    exit 1
fi

echo ""

# ============================================
# 测试 2: Wanda 剪枝功能测试
# ============================================
echo "========================================="
echo "测试 2: 稀疏度评分 + Wanda 剪枝"
echo "========================================="

python natural_niches_sparsity_aware_fn.py \
    --debug_models \
    --pop_size 4 \
    --total_forward_passes 10 \
    --runs 1 \
    --omega 0.5 \
    --beta 0.5 \
    --tau 1.0 \
    --pruning_sparsity 0.3 \
    --pruning_method wanda \
    --output_dir "$TEST_OUTPUT_DIR/test2_with_pruning" \
    --log_sparsity_stats

if [ $? -eq 0 ]; then
    echo "✅ 测试 2 通过"
else
    echo "❌ 测试 2 失败（可能是 Wanda 不可用）"
    echo "   尝试使用 magnitude 剪枝..."
    
    # 备用：使用 magnitude 剪枝
    python natural_niches_sparsity_aware_fn.py \
        --debug_models \
        --pop_size 4 \
        --total_forward_passes 10 \
        --runs 1 \
        --omega 0.5 \
        --beta 0.5 \
        --tau 1.0 \
        --pruning_sparsity 0.3 \
        --pruning_method magnitude \
        --output_dir "$TEST_OUTPUT_DIR/test2_magnitude_fallback" \
        --log_sparsity_stats
    
    if [ $? -eq 0 ]; then
        echo "✅ 测试 2 通过（使用 magnitude 替代）"
    else
        echo "❌ 测试 2 完全失败"
        exit 1
    fi
fi

echo ""

# ============================================
# 测试 3: 不同参数组合测试
# ============================================
echo "========================================="
echo "测试 3: 不同的 omega/beta 组合"
echo "========================================="

for omega in 0.2 0.5 0.8; do
    beta=$(python3 -c "print(round(1.0 - $omega, 1))")
    
    echo "  → 测试 omega=$omega, beta=$beta"
    
    python natural_niches_sparsity_aware_fn.py \
        --debug_models \
        --pop_size 4 \
        --total_forward_passes 5 \
        --runs 1 \
        --omega $omega \
        --beta $beta \
        --pruning_sparsity 0.0 \
        --output_dir "$TEST_OUTPUT_DIR/test3_omega_${omega}" \
        > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "    ✅ omega=$omega 通过"
    else
        echo "    ❌ omega=$omega 失败"
    fi
done

echo ""

# ============================================
# 测试结果总结
# ============================================
echo "========================================="
echo "测试完成！"
echo "========================================="
echo ""
echo "📊 结果文件："
ls -lh "$TEST_OUTPUT_DIR"/*/*pkl 2>/dev/null || echo "   (未找到 .pkl 文件)"
echo ""
echo "📂 完整结果目录: $TEST_OUTPUT_DIR"
echo ""

# ============================================
# 验证结果文件
# ============================================
echo "========================================="
echo "验证保存的模型参数"
echo "========================================="

# 创建简单的 Python 验证脚本
cat > /tmp/verify_results.py << 'EOF'
import jax.numpy as jnp
import glob
import os
import sys

test_dir = sys.argv[1]
npz_files = glob.glob(os.path.join(test_dir, "**/*best_model*.npz"), recursive=True)

print(f"\n找到 {len(npz_files)} 个模型文件:")
for npz_file in npz_files:
    try:
        data = jnp.load(npz_file)
        params = data["params"]
        sparsity = jnp.mean(jnp.abs(params) < 1e-10)
        
        print(f"\n📄 {os.path.basename(npz_file)}")
        print(f"   参数总数: {params.size:,}")
        print(f"   稀疏度: {sparsity:.4f}")
        print(f"   非零参数: {jnp.sum(jnp.abs(params) >= 1e-10):,}")
    except Exception as e:
        print(f"\n❌ 无法读取 {npz_file}: {e}")

if not npz_files:
    print("\n⚠️  未找到任何模型文件")
EOF

python /tmp/verify_results.py "$TEST_OUTPUT_DIR"

echo ""
echo "========================================="
echo "🎉 所有测试完成！"
echo "========================================="

