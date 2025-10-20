#!/usr/bin/env bash
set -euo pipefail

# 🚀 服务器端设置脚本 - 在服务器上运行这个
# 检查BFCL环境是否就绪

# ============================================================================
# Workspace setup (match run_sparsity_single_node.sh)
# ============================================================================
ROOTPATH=${ROOTPATH:-/mnt/shared-storage-user/yefei}

echo "========================================"
echo "🚀 服务器端BFCL设置"
echo "========================================"
echo ""
echo "工作目录: $ROOTPATH/SparseFusion"
echo ""

# Change to workspace
if [[ -d "$ROOTPATH/SparseFusion" ]]; then
  cd "$ROOTPATH/SparseFusion"
else
  echo "❌ 目录不存在: $ROOTPATH/SparseFusion" >&2
  echo "请先克隆仓库或设置正确的ROOTPATH" >&2
  exit 1
fi

# 1. 检查git同步
echo ""
echo "步骤1: 检查代码版本"
echo "----------------------------------------"
git log --oneline -1
git status

# 2. 检查BFCL数据
echo ""
echo "步骤2: 检查BFCL数据"
echo "----------------------------------------"

if [ -f "bfcl/data/bfcl_test_200.json" ]; then
    FILE_SIZE=$(ls -lh bfcl/data/bfcl_test_200.json | awk '{print $5}')
    echo "✅ BFCL数据集已存在: $FILE_SIZE"
    
    # 验证数据
    SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('bfcl/data/bfcl_test_200.json'))))")
    echo "✅ 数据样本数: $SAMPLE_COUNT"
else
    echo "❌ BFCL数据集不存在"
    echo "GitHub应该已包含该文件，检查git pull是否成功"
    exit 1
fi

# 3. 检查模型
echo ""
echo "步骤3: 检查模型文件"
echo "----------------------------------------"

if [ -d "models/Qwen2.5-0.5B-Instruct" ]; then
    echo "✅ 模型存在: models/Qwen2.5-0.5B-Instruct"
else
    echo "⚠️  模型不存在: models/Qwen2.5-0.5B-Instruct"
    echo "请确保模型路径正确，或修改scripts/experiments/RUN_BFCL_NOW.sh中的--model1_path参数"
fi

# 4. 检查Python依赖
echo ""
echo "步骤4: 检查Python依赖"
echo "----------------------------------------"

python3 -c "
import sys
missing = []
try:
    import jax
    print('✅ jax')
except ImportError:
    missing.append('jax')
    print('❌ jax')

try:
    import torch
    print('✅ torch')
except ImportError:
    missing.append('torch')
    print('❌ torch')

try:
    import transformers
    print('✅ transformers')
except ImportError:
    missing.append('transformers')
    print('❌ transformers')

try:
    from datasets import load_dataset
    print('✅ datasets')
except ImportError:
    missing.append('datasets')
    print('❌ datasets')

if missing:
    print(f'\\n需要安装: pip install {\" \".join(missing)}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "请安装缺失的依赖"
    exit 1
fi

# 5. 测试导入
echo ""
echo "步骤5: 测试BFCL模块导入"
echo "----------------------------------------"

python3 -c "
import sys
sys.path.insert(0, '.')

try:
    from bfcl_data_utils import load_bfcl_dataset
    print('✅ bfcl_data_utils 导入成功')
except Exception as e:
    print(f'❌ bfcl_data_utils 导入失败: {e}')
    sys.exit(1)

try:
    from bfcl_eval_utils import extract_function_call, evaluate_function_call
    print('✅ bfcl_eval_utils 导入成功')
except Exception as e:
    print(f'❌ bfcl_eval_utils 导入失败: {e}')
    sys.exit(1)

print('✅ 所有BFCL模块可用')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "模块导入失败，检查代码完整性"
    exit 1
fi

# 6. 给脚本添加执行权限
echo ""
echo "步骤6: 设置脚本权限"
echo "----------------------------------------"
chmod +x scripts/experiments/RUN_BFCL_NOW.sh
chmod +x scripts/experiments/run_bfcl_quick_test.sh
chmod +x scripts/tests/test_bfcl.sh
echo "✅ 脚本权限已设置"

# 7. 环境变量检查
echo ""
echo "步骤7: 检查环境变量"
echo "----------------------------------------"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "⚠️  CUDA_VISIBLE_DEVICES 未设置"
    echo "建议: export CUDA_VISIBLE_DEVICES=0"
else
    echo "✅ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# 8. 磁盘空间检查
echo ""
echo "步骤8: 检查磁盘空间"
echo "----------------------------------------"
df -h . | tail -1

echo ""
echo "========================================"
echo "✅ 服务器设置完成!"
echo "========================================"
echo ""
echo "现在可以运行实验:"
echo ""
echo "  方式1 (快速测试 - 2分钟):"
echo "    bash scripts/tests/test_bfcl.sh"
echo ""
echo "  方式2 (小规模验证 - 1小时):"
echo "    bash scripts/experiments/run_bfcl_quick_test.sh"
echo ""
echo "  方式3 (完整实验 - 8-12小时):"
echo "    bash scripts/experiments/RUN_BFCL_NOW.sh"
echo ""
echo "  方式4 (后台运行):"
echo "    nohup bash scripts/experiments/RUN_BFCL_NOW.sh > bfcl_run.log 2>&1 &"
echo "    tail -f bfcl_run.log"
echo ""
