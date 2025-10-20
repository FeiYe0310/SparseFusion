#!/bin/bash
# BFCL功能测试脚本

echo "========================================"
echo "🧪 BFCL Integration Test"
echo "========================================"

# 设置代理
export https_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128
export http_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128

echo ""
echo "步骤1: 测试Function Call解析器和评估器"
echo "----------------------------------------"
python bfcl_eval_utils.py

if [ $? -ne 0 ]; then
    echo "❌ Function Call解析器测试失败"
    exit 1
fi

echo ""
echo "✅ Function Call解析器测试通过"
echo ""

echo "步骤2: 测试BFCL数据加载"
echo "----------------------------------------"
# 创建临时测试脚本
cat > /tmp/test_bfcl_data.py << 'EOFTEST'
from transformers import AutoTokenizer
from bfcl_data_utils import load_bfcl_dataset, bfcl_collate_fn
from torch.utils.data import DataLoader

print("加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-0.5B-Instruct")

print("加载BFCL数据集...")
dataset = load_bfcl_dataset(
    data_path="bfcl/data/bfcl_test_simple.json",
    tokenizer=tokenizer,
    subset_size=5
)

print(f"✓ 数据集加载成功: {len(dataset)} 样本")
print(f"✓ 样本keys: {dataset[0].keys()}")
print(f"✓ Ground truth示例: {dataset[0]['ground_truth']}")

# 测试DataLoader
dataloader = DataLoader(dataset, batch_size=2, collate_fn=bfcl_collate_fn)
batch = next(iter(dataloader))
print(f"✓ Batch shape: {batch['input_ids'].shape}")
print(f"✓ Ground truths: {batch['ground_truth']}")

print("\n✅ 所有数据加载测试通过!")
EOFTEST

python /tmp/test_bfcl_data.py

if [ $? -ne 0 ]; then
    echo "❌ BFCL数据加载测试失败"
    exit 1
fi

echo ""
echo "========================================"
echo "✅ 所有BFCL功能测试通过!"
echo "========================================"
echo ""
echo "接下来可以运行完整的实验："
echo "  bash scripts/experiments/run_bfcl_quick_test.sh  # 快速验证"
echo "  bash scripts/experiments/run_bfcl_full_exp.sh    # 完整实验"
