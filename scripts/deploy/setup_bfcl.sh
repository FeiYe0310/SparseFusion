#!/bin/bash
# BFCL官方数据集下载和安装脚本

echo "========================================"
echo "🎯 BFCL (Berkeley Function Calling Leaderboard) Setup"
echo "========================================"

# 1. 设置代理
export https_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128
export http_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128

echo ""
echo "步骤1: 下载BFCL官方仓库"
echo "----------------------------------------"

# 检查是否已存在
if [ -d "bfcl/gorilla" ]; then
    echo "⚠️  gorilla仓库已存在，跳过下载"
else
    cd bfcl
    git clone https://github.com/ShishirPatil/gorilla.git --depth 1
    cd ..
    echo "✅ BFCL仓库下载完成"
fi

echo ""
echo "步骤2: 安装BFCL依赖包"
echo "----------------------------------------"

# 安装BFCL评估工具
pip install bfcl -q

echo "✅ BFCL包安装完成"

echo ""
echo "步骤3: 准备BFCL数据集"
echo "----------------------------------------"

# 查找BFCL数据文件
if [ -d "bfcl/gorilla/berkeley-function-call-leaderboard" ]; then
    BFCL_DATA_DIR="bfcl/gorilla/berkeley-function-call-leaderboard/data"
    
    if [ -d "$BFCL_DATA_DIR" ]; then
        echo "✅ 找到BFCL数据目录: $BFCL_DATA_DIR"
        ls -lh $BFCL_DATA_DIR/*.json 2>/dev/null | head -10
    else
        echo "⚠️  未找到data目录"
    fi
else
    echo "❌ 未找到berkeley-function-call-leaderboard目录"
fi

echo ""
echo "步骤4: 转换BFCL数据格式"
echo "----------------------------------------"

# 创建转换脚本
python3 << 'EOFPYTHON'
import json
import os

# 查找BFCL数据文件
bfcl_data_dir = "bfcl/gorilla/berkeley-function-call-leaderboard/data"

if not os.path.exists(bfcl_data_dir):
    print("❌ 找不到BFCL数据目录")
    print("请手动下载: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard")
else:
    # 列出所有JSON文件
    json_files = [f for f in os.listdir(bfcl_data_dir) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个数据文件:")
    for f in json_files[:5]:
        print(f"  - {f}")
    
    if json_files:
        print(f"\n✅ BFCL数据准备完成")
        print(f"数据目录: {bfcl_data_dir}")
    else:
        print("⚠️  未找到JSON数据文件")
EOFPYTHON

echo ""
echo "========================================"
echo "✅ BFCL设置完成!"
echo "========================================"
echo ""
echo "下一步:"
echo "  1. 查看数据: ls -lh bfcl/gorilla/berkeley-function-call-leaderboard/data/"
echo "  2. 转换数据格式（如需要）"
echo "  3. 运行BFCL测试: bash scripts/experiments/run_bfcl_quick_test.sh"
