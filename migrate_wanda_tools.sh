#!/bin/bash

# Wanda 工具迁移脚本
# 自动将 Wanda 剪枝工具迁移到 SparseFusion 项目内

echo "========================================="
echo " Wanda 工具迁移脚本"
echo "========================================="
echo ""

# 设置路径
SPARSEFUSION_DIR="/fs-computility/pdz-grp1/yefei.p/SparseFusion"
WANDA_LIB_DIR="/fs-computility/pdz-grp1/yefei.p/wanda/lib"
PRUNING_DIR="$SPARSEFUSION_DIR/pruning"

# 检查源目录是否存在
if [ ! -d "$WANDA_LIB_DIR" ]; then
    echo "❌ 错误：Wanda 源目录不存在"
    echo "   路径：$WANDA_LIB_DIR"
    exit 1
fi

echo "📂 源目录：$WANDA_LIB_DIR"
echo "📂 目标目录：$PRUNING_DIR"
echo ""

# 检查目标目录
if [ -d "$PRUNING_DIR" ]; then
    echo "⚠️  警告：pruning/ 目录已存在"
    read -p "   是否覆盖现有文件？(y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 取消迁移"
        exit 1
    fi
fi

echo "========================================="
echo " 步骤 1: 备份现有文件（如果存在）"
echo "========================================="

if [ -d "$PRUNING_DIR" ]; then
    BACKUP_DIR="$PRUNING_DIR.backup.$(date +%Y%m%d_%H%M%S)"
    echo "📦 备份到：$BACKUP_DIR"
    cp -r "$PRUNING_DIR" "$BACKUP_DIR"
    echo "✅ 备份完成"
else
    echo "📁 pruning/ 目录不存在，无需备份"
fi

echo ""

# 创建目标目录
mkdir -p "$PRUNING_DIR"

echo "========================================="
echo " 步骤 2: 复制核心文件"
echo "========================================="

# 需要复制的文件列表
FILES_TO_COPY=(
    "prune.py"
    "data.py"
    "layerwrapper.py"
    "sparsegpt.py"
)

for file in "${FILES_TO_COPY[@]}"; do
    src="$WANDA_LIB_DIR/$file"
    dst="$PRUNING_DIR/$file"
    
    if [ -f "$src" ]; then
        echo "  → 复制 $file"
        cp "$src" "$dst"
        
        if [ -f "$dst" ]; then
            echo "    ✅ 完成"
        else
            echo "    ❌ 失败"
            exit 1
        fi
    else
        echo "  ⚠️  警告：$file 不存在，跳过"
    fi
done

echo ""

echo "========================================="
echo " 步骤 3: 创建 __init__.py"
echo "========================================="

cat > "$PRUNING_DIR/__init__.py" << 'EOF'
"""
Pruning tools for SparseFusion

Migrated from Wanda project: https://github.com/locuslab/wanda
"""

from .prune import (
    prune_wanda,
    prune_magnitude,
    prune_sparsegpt,
    check_sparsity,
    find_layers,
)

from .data import get_loaders

__all__ = [
    "prune_wanda",
    "prune_magnitude", 
    "prune_sparsegpt",
    "check_sparsity",
    "find_layers",
    "get_loaders",
]
EOF

echo "✅ 创建 __init__.py"
echo ""

echo "========================================="
echo " 步骤 4: 修改 natural_niches_sparsity_aware_fn.py"
echo "========================================="

MAIN_FILE="$SPARSEFUSION_DIR/natural_niches_sparsity_aware_fn.py"

if [ ! -f "$MAIN_FILE" ]; then
    echo "❌ 错误：找不到 $MAIN_FILE"
    exit 1
fi

# 备份原文件
echo "📦 备份原文件"
cp "$MAIN_FILE" "$MAIN_FILE.backup.$(date +%Y%m%d_%H%M%S)"

# 修改导入路径
echo "🔧 修改导入路径..."

# 删除外部路径添加
sed -i '/sys.path.insert.*wanda/d' "$MAIN_FILE"

# 修改导入语句
sed -i 's/from lib.prune import/from pruning.prune import/g' "$MAIN_FILE"
sed -i 's/from lib.data import/from pruning.data import/g' "$MAIN_FILE"

echo "✅ 修改完成"
echo ""

echo "========================================="
echo " 步骤 5: 验证迁移"
echo "========================================="

# 检查文件是否存在
echo "📋 检查文件..."
for file in "${FILES_TO_COPY[@]}"; do
    if [ -f "$PRUNING_DIR/$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (缺失)"
    fi
done

if [ -f "$PRUNING_DIR/__init__.py" ]; then
    echo "  ✅ __init__.py"
else
    echo "  ❌ __init__.py (缺失)"
fi

echo ""

# 检查导入是否正确修改
echo "📋 检查导入修改..."
if grep -q "sys.path.insert.*wanda" "$MAIN_FILE"; then
    echo "  ⚠️  警告：仍然存在外部路径引用"
else
    echo "  ✅ 外部路径引用已删除"
fi

if grep -q "from pruning.prune import" "$MAIN_FILE"; then
    echo "  ✅ 新导入路径已配置"
else
    echo "  ⚠️  警告：未找到新导入路径"
fi

echo ""

echo "========================================="
echo " 步骤 6: 测试导入"
echo "========================================="

cd "$SPARSEFUSION_DIR"

echo "🧪 测试 Python 导入..."
python3 -c "
import sys
sys.path.insert(0, '$SPARSEFUSION_DIR')

try:
    from pruning.prune import prune_wanda
    from pruning.data import get_loaders
    print('✅ 导入成功')
except ImportError as e:
    print(f'❌ 导入失败: {e}')
    sys.exit(1)
"

IMPORT_STATUS=$?

echo ""

echo "========================================="
echo " 迁移完成！"
echo "========================================="

if [ $IMPORT_STATUS -eq 0 ]; then
    echo "✅ 所有检查通过"
    echo ""
    echo "📁 迁移的文件："
    ls -lh "$PRUNING_DIR"
    echo ""
    echo "📝 备份文件："
    ls -t "$SPARSEFUSION_DIR"/*.backup.* 2>/dev/null | head -1
    echo ""
    echo "🚀 现在可以运行测试："
    echo "   cd $SPARSEFUSION_DIR"
    echo "   python main_sparsity_aware.py --debug_models --pop_size 4 --total_forward_passes 10"
else
    echo "⚠️  迁移完成，但导入测试失败"
    echo "   请手动检查 $MAIN_FILE"
    echo ""
    echo "   如果需要恢复，运行："
    echo "   mv $MAIN_FILE.backup.* $MAIN_FILE"
fi

echo ""
echo "========================================="

