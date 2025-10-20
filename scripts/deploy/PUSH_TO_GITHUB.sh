#!/bin/bash
# 提交BFCL集成代码到GitHub

echo "========================================"
echo "📤 推送BFCL代码到GitHub"
echo "========================================"

# 1. 查看当前状态
echo ""
echo "步骤1: 查看修改的文件"
echo "----------------------------------------"
git status

echo ""
echo "步骤2: 添加所有BFCL相关文件"
echo "----------------------------------------"

# 添加核心代码文件
git add natural_niches_sparsity_aware_fn.py
git add main_sparsity_aware.py
git add bfcl_data_utils.py
git add bfcl_eval_utils.py

# 添加BFCL数据（只提交转换后的数据，不提交gorilla仓库）
git add bfcl/data/bfcl_test_200.json
git add bfcl/data/bfcl_test_simple.json

# 添加脚本和文档
git add tools/convert_bfcl_data.py
git add scripts/deploy/setup_bfcl.sh
git add scripts/experiments/RUN_BFCL_NOW.sh
git add scripts/experiments/run_bfcl_quick_test.sh
git add scripts/experiments/run_bfcl_full_exp.sh
git add scripts/tests/test_bfcl.sh
git add docs/BFCL_INTEGRATION_DESIGN.md
git add docs/BFCL_QUICK_START.md

# 添加.gitignore排除gorilla仓库
echo "bfcl/gorilla/" >> .gitignore
echo "generate_bfcl_data.py" >> .gitignore
git add .gitignore

echo ""
echo "步骤3: 提交"
echo "----------------------------------------"
git commit -m "🎯 Add BFCL (Berkeley Function Calling Leaderboard) multi-task evaluation

- Implement BFCL evaluation function (function call extraction + AST matching)
- Integrate multi-task evaluation (GSM8K + BFCL)
- Add BFCL official data conversion (258 samples)
- Add command-line arguments for BFCL configuration
- Fully compatible with existing Natural Niches workflow
- Update checkpoint frequency to save disk space

New files:
- bfcl_data_utils.py: BFCL data loading and preprocessing
- bfcl_eval_utils.py: Function call parser and evaluator
- bfcl/data/bfcl_test_200.json: Converted BFCL dataset
- scripts/experiments/RUN_BFCL_NOW.sh: One-click run script
- BFCL_QUICK_START.md: Usage guide

Modified files:
- natural_niches_sparsity_aware_fn.py: Add BFCL evaluation and multi-task support
- main_sparsity_aware.py: Add BFCL command-line arguments
"

echo ""
echo "步骤4: 推送到GitHub"
echo "----------------------------------------"
git push origin main

echo ""
echo "========================================"
echo "✅ 代码已推送到GitHub!"
echo "========================================"
echo ""
echo "下一步: 在服务器上执行"
echo "  cd /path/to/your/server/SparseFusion"
echo "  git pull origin main"
echo "  bash scripts/deploy/SERVER_SETUP.sh"
