# 🚀 部署到服务器 - 完整操作指南

## 📋 **部署流程总览**

```
当前机器 (开发环境)          服务器 (运行环境)
      ↓                            ↓
  1. 提交代码到GitHub         4. 拉取代码
  2. 推送数据                 5. 设置环境
  3. 推送到远程              6. 运行实验
```

---

## 🔄 **方案A: 代码通过GitHub，数据也通过GitHub（推荐）**

### **优点：**
- ✅ 最简单，一次同步所有内容
- ✅ BFCL数据已转换好(380KB)，可以直接提交
- ✅ 服务器端无需额外处理数据

### **步骤：**

#### **在当前机器（开发环境）：**

```bash
cd /fs-computility/pdz-grp1/yefei.p/SparseFusion

# 1. 提交并推送所有代码和数据
bash PUSH_TO_GITHUB.sh

# 或手动执行：
git add .
git commit -m "🎯 Add BFCL multi-task evaluation support"
git push origin main
```

#### **在服务器：**

```bash
# 1. 进入项目目录
cd /path/to/your/SparseFusion

# 2. 拉取最新代码
git pull origin main

# 3. 运行设置脚本
bash SERVER_SETUP.sh

# 4. 运行实验
bash RUN_BFCL_NOW.sh
```

**就这么简单！** 数据随代码一起同步。

---

## 🔄 **方案B: 代码通过GitHub，数据在服务器重新生成**

### **优点：**
- ✅ 不污染Git仓库（数据不提交）
- ✅ 使用最新的BFCL官方数据

### **缺点：**
- ⚠️ 需要在服务器下载gorilla仓库（可能需要代理）

### **步骤：**

#### **在当前机器（开发环境）：**

```bash
cd /fs-computility/pdz-grp1/yefei.p/SparseFusion

# 提交代码（不包含大数据文件）
git add natural_niches_sparsity_aware_fn.py
git add main_sparsity_aware.py
git add bfcl_data_utils.py
git add bfcl_eval_utils.py
git add convert_bfcl_data.py
git add *.sh *.md

git commit -m "🎯 Add BFCL multi-task evaluation (code only)"
git push origin main
```

#### **在服务器：**

```bash
cd /path/to/your/SparseFusion

# 1. 拉取代码
git pull origin main

# 2. 下载BFCL数据并转换
bash setup_bfcl.sh  # 会下载gorilla仓库
python convert_bfcl_data.py  # 转换数据

# 3. 设置环境
bash SERVER_SETUP.sh

# 4. 运行实验
bash RUN_BFCL_NOW.sh
```

---

## 📊 **数据文件说明**

| 文件 | 大小 | 是否提交GitHub | 说明 |
|------|------|----------------|------|
| `bfcl/data/bfcl_test_200.json` | 380KB | ✅ 推荐 | 转换后的数据，可提交 |
| `bfcl/data/bfcl_test_simple.json` | 9KB | ✅ 可以 | 测试数据 |
| `bfcl/gorilla/` | 42MB | ❌ 不要 | BFCL官方仓库，太大 |

**推荐做法：**
```bash
# .gitignore中添加：
echo "bfcl/gorilla/" >> .gitignore

# 提交转换后的数据：
git add bfcl/data/bfcl_test_200.json
```

---

## 🚀 **完整命令序列（推荐流程）**

### **当前机器：**

```bash
# 1. 进入项目目录
cd /fs-computility/pdz-grp1/yefei.p/SparseFusion

# 2. 检查修改
git status

# 3. 添加文件（包含BFCL数据）
git add natural_niches_sparsity_aware_fn.py
git add main_sparsity_aware.py
git add bfcl_data_utils.py
git add bfcl_eval_utils.py
git add bfcl/data/bfcl_test_200.json
git add convert_bfcl_data.py
git add *.sh
git add BFCL*.md

# 4. 排除gorilla仓库
echo "bfcl/gorilla/" >> .gitignore
git add .gitignore

# 5. 提交
git commit -m "🎯 Add BFCL multi-task evaluation

Features:
- Multi-task evaluation (GSM8K + BFCL)
- BFCL function call parser and evaluator
- 258 BFCL samples included
- One-click run scripts
- Full documentation

Usage:
  bash RUN_BFCL_NOW.sh
"

# 6. 推送
git push origin main
```

### **服务器：**

```bash
# 1. SSH登录服务器
ssh your_server

# 2. 进入项目目录
cd /path/to/your/SparseFusion  # ← 修改为实际路径

# 3. 拉取代码
git pull origin main

# 4. 检查代码是否完整
ls -lh bfcl/data/bfcl_test_200.json  # 应该看到380KB
ls -lh bfcl_*.py  # 应该看到两个py文件

# 5. 运行设置脚本
bash SERVER_SETUP.sh

# 6a. 快速测试（2分钟）
bash test_bfcl.sh

# 6b. 或小规模验证（1小时）
bash run_bfcl_quick_test.sh

# 6c. 或完整实验（8-12小时，后台运行）
nohup bash RUN_BFCL_NOW.sh > bfcl_run.log 2>&1 &
tail -f bfcl_run.log  # 查看日志

# 查看后台任务
jobs
ps aux | grep python | grep main_sparsity
```

---

## ⚙️ **服务器环境配置**

### **必须配置：**

```bash
# 1. 设置GPU
export CUDA_VISIBLE_DEVICES=0  # 或你的GPU编号

# 2. 设置JAX
export JAX_PLATFORM_NAME=cpu   # 或 gpu

# 3. 设置代理（如果需要）
export https_proxy=YOUR_PROXY
export http_proxy=YOUR_PROXY
```

### **可选配置（加速）：**

```bash
# 使用更快的GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 增加并行度
# 修改RUN_BFCL_NOW.sh中的POP_SIZE
```

---

## 🐛 **常见问题**

### **Q1: git push失败？**

```bash
# 检查远程仓库
git remote -v

# 如果没有远程仓库，添加：
git remote add origin YOUR_GITHUB_REPO_URL

# 强制推送（谨慎使用）
git push -f origin main
```

### **Q2: 服务器上git pull失败？**

```bash
# 查看冲突
git status

# 放弃本地修改
git reset --hard origin/main

# 或保存本地修改
git stash
git pull
git stash pop
```

### **Q3: BFCL数据文件缺失？**

```bash
# 方案1: 重新git pull
git pull origin main

# 方案2: 手动下载并转换
bash setup_bfcl.sh
python convert_bfcl_data.py

# 方案3: 从当前机器scp传输
# 在当前机器：
scp bfcl/data/bfcl_test_200.json user@server:/path/to/SparseFusion/bfcl/data/
```

### **Q4: 模型文件不存在？**

```bash
# 检查模型路径
ls models/

# 修改RUN_BFCL_NOW.sh使用正确的模型路径
# 将--model1_path改为实际路径
```

---

## 📊 **监控运行状态**

```bash
# 查看实时日志
tail -f bfcl_run.log

# 查看GPU使用
nvidia-smi

# 查看进程
ps aux | grep python

# 查看结果
ls -lh results_bfcl_*/

# 中途查看结果
python plot_training_curves.py --input results_bfcl_*/*.pkl
```

---

## ✅ **验证清单**

部署到服务器前检查：

**当前机器：**
- [ ] 代码已提交到Git
- [ ] BFCL数据已添加（或准备在服务器生成）
- [ ] .gitignore已排除gorilla目录
- [ ] 已推送到GitHub

**服务器：**
- [ ] Git pull成功
- [ ] BFCL数据存在（380KB）
- [ ] 模型文件存在
- [ ] Python依赖完整
- [ ] GPU可用
- [ ] 磁盘空间充足（至少10GB）

---

## 🎯 **总结：最简单的部署流程**

```bash
# === 当前机器 ===
cd /fs-computility/pdz-grp1/yefei.p/SparseFusion
bash PUSH_TO_GITHUB.sh  # 一键提交推送

# === 服务器 ===
cd /path/to/SparseFusion
git pull origin main          # 拉取代码
bash SERVER_SETUP.sh          # 设置环境
nohup bash RUN_BFCL_NOW.sh > run.log 2>&1 &  # 后台运行
tail -f run.log               # 监控日志
```

**就这4个命令！** 🚀

