# 🚀 服务器端快速启动指南

## ✅ 代码已推送到GitHub！

提交信息: `c989828 - Add BFCL multi-task evaluation`
- 21个文件已提交
- BFCL数据 (380KB) 已包含
- 所有脚本和文档已就绪

---

## 📋 服务器端操作步骤

### **1️⃣ SSH登录服务器**

```bash
ssh your_username@your_server_address
```

### **2️⃣ 进入项目目录**

```bash
# 如果还没有克隆仓库:
git clone https://github.com/FeiYe0310/SparseFusion.git
cd SparseFusion

# 如果已有仓库:
cd /path/to/your/SparseFusion
git pull origin main
```

### **3️⃣ 验证代码完整性**

```bash
# 检查最新提交
git log --oneline -1
# 应该看到: c989828 🎯 Add BFCL...

# 检查BFCL数据
ls -lh bfcl/data/bfcl_test_200.json
# 应该看到: 380K

# 检查BFCL代码
ls -lh bfcl_*.py
# 应该看到: bfcl_data_utils.py, bfcl_eval_utils.py

# 检查脚本
ls -lh *.sh | grep -E "(SERVER|RUN_BFCL|test_bfcl)"
# 应该看到: SERVER_SETUP.sh, RUN_BFCL_NOW.sh, test_bfcl.sh等
```

### **4️⃣ 运行设置脚本**

```bash
bash SERVER_SETUP.sh
```

**这个脚本会自动：**
- ✅ 检查代码版本
- ✅ 验证BFCL数据 (258个样本)
- ✅ 检查模型路径
- ✅ 检查Python依赖 (jax, torch, transformers, datasets)
- ✅ 测试BFCL模块导入
- ✅ 设置脚本权限
- ✅ 检查GPU和磁盘空间

**预期输出：**
```
========================================
✅ 服务器设置完成!
========================================
```

### **5️⃣ 选择运行模式**

#### **方式A: 单元测试（2分钟，推荐先跑）**

```bash
bash test_bfcl.sh
```

**测试内容：**
- Function call 解析器
- AST 匹配评估器
- 数据加载

**预期输出：**
```
✅ 测试1: 正确的function call
✅ 测试2: 格式错误的输出
✅ 测试3: 完全匹配
✅ 测试4: 参数错误
✅ 所有测试通过！
```

#### **方式B: 快速验证实验（1-2小时）**

```bash
bash run_bfcl_quick_test.sh
```

**配置：**
- pop_size=2
- total_forward_passes=100
- eval_subset_size=30 (GSM8K 30 + BFCL 30)
- 输出目录: `results_bfcl_quick/`

#### **方式C: 完整实验（8-12小时，后台运行）**

```bash
# 前台运行（可以看到实时日志）
bash RUN_BFCL_NOW.sh

# 或后台运行
nohup bash RUN_BFCL_NOW.sh > bfcl_run.log 2>&1 &

# 查看日志
tail -f bfcl_run.log
```

**配置：**
- pop_size=5
- total_forward_passes=3000
- eval_subset_size=30
- omega=0.8, beta=0.4, pruning_sparsity=0.3
- 输出目录: `results_bfcl_multi_task/`

---

## ⚙️ 可选配置修改

### **修改代理（如果服务器需要）**

编辑 `SERVER_SETUP.sh` 或 `RUN_BFCL_NOW.sh`:

```bash
export https_proxy=http://your_proxy:port
export http_proxy=http://your_proxy:port
```

### **修改模型路径**

编辑 `RUN_BFCL_NOW.sh`:

```bash
--model1_path /your/path/to/model \
```

### **修改GPU设置**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用多GPU
```

### **调整实验参数**

编辑 `RUN_BFCL_NOW.sh`:

```bash
--pop_size 10 \              # 增大种群
--eval_subset_size 50 \      # 增加采样数
--gsm8k_weight 0.6 \         # 调整任务权重
--bfcl_weight 0.4 \
```

---

## 📊 监控和查看结果

### **查看运行状态**

```bash
# 实时日志
tail -f bfcl_run.log

# GPU使用
watch -n 1 nvidia-smi

# 进程
ps aux | grep python | grep main_sparsity

# 后台任务
jobs
```

### **查看中间结果**

```bash
# 列出结果目录
ls -lh results_bfcl_*/

# 查看results.pkl（如果有plotting脚本）
python plot_training_curves.py --input results_bfcl_*/*.pkl
```

### **停止实验**

```bash
# 如果是前台运行: Ctrl+C

# 如果是后台运行:
# 1. 查找进程ID
ps aux | grep python | grep main_sparsity

# 2. 终止进程
kill <PID>

# 或用pkill
pkill -f main_sparsity_aware
```

---

## 🐛 常见问题

### **Q1: git pull 失败？**

```bash
# 查看状态
git status

# 如果有本地修改，暂存：
git stash

# 强制同步远程
git fetch origin main
git reset --hard origin/main

# 恢复暂存（如果需要）
git stash pop
```

### **Q2: BFCL数据不存在？**

```bash
# 重新拉取
git pull origin main

# 检查文件
ls -lh bfcl/data/

# 如果还是没有，手动从GitHub下载
wget https://github.com/FeiYe0310/SparseFusion/raw/main/bfcl/data/bfcl_test_200.json \
  -O bfcl/data/bfcl_test_200.json
```

### **Q3: 模型加载失败？**

```bash
# 检查模型路径
ls models/

# 修改RUN_BFCL_NOW.sh中的模型路径
# 或者下载模型:
cd models
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
```

### **Q4: 磁盘空间不足？**

```bash
# 检查空间
df -h .

# 删除旧checkpoint（已设置1000步一存，占用小很多）
rm -rf results_*/checkpoint_*.pkl

# 或只保留最终模型
find results_* -name "checkpoint_*.pkl" -delete
```

### **Q5: 依赖缺失？**

```bash
# 安装缺失的库
pip install jax torch transformers datasets

# 或使用conda
conda install jax pytorch transformers datasets -c conda-forge
```

---

## 📈 预期结果

### **快速测试 (100步):**
- 运行时间: ~1-2小时
- 检查点: 无（太短）
- 最终输出: `results_bfcl_quick/best_model.pkl`, `results.pkl`

### **完整实验 (3000步):**
- 运行时间: ~8-12小时
- 检查点: 每1000步
- 最终输出:
  - `results_bfcl_multi_task/best_model.pkl`
  - `results_bfcl_multi_task/results.pkl`
  - 训练曲线: fitness_history, sparsity_history, best_fitness_history

### **日志示例:**

```
========================================
🚀 BFCL多任务评估 - Natural Niches
========================================
配置: pop_size=5, omega=0.8, beta=0.4
使用BFCL评估: True
GSM8K权重: 0.5, BFCL权重: 0.5
BFCL数据: bfcl/data/bfcl_test_200.json
----------------------------------------

迭代 1/3000
  ⏱️  评估耗时: 35.2s
  📊 当前fitness: 0.45 (BFCL: 0.38, GSM8K: 0.52)
  🎯 稀疏度: 28.3%
  📈 最佳fitness: 0.45

迭代 100/3000
  ⏱️  评估耗时: 32.1s
  📊 当前fitness: 0.62 (BFCL: 0.58, GSM8K: 0.66)
  🎯 稀疏度: 29.7%
  📈 最佳fitness: 0.68
  💾 保存checkpoint...
```

---

## 🎯 总结：最简化流程

```bash
# 在服务器上执行这4个命令：

cd /path/to/SparseFusion
git pull origin main
bash SERVER_SETUP.sh
nohup bash RUN_BFCL_NOW.sh > run.log 2>&1 &
tail -f run.log
```

**就这么简单！** 🚀

---

## 📞 需要帮助？

- 查看详细部署文档: `DEPLOY_TO_SERVER.md`
- 查看BFCL设计: `BFCL_INTEGRATION_DESIGN.md`
- 查看快速开始: `BFCL_QUICK_START.md`

祝实验顺利！ 🎉

