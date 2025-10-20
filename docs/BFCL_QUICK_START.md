# 🎯 BFCL集成 - 快速开始指南

## ✅ **当前状态**

- ✅ BFCL数据集已下载并转换 (258个样本)
- ✅ 评估函数已实现 (Function Call解析 + AST匹配)
- ✅ 多任务评估已集成 (GSM8K + BFCL)
- ✅ 代码已完全集成到Natural Niches

---

## 🚀 **立即运行 - 3步搞定**

### **方式1: 一键运行（推荐）**

```bash
cd /fs-computility/pdz-grp1/yefei.p/SparseFusion

# 直接运行！
bash scripts/experiments/RUN_BFCL_NOW.sh
```

**这个脚本会：**
- ✅ 自动配置代理
- ✅ 设置环境变量
- ✅ 运行GSM8K + BFCL多任务实验
- ✅ Pop size = 5, 3000步, 每任务采样30个

---

### **方式2: 手动运行（更灵活）**

```bash
# 1. 设置代理（必须）
export https_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128
export http_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128

# 2. 设置环境
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=0

# 3. 运行实验
python main_sparsity_aware.py \
    --model1_path models/Qwen2.5-0.5B-Instruct \
    --model2_path models/Qwen2.5-0.5B-Instruct \
    --pop_size 5 \
    --total_forward_passes 3000 \
    --runs 1 \
    --omega 0.7 \
    --beta 0.3 \
    --pruning_sparsity 0.3 \
    --eval_subset_size 30 \
    --use_bfcl_eval \
    --bfcl_data_path bfcl/data/bfcl_test_200.json \
    --gsm8k_weight 0.5 \
    --bfcl_weight 0.5 \
    --output_dir results_bfcl_test
```

---

## 📊 **评估机制详解**

### **数据量**

```
GSM8K数据集:  200个训练样本
BFCL数据集:   258个训练样本
```

### **每次迭代评估**

```
--eval_subset_size 30 时:

每个模型每次迭代评估:
├─ GSM8K: 从200个中随机采样30个 → 30个分数
├─ BFCL:  从258个中随机采样30个 → 30个分数
└─ 总计: 60个任务分数

Competitive Normalization:
├─ 5个模型 × 60个任务
└─ 输出: 5个fitness分数
```

### **与GSM8K完全一致的机制**

```python
# GSM8K评估
gsm8k_scores = [1.0, 0.0, 1.0, ...]  # 30个
# BFCL评估  
bfcl_scores = [1.0, 1.0, 0.0, ...]   # 30个

# 拼接
all_scores = [gsm8k的30个, bfcl的30个]  # 60个

# Competitive Normalization (所有60个任务一起竞争)
fitness = competitive_normalize(all_scores)  # → 单个fitness值
```

---

## 🔧 **参数调整**

### **调整采样数量**

```bash
--eval_subset_size 50  # GSM8K 50个 + BFCL 50个 = 100个任务
```

### **调整任务权重**（暂未实现，目前平等对待）

```bash
--gsm8k_weight 0.6  # 将来可以调整采样比例
--bfcl_weight 0.4
```

### **只用BFCL（不用GSM8K）**

这需要修改代码，当前必须两个任务都用。

---

## 📁 **数据文件说明**

| 文件 | 样本数 | 用途 |
|------|--------|------|
| `bfcl/data/bfcl_test_simple.json` | 8个 | 测试用（太少） |
| `bfcl/data/bfcl_test_200.json` | 258个 | **正式实验用** ✅ |
| `bfcl/gorilla/...` | 官方数据 | BFCL原始数据 |

---

## 🐛 **常见问题**

### **Q1: BFCL数据集只有8个样本怎么办？**

**A:** 已解决！运行`python tools/convert_bfcl_data.py`转换官方数据，现在有258个样本。

### **Q2: 如何验证BFCL评估是否工作？**

**A:** 查看运行日志：
```bash
# 应该看到这样的输出:
🎯 加载BFCL数据集...
  数据路径: bfcl/data/bfcl_test_200.json
  ✓ BFCL数据集加载完成: 258 样本

🎯 创建多任务评估函数 (GSM8K + BFCL)
  GSM8K权重: 0.5
  BFCL权重: 0.5
  每个任务采样: 30 样本
```

### **Q3: 如何查看结果？**

**A:** 
```bash
# 查看保存的结果
ls -lh results_bfcl_*/

# 查看JSON日志
cat results_bfcl_*/*.json | grep -A10 "test_evaluations"

# 绘制曲线
python tools/plot_training_curves.py --input results_bfcl_*/*.pkl
```

---

## ⚠️ **重要提示**

1. **必须设置代理**：否则无法加载tokenizer
2. **数据集路径**：确保`bfcl/data/bfcl_test_200.json`存在
3. **模型路径**：确保`models/Qwen2.5-0.5B-Instruct`存在
4. **GPU内存**：建议至少20GB显存

---

## 📈 **预期运行时间**

```
Pop size=5, 3000 forward passes, eval_subset=30:

预计时间: 8-12小时

每次迭代:
├─ 父代选择: ~1s
├─ 剪枝: ~10s
├─ 交叉变异: ~2s
├─ 评估 (GSM8K 30个 + BFCL 30个): ~40s
└─ Archive更新: ~1s
总计: ~54s/iteration

3000步 × 54s ≈ 45小时 / 5个模型并行 ≈ 9小时
```

---

## ✅ **验证清单**

运行前确认：

- [ ] BFCL数据集存在: `ls bfcl/data/bfcl_test_200.json`
- [ ] 模型存在: `ls models/Qwen2.5-0.5B-Instruct/`
- [ ] 代理已设置: `echo $https_proxy`
- [ ] GPU可用: `nvidia-smi`

全部OK后运行：
```bash
bash scripts/experiments/RUN_BFCL_NOW.sh
```

---

## 🎯 **就这么简单！**

一键运行，坐等结果！ 🚀
