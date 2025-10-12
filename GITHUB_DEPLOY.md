# GitHub 部署指南

## 📦 上传到 GitHub

```bash
cd /fs-computility/pdz-grp1/yefei.p/SparseFusion

# 初始化 git
git init

# 添加所有文件（.gitignore 会自动排除大文件）
git add .

# 提交
git commit -m "Initial commit: Sparsity-Aware Natural Niches"

# 关联远程仓库（替换为您的 GitHub 仓库地址）
git remote add origin https://github.com/YOUR_USERNAME/SparseFusion.git

# 推送
git push -u origin main
```

## 🔽 集群上克隆

```bash
# SSH 到集群
ssh user@cluster.address

# 克隆代码
git clone https://github.com/YOUR_USERNAME/SparseFusion.git
cd SparseFusion

# 准备运行环境
pip install torch jax transformers datasets numpy tqdm

# 测试运行
python main_sparsity_aware.py --debug_models --pop_size 2 --total_forward_passes 2
```

## 📋 检查要上传的文件

```bash
# 查看将要上传的文件
git status

# 确认排除了大文件
git ls-files | grep -E '(models|datasets|results|__pycache__|\.pkl|\.npz)'
# 应该没有输出
```
