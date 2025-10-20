# 🚀 GitHub同步 - 完整操作指南

## 📤 推送代码到GitHub（带代理配置）

### **一键执行：**

```bash
cd /fs-computility/pdz-grp1/yefei.p/SparseFusion

# 1. 配置代理
export https_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128
export http_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128

# 2. 检查状态
git status

# 3. 添加所有修改
git add .

# 4. 提交
git commit -m "🔧 Update BFCL scripts with proxy configuration"

# 5. 推送
git push origin main
```

---

## 📥 服务器端拉取代码

### **在服务器执行：**

```bash
cd /path/to/SparseFusion

# 如果需要代理拉取
export https_proxy=YOUR_SERVER_PROXY
export http_proxy=YOUR_SERVER_PROXY

# 拉取代码
git pull origin main

# 检查
ls -lh bfcl/data/bfcl_test_200.json
```

---

## ✅ 现在执行推送

```bash
# 配置代理并推送
export https_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128
export http_proxy=https://yefei.p:AolVNTblt0qW28ZwuxuiwM0zKdisK5oSwW1sa3N8UHb1DclJo6kl67yzxe5g@volc-proxy.pjlab.org.cn:13128
git push origin main
```

