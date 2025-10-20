# BFCL (Berkeley Function Calling Leaderboard) 集成设计

## 📋 **设计目标**

将BFCL的Function Calling评估集成到Natural Niches的fitness评估中，用于评估模型的工具调用能力。

---

## 🏗️ **架构设计**

### 1️⃣ **评估维度**

在现有的GSM8K评估基础上，增加BFCL评估维度：

```python
fitness_scores = {
    "gsm8k_accuracy": 0.75,      # 原有的数学推理能力
    "bfcl_accuracy": 0.82,        # 新增：工具调用准确率
    "bfcl_ast_accuracy": 0.78,    # 新增：AST语法正确率
    "bfcl_exec_accuracy": 0.85,   # 新增：可执行性准确率
}

# 加权总分
total_fitness = α₁ * gsm8k + α₂ * bfcl + α₃ * sparsity
```

### 2️⃣ **BFCL评估流程**

根据[BFCL Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)的设计：

```
输入: Function Definition + User Query
  ↓
模型生成: Function Call (JSON格式)
  ↓
评估指标:
  ├─ AST Match: 语法树匹配（参数顺序无关）
  ├─ Exec Match: 可执行性匹配（实际调用结果）
  └─ Overall: 综合准确率
```

---

## 💻 **实现方案**

### 方案A：多任务评估（推荐）

**优点：**
- ✅ 全面评估模型能力（数学 + 工具调用）
- ✅ 符合真实应用场景
- ✅ 可以为不同任务设置权重

**实现：**

```python
def create_multi_task_evaluation_fn(
    model_skeleton: torch.nn.Module,
    param_shapes: list,
    gsm8k_dataset,      # GSM8K数据集
    bfcl_dataset,       # BFCL数据集
    tokenizer: AutoTokenizer,
    task_weights: dict = {"gsm8k": 0.5, "bfcl": 0.5},
    eval_subset_size: int = None,
):
    """
    多任务评估函数：同时评估GSM8K和BFCL
    
    Args:
        task_weights: 任务权重，例如 {"gsm8k": 0.5, "bfcl": 0.5}
        eval_subset_size: 每个任务采样的样本数
    
    Returns:
        scores: shape (num_samples,) 包含所有任务的得分
    """
    
    # 创建两个评估函数
    gsm8k_eval_fn = create_evaluation_fn_for_llm(
        model_skeleton, param_shapes, gsm8k_dataset, tokenizer,
        eval_subset_size=eval_subset_size
    )
    
    bfcl_eval_fn = create_bfcl_evaluation_fn(
        model_skeleton, param_shapes, bfcl_dataset, tokenizer,
        eval_subset_size=eval_subset_size
    )
    
    def evaluation_fn(flat_params: jnp.ndarray) -> jnp.ndarray:
        # 评估两个任务
        gsm8k_scores = gsm8k_eval_fn(flat_params)  # shape: (n1,)
        bfcl_scores = bfcl_eval_fn(flat_params)    # shape: (n2,)
        
        # 方式1：拼接所有分数（保持per-sample粒度）
        all_scores = jnp.concatenate([gsm8k_scores, bfcl_scores])
        
        # 方式2：加权平均后广播（如果需要统一维度）
        # weighted_score = (
        #     task_weights["gsm8k"] * jnp.mean(gsm8k_scores) +
        #     task_weights["bfcl"] * jnp.mean(bfcl_scores)
        # )
        # all_scores = jnp.full(len(gsm8k_scores), weighted_score)
        
        return all_scores
    
    return evaluation_fn
```

### 方案B：单独BFCL评估

**优点：**
- ✅ 专注于工具调用能力
- ✅ 实现简单
- ✅ 可以快速验证

**实现：**

```python
def create_bfcl_evaluation_fn(
    model_skeleton: torch.nn.Module,
    param_shapes: list,
    bfcl_dataset,       # BFCL测试数据
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    eval_subset_size: int = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    创建BFCL评估函数
    
    BFCL评估流程：
    1. 输入：function definitions + user query
    2. 模型生成：function call (JSON格式)
    3. 评估：
       - AST Match: 参数和函数名匹配（顺序无关）
       - Exec Match: 实际执行结果匹配
       - Overall: 综合得分
    
    Returns:
        evaluation_fn: 返回shape (num_samples,) 的得分数组
    """
    base_model = (
        model_skeleton.module if hasattr(model_skeleton, "module") else model_skeleton
    )
    device = next(base_model.parameters()).device
    iteration_counter = {'count': 0}
    
    def evaluation_fn(flat_params: jnp.ndarray) -> jnp.ndarray:
        """评估单个模型在BFCL上的表现"""
        
        # 1. 加载参数到模型
        load_flat_params_to_model(base_model, flat_params, param_shapes)
        base_model.eval()
        
        # 2. 随机采样数据（如果启用subset evaluation）
        if eval_subset_size is not None and eval_subset_size < len(bfcl_dataset):
            import random
            iteration_counter['count'] += 1
            random.seed(42 + iteration_counter['count'])
            indices = random.sample(range(len(bfcl_dataset)), eval_subset_size)
            from torch.utils.data import Subset
            eval_dataset = Subset(bfcl_dataset, indices)
        else:
            eval_dataset = bfcl_dataset
        
        # 3. 创建DataLoader
        from torch.utils.data import DataLoader
        data_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=bfcl_collate_fn
        )
        
        # 4. 批量评估
        all_scores = []
        
        with torch.no_grad():
            for batch in data_loader:
                # batch包含:
                # - input_ids: 包含function definitions + user query
                # - ground_truth_calls: 正确的function call
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                ground_truth_calls = batch["ground_truth_calls"]  # List[dict]
                
                # 生成function call
                generated_ids = base_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                
                # 解码生成的文本
                generated_texts = tokenizer.batch_decode(
                    generated_ids[:, input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                
                # 评估每个样本
                for gen_text, gt_call in zip(generated_texts, ground_truth_calls):
                    # 提取生成的function call（JSON格式）
                    pred_call = extract_function_call(gen_text)
                    
                    # 评估准确性
                    score = evaluate_function_call(pred_call, gt_call)
                    all_scores.append(score)
        
        return jnp.array(all_scores)
    
    return evaluation_fn


def extract_function_call(text: str) -> dict:
    """
    从生成的文本中提取function call
    
    示例输入:
        "I will call the function: calculate_area(length=5, width=3)"
    
    示例输出:
        {
            "name": "calculate_area",
            "arguments": {"length": 5, "width": 3}
        }
    """
    import json
    import re
    
    # 尝试提取JSON格式
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    # 尝试提取函数调用格式: func_name(arg1=val1, arg2=val2)
    func_match = re.search(r'(\w+)\((.*?)\)', text)
    if func_match:
        func_name = func_match.group(1)
        args_str = func_match.group(2)
        
        # 解析参数
        args = {}
        for arg in args_str.split(','):
            if '=' in arg:
                key, val = arg.split('=', 1)
                args[key.strip()] = eval(val.strip())  # 注意：生产环境需要更安全的解析
        
        return {
            "name": func_name,
            "arguments": args
        }
    
    return {"name": None, "arguments": {}}


def evaluate_function_call(pred_call: dict, gt_call: dict) -> float:
    """
    评估function call的准确性
    
    BFCL评估标准：
    1. AST Match: 函数名和参数匹配（参数顺序无关）
    2. Exec Match: 实际执行结果匹配
    
    Args:
        pred_call: 预测的function call
        gt_call: ground truth function call
    
    Returns:
        score: 0.0 (错误) 或 1.0 (正确)
    """
    # 检查函数名
    if pred_call.get("name") != gt_call.get("name"):
        return 0.0
    
    # 检查参数（顺序无关）
    pred_args = pred_call.get("arguments", {})
    gt_args = gt_call.get("arguments", {})
    
    # 参数数量必须一致
    if len(pred_args) != len(gt_args):
        return 0.0
    
    # 每个参数的值必须匹配
    for key, gt_val in gt_args.items():
        if key not in pred_args:
            return 0.0
        
        pred_val = pred_args[key]
        
        # 数值比较（允许浮点误差）
        if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
            if abs(pred_val - gt_val) > 1e-6:
                return 0.0
        # 字符串比较
        elif str(pred_val).strip() != str(gt_val).strip():
            return 0.0
    
    return 1.0


def bfcl_collate_fn(batch):
    """BFCL数据集的collate函数"""
    import torch
    
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    ground_truth_calls = [item["ground_truth_call"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ground_truth_calls": ground_truth_calls
    }
```

---

## 📊 **数据准备**

### BFCL数据集格式

```python
# 数据集示例
bfcl_data = [
    {
        "functions": [
            {
                "name": "calculate_area",
                "description": "Calculate the area of a rectangle",
                "parameters": {
                    "length": {"type": "number", "description": "Length of rectangle"},
                    "width": {"type": "number", "description": "Width of rectangle"}
                }
            }
        ],
        "user_query": "What is the area of a rectangle with length 5 and width 3?",
        "ground_truth_call": {
            "name": "calculate_area",
            "arguments": {"length": 5, "width": 3}
        }
    },
    # ... more samples
]

# 预处理为Hugging Face Dataset格式
from datasets import Dataset

def preprocess_bfcl(example, tokenizer):
    """将BFCL数据转换为模型输入格式"""
    
    # 构建输入prompt
    functions_text = format_functions(example["functions"])
    user_query = example["user_query"]
    
    prompt = f"""Available functions:
{functions_text}

User query: {user_query}

Generate the appropriate function call:"""
    
    # Tokenize
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    
    return {
        "input_ids": encoded["input_ids"][0],
        "attention_mask": encoded["attention_mask"][0],
        "ground_truth_call": example["ground_truth_call"]
    }

def format_functions(functions):
    """格式化function definitions"""
    formatted = []
    for func in functions:
        params_str = ", ".join([
            f"{name}: {info['type']}" 
            for name, info in func["parameters"].items()
        ])
        formatted.append(f"- {func['name']}({params_str}): {func['description']}")
    return "\n".join(formatted)

# 加载和预处理
bfcl_dataset = Dataset.from_list(bfcl_data)
tokenized_bfcl = bfcl_dataset.map(
    lambda x: preprocess_bfcl(x, tokenizer),
    remove_columns=bfcl_dataset.column_names
)
```

---

## 🔧 **集成到Natural Niches**

### 修改主评估函数

```python
# 在 run_natural_niches_sparsity_aware() 中

# 原有的GSM8K评估
train_eval_fn = create_evaluation_fn_for_llm(
    model_skeleton,
    param_shapes,
    tokenized_train_dataset,  # GSM8K
    tokenizer,
    eval_subset_size=eval_subset_size,
)

# 新增：BFCL评估（可选）
if use_bfcl_eval:
    bfcl_eval_fn = create_bfcl_evaluation_fn(
        model_skeleton,
        param_shapes,
        tokenized_bfcl_dataset,
        tokenizer,
        eval_subset_size=eval_subset_size,
    )
    
    # 组合评估
    def combined_eval_fn(flat_params):
        gsm8k_scores = train_eval_fn(flat_params)
        bfcl_scores = bfcl_eval_fn(flat_params)
        
        # 拼接分数（保持per-sample粒度）
        return jnp.concatenate([gsm8k_scores, bfcl_scores])
    
    train_eval_fn = combined_eval_fn
    num_tasks = len(tokenized_train_dataset) + len(tokenized_bfcl_dataset)
```

---

## ⚡ **性能优化建议**

### 1. 使用Subset Evaluation

```python
# GSM8K: 30个样本
# BFCL: 30个样本
# 总共: 60个样本/次评估

eval_subset_size = 30  # 每个任务采样30个
```

### 2. 批量处理

```python
batch_size = 16  # 增大batch size加速
```

### 3. 缓存Function Definitions

```python
# 预先格式化function definitions，避免重复处理
cached_function_prompts = precompute_function_prompts(bfcl_dataset)
```

---

## 📈 **实验设置建议**

### 阶段1：验证BFCL评估

```bash
# 先单独测试BFCL评估
python main_sparsity_aware.py \
    --model1_path models/Qwen2.5-0.5B-Instruct \
    --model2_path models/Qwen2.5-0.5B-Instruct \
    --pop_size 2 \
    --total_forward_passes 100 \
    --eval_subset_size 30 \
    --use_bfcl_eval \
    --bfcl_weight 1.0  # 只用BFCL
```

### 阶段2：多任务联合训练

```bash
# GSM8K + BFCL 联合评估
python main_sparsity_aware.py \
    --model1_path models/Qwen2.5-0.5B-Instruct \
    --model2_path models/Qwen2.5-0.5B-Instruct \
    --pop_size 5 \
    --total_forward_passes 3000 \
    --eval_subset_size 30 \
    --use_multi_task_eval \
    --gsm8k_weight 0.5 \
    --bfcl_weight 0.5
```

---

## 🎯 **预期效果**

### Fitness Score组成

```python
# 原始fitness (只有GSM8K)
fitness_old = accuracy_gsm8k

# 新fitness (GSM8K + BFCL)
fitness_new = (
    0.5 * accuracy_gsm8k + 
    0.5 * accuracy_bfcl
)

# 加上sparsity
total_score = omega * fitness_new + beta * sparsity
```

### 预期训练曲线

```
Iteration | GSM8K Acc | BFCL Acc | Total Fitness | Sparsity
---------|-----------|----------|---------------|----------
   100   |   0.15    |   0.10   |     0.125     |   0.35
   500   |   0.25    |   0.20   |     0.225     |   0.28
  1000   |   0.32    |   0.28   |     0.300     |   0.22
  3000   |   0.38    |   0.35   |     0.365     |   0.18
```

---

## 📝 **命令行参数扩展**

```python
# 在 main_sparsity_aware.py 中添加
parser.add_argument('--use_bfcl_eval', action='store_true',
                   help='Enable BFCL function calling evaluation')
parser.add_argument('--bfcl_data_path', type=str, 
                   default='datasets/bfcl_test.json',
                   help='Path to BFCL test dataset')
parser.add_argument('--gsm8k_weight', type=float, default=0.5,
                   help='Weight for GSM8K task')
parser.add_argument('--bfcl_weight', type=float, default=0.5,
                   help='Weight for BFCL task')
```

---

## 🚀 **实施步骤**

1. **准备BFCL数据集**
   - 下载BFCL test set
   - 转换为Hugging Face Dataset格式
   - 预处理和tokenize

2. **实现评估函数**
   - `create_bfcl_evaluation_fn()`
   - `extract_function_call()`
   - `evaluate_function_call()`

3. **集成到主循环**
   - 修改`run_natural_niches_sparsity_aware()`
   - 添加多任务评估逻辑

4. **测试验证**
   - 单独测试BFCL评估
   - 验证多任务评估
   - 性能benchmarking

5. **全量实验**
   - 运行完整的进化算法
   - 对比单任务vs多任务效果

---

## 💡 **关键注意事项**

1. **数据维度一致性**
   - GSM8K: 200个样本 → 采样30个
   - BFCL: 100个样本 → 采样30个
   - 总任务数: 60个样本/次

2. **评估速度**
   - BFCL生成可能比GSM8K慢（需要生成JSON格式）
   - 建议使用较小的`max_new_tokens`（256足够）

3. **Prompt Engineering**
   - Function definitions的格式很重要
   - 需要引导模型生成标准JSON格式

4. **错误处理**
   - 模型可能生成不合法的JSON
   - 需要robust的parsing逻辑


