"""
BFCL数据加载和预处理工具
用于Berkeley Function Calling Leaderboard数据集
"""

import json
from typing import Dict, List, Any
from datasets import Dataset
from transformers import AutoTokenizer


def format_functions(functions: List[Dict[str, Any]]) -> str:
    """
    将function definitions格式化为可读的文本
    
    Args:
        functions: 函数定义列表
        
    Returns:
        格式化后的函数描述文本
    
    Example:
        Input: [{"name": "get_weather", "description": "Get weather", ...}]
        Output: "- get_weather(location: string, unit: string): Get weather\\n..."
    """
    formatted_lines = []
    
    for func in functions:
        func_name = func.get("name", "unknown")
        func_desc = func.get("description", "No description")
        
        # 提取参数信息
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])
        
        # 构建参数字符串
        param_strs = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "any")
            is_required = " (required)" if param_name in required else " (optional)"
            param_desc = param_info.get("description", "")
            param_strs.append(f"{param_name}: {param_type}{is_required} - {param_desc}")
        
        # 格式化单个函数
        if param_strs:
            params_text = "\\n    ".join(param_strs)
            formatted_lines.append(
                f"Function: {func_name}\\n"
                f"  Description: {func_desc}\\n"
                f"  Parameters:\\n    {params_text}"
            )
        else:
            formatted_lines.append(
                f"Function: {func_name}\\n"
                f"  Description: {func_desc}\\n"
                f"  Parameters: None"
            )
    
    return "\\n\\n".join(formatted_lines)


def create_bfcl_prompt(functions: List[Dict], user_query: str, tokenizer: AutoTokenizer) -> str:
    """
    创建BFCL任务的prompt
    
    Args:
        functions: 可用的函数列表
        user_query: 用户查询
        tokenizer: tokenizer（用于获取chat template）
        
    Returns:
        完整的prompt文本
    """
    functions_text = format_functions(functions)
    
    # 使用chat template格式（如果模型支持）
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can call functions. "
                    "Based on the user's request, you should call the appropriate function "
                    "by outputting a JSON object in the format: "
                    '{"name": "function_name", "arguments": {"arg1": "value1", "arg2": "value2"}}'
                )
            },
            {
                "role": "user",
                "content": (
                    f"Available functions:\\n{functions_text}\\n\\n"
                    f"User request: {user_query}\\n\\n"
                    f"Please call the appropriate function:"
                )
            }
        ]
        
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback to simple format
            prompt = (
                f"Available functions:\\n{functions_text}\\n\\n"
                f"User request: {user_query}\\n\\n"
                f"Function call (JSON format):"
            )
    else:
        # Simple format for models without chat template
        prompt = (
            f"Available functions:\\n{functions_text}\\n\\n"
            f"User request: {user_query}\\n\\n"
            f"Function call (JSON format):"
        )
    
    return prompt


def preprocess_bfcl_example(example: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int = 512) -> Dict[str, Any]:
    """
    预处理单个BFCL样本
    
    Args:
        example: BFCL数据样本
        tokenizer: tokenizer
        max_length: 最大序列长度
        
    Returns:
        处理后的样本，包含input_ids, attention_mask, ground_truth
    """
    # 创建prompt
    prompt = create_bfcl_prompt(
        functions=example["functions"],
        user_query=example["user_query"],
        tokenizer=tokenizer
    )
    
    # Tokenize
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None  # 返回list而不是tensor
    )
    
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "ground_truth": example["ground_truth"],
        "id": example.get("id", "unknown")
    }


def load_bfcl_dataset(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    subset_size: int = None
) -> Dataset:
    """
    加载和预处理BFCL数据集
    
    Args:
        data_path: BFCL数据文件路径（JSON格式）
        tokenizer: tokenizer
        max_length: 最大序列长度
        subset_size: 如果指定，只加载前N个样本
        
    Returns:
        Hugging Face Dataset对象
    """
    # 加载JSON数据
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 限制数据集大小
    if subset_size is not None and subset_size < len(raw_data):
        raw_data = raw_data[:subset_size]
    
    # 转换为Dataset
    dataset = Dataset.from_list(raw_data)
    
    # 预处理
    def preprocess_fn(example):
        return preprocess_bfcl_example(example, tokenizer, max_length)
    
    tokenized_dataset = dataset.map(
        preprocess_fn,
        remove_columns=dataset.column_names,
        desc="Tokenizing BFCL dataset",
        features=None,  # 禁用自动类型推断，保持原始类型
    )
    
    return tokenized_dataset


def bfcl_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    BFCL数据集的collate函数
    
    Args:
        batch: 批次数据
        
    Returns:
        整理后的批次数据
    """
    import torch
    
    # Stack input_ids and attention_mask
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    
    # Keep ground_truth and id as lists
    ground_truths = [item["ground_truth"] for item in batch]
    ids = [item.get("id", "unknown") for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ground_truth": ground_truths,
        "id": ids
    }


# ========== 测试代码 ==========
if __name__ == "__main__":
    """测试BFCL数据加载"""
    from transformers import AutoTokenizer
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-0.5B-Instruct")
    
    # 测试数据路径
    data_path = "bfcl/data/bfcl_test_simple.json"
    
    # 加载数据集
    print("Loading BFCL dataset...")
    dataset = load_bfcl_dataset(data_path, tokenizer, subset_size=5)
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # 查看第一个样本
    sample = dataset[0]
    print(f"\\n✓ Sample keys: {sample.keys()}")
    print(f"✓ Input shape: {len(sample['input_ids'])}")
    print(f"✓ Ground truth: {sample['ground_truth']}")
    
    # 测试collate_fn
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=bfcl_collate_fn)
    batch = next(iter(dataloader))
    
    print(f"\\n✓ Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"✓ Batch ground_truths: {batch['ground_truth']}")
    
    print("\\n✅ All tests passed!")

