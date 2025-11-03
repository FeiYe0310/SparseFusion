"""
Unified Evaluation Utilities for SparseFusion

All evaluation utilities consolidated in one place:
- GSM8K: Math reasoning
- BFCL: Berkeley Function Calling Leaderboard  
- MBPP: Mostly Basic Python Problems
- DoT: Depth-of-Thought tasks

Import everything from this module:
    from utils.eval_utils import extract_answer, MBPPDataset, extract_function_call, ...
"""

import re
import json
import os
import random
import torch
from typing import Dict, List, Any, Optional
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

# ==============================================================================
# GSM8K Utilities
# ==============================================================================

def extract_answer(text: str) -> str:
    """从GSM8K生成文本中提取数字答案（####后面）"""
    if "####" in text:
        answer = text.split("####")[-1].strip()
    else:
        answer = text.strip()
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer.replace(",", ""))
    return numbers[-1] if numbers else ""

# ==============================================================================
# BFCL (Berkeley Function Calling) Utilities  
# ==============================================================================

def extract_function_call(text: str) -> Dict[str, Any]:
    """从模型输出中提取function call"""
    text = text.strip()
    
    # 方法1: JSON格式
    try:
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "name" in data and "arguments" in data:
                return {"name": data["name"], "arguments": data["arguments"]}
            elif "function" in data and "parameters" in data:
                return {"name": data["function"], "arguments": data["parameters"]}
            elif "name" in data and "parameters" in data:
                return {"name": data["name"], "arguments": data["parameters"]}
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    
    # 方法2: 函数调用格式 func(arg1=val1, arg2=val2)
    func_call_match = re.search(r'(\w+)\s*\((.*?)\)', text, re.DOTALL)
    if func_call_match:
        func_name = func_call_match.group(1)
        args_str = func_call_match.group(2).strip()
        args = {}
        if args_str:
            arg_pattern = r'(\w+)\s*[:=]\s*([^,]+)'
            for match in re.finditer(arg_pattern, args_str):
                key = match.group(1).strip()
                value_str = match.group(2).strip()
                try:
                    if (value_str.startswith('"') and value_str.endswith('"')) or \
                       (value_str.startswith("'") and value_str.endswith("'")):
                        value = value_str[1:-1]
                    elif '.' in value_str:
                        value = float(value_str)
                    elif value_str.isdigit() or (value_str.startswith('-') and value_str[1:].isdigit()):
                        value = int(value_str)
                    elif value_str.lower() == 'true':
                        value = True
                    elif value_str.lower() == 'false':
                        value = False
                    elif value_str.lower() in ('null', 'none'):
                        value = None
                    else:
                        value = value_str
                    args[key] = value
                except (ValueError, AttributeError):
                    args[key] = value_str
        return {"name": func_name, "arguments": args}
    
    # Fallback
    return {"name": None, "arguments": {}}


def evaluate_function_call(predicted: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """评估function call是否正确（AST matching）"""
    if predicted["name"] != expected["name"]:
        return False
    pred_args = predicted.get("arguments", {})
    exp_args = expected.get("arguments", {})
    
    if set(pred_args.keys()) != set(exp_args.keys()):
        return False
    
    for key in exp_args:
        if pred_args[key] != exp_args[key]:
            return False
    
    return True


def format_functions(functions: List[Dict[str, Any]]) -> str:
    """格式化function definitions为可读文本"""
    formatted_lines = []
    for func in functions:
        func_name = func.get("name", "unknown")
        func_desc = func.get("description", "No description")
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])
        
        param_strs = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "any")
            is_required = " (required)" if param_name in required else " (optional)"
            param_desc = param_info.get("description", "")
            param_strs.append(f"{param_name}: {param_type}{is_required} - {param_desc}")
        
        if param_strs:
            params_text = "\n    ".join(param_strs)
            formatted_lines.append(
                f"Function: {func_name}\n  Description: {func_desc}\n  Parameters:\n    {params_text}"
            )
        else:
            formatted_lines.append(
                f"Function: {func_name}\n  Description: {func_desc}\n  Parameters: None"
            )
    return "\n\n".join(formatted_lines)


def load_bfcl_dataset(data_path: str, tokenizer: AutoTokenizer) -> Dataset:
    """加载BFCL数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    
    processed_data = []
    for item in data:
        functions = item.get("functions", item.get("available_functions", []))
        query = item.get("query", item.get("user_request", ""))
        expected_call = item.get("expected_function_call", item.get("ground_truth", {}))
        
        functions_text = format_functions(functions)
        prompt = f"Available functions:\n{functions_text}\n\nUser request: {query}\n\nPlease call the appropriate function:"
        
        processed_data.append({
            "prompt": prompt,
            "functions": functions,
            "query": query,
            "expected_call": expected_call,
        })
    
    return Dataset.from_list(processed_data)


def bfcl_collate_fn(batch):
    """BFCL任务的collate函数"""
    return {
        "prompts": [item["prompt"] for item in batch],
        "expected_calls": [item["expected_call"] for item in batch],
        "functions": [item["functions"] for item in batch],
    }

# ==============================================================================
# MBPP (Mostly Basic Python Problems) Utilities
# ==============================================================================

class MBPPDataset(TorchDataset):
    """MBPP数据集类"""
    
    def __init__(self, data_path: str, tokenizer=None, split="test"):
        self.data = self._load_data(data_path, split)
        self.tokenizer = tokenizer
    
    def _load_data(self, data_path: str, split: str) -> List[Dict]:
        if os.path.exists(data_path):
            if os.path.isdir(data_path):
                ds = load_from_disk(data_path)
                chosen_split = split if split in ds else ('test' if 'test' in ds else 'train')
                data = list(ds[chosen_split])
            else:
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f] if data_path.endswith('.jsonl') else json.load(f)
        else:
            try:
                dataset = load_dataset(data_path, 'sanitized', split=split)
                data = list(dataset)
            except Exception:
                dataset = load_dataset(data_path, split=split)
                data = list(dataset)
        
        return data if isinstance(data, list) else [data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get("prompt") or item.get("text") or item.get("description") or ""
        
        return {
            "task_id": item.get("task_id", idx),
            "prompt": prompt,
            "text": item.get("text", ""),
            "test_list": item.get("test_list", []),
            "test_setup_code": item.get("test_setup_code", ""),
            "test_import": item.get("test_imports", ""),
            "challenge_test_list": item.get("challenge_test_list", []),
            "reference_code": item.get("code", ""),
        }


def mbpp_collate_fn(batch):
    """MBPP任务的collate函数"""
    return {
        "task_ids": [item["task_id"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "test_lists": [item["test_list"] for item in batch],
        "setup_codes": [item["test_setup_code"] for item in batch],
        "test_imports": [item["test_import"] for item in batch],
        "reference_codes": [item["reference_code"] for item in batch],
    }

# ==============================================================================
# DoT (Depth-of-Thought) Tasks Utilities
# ==============================================================================

def generate_mult_dataset(num_samples: int, digits: int = 4, seed: int = 42) -> List[Dict]:
    """生成多位数乘法数据集"""
    assert digits in (4, 5), "digits must be 4 or 5"
    rng = random.Random(seed)
    low, high = 10 ** (digits - 1), 10 ** digits - 1
    data = []
    for _ in range(num_samples):
        a, b = rng.randint(low, high), rng.randint(low, high)
        prompt = f"Compute the product: {a} × {b}. Only output the final integer, no text."
        data.append({"prompt": prompt, "gold": a * b})
    return data


def generate_bool_dataset(num_samples: int, num_vars: int = 4, seed: int = 42) -> List[Dict]:
    """生成布尔逻辑表达式数据集"""
    rng = random.Random(seed)
    
    def _rand_bool_expr(vars_vals: Dict[str, bool], max_depth: int = 2):
        if max_depth <= 0 or rng.random() < 0.4:
            var = rng.choice(list(vars_vals.keys()))
            if rng.random() < 0.5:
                return (f"not {var}", not vars_vals[var])
            return (var, vars_vals[var])
        
        left_expr, left_val = _rand_bool_expr(vars_vals, max_depth - 1)
        right_expr, right_val = _rand_bool_expr(vars_vals, max_depth - 1)
        op = rng.choice(["and", "or", "xor"])
        
        if op == "and":
            result_val = left_val and right_val
        elif op == "or":
            result_val = left_val or right_val
        else:
            result_val = left_val != right_val
        
        return (f"({left_expr} {op} {right_expr})", result_val)
    
    data = []
    var_names = [f"v{i}" for i in range(num_vars)]
    for _ in range(num_samples):
        vars_vals = {v: rng.choice([True, False]) for v in var_names}
        expr_str, gold = _rand_bool_expr(vars_vals, 2)
        var_lines = [f"{v} = {'True' if val else 'False'}" for v, val in vars_vals.items()]
        prompt = "Given:\n" + "\n".join(var_lines) + f"\n\nEvaluate: {expr_str}\nOnly output True or False, no text."
        data.append({"prompt": prompt, "gold": gold})
    return data


def parse_int_from_text(text: str) -> int:
    """从生成文本中解析整数"""
    numbers = re.findall(r'-?\d+', text.replace(',', ''))
    return int(numbers[-1]) if numbers else 0


def parse_bool_from_text(text: str) -> bool:
    """从生成文本中解析布尔值"""
    text_lower = text.strip().lower()
    return 'true' in text_lower if 'true' in text_lower or 'false' in text_lower else False


def dot_collate(batch):
    """DoT任务的collate函数"""
    return {
        "prompts": [item["prompt"] for item in batch],
        "golds": [item["gold"] for item in batch],
    }

# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    # GSM8K
    "extract_answer",
    # BFCL
    "extract_function_call",
    "evaluate_function_call",
    "format_functions",
    "load_bfcl_dataset",
    "bfcl_collate_fn",
    # MBPP
    "MBPPDataset",
    "mbpp_collate_fn",
    # DoT
    "generate_mult_dataset",
    "generate_bool_dataset",
    "parse_int_from_text",
    "parse_bool_from_text",
    "dot_collate",
]
