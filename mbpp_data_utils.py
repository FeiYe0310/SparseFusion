#!/usr/bin/env python3
"""
MBPP数据加载与处理工具
支持MBPP (Mostly Basic Python Problems)数据集的加载、预处理和批处理
"""

import os
import json
import torch
from typing import Dict, List, Any
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk


class MBPPDataset(Dataset):
    """
    MBPP数据集类，支持从HuggingFace datasets加载或本地文件加载
    """
    
    def __init__(self, data_path: str, tokenizer=None, split="test"):
        """
        Args:
            data_path: MBPP数据集标识符 (如 'mbpp') 或本地文件路径
            tokenizer: HuggingFace tokenizer
            split: 数据集划分 ('test', 'train', 'validation')
        """
        self.data = self._load_data(data_path, split)
        self.tokenizer = tokenizer
    
    def _load_data(self, data_path: str, split: str) -> List[Dict]:
        """从HuggingFace datasets或本地文件加载MBPP数据"""
        if os.path.exists(data_path):
            # 本地路径：区分目录(HF保存的dataset)与文件(JSON/JSONL)
            if os.path.isdir(data_path):
                print(f"Loading MBPP data from local HF dataset dir: {data_path}")
                ds = load_from_disk(data_path)
                # 选择split
                chosen_split = split if split in ds else (
                    'test' if 'test' in ds else ('validation' if 'validation' in ds else 'train')
                )
                data = list(ds[chosen_split])
            else:
                print(f"Loading MBPP data from local file: {data_path}")
                with open(data_path, 'r', encoding='utf-8') as f:
                    if data_path.endswith('.jsonl'):
                        data = [json.loads(line) for line in f if line.strip()]
                    else:
                        data = json.load(f)
        else:
            # 从HuggingFace datasets加载
            print(f"Loading MBPP data from HuggingFace Hub: '{data_path}' (split: {split})")
            try:
                dataset = load_dataset(data_path, 'sanitized', split=split)
                data = list(dataset)
            except Exception as e:
                print(f"Failed to load from HuggingFace Hub. Error: {e}")
                # Fallback to default mbpp if 'sanitized' fails
                try:
                    dataset = load_dataset(data_path, split=split)
                    data = list(dataset)
                except Exception as e_fallback:
                    print(f"Fallback to default MBPP also failed. Error: {e_fallback}")
                    raise ValueError("Could not load MBPP dataset from Hub or local path.")

        if isinstance(data, dict):
            data = [data]
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建prompt（指导模型生成代码）
        prompt = self._build_prompt(item)
        
        return {
            "task_id": item.get("task_id", idx),
            "prompt": prompt,
            "text": item.get("text", ""),
            "test_list": item.get("test_list", []),
            "test_setup_code": item.get("test_setup_code", ""),
            "challenge_test_list": item.get("challenge_test_list", []),
            "reference_code": item.get("code", ""),  # 参考实现
        }
    
    def _build_prompt(self, item: Dict) -> str:
        """
        使用官方字段生成提示：严格返回数据集中已有的英文描述，不做包装。
        优先顺序：prompt > text > description > task_description
        """
        for key in ("prompt", "text", "description", "task_description"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""


def mbpp_collate_fn(batch, tokenizer, max_length=512):
    """
    MBPP批处理函数
    
    Args:
        batch: 一批样本
        tokenizer: HuggingFace tokenizer
        max_length: 最大序列长度
    
    Returns:
        批处理后的字典，包含：
        - input_ids: tensor (batch_size, seq_len)
        - attention_mask: tensor (batch_size, seq_len)
        - test_list: 测试用例列表
        - task_ids: 任务ID列表
    """
    prompts = [item["prompt"] for item in batch]
    
    # Tokenize
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "test_list": [item.get("test_list", []) for item in batch],
        "test_setup_code": [item.get("test_setup_code", "") for item in batch],
        "test_imports": [item.get("test_imports", "") for item in batch],
        "reference_code": [item.get("reference_code", "") for item in batch],
        "task_ids": [item["task_id"] for item in batch],
        "prompts": prompts,  # 保留原始prompt用于调试
    }

