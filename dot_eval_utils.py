#!/usr/bin/env python3
"""
DoT任务（4x4/5x5乘法、布尔逻辑）的在线数据生成与评测辅助工具。
提供：
- 随机生成多位数乘法与布尔逻辑样本（可控数量与随机种子）
- Prompt构建（要求仅输出最终答案）
- 预测输出解析（数值/布尔标准化）
- 简单collate以便tokenizer批处理
"""

import random
import re
from typing import List, Dict, Tuple


# ========== 数据生成 ==========

def generate_mult_dataset(num_samples: int, digits: int = 4, seed: int = 42) -> List[Dict]:
    assert digits in (4, 5), "digits must be 4 or 5"
    rng = random.Random(seed)
    low = 10 ** (digits - 1)
    high = 10 ** digits - 1
    data: List[Dict] = []
    for _ in range(num_samples):
        a = rng.randint(low, high)
        b = rng.randint(low, high)
        gold = a * b
        prompt = (
            f"Compute the product: {a} × {b}. Only output the final integer, no text."
        )
        data.append({"prompt": prompt, "gold": gold})
    return data


def _rand_bool_expr_with_val(
    rng: random.Random, vars_vals: Dict[str, bool], max_depth: int = 2
) -> tuple[str, bool]:
    """
    递归同时生成布尔表达式字符串与其真值，避免运行时 eval。
    支持一元 not 与二元 and/or/xor。
    """
    # 原子
    if max_depth <= 0 or rng.random() < 0.4:
        var = rng.choice(list(vars_vals.keys()))
        # 可选一元 not
        if rng.random() < 0.5:
            return (f"not {var}", (not vars_vals[var]))
        return (var, vars_vals[var])

    # 二元组合
    left_expr, left_val = _rand_bool_expr_with_val(rng, vars_vals, max_depth - 1)
    right_expr, right_val = _rand_bool_expr_with_val(rng, vars_vals, max_depth - 1)
    op = rng.choice(["and", "or", "xor"])  # 支持xor
    if op == "and":
        return (f"({left_expr} and {right_expr})", (left_val and right_val))
    if op == "or":
        return (f"({left_expr} or {right_expr})", (left_val or right_val))
    # xor
    return (f"({left_expr} xor {right_expr})", (left_val != right_val))


def generate_bool_dataset(num_samples: int, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    data: List[Dict] = []
    for _ in range(num_samples):
        vals = {
            "A": bool(rng.getrandbits(1)),
            "B": bool(rng.getrandbits(1)),
            "C": bool(rng.getrandbits(1)),
        }
        expr, gold = _rand_bool_expr_with_val(rng, vals, max_depth=2)
        prompt = (
            f"Given A={vals['A']}, B={vals['B']}, C={vals['C']}, evaluate: {expr}. "
            f"Answer 'True' or 'False' only."
        )
        data.append({"prompt": prompt, "gold": gold})
    return data


# ========== 解析与标准化 ==========

def parse_int_from_text(text: str) -> int | None:
    # 提取首个有符号整数
    m = re.search(r"[-+]?\d+", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def parse_bool_from_text(text: str) -> bool | None:
    t = text.strip().lower()
    if t in ("true", "1", "yes"):  # 宽松一些
        return True
    if t in ("false", "0", "no"):
        return False
    # 兜底：提取第一个单词
    m = re.search(r"\b(true|false|1|0)\b", t)
    if m:
        return True if m.group(1) in ("true", "1") else False
    return None


# ========== 批处理辅助 ==========

def dot_collate(prompts: List[str], tokenizer, max_length: int = 256) -> Dict:
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
    }
