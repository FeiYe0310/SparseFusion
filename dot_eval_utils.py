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


def _rand_bool_expr(rng: random.Random, vars_vals: Dict[str, bool], max_depth: int = 2) -> str:
    # 简单递归生成：原子(Var 或 not Var) 或 二元 (expr op expr)
    if max_depth <= 0:
        var = rng.choice(list(vars_vals.keys()))
        if rng.random() < 0.3:
            return f"not {var}"
        return var
    # 二元或一元
    if rng.random() < 0.4:
        var = rng.choice(list(vars_vals.keys()))
        if rng.random() < 0.5:
            return f"not {var}"
        return var
    left = _rand_bool_expr(rng, vars_vals, max_depth - 1)
    right = _rand_bool_expr(rng, vars_vals, max_depth - 1)
    op = rng.choice(["and", "or", "xor"])  # 支持xor
    return f"({left} {op} {right})"


def generate_bool_dataset(num_samples: int, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    data: List[Dict] = []
    for _ in range(num_samples):
        vals = {"A": bool(rng.getrandbits(1)), "B": bool(rng.getrandbits(1)), "C": bool(rng.getrandbits(1))}
        expr = _rand_bool_expr(rng, vals, max_depth=2)
        # 计算gold
        def _val(name: str) -> bool:
            return vals[name]
        # 安全求值：仅允许 and/or/not/xor/A/B/C/括号/空格
        safe_expr = expr.replace("xor", "^")
        local = {
            "A": vals["A"],
            "B": vals["B"],
            "C": vals["C"],
            "and": lambda x, y: x and y,  # 未被eval直接调用，这里只是占位
            "or": lambda x, y: x or y,
            "not": lambda x: (not x),
        }
        # 将 ^ 作为异或：转为Python按位异或，再映射到bool
        # 我们用替换方案：A ^ B -> (A) ^ (B)
        # 最终用 eval 仅在包含 A/B/C/()/^/not/and/or 的上下文中。
        expr_eval = safe_expr
        # 评估：将 A/B/C 替换为 True/False 字面值
        expr_eval = (expr_eval
                     .replace("A", str(vals["A"]))
                     .replace("B", str(vals["B"]))
                     .replace("C", str(vals["C"]))
                     )
        # 将 xor(^) 映射为不等：x ^ y 等价于 (x != y)
        expr_eval = re.sub(r"\^", " != ", expr_eval)
        gold = bool(eval(expr_eval))  # 安全前提：受控生成
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
