"""
Evaluation Utilities for SparseFusion

This module contains evaluation function utilities.

Note: The actual evaluation functions (create_evaluation_fn_for_llm, 
create_bfcl_evaluation_fn, create_mbpp_evaluation_fn, etc.) are currently
kept in natural_niches_sparsity_aware_fn.py due to their complexity and
tight coupling with the main evolution loop. They can be migrated here
in future refactoring.

This module currently provides:
- Answer extraction utilities for GSM8K
- Helper functions for evaluation
"""

import re
from typing import Optional


def extract_answer(text: str) -> str:
    """
    从GSM8K生成的文本中提取数字答案
    GSM8K的标准格式：答案在####后面
    
    Args:
        text: 生成的文本
    
    Returns:
        提取的数字答案字符串
    """
    # 尝试找到####后的答案
    if "####" in text:
        answer = text.split("####")[-1].strip()
    else:
        # 如果没有####，尝试提取最后一个数字
        answer = text.strip()

    # 提取数字（可能带逗号、小数点）
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer.replace(",", ""))
    if numbers:
        return numbers[-1]  # 返回最后一个数字
    return ""


# TODO: Future refactoring - migrate these functions from main file:
# - create_evaluation_fn_for_llm (GSM8K evaluation with generation)
# - create_bfcl_evaluation_fn (BFCL function calling evaluation)
# - create_mbpp_evaluation_fn (MBPP code generation evaluation)  
# - create_multi_task_evaluation_fn (Multi-task weighted evaluation)
# - create_dot_eval_fn (DoT task evaluation for mult4/mult5/bool)

