"""
BFCL评估工具
包含Function Call解析、验证和评估功能
"""

import re
import json
from typing import Dict, Any, Optional, Union


def extract_function_call(text: str) -> Dict[str, Any]:
    """
    从模型生成的文本中提取function call
    
    支持多种格式：
    1. JSON格式: {"name": "func", "arguments": {...}}
    2. 函数调用格式: func_name(arg1=val1, arg2=val2)
    3. 纯JSON格式: {"function": "func", "parameters": {...}}
    
    Args:
        text: 模型生成的文本
        
    Returns:
        标准化的function call字典: {"name": str, "arguments": dict}
        如果解析失败，返回 {"name": None, "arguments": {}}
    
    Examples:
        >>> extract_function_call('{"name": "get_weather", "arguments": {"location": "Beijing"}}')
        {'name': 'get_weather', 'arguments': {'location': 'Beijing'}}
        
        >>> extract_function_call('get_weather(location="Beijing", unit="celsius")')
        {'name': 'get_weather', 'arguments': {'location': 'Beijing', 'unit': 'celsius'}}
    """
    text = text.strip()
    
    # 方法1: 尝试直接解析JSON格式
    try:
        # 提取第一个完整的JSON对象
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)
            
            # 标准化字段名
            if "name" in data and "arguments" in data:
                return {"name": data["name"], "arguments": data["arguments"]}
            elif "function" in data and "parameters" in data:
                return {"name": data["function"], "arguments": data["parameters"]}
            elif "name" in data and "parameters" in data:
                return {"name": data["name"], "arguments": data["parameters"]}
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    
    # 方法2: 解析函数调用格式 func_name(arg1=val1, arg2=val2)
    func_call_match = re.search(r'(\w+)\s*\((.*?)\)', text, re.DOTALL)
    if func_call_match:
        func_name = func_call_match.group(1)
        args_str = func_call_match.group(2).strip()
        
        # 解析参数
        args = {}
        if args_str:
            # 分割参数（处理逗号分隔）
            # 注意：这里简化处理，不处理嵌套的逗号
            arg_pattern = r'(\w+)\s*[:=]\s*([^,]+)'
            for match in re.finditer(arg_pattern, args_str):
                key = match.group(1).strip()
                value_str = match.group(2).strip()
                
                # 尝试解析值
                try:
                    # 去除引号
                    if (value_str.startswith('"') and value_str.endswith('"')) or \
                       (value_str.startswith("'") and value_str.endswith("'")):
                        value = value_str[1:-1]
                    # 尝试转换为数字
                    elif '.' in value_str:
                        value = float(value_str)
                    elif value_str.isdigit() or (value_str.startswith('-') and value_str[1:].isdigit()):
                        value = int(value_str)
                    # 布尔值
                    elif value_str.lower() == 'true':
                        value = True
                    elif value_str.lower() == 'false':
                        value = False
                    # null/None
                    elif value_str.lower() in ('null', 'none'):
                        value = None
                    else:
                        value = value_str
                    
                    args[key] = value
                except (ValueError, AttributeError):
                    args[key] = value_str
        
        return {"name": func_name, "arguments": args}
    
    # 方法3: 提取可能的函数名（作为fallback）
    func_name_match = re.search(r'(?:function|call|invoke)[\s:]+(\w+)', text, re.IGNORECASE)
    if func_name_match:
        return {"name": func_name_match.group(1), "arguments": {}}
    
    # 解析失败
    return {"name": None, "arguments": {}}


def evaluate_function_call(
    pred_call: Dict[str, Any],
    gt_call: Dict[str, Any],
    strict: bool = True
) -> float:
    """
    评估function call的准确性
    
    评估标准（基于BFCL AST Match）:
    1. 函数名必须完全匹配
    2. 所有必需参数必须存在
    3. 参数值必须匹配（支持类型转换）
    4. 可选：参数顺序无关
    
    Args:
        pred_call: 预测的function call
        gt_call: ground truth function call
        strict: 是否严格模式（要求所有参数完全匹配，包括可选参数）
        
    Returns:
        score: 1.0 (完全正确) 或 0.0 (错误)
        
    Examples:
        >>> pred = {"name": "get_weather", "arguments": {"location": "Beijing"}}
        >>> gt = {"name": "get_weather", "arguments": {"location": "Beijing"}}
        >>> evaluate_function_call(pred, gt)
        1.0
        
        >>> pred = {"name": "wrong_func", "arguments": {"location": "Beijing"}}
        >>> evaluate_function_call(pred, gt)
        0.0
    """
    # 1. 检查函数名
    pred_name = pred.get("name")
    gt_name = gt.get("name")
    
    if pred_name is None or pred_name != gt_name:
        return 0.0
    
    # 2. 检查参数
    pred_args = pred.get("arguments", {})
    gt_args = gt.get("arguments", {})
    
    # 严格模式：参数数量必须一致
    if strict and len(pred_args) != len(gt_args):
        return 0.0
    
    # 非严格模式：只要求ground truth中的参数都存在
    for key, gt_val in gt_args.items():
        if key not in pred_args:
            return 0.0
        
        pred_val = pred_args[key]
        
        # 值比较（支持类型转换）
        if not _values_match(pred_val, gt_val):
            return 0.0
    
    # 严格模式：检查是否有多余的参数
    if strict:
        for key in pred_args:
            if key not in gt_args:
                return 0.0
    
    return 1.0


def _values_match(pred_val: Any, gt_val: Any, tolerance: float = 1e-6) -> bool:
    """
    比较两个值是否匹配（支持类型转换和数值容差）
    
    Args:
        pred_val: 预测值
        gt_val: ground truth值
        tolerance: 数值比较的容差
        
    Returns:
        是否匹配
    """
    # 类型完全相同的情况
    if type(pred_val) == type(gt_val):
        if isinstance(gt_val, (int, float)):
            return abs(pred_val - gt_val) <= tolerance
        else:
            return pred_val == gt_val
    
    # 数值类型的比较（int vs float）
    if isinstance(pred_val, (int, float)) and isinstance(gt_val, (int, float)):
        return abs(float(pred_val) - float(gt_val)) <= tolerance
    
    # 字符串比较（忽略大小写和首尾空格）
    if isinstance(pred_val, str) and isinstance(gt_val, str):
        return pred_val.strip().lower() == gt_val.strip().lower()
    
    # 字符串与数字的转换
    if isinstance(pred_val, str) and isinstance(gt_val, (int, float)):
        try:
            return abs(float(pred_val) - float(gt_val)) <= tolerance
        except ValueError:
            return False
    
    if isinstance(pred_val, (int, float)) and isinstance(gt_val, str):
        try:
            return abs(float(pred_val) - float(gt_val)) <= tolerance
        except ValueError:
            return False
    
    # 其他情况：字符串化后比较
    return str(pred_val).strip() == str(gt_val).strip()


def batch_evaluate_bfcl(
    predictions: list,
    ground_truths: list,
    strict: bool = True
) -> Dict[str, float]:
    """
    批量评估BFCL任务
    
    Args:
        predictions: 预测的function calls列表
        ground_truths: ground truth function calls列表
        strict: 是否严格模式
        
    Returns:
        评估指标字典，包含:
        - accuracy: 总体准确率
        - correct_count: 正确数量
        - total_count: 总数量
        - per_sample_scores: 每个样本的得分
    """
    assert len(predictions) == len(ground_truths), \
        f"Predictions and ground truths must have same length: {len(predictions)} vs {len(ground_truths)}"
    
    scores = []
    for pred, gt in zip(predictions, ground_truths):
        score = evaluate_function_call(pred, gt, strict=strict)
        scores.append(score)
    
    correct_count = sum(scores)
    total_count = len(scores)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "per_sample_scores": scores
    }


# ========== 测试代码 ==========
if __name__ == "__main__":
    """测试Function Call解析和评估功能"""
    
    print("=" * 60)
    print("测试 extract_function_call()")
    print("=" * 60)
    
    # 测试用例1: JSON格式
    test_cases = [
        ('{"name": "get_weather", "arguments": {"location": "Beijing"}}', 
         {"name": "get_weather", "arguments": {"location": "Beijing"}}),
        
        ('get_weather(location="Beijing", unit="celsius")',
         {"name": "get_weather", "arguments": {"location": "Beijing", "unit": "celsius"}}),
        
        ('calculate_area(length=5, width=3)',
         {"name": "calculate_area", "arguments": {"length": 5, "width": 3}}),
        
        ('I will call the function get_weather(location="Tokyo")',
         {"name": "get_weather", "arguments": {"location": "Tokyo"}}),
        
        ('random text without function call',
         {"name": None, "arguments": {}}),
    ]
    
    for i, (text, expected) in enumerate(test_cases, 1):
        result = extract_function_call(text)
        status = "✓" if result == expected else "✗"
        print(f"\n测试 {i}: {status}")
        print(f"  输入: {text[:60]}...")
        print(f"  预期: {expected}")
        print(f"  实际: {result}")
    
    print("\n" + "=" * 60)
    print("测试 evaluate_function_call()")
    print("=" * 60)
    
    # 测试用例2: 评估
    eval_cases = [
        # (pred, gt, expected_score, description)
        ({"name": "get_weather", "arguments": {"location": "Beijing"}},
         {"name": "get_weather", "arguments": {"location": "Beijing"}},
         1.0, "完全匹配"),
        
        ({"name": "wrong_func", "arguments": {"location": "Beijing"}},
         {"name": "get_weather", "arguments": {"location": "Beijing"}},
         0.0, "函数名错误"),
        
        ({"name": "calc", "arguments": {"a": 5, "b": 3}},
         {"name": "calc", "arguments": {"a": 5, "b": 3}},
         1.0, "数值参数匹配"),
        
        ({"name": "calc", "arguments": {"a": 5}},
         {"name": "calc", "arguments": {"a": 5, "b": 3}},
         0.0, "缺少参数"),
        
        ({"name": "calc", "arguments": {"a": "5", "b": "3"}},
         {"name": "calc", "arguments": {"a": 5, "b": 3}},
         1.0, "字符串数字转换"),
        
        ({"name": "send", "arguments": {"to": "JOHN@EXAMPLE.COM"}},
         {"name": "send", "arguments": {"to": "john@example.com"}},
         1.0, "大小写不敏感"),
    ]
    
    for i, (pred, gt, expected_score, desc) in enumerate(eval_cases, 1):
        result = evaluate_function_call(pred, gt)
        status = "✓" if result == expected_score else "✗"
        print(f"\n测试 {i}: {status} - {desc}")
        print(f"  预期分数: {expected_score}")
        print(f"  实际分数: {result}")
    
    print("\n" + "=" * 60)
    print("测试 batch_evaluate_bfcl()")
    print("=" * 60)
    
    predictions = [
        {"name": "func1", "arguments": {"a": 1}},
        {"name": "func2", "arguments": {"b": 2}},
        {"name": "wrong", "arguments": {"c": 3}},
    ]
    
    ground_truths = [
        {"name": "func1", "arguments": {"a": 1}},
        {"name": "func2", "arguments": {"b": 2}},
        {"name": "func3", "arguments": {"c": 3}},
    ]
    
    results = batch_evaluate_bfcl(predictions, ground_truths)
    print(f"\n总体准确率: {results['accuracy']:.2%}")
    print(f"正确数量: {results['correct_count']}/{results['total_count']}")
    print(f"每个样本分数: {results['per_sample_scores']}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成!")
    print("=" * 60)

