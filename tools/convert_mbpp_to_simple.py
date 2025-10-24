#!/usr/bin/env python3
"""
转换MBPP官方数据到简化格式

官方MBPP数据通常格式如下：
{
    "task_id": int,
    "text": str,  # 问题描述
    "code": str,  # 参考实现
    "test_list": [str],  # 测试用例
    "test_setup_code": str,  # 可选的前置代码
    "challenge_test_list": [str]  # 可选的额外测试
}

使用方法:
    python tools/convert_mbpp_to_simple.py \
        --input mbpp_original.jsonl \
        --output mbpp/data/mbpp_test.json \
        --limit 100
"""

import json
import argparse
from pathlib import Path


def convert_mbpp_data(input_file: str, output_file: str, limit: int = None):
    """转换MBPP数据到简化格式"""
    
    print(f"读取MBPP数据: {input_file}")
    
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.jsonl'):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    
    print(f"原始数据: {len(data)} 条")
    
    # 限制数量
    if limit and limit < len(data):
        data = data[:limit]
        print(f"限制到: {limit} 条")
    
    # 转换格式（保持字段不变，确保必需字段存在）
    converted = []
    for item in data:
        converted_item = {
            "task_id": item.get("task_id", item.get("id", len(converted))),
            "text": item.get("text", item.get("prompt", "")),
            "code": item.get("code", item.get("solution", "")),
            "test_list": item.get("test_list", item.get("tests", [])),
            "test_setup_code": item.get("test_setup_code", ""),
            "challenge_test_list": item.get("challenge_test_list", [])
        }
        converted.append(converted_item)
    
    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已保存到: {output_file}")
    print(f"   包含 {len(converted)} 条数据")


def main():
    parser = argparse.ArgumentParser(description='转换MBPP数据到简化格式')
    parser.add_argument('--input', type=str, required=True,
                       help='输入文件路径（JSONL或JSON）')
    parser.add_argument('--output', type=str, default='mbpp/data/mbpp_test.json',
                       help='输出文件路径')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制数据条数（用于快速测试）')
    
    args = parser.parse_args()
    
    convert_mbpp_data(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()

