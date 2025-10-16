"""
转换BFCL官方数据到我们需要的格式
从JSONL格式转换为标准JSON数组，并扩充到200个样本
"""

import json

# 读取BFCL官方数据（JSONL格式）
bfcl_file = "bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_simple.json"

print(f"读取: {bfcl_file}")

# 读取JSONL格式（每行一个JSON对象）
data_items = []
with open(bfcl_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data_items.append(json.loads(line))

print(f"✓ 读取了 {len(data_items)} 个样本")

# 转换为我们的格式
converted_data = []
for idx, item in enumerate(data_items):
    # BFCL格式转换为我们的格式
    converted_item = {
        "id": f"bfcl_v4_{idx+1}",
        "functions": item.get("function", []),  # BFCL的function字段
        "user_query": item.get("question", ""),  # BFCL的question字段
        "ground_truth": item.get("ground_truth", [{}])[0] if item.get("ground_truth") else {}
    }
    converted_data.append(converted_item)

print(f"✓ 转换了 {len(converted_data)} 个样本")

# 如果样本不够200个，重复使用
while len(converted_data) < 200:
    for item in data_items[:min(200-len(converted_data), len(data_items))]:
        idx = len(converted_data)
        converted_item = {
            "id": f"bfcl_v4_{idx+1}",
            "functions": item.get("function", []),
            "user_query": item.get("question", ""),
            "ground_truth": item.get("ground_truth", [{}])[0] if item.get("ground_truth") else {}
        }
        converted_data.append(converted_item)

print(f"✓ 扩充到 {len(converted_data)} 个样本")

# 保存为标准JSON
output_file = "bfcl/data/bfcl_test_200.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print(f"✅ 保存到: {output_file}")
print(f"\n示例数据:")
print(json.dumps(converted_data[0], indent=2, ensure_ascii=False)[:500])

