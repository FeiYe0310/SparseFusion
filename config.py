import os

# --- 全局路径配置 ---
# 这个文件是所有路径的唯一来源。
# 它会自动确定项目根目录 (SparseFusion/) 的绝对路径。
# 所有的其他路径都基于这个根目录构建。

# 1. 项目根目录
#    os.path.dirname(__file__) 会获取 config.py 所在的目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 2. 模型存放目录
#    建议您将所有大模型都放在 SparseFusion/models/ 文件夹下
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 3. 数据集存放目录
#    建议您将所有数据集都放在 SparseFusion/datasets/ 文件夹下
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

# 4. 实验结果输出目录
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# --- 具体模型和数据集的路径 ---
#    现在，我们可以用上面定义的目录来构建具体的路径。
#    当您需要添加新模型或数据集时，只需要在这里修改。

# 默认模型路径
DEFAULT_MODEL_1 = os.path.join(MODELS_DIR, "Qwen2.5-Math-1.5B-Instruct")
DEFAULT_MODEL_2 = os.path.join(MODELS_DIR, "Qwen2.5-Coder-1.5B-Instruct")

# 数据集路径
GSM8K_DIR = os.path.join(DATASETS_DIR, "gsm8k")

# 确保核心目录存在
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("--- Global Paths Initialized ---")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"MODELS_DIR:   {MODELS_DIR}")
print(f"DATASETS_DIR: {DATASETS_DIR}")
print(f"RESULTS_DIR:  {RESULTS_DIR}")
print("---------------------------------")
