"""
预设配置文件

提供常用的配置预设，方便快速启动实验。
"""

from .model_config import ModelConfig
from .training_config import TrainingConfig
from .evaluation_config import EvaluationConfig
from .sparsity_config import SparsityConfig
from .task_config import TaskConfig


def get_default_config():
    """默认配置（GSM8K单任务）"""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "evaluation": EvaluationConfig(),
        "sparsity": SparsityConfig(),
        "task": TaskConfig(),
    }


def get_quick_test_config():
    """快速测试配置（小规模，用于调试）"""
    return {
        "model": ModelConfig(debug_models=True),
        "training": TrainingConfig(
            pop_size=8,
            total_forward_passes=100,
            runs=1,
        ),
        "evaluation": EvaluationConfig(
            batch_size=4,
            eval_subset_size=10,
        ),
        "sparsity": SparsityConfig(
            omega=0.5,
            beta=0.5,
            pruning_sparsity=0.0,
        ),
        "task": TaskConfig(),
    }


def get_multi_task_config():
    """多任务配置（GSM8K + MBPP）"""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(
            pop_size=16,
            total_forward_passes=10000,
        ),
        "evaluation": EvaluationConfig(
            batch_size=16,
            eval_subset_size=15,
            mbpp_qwen_chat=True,
            gsm8k_qwen_chat=True,
        ),
        "sparsity": SparsityConfig(
            omega=0.5,
            beta=0.5,
            use_dynamic_sparsity=True,
            sparsity_min=0.1,
            sparsity_max=0.6,
        ),
        "task": TaskConfig(
            use_gsm8k=True,
            use_mbpp_eval=True,
            gsm8k_weight=0.5,
            mbpp_weight=0.5,
        ),
    }


def get_dynamic_sparsity_config():
    """动态稀疏度配置"""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "evaluation": EvaluationConfig(batch_size=16),
        "sparsity": SparsityConfig(
            omega=0.5,
            beta=0.5,
            use_dynamic_sparsity=True,
            sparsity_min=0.1,
            sparsity_max=0.6,
            sparsity_t0=100,
            sparsity_t_mult=2,
            pruning_method="wanda",
        ),
        "task": TaskConfig(),
    }


def get_high_performance_config():
    """高性能配置（大batch，无few-shot）"""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(
            pop_size=20,
            total_forward_passes=50000,
            distributed=True,
            archive_backend="gpu",
        ),
        "evaluation": EvaluationConfig(
            batch_size=32,
            eval_subset_size=30,
            gsm8k_qwen_chat=False,  # 不使用few-shot加速
        ),
        "sparsity": SparsityConfig(
            omega=0.5,
            beta=0.5,
            pruning_sparsity=0.3,
        ),
        "task": TaskConfig(),
    }


# 预设名称映射
PRESETS = {
    "default": get_default_config,
    "quick_test": get_quick_test_config,
    "multi_task": get_multi_task_config,
    "dynamic_sparsity": get_dynamic_sparsity_config,
    "high_performance": get_high_performance_config,
}


def load_preset(preset_name: str):
    """
    加载预设配置
    
    Args:
        preset_name: 预设名称，可选：
            - "default": 默认配置
            - "quick_test": 快速测试
            - "multi_task": 多任务学习
            - "dynamic_sparsity": 动态稀疏度
            - "high_performance": 高性能配置
    
    Returns:
        配置字典
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )
    
    return PRESETS[preset_name]()


def print_config(config_dict):
    """打印配置摘要"""
    print("\n" + "=" * 70)
    print("📋 Configuration Summary")
    print("=" * 70)
    
    for section_name, section_config in config_dict.items():
        print(f"\n[{section_name.upper()}]")
        config_data = section_config.to_dict()
        for key, value in config_data.items():
            print(f"  {key}: {value}")
    
    print("=" * 70 + "\n")

预设配置文件

提供常用的配置预设，方便快速启动实验。
"""

from .model_config import ModelConfig
from .training_config import TrainingConfig
from .evaluation_config import EvaluationConfig
from .sparsity_config import SparsityConfig
from .task_config import TaskConfig


def get_default_config():
    """默认配置（GSM8K单任务）"""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "evaluation": EvaluationConfig(),
        "sparsity": SparsityConfig(),
        "task": TaskConfig(),
    }


def get_quick_test_config():
    """快速测试配置（小规模，用于调试）"""
    return {
        "model": ModelConfig(debug_models=True),
        "training": TrainingConfig(
            pop_size=8,
            total_forward_passes=100,
            runs=1,
        ),
        "evaluation": EvaluationConfig(
            batch_size=4,
            eval_subset_size=10,
        ),
        "sparsity": SparsityConfig(
            omega=0.5,
            beta=0.5,
            pruning_sparsity=0.0,
        ),
        "task": TaskConfig(),
    }


def get_multi_task_config():
    """多任务配置（GSM8K + MBPP）"""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(
            pop_size=16,
            total_forward_passes=10000,
        ),
        "evaluation": EvaluationConfig(
            batch_size=16,
            eval_subset_size=15,
            mbpp_qwen_chat=True,
            gsm8k_qwen_chat=True,
        ),
        "sparsity": SparsityConfig(
            omega=0.5,
            beta=0.5,
            use_dynamic_sparsity=True,
            sparsity_min=0.1,
            sparsity_max=0.6,
        ),
        "task": TaskConfig(
            use_gsm8k=True,
            use_mbpp_eval=True,
            gsm8k_weight=0.5,
            mbpp_weight=0.5,
        ),
    }


def get_dynamic_sparsity_config():
    """动态稀疏度配置"""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "evaluation": EvaluationConfig(batch_size=16),
        "sparsity": SparsityConfig(
            omega=0.5,
            beta=0.5,
            use_dynamic_sparsity=True,
            sparsity_min=0.1,
            sparsity_max=0.6,
            sparsity_t0=100,
            sparsity_t_mult=2,
            pruning_method="wanda",
        ),
        "task": TaskConfig(),
    }


def get_high_performance_config():
    """高性能配置（大batch，无few-shot）"""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(
            pop_size=20,
            total_forward_passes=50000,
            distributed=True,
            archive_backend="gpu",
        ),
        "evaluation": EvaluationConfig(
            batch_size=32,
            eval_subset_size=30,
            gsm8k_qwen_chat=False,  # 不使用few-shot加速
        ),
        "sparsity": SparsityConfig(
            omega=0.5,
            beta=0.5,
            pruning_sparsity=0.3,
        ),
        "task": TaskConfig(),
    }


# 预设名称映射
PRESETS = {
    "default": get_default_config,
    "quick_test": get_quick_test_config,
    "multi_task": get_multi_task_config,
    "dynamic_sparsity": get_dynamic_sparsity_config,
    "high_performance": get_high_performance_config,
}


def load_preset(preset_name: str):
    """
    加载预设配置
    
    Args:
        preset_name: 预设名称，可选：
            - "default": 默认配置
            - "quick_test": 快速测试
            - "multi_task": 多任务学习
            - "dynamic_sparsity": 动态稀疏度
            - "high_performance": 高性能配置
    
    Returns:
        配置字典
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )
    
    return PRESETS[preset_name]()


def print_config(config_dict):
    """打印配置摘要"""
    print("\n" + "=" * 70)
    print("📋 Configuration Summary")
    print("=" * 70)
    
    for section_name, section_config in config_dict.items():
        print(f"\n[{section_name.upper()}]")
        config_data = section_config.to_dict()
        for key, value in config_data.items():
            print(f"  {key}: {value}")
    
    print("=" * 70 + "\n")

