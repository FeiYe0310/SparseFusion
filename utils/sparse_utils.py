"""
Sparsity Utilities for SparseFusion

This module contains all sparsity-related functions:
- Dynamic sparsity scheduling with cosine annealing
- Sparsity computation and scoring
- Magnitude and Wanda pruning methods
"""

import jax.numpy as jnp
import torch
import torch.nn as nn
import numpy as np
import math
from typing import List
from transformers import AutoTokenizer


def calculate_dynamic_sparsity(
    current_iteration: int, eta_min: float, eta_max: float, t0: int, t_mult: int
) -> float:
    """
    使用带热重启的余弦退火（修改为正弦）计算动态剪枝稀疏度

    基于论文: SGDR: Stochastic Gradient Descent with Warm Restarts (https://arxiv.org/abs/1608.03983)
    修改点：使用 sin(pi/2) 替代原文的 cos，实现 warm-up 效果

    公式：
        eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + sin(T_cur/T_i * pi/2))

    其中：
        - T_cur: 当前周期内已经过的迭代次数
        - T_i: 当前周期的总迭代次数
        - 每个重启后，T_i *= t_mult（周期长度变化）

    设计思路：
        1. 使用 sin(pi/2) 让稀疏度从 eta_min 平滑增长到 eta_max（warm-up）
        2. 在进化初期使用小稀疏度，保护高fitness但低sparsity的优秀个体
        3. 随着进化推进，逐渐探索更高的稀疏度空间
        4. 周期性重启避免算法陷入特定稀疏度的"舒适区"
        5. 增加种群在不同稀疏生态位(sparsity niches)上的多样性

    Args:
        current_iteration: 当前的进化代数（从0开始）
        eta_min: 稀疏度的最小值（例如 0.1）
        eta_max: 稀疏度的最大值（例如 0.6）
        t0: 第一个周期的长度（迭代次数，例如 100）
        t_mult: 每次重启后周期长度的乘数（例如 2 表示每次翻倍，1 表示固定周期）

    Returns:
        计算出的当前稀疏度值（在 [eta_min, eta_max] 范围内）

    示例：
        >>> # 第一个周期100次迭代，每次周期翻倍，稀疏度在0.1-0.6之间变化
        >>> for i in range(400):
        ...     sparsity = calculate_dynamic_sparsity(i, 0.1, 0.6, 100, 2)
        ...     print(f"Iter {i}: sparsity={sparsity:.4f}")

        输出：
        Iter 0: sparsity=0.1000    # 第1周期开始（0-99），从最小值开始
        Iter 50: sparsity=0.2853   # 第1周期中点
        Iter 99: sparsity=0.6000   # 第1周期结束，达到最大值
        Iter 100: sparsity=0.1000  # 第2周期开始（100-299），重启！
        Iter 150: sparsity=0.2146  # 第2周期25%处
        Iter 200: sparsity=0.3500  # 第2周期50%处
        Iter 299: sparsity=0.6000  # 第2周期结束
        Iter 300: sparsity=0.1000  # 第3周期开始（300-699），重启！
    """
    t_i = float(t0)
    t_cur = float(current_iteration)

    # 确定当前是第几个周期以及在该周期内的位置
    while t_cur >= t_i:
        t_cur -= t_i
        t_i *= t_mult

    # 应用正弦预热公式
    # T_cur / T_i: 从 0 增长到 1（表示在周期内的进度）
    # * pi/2: 映射到 [0, pi/2]
    # sin(...): 从 0 增长到 1（正弦在 [0, pi/2] 区间单调递增）
    # (1 + sin(...)): 从 1 增长到 2
    # 0.5 * (1 + sin(...)): 从 0.5 增长到 1
    # eta_min + (eta_max - eta_min) * ...: 从 eta_min 增长到 eta_max
    sparsity_ratio = 0.5 * (1 + math.sin((t_cur / t_i) * math.pi / 2))

    current_sparsity = eta_min + (eta_max - eta_min) * sparsity_ratio

    # 安全边界检查（理论上不需要，但保险起见）
    return max(eta_min, min(current_sparsity, eta_max))


def compute_sparsity(params: jnp.ndarray, epsilon: float = 1e-10) -> float:
    """
    计算参数的稀疏度（近零参数的比例）

    Args:
        params: JAX参数数组
        epsilon: 判断为零的阈值

    Returns:
        稀疏度（0-1之间）
    """
    near_zero = jnp.abs(params) < epsilon
    return jnp.mean(near_zero.astype(jnp.float32))


def compute_sparsity_scores(
    archive: jnp.ndarray, tau: float, epsilon: float
) -> jnp.ndarray:
    """
    计算档案中所有个体的归一化稀疏度得分

    使用Softmax归一化确保：
    1. 所有得分和为1
    2. 避免除零问题

    Args:
        archive: 档案参数矩阵 (pop_size, num_params)
        tau: Softmax温度参数
        epsilon: 零参数阈值

    Returns:
        归一化的稀疏度得分 (pop_size,)
    """
    pop_size = archive.shape[0]
    sparsities = jnp.array(
        [compute_sparsity(archive[i], epsilon) for i in range(pop_size)]
    )

    # Softmax归一化
    exp_sparsities = jnp.exp(sparsities / tau)
    normalized = exp_sparsities / jnp.sum(exp_sparsities)

    return normalized


def prune_magnitude(
    jax_flat_params: jnp.ndarray, sparsity_ratio: float, sample_size: int = 10000
) -> jnp.ndarray:
    """
    快速Magnitude剪枝：使用采样估计阈值
    不需要排序整个数组，速度快

    Args:
        jax_flat_params: JAX扁平化参数数组
        sparsity_ratio: 目标稀疏度 (0.0-1.0)
        sample_size: 采样大小（用于估计阈值）

    Returns:
        剪枝后的JAX参数数组
    """
    if sparsity_ratio <= 0.0:
        return jax_flat_params

    # 保存原始dtype
    original_dtype = jax_flat_params.dtype

    # 转换为float32进行计算
    params_float = jax_flat_params.astype(jnp.float32)
    abs_params = jnp.abs(params_float)

    # 快速方法：采样估计阈值（避免对全部参数排序）
    total_size = abs_params.size
    if total_size > sample_size:
        # 随机采样一部分参数来估计阈值
        sample_indices = jnp.linspace(0, total_size - 1, sample_size, dtype=jnp.int32)
        sampled_abs = abs_params.flatten()[sample_indices]
        threshold = jnp.percentile(sampled_abs, sparsity_ratio * 100)
    else:
        # 参数量小，直接计算
        threshold = jnp.percentile(abs_params, sparsity_ratio * 100)

    # 剪枝：小于阈值的设为0
    pruned_params = jnp.where(abs_params < threshold, 0.0, params_float)

    # 转回原始dtype
    pruned_params = pruned_params.astype(original_dtype)

    return pruned_params


def find_linear_layers(module: nn.Module, layers: List = None, name: str = "") -> dict:
    """
    递归查找模块中的特定层类型（默认为Linear层）

    Args:
        module: PyTorch模块
        layers: 要查找的层类型列表
        name: 当前模块名称

    Returns:
        找到的层字典 {name: layer}
    """
    if layers is None:
        layers = [nn.Linear]
    
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_linear_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res


def prune_model_weights(
    pytorch_model: nn.Module, sparsity_ratio: float
) -> nn.Module:
    """
    直接对PyTorch模型的权重进行剪枝（基于magnitude）
    不需要校准数据，不需要forward pass
    保留Wanda的层级遍历逻辑

    Args:
        pytorch_model: PyTorch模型
        sparsity_ratio: 目标稀疏度

    Returns:
        剪枝后的模型（in-place修改）
    """
    # 遍历所有transformer层
    if hasattr(pytorch_model, "model") and hasattr(pytorch_model.model, "layers"):
        layers = pytorch_model.model.layers

        for i in range(len(layers)):
            layer = layers[i]
            subset = find_linear_layers(layer)

            # 对每个线性层的权重进行剪枝
            for name in subset:
                W = subset[name].weight.data
                original_dtype = W.dtype

                # 优化：尽量在GPU上计算，避免CPU-GPU传输
                # 1. 转float32（GPU上）
                W_float = W.float()

                # 2. 计算magnitude（GPU上）
                W_metric = torch.abs(W_float)

                # 3. 计算阈值（使用torch.kthvalue在GPU上，避免sort）
                k = int(W.numel() * sparsity_ratio)
                if k > 0 and k < W.numel():
                    # kthvalue在GPU上计算，比CPU numpy快很多
                    thresh = torch.kthvalue(W_metric.flatten(), k).values
                else:
                    thresh = 0.0

                # 4. 应用剪枝（GPU上）
                mask = W_metric <= thresh
                W_float[mask] = 0

                # 5. 转回原始dtype（GPU上）
                W_pruned = W_float.to(original_dtype)

                # 更新权重（无需移动设备，全程GPU）
                subset[name].weight.data = W_pruned

    return pytorch_model


def prune_with_wanda(
    jax_flat_params: jnp.ndarray,
    model_skeleton: nn.Module,
    param_shapes: List,
    tokenizer: AutoTokenizer,
    sparsity_ratio: float,
    device: torch.device,
    nsamples: int = 128,
    jax_to_pytorch_fn=None,  # 需要外部传入转换函数
) -> jnp.ndarray:
    """
    对JAX参数进行剪枝（纯剪枝逻辑，无校准）

    流程：
    1. JAX flat params → PyTorch model
    2. 应用magnitude剪枝（Wanda风格的层级遍历）
    3. PyTorch model → JAX flat params

    Args:
        jax_flat_params: JAX扁平化参数数组
        model_skeleton: PyTorch模型骨架
        param_shapes: 参数形状列表
        tokenizer: 未使用（保留接口兼容性）
        sparsity_ratio: 目标稀疏度
        device: 计算设备
        nsamples: 未使用（保留接口兼容性）
        jax_to_pytorch_fn: JAX转PyTorch的函数（从helper_fn导入）

    Returns:
        剪枝后的JAX参数数组
    """
    if jax_to_pytorch_fn is None:
        raise ValueError("Must provide jax_to_pytorch_fn for conversion")
    
    # 步骤1: 转换为PyTorch模型
    base_model = (
        model_skeleton.module if hasattr(model_skeleton, "module") else model_skeleton
    )
    pytorch_model = jax_to_pytorch_fn(
        jax_flat_params, base_model, param_shapes
    )
    pytorch_model.eval()

    # 步骤2: 直接剪枝（不需要校准数据）
    pytorch_model = prune_model_weights(pytorch_model, sparsity_ratio)

    # 步骤3: 转回JAX参数
    pruned_params = []
    for param in pytorch_model.parameters():
        # 关键：先.float()再.numpy()，避免bfloat16错误
        pruned_params.append(param.detach().cpu().float().numpy().flatten())

    return jnp.array(np.concatenate(pruned_params)).astype(jnp.bfloat16)

