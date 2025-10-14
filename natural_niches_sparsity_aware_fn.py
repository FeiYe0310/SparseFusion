"""
Natural Niches with Sparsity-Aware Selection and Wanda Pruning

This module extends the original Natural Niches algorithm by:
1. Adding sparsity scoring alongside fitness scoring
2. Integrating Wanda pruning for active sparsification
3. Using Total Score (fitness + sparsity) for selection and archiving

The core architecture and evaluation logic remain IDENTICAL to the original.
"""

import jax

# Enable 64-bit precision for JAX. This is crucial for handling large numbers,
# such as the total parameter count of a 7B model, which exceeds the int32 limit.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer
from datasets import load_from_disk
from typing import Callable
import os
import math
import sys
import pickle
from contextlib import nullcontext

# Add current directory to path for importing lib/ module
# This ensures the code is portable across different environments
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from jax import make_array_from_callback
except ImportError:  # pragma: no cover - back-compat with older JAX
    Mesh = None
    NamedSharding = None
    PartitionSpec = None
    make_array_from_callback = None

try:
    default_device_ctx = jax.default_device
except AttributeError:  # pragma: no cover - older JAX
    from contextlib import contextmanager

    def default_device_ctx(_device):
        @contextmanager
        def _noop():
            yield

        return _noop()

# --- Imports for Multi-GPU ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from helper_fn import (
    crossover,
    crossover_without_splitpoint,
    mutate,
    get_pre_trained_models_and_skeleton,
    jax_flattened_to_pytorch_model,
)
from config import GSM8K_DIR, RESULTS_DIR


def _init_distributed_if_needed() -> tuple[int, int]:
    """Initialise torch.distributed with sensible defaults and backend selection."""

    if not dist.is_available():
        raise RuntimeError(
            "torch.distributed is not available in this build of PyTorch"
        )

    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    # Provide defaults so that single-process runs do not crash when torchrun env vars are missing.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("LOCAL_RANK", os.environ.get("RANK", "0"))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = os.environ.get("TORCH_BACKEND")
    if not backend:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Set a very long timeout (7 days) to prevent NCCL timeout errors
    # Default is 10 minutes (600s), which is too short for long computations
    from datetime import timedelta
    timeout = timedelta(days=7)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )

    return rank, world_size


# ==============================================================================
# SPARSITY-RELATED FUNCTIONS (NEW)
# ==============================================================================

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


def compute_sparsity_scores(archive: jnp.ndarray, tau: float, epsilon: float) -> jnp.ndarray:
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
    sparsities = jnp.array([compute_sparsity(archive[i], epsilon) for i in range(pop_size)])
    
    # Softmax归一化
    exp_sparsities = jnp.exp(sparsities / tau)
    normalized = exp_sparsities / jnp.sum(exp_sparsities)
    
    return normalized


def compute_normalized_fitness(scores: jnp.ndarray, alpha: float, num_tasks: int) -> jnp.ndarray:
    """
    计算归一化的fitness得分
    
    原始fitness仅求和不归一化，在引入sparsity score后需要归一化以对齐数量级。
    
    Formula: F_i = Σ(f_i,j) / M^α
    
    Args:
        scores: 得分矩阵 (pop_size, num_tasks)，已经过竞争归一化
        alpha: 归一化指数
        num_tasks: 总任务数M
    
    Returns:
        归一化的fitness (pop_size,)
    """
    # 确保scores是float类型（可能是bool）
    scores_float = scores.astype(jnp.float32)
    
    # 应用竞争归一化（与原始update_archive相同）
    z = jnp.sum(scores_float, axis=0) ** alpha
    z = jnp.where(z, z, 1)  # 避免除零
    normalized_scores = scores_float / z[None, :]
    
    # 求和并除以M^α
    fitness = jnp.sum(normalized_scores, axis=1) / (num_tasks ** alpha)
    
    return fitness


def compute_total_scores(
    archive: jnp.ndarray,
    scores: jnp.ndarray,
    omega: float,
    beta: float,
    tau: float,
    alpha: float,
    num_tasks: int,
    epsilon: float
) -> jnp.ndarray:
    """
    计算总分 = ω×Fitness + β×Sparsity
    
    **重要惩罚机制**: 对稀疏度 > 0.5 的个体给予极低分数（-1e6），
    确保它们会被优先替换掉，避免archive被全0个体占据。
    
    Args:
        archive: 档案参数矩阵
        scores: 性能得分矩阵
        omega: Fitness权重
        beta: Sparsity权重
        tau: Softmax温度
        alpha: Fitness归一化指数
        num_tasks: 任务总数
        epsilon: 零参数阈值
    
    Returns:
        总分 (pop_size,) - 过于稀疏的个体会得到极低分
    """
    pop_size = archive.shape[0]
    
    # 计算每个个体的稀疏度
    sparsities = jnp.array([compute_sparsity(archive[i], epsilon) for i in range(pop_size)])
    
    # 计算正常的fitness和sparsity scores
    fitness = compute_normalized_fitness(scores, alpha, num_tasks)
    sparsity_scores = compute_sparsity_scores(archive, tau, epsilon)
    
    # 计算总分
    total = omega * fitness + beta * sparsity_scores
    
    # 惩罚机制：稀疏度 > 0.5 的个体给予极低分数
    penalty_mask = sparsities > 0.5
    total = jnp.where(penalty_mask, -1e6, total)
    
    return total


# ==============================================================================
# WANDA PRUNING INTEGRATION (NEW)
# ==============================================================================

def prune_magnitude(
    jax_flat_params: jnp.ndarray,
    sparsity_ratio: float,
    sample_size: int = 10000
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
        sample_indices = jnp.linspace(0, total_size-1, sample_size, dtype=jnp.int32)
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


def prune_model_weights(
    pytorch_model: torch.nn.Module,
    sparsity_ratio: float
) -> torch.nn.Module:
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
    import torch.nn as nn
    
    def find_layers(module, layers=[nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res
    
    # 遍历所有transformer层
    if hasattr(pytorch_model, 'model') and hasattr(pytorch_model.model, 'layers'):
        layers = pytorch_model.model.layers
        
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)
            
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
    model_skeleton: torch.nn.Module,
    param_shapes: list,
    tokenizer: AutoTokenizer,
    sparsity_ratio: float,
    device: torch.device,
    nsamples: int = 128
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
    
    Returns:
        剪枝后的JAX参数数组
    """
    # 步骤1: 转换为PyTorch模型
    base_model = (
        model_skeleton.module if hasattr(model_skeleton, "module") else model_skeleton
    )
    pytorch_model = jax_flattened_to_pytorch_model(
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


# ==============================================================================
# EVALUATION FUNCTION (IDENTICAL TO ORIGINAL)
# ==============================================================================

def extract_answer(text: str) -> str:
    """
    从GSM8K生成的文本中提取数字答案
    GSM8K的标准格式：答案在####后面
    """
    import re
    
    # 尝试找到####后的答案
    if '####' in text:
        answer = text.split('####')[-1].strip()
    else:
        # 如果没有####，尝试提取最后一个数字
        answer = text.strip()
    
    # 提取数字（可能带逗号、小数点）
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer.replace(',', ''))
    if numbers:
        return numbers[-1]  # 返回最后一个数字
    return ""


def create_evaluation_fn_for_llm(
    model_skeleton: torch.nn.Module,
    param_shapes: list,
    tokenized_dataset,
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Creates an evaluation function for GSM8K using **generation + exact match**.
    
    真实评估流程：
    1. 模型生成完整答案（使用 model.generate()）
    2. 从生成的文本中提取数字答案
    3. 与ground truth进行精确匹配
    """
    base_model = (
        model_skeleton.module if hasattr(model_skeleton, "module") else model_skeleton
    )
    device = next(base_model.parameters()).device

    def collate_fn(batch):
        """
        自定义collate函数：
        - input_ids, attention_mask -> stack成tensor
        - answer_text -> 保持为字符串列表
        """
        import torch
        input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
        attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
        answer_texts = [item["answer_text"] for item in batch]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_text": answer_texts
        }
    
    def evaluation_fn(flat_params: jnp.ndarray) -> jnp.ndarray:
        # Get device from model
        device = next(base_model.parameters()).device
        
        # Restore parameters into the raw model (not the DDP wrapper)
        restored_model = jax_flattened_to_pytorch_model(
            flat_params, base_model, param_shapes
        )
        restored_model.eval()

        # Sampler ensures each GPU gets a different slice of data when distributed
        if distributed:
            sampler = DistributedSampler(
                tokenized_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
            data_loader = DataLoader(
                tokenized_dataset, 
                batch_size=batch_size, 
                sampler=sampler,
                collate_fn=collate_fn
            )
        else:
            data_loader = DataLoader(
                tokenized_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                collate_fn=collate_fn
            )

        local_scores = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                # 原始答案文本（用于提取ground truth）
                answer_texts = batch["answer_text"]
                
                # 生成答案（最多256个token，保证完整推理过程）
                generated_ids = restored_model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=False,  # 贪婪解码，保证可重复
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # 解码生成的文本
                generated_texts = tokenizer.batch_decode(
                    generated_ids[:, input_ids.shape[1]:],  # 只要新生成的部分
                    skip_special_tokens=True
                )
                
                # 对比预测答案和ground truth
                batch_scores = []
                for gen_text, gt_text in zip(generated_texts, answer_texts):
                    # 从生成文本提取预测答案
                    pred_answer = extract_answer(gen_text)
                    
                    # 从ground truth文本提取答案
                    gt_answer = extract_answer(gt_text)
                    
                    # 精确匹配
                    is_correct = (pred_answer == gt_answer) and (pred_answer != "")
                    batch_scores.append(1.0 if is_correct else 0.0)
                
                local_scores.extend(batch_scores)

        if not local_scores:
            return jnp.zeros(len(tokenized_dataset))

        # 转为tensor
        local_results_tensor = torch.tensor(local_scores, dtype=torch.float32)

        if distributed:
            # Move to GPU for NCCL backend
            local_results_tensor = local_results_tensor.to(device)
            gathered_tensors = [
                torch.empty_like(local_results_tensor) for _ in range(world_size)
            ]
            dist.all_gather(gathered_tensors, local_results_tensor)
            full_results_tensor = torch.cat(gathered_tensors)[: len(tokenized_dataset)]
            # Move back to CPU for numpy conversion
            full_results_tensor = full_results_tensor.cpu()
        else:
            full_results_tensor = local_results_tensor[: len(tokenized_dataset)]

        return jnp.array(full_results_tensor.numpy())

    return evaluation_fn


# ==============================================================================
# PARENT SELECTION (MODIFIED TO USE TOTAL SCORE)
# ==============================================================================

def sample_parents_with_sparsity(
    archive: jnp.ndarray,
    scores: jnp.ndarray,
    rand_key: jnp.ndarray,
    alpha: float,
    num_tasks: int,
    omega: float,
    beta: float,
    tau: float,
    use_matchmaker: bool,
    epsilon: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    选择父代（基于Total Score = ω×Fitness + β×Sparsity）
    
    与原始sample_parents的区别：使用total_scores代替fitness
    
    Args:
        archive: 档案参数矩阵
        scores: 性能得分矩阵
        rand_key: JAX随机密钥
        alpha: Fitness归一化指数
        num_tasks: 任务总数
        omega, beta, tau: 总分计算参数
        use_matchmaker: 是否使用matchmaker
        epsilon: 零参数阈值
    
    Returns:
        两个父代参数
    """
    k1, k2 = jax.random.split(rand_key)
    
    # 计算Total Score
    total_scores = compute_total_scores(
        archive, scores, omega, beta, tau, alpha, num_tasks, epsilon
    )
    
    # 归一化为概率
    probs = total_scores / jnp.sum(total_scores)
    
    # 第一个父代
    if use_matchmaker:
        parent_1_idx = jax.random.choice(k1, probs.size, shape=(1,), p=probs)[0]
        
        # 第二个父代：基于与第一个父代的互补性
        # 计算每个个体与parent_1在各任务上的互补得分
        z = scores.sum(axis=0)
        z = jnp.where(z, z, 1) ** alpha
        fitness_matrix = scores.astype(jnp.float32) / z[None, :]
        
        match_score = jnp.maximum(
            0, fitness_matrix - fitness_matrix[parent_1_idx, :]
        ).sum(axis=1)
        
        match_probs = match_score / jnp.sum(match_score)
        parent_2_idx = jax.random.choice(k2, match_probs.size, shape=(1,), p=match_probs)[0]
    else:
        parent_2_idx, parent_1_idx = jax.random.choice(
            k1, probs.size, shape=(2,), p=probs
        )
    
    return archive[parent_1_idx], archive[parent_2_idx]


# ==============================================================================
# ARCHIVE UPDATE (MODIFIED TO USE TOTAL SCORE)
# ==============================================================================

def update_archive_with_sparsity(
    score: jnp.ndarray,
    param: jnp.ndarray,
    archive: jnp.ndarray,
    scores: jnp.ndarray,
    alpha: float,
    omega: float,
    beta: float,
    tau: float,
    num_tasks: int,
    epsilon: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    更新档案（基于Total Score）
    
    与原始update_archive的区别：使用total_scores代替fitness
    
    Args:
        score: 新个体的性能得分
        param: 新个体的参数
        archive: 当前档案
        scores: 当前性能得分矩阵
        alpha: Fitness归一化指数
        omega, beta, tau: 总分计算参数
        num_tasks: 任务总数
        epsilon: 零参数阈值
    
    Returns:
        更新后的档案和得分矩阵
    """
    # 扩展档案以包含新个体
    ext_archive = jnp.concatenate([archive, param[None, ...]], axis=0)
    ext_scores = jnp.concatenate([scores, score[None, ...]], axis=0)
    
    # 计算所有个体的Total Score
    total_scores = compute_total_scores(
        ext_archive, ext_scores, omega, beta, tau, alpha, num_tasks, epsilon
    )
    
    # 找到最差个体
    worst_ix = jnp.asarray(jnp.argmin(total_scores), dtype=jnp.int32)
    scores_len = jnp.asarray(scores.shape[0], dtype=jnp.int32)
    update_mask = worst_ix < scores_len
    
    # 替换最差个体
    row_selector = (jnp.arange(archive.shape[0], dtype=jnp.int32) == worst_ix) & update_mask
    row_selector = row_selector[:, None]
    
    archive = jnp.where(row_selector, param[None, :], archive)
    scores = jnp.where(row_selector[: scores.shape[0], :], score[None, :], scores)
    
    return archive, scores


# ==============================================================================
# MAIN EVOLUTION FUNCTION (MODIFIED TO INTEGRATE PRUNING)
# ==============================================================================

def run_natural_niches_sparsity_aware(
    runs: int,
    pop_size: int,
    total_forward_passes: int,
    store_train_results: bool,
    no_matchmaker: bool,
    no_crossover: bool,
    no_splitpoint: bool,
    alpha: float = 1.0,
    omega: float = 0.5,
    beta: float = 0.5,
    tau: float = 1.0,
    epsilon: float = 1e-10,
    pruning_sparsity: float = 0.0,
    pruning_method: str = "wanda",
    use_pre_trained: bool = False,
    model1_path: str = "",
    model2_path: str = "",
    distributed: bool = False,
    archive_backend: str = "gpu",
    log_sparsity_stats: bool = False,
) -> list:
    """
    Run Natural Niches with Sparsity-Aware Selection and Wanda Pruning
    
    新增参数:
        omega: Fitness权重 (default: 0.5)
        beta: Sparsity权重 (default: 0.5)
        tau: Softmax温度 (default: 1.0)
        epsilon: 零参数阈值 (default: 1e-10)
        pruning_sparsity: Wanda剪枝目标稀疏度 (default: 0.0 = 不剪枝)
        pruning_method: 剪枝方法 'wanda' 或 'magnitude' (default: 'wanda')
        log_sparsity_stats: 是否记录稀疏度统计 (default: False)
    
    其他参数与原始run_natural_niches相同。
    """
    archive_backend = archive_backend.lower()
    if archive_backend not in {"gpu", "cpu"}:
        raise ValueError("archive_backend must be 'gpu' or 'cpu'")

    use_matchmaker, use_crossover, use_splitpoint = (
        not no_matchmaker,
        not no_crossover,
        not no_splitpoint,
    )
    
    # 确定是否启用剪枝
    enable_pruning = pruning_sparsity > 0.0

    # --- Multi-GPU Distributed Setup (IDENTICAL TO ORIGINAL) ---
    if distributed:
        rank, world_size = _init_distributed_if_needed()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size = 0, 1
        local_rank = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    is_main_process = rank == 0
    dist_enabled = distributed and world_size > 1

    # --- LLM & Data Loading (IDENTICAL TO ORIGINAL) ---
    if is_main_process:
        print("Loading tokenizer and models...")

    (
        model_1,
        model_2,
        param_shapes,
        tokenizer,
        model_skeleton,
    ) = get_pre_trained_models_and_skeleton(
        model1_path,
        model2_path,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 对于decoder-only模型，生成时必须用左填充
    tokenizer.padding_side = 'left'
    
    # 清除模型默认的采样参数（我们使用贪婪解码）
    if hasattr(model_skeleton, 'generation_config'):
        model_skeleton.generation_config.temperature = None
        model_skeleton.generation_config.top_p = None
        model_skeleton.generation_config.top_k = None

    num_params_llm = model_1.shape[0]

    if is_main_process:
        print("Loading and preprocessing GSM8K dataset from local directory...")
        if dist_enabled:
            load_from_disk(GSM8K_DIR)

    if dist_enabled:
        dist.barrier()

    dataset = load_from_disk(GSM8K_DIR)

    def preprocess_function(examples):
        """
        预处理GSM8K数据用于生成式评估：
        - input_ids: 只包含问题（用于model.generate()）
        - answer_text: 原始答案文本（用于提取ground truth）
        """
        # 只tokenize问题部分
        questions = examples["question"]
        model_inputs = tokenizer(
            questions, 
            max_length=256,  # GSM8K问题有时较长
            padding="max_length", 
            truncation=True
        )
        
        # 保存原始答案文本（不tokenize，后续直接用于提取答案）
        model_inputs["answer_text"] = examples["answer"]
        
        return model_inputs

    # 训练集：前1319个样本；测试集：前50个样本（快速评估）
    tokenized_train_dataset = (
        dataset["train"]
        .select(range(1319))
        .map(preprocess_function, batched=True, remove_columns=["question", "answer"])
    )
    tokenized_test_dataset = (
        dataset["test"]
        .select(range(50))
        .map(preprocess_function, batched=True, remove_columns=["question", "answer"])
    )
    # 不使用set_format，保持原始格式
    # DataLoader会自动将input_ids等转为tensor，answer_text保持为字符串列表
    
    num_tasks = len(tokenized_train_dataset)

    # --- Evaluation Setup (IDENTICAL TO ORIGINAL) ---
    if is_main_process:
        print("Setting up evaluation environment...")
    model_skeleton.to(device)

    if dist_enabled:
        model_skeleton.to(torch.float32)
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[device.index], output_device=device.index)
        model_skeleton = DDP(model_skeleton, **ddp_kwargs)
        model_skeleton.to(torch.bfloat16)
    else:
        model_skeleton.to(torch.bfloat16)

    model_skeleton.eval()

    train_eval_fn = create_evaluation_fn_for_llm(
        model_skeleton,
        param_shapes,
        tokenized_train_dataset,
        tokenizer,
        distributed=dist_enabled,
        world_size=world_size,
        rank=rank,
    )
    test_eval_fn = create_evaluation_fn_for_llm(
        model_skeleton,
        param_shapes,
        tokenized_test_dataset,
        tokenizer,
        distributed=dist_enabled,
        world_size=world_size,
        rank=rank,
    )

    # --- Archive Sharding Setup (IDENTICAL TO ORIGINAL) ---
    archive_sharding = None
    scores_sharding = None
    mesh_context = nullcontext()
    cpu_archive_device = None

    if archive_backend == "gpu" and Mesh and NamedSharding and PartitionSpec:
        available_jax_devices = jax.devices()
        if len(available_jax_devices) > 1 and pop_size > 1:
            shard_axis_size = min(len(available_jax_devices), pop_size)
            if pop_size % shard_axis_size != 0:
                shard_axis_size = math.gcd(pop_size, shard_axis_size)

            if shard_axis_size and shard_axis_size > 1:
                shard_mesh = Mesh(
                    np.array(available_jax_devices[:shard_axis_size]),
                    ("archive_shard",),
                )
                archive_sharding = NamedSharding(
                    shard_mesh, PartitionSpec("archive_shard", None)
                )
                scores_sharding = NamedSharding(
                    shard_mesh, PartitionSpec("archive_shard", None)
                )
                mesh_context = shard_mesh
                if is_main_process:
                    print(
                        f"Sharding archive across {shard_axis_size} JAX devices."
                    )
    elif archive_backend == "cpu":
        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise RuntimeError("No CPU devices available for archive_backend='cpu'")
        cpu_archive_device = cpu_devices[0]

    with mesh_context:
        if is_main_process:
            print("Setup complete. Starting evolution with Sparsity-Aware Selection.")
            if enable_pruning:
                print(f"🔪 Pruning ENABLED: method={pruning_method}, target_sparsity={pruning_sparsity}")
            print(f"📊 Scoring weights: ω={omega} (fitness), β={beta} (sparsity)")

        # --- JIT Compilation of Update Function ---
        if archive_sharding is not None:
            update_archive_fn = jax.jit(
                update_archive_with_sparsity,
                static_argnums=(4, 5, 6, 7, 8, 9),  # alpha, omega, beta, tau, num_tasks, epsilon
                in_shardings=(
                    None,
                    None,
                    archive_sharding,
                    scores_sharding,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                out_shardings=(archive_sharding, scores_sharding),
            )
        elif archive_backend == "cpu":
            update_archive_fn = jax.jit(
                update_archive_with_sparsity,
                static_argnums=(4, 5, 6, 7, 8, 9),
                backend="cpu"
            )
        else:
            update_archive_fn = jax.jit(
                update_archive_with_sparsity,
                static_argnums=(4, 5, 6, 7, 8, 9)
            )

        results = []

        if is_main_process:
            # --- Archive Initialization (Main Process Only) ---
            print(f"--- Initializing Archive on Main Process (Rank {rank}) ---")
            archive_shape = (pop_size, num_params_llm)
            scores_shape = (pop_size, len(tokenized_train_dataset))
            if archive_sharding is not None:
                archive = _sharded_zeros(archive_shape, jnp.bfloat16, archive_sharding)
                scores = _sharded_zeros(scores_shape, jnp.float32, scores_sharding)
            elif archive_backend == "cpu":
                with default_device_ctx(cpu_archive_device):
                    archive = jnp.zeros(archive_shape, dtype=jnp.bfloat16)
                    scores = jnp.zeros(scores_shape, dtype=jnp.float32)
            else:
                archive = jnp.zeros(archive_shape, dtype=jnp.bfloat16)
                scores = jnp.zeros(scores_shape, dtype=jnp.float32)

            # Evaluate initial models to populate the archive
            for model in (model_1, model_2):
                score = train_eval_fn(model)
                archive, scores = update_archive_fn(
                    score, model, archive, scores, alpha, omega, beta, tau, num_tasks, epsilon
                )
        else:
            archive = None
            scores = None
            if dist_enabled:
                for model in (model_1, model_2):
                    train_eval_fn(model)

        if dist_enabled:
            dist.barrier()

        # --- Main Evolution Loop ---
        for run in range(runs):
            if is_main_process:
                print(f"--- Starting Run {run+1}/{runs} ---")
                results.append(defaultdict(list))

            seed = 42 + run
            key = jax.random.PRNGKey(seed)

            progress_bar = tqdm(
                range(total_forward_passes),
                desc="Forward passes",
                disable=not is_main_process,
            )
            
            for i in progress_bar:
                k1, k2, k3, key = jax.random.split(key, 4)

                # --- Child Generation (Main Process Only) ---
                if is_main_process:
                    # Monitor JAX computation time to detect hangs
                    import time
                    start_time = time.time()
                    
                    # 1. Select parents based on Total Score
                    parents_bf16 = sample_parents_with_sparsity(
                        archive, scores, k1, alpha, num_tasks, omega, beta, tau, use_matchmaker, epsilon
                    )
                    parents_f32 = (
                        parents_bf16[0].astype(jnp.float32),
                        parents_bf16[1].astype(jnp.float32),
                    )
                    
                    # Check if JAX computation is taking too long
                    jax_time = time.time() - start_time
                    if jax_time > 60:  # Warning if JAX computation takes >60 seconds
                        print(f"⚠️  WARNING: JAX parent selection took {jax_time:.1f}s (may cause timeout)")
                    
                    # 2. Apply pruning to parents (NEW)
                    # 使用PyTorch+NumPy的剪枝方案（已修复bfloat16问题）
                    if enable_pruning:
                        try:
                            parents_f32 = (
                                prune_with_wanda(
                                    parents_f32[0].astype(jnp.bfloat16),
                                    model_skeleton,
                                    param_shapes,
                                    tokenizer,
                                    pruning_sparsity,
                                    device
                                ).astype(jnp.float32),
                                prune_with_wanda(
                                    parents_f32[1].astype(jnp.bfloat16),
                                    model_skeleton,
                                    param_shapes,
                                    tokenizer,
                                    pruning_sparsity,
                                    device
                                ).astype(jnp.float32),
                            )
                        except Exception as e:
                            print(f"⚠️  Pruning failed at iteration {i}: {e}")
                            # Continue without pruning

                    # 3. Crossover
                    if use_crossover:
                        if use_splitpoint:
                            child_f32 = crossover(parents_f32, k2)
                        else:
                            child_f32 = crossover_without_splitpoint(parents_f32, k2)
                    else:
                        child_f32 = parents_f32[0]

                    # 4. Mutation
                    child_f32 = mutate(child_f32, k3)
                    child_bf16_main = child_f32.astype(jnp.bfloat16)
                    
                    # Log sparsity if requested
                    if log_sparsity_stats and (i + 1) % 10 == 0:
                        child_sparsity = compute_sparsity(child_bf16_main, epsilon)
                        print(f"\n[Iter {i+1}] Child sparsity: {child_sparsity:.4f}")
                else:
                    child_bf16_main = None

                # --- Broadcast Child (IDENTICAL TO ORIGINAL) ---
                if dist_enabled:
                    if is_main_process:
                        child_tensor = torch.from_numpy(
                            np.array(child_bf16_main.astype(jnp.float32))
                        )
                    else:
                        child_tensor = torch.empty(num_params_llm, dtype=torch.float32)

                    # Move to GPU for NCCL backend
                    child_tensor = child_tensor.to(device)
                    dist.broadcast(child_tensor, src=0)
                    # Move back to CPU for numpy conversion
                    child_tensor = child_tensor.cpu()
                    child_bf16 = jnp.array(child_tensor.numpy()).astype(jnp.bfloat16)
                else:
                    child_bf16 = child_bf16_main

                # --- Evaluation (All Processes) ---
                score = train_eval_fn(child_bf16)

                # --- Archive Update (Main Process Only) ---
                if is_main_process:
                    archive, scores = update_archive_fn(
                        score, child_bf16, archive, scores, alpha, omega, beta, tau, num_tasks, epsilon
                    )
                    
                    # Record iteration statistics (every 10 steps to reduce overhead)
                    if (i + 1) % 10 == 0 or i == 0:
                        # Compute all archive statistics efficiently
                        archive_fitness_vals = jnp.mean(scores, axis=1)
                        archive_sparsity_vals = jnp.array([compute_sparsity(archive[j], epsilon) for j in range(pop_size)])
                        archive_total_scores = compute_total_scores(
                            archive, scores, omega, beta, tau, alpha, num_tasks, epsilon
                        )
                        
                        iteration_stats = {
                            "iteration": i + 1,
                            "child_fitness": float(jnp.mean(score)),
                            "child_sparsity": float(compute_sparsity(child_bf16, epsilon)),
                            "archive_fitness_mean": float(jnp.mean(archive_fitness_vals)),
                            "archive_fitness_max": float(jnp.max(archive_fitness_vals)),
                            "archive_sparsity_mean": float(jnp.mean(archive_sparsity_vals)),
                            "archive_sparsity_min": float(jnp.min(archive_sparsity_vals)),
                            "archive_total_score_mean": float(jnp.mean(archive_total_scores)),
                            "archive_total_score_max": float(jnp.max(archive_total_scores)),
                        }
                        results[run]["iterations"].append(iteration_stats)

                # --- GPU Memory Cleanup (Every 100 steps to prevent memory leak) ---
                if (i + 1) % 100 == 0:
                    if is_main_process:
                        print(f"[Step {i+1}] Cleaning GPU memory...")
                    torch.cuda.empty_cache()
                    if dist_enabled:
                        dist.barrier()  # Ensure all processes clean memory together

                if dist_enabled:
                    dist.barrier()

                # --- Periodic Full Archive Evaluation (IDENTICAL TO ORIGINAL) ---
                if (i + 1) % 10 == 0:
                    if is_main_process:
                        print(
                            f"\n--- [Step {i+1}/{total_forward_passes}] Evaluating full archive ---"
                        )

                    for j in range(pop_size):
                        if dist_enabled:
                            if is_main_process:
                                individual_params = archive[j]
                                params_tensor = torch.from_numpy(
                                    np.array(individual_params.astype(jnp.float32))
                                )
                            else:
                                params_tensor = torch.empty(
                                    num_params_llm, dtype=torch.float32
                                )

                            # Move to GPU for NCCL backend
                            params_tensor = params_tensor.to(device)
                            dist.broadcast(params_tensor, src=0)
                            # Move back to CPU for numpy conversion
                            params_tensor = params_tensor.cpu()
                            params_bf16 = jnp.array(params_tensor.numpy()).astype(
                                jnp.bfloat16
                            )
                        else:
                            params_bf16 = archive[j]

                        test_scores_vector = test_eval_fn(params_bf16)

                        if is_main_process:
                            acc = jnp.mean(test_scores_vector)
                            
                            # Log sparsity alongside accuracy
                            if log_sparsity_stats:
                                ind_sparsity = compute_sparsity(params_bf16, epsilon)
                                print(
                                    f"  > Archive Individual {j+1}/{pop_size} | "
                                    f"Test Acc: {acc:.4f} | Sparsity: {ind_sparsity:.4f}"
                                )
                            else:
                                print(
                                    f"  > Archive Individual {j+1}/{pop_size} | Test Accuracy: {acc:.4f}"
                                )
                            
                            # Record test evaluation results
                            if "test_evaluations" not in results[run]:
                                results[run]["test_evaluations"] = []
                            
                            test_eval_stats = {
                                "iteration": i + 1,
                                "individual": j + 1,
                                "test_accuracy": float(acc),
                                "sparsity": float(compute_sparsity(params_bf16, epsilon)) if log_sparsity_stats else None
                            }
                            results[run]["test_evaluations"].append(test_eval_stats)

                # --- Periodic Checkpoint Save (Every 500 steps to prevent data loss) ---
                if (i + 1) % 500 == 0 and is_main_process:
                    from datetime import datetime
                    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    checkpoint_path = os.path.join(
                        checkpoint_dir, 
                        f"checkpoint_run{run+1}_step{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    )
                    
                    checkpoint_data = {
                        "iteration": i + 1,
                        "run": run + 1,
                        "archive": np.array(archive),
                        "scores": np.array(scores),
                        "results": results,
                    }
                    
                    with open(checkpoint_path, "wb") as f:
                        pickle.dump(checkpoint_data, f)
                    
                    print(f"💾 Checkpoint saved: {checkpoint_path}")

        if dist_enabled:
            dist.barrier()

        # --- Save Final Best Model (Main Process Only) ---
        if is_main_process:
            if runs > 0:
                # Find best based on Total Score
                total_scores = compute_total_scores(
                    archive, scores, omega, beta, tau, alpha, num_tasks, epsilon
                )
                best_individual_idx = jnp.argmax(total_scores)
                best_params = archive[best_individual_idx]

                os.makedirs(RESULTS_DIR, exist_ok=True)

                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    RESULTS_DIR, f"best_model_sparsity_aware_run_{run+1}_{timestamp}.npz"
                )

                print(f"\n🏆 Saving the best model (by Total Score) from the last run to: {save_path}")
                
                # Log final statistics
                best_sparsity = compute_sparsity(best_params, epsilon)
                best_fitness = float(jnp.mean(scores[best_individual_idx]))
                best_total_score = float(total_scores[best_individual_idx])
                
                print(f"   Final fitness: {best_fitness:.4f}")
                print(f"   Final sparsity: {best_sparsity:.4f}")
                print(f"   Final total score: {best_total_score:.4f}")
                
                # Record final best model info
                results[run]["final_best_model"] = {
                    "save_path": save_path,
                    "fitness": best_fitness,
                    "sparsity": float(best_sparsity),
                    "total_score": best_total_score,
                    "individual_idx": int(best_individual_idx)
                }
                
                jnp.savez(save_path, params=best_params)
                print("✅ Model saved successfully.")

        return results


def _sharded_zeros(shape: tuple[int, ...], dtype, sharding):
    """Helper function for sharded array initialization (IDENTICAL TO ORIGINAL)"""
    if sharding is None or make_array_from_callback is None:
        return jnp.zeros(shape, dtype=dtype)

    def _cb(index):
        slice_dims = []
        for dim_idx, dim_size in zip(index, shape):
            if isinstance(dim_idx, slice):
                start = 0 if dim_idx.start is None else dim_idx.start
                stop = dim_size if dim_idx.stop is None else dim_idx.stop
                slice_dims.append(stop - start)
            else:
                slice_dims.append(1)
        return jnp.zeros(tuple(slice_dims), dtype=dtype)

    return make_array_from_callback(shape, sharding, _cb)

