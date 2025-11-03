"""
Natural Niches Evolution Utilities

This module contains the core evolution algorithm logic:
- Fitness computation and normalization
- Total score calculation (fitness + sparsity)
- Parent selection with sparsity awareness
- Archive update with sparsity awareness
"""

import jax
import jax.numpy as jnp
import os
import time
from typing import Tuple

# Import sparsity utilities from the same package
from .sparse_utils import compute_sparsity, compute_sparsity_scores


# Selection profiling (module-level cache)
_last_select_profile = None


def compute_normalized_fitness(
    scores: jnp.ndarray, alpha: float, num_tasks: int
) -> jnp.ndarray:
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
    fitness = jnp.sum(normalized_scores, axis=1) / (num_tasks**alpha)

    return fitness


def compute_total_scores(
    archive: jnp.ndarray,
    scores: jnp.ndarray,
    omega: float,
    beta: float,
    tau: float,
    alpha: float,
    num_tasks: int,
    epsilon: float,
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
    sparsities = jnp.array(
        [compute_sparsity(archive[i], epsilon) for i in range(pop_size)]
    )

    # 计算正常的fitness和sparsity scores
    fitness = compute_normalized_fitness(scores, alpha, num_tasks)
    sparsity_scores = compute_sparsity_scores(archive, tau, epsilon)

    # 计算总分
    total = omega * fitness + beta * sparsity_scores

    # 惩罚机制：稀疏度 > 0.5 的个体给予极低分数
    penalty_mask = sparsities > 0.5
    total = jnp.where(penalty_mask, -1e6, total)

    return total


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
    epsilon: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, int, int]:
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
        两个父代参数以及它们的索引 (parent1, parent2, idx1, idx2)
    """
    k1, k2 = jax.random.split(rand_key)

    # 细粒度耗时统计
    do_profile = os.environ.get("TIME_PROFILE", "0") == "1"
    prof = {"path": ("matchmaker" if use_matchmaker else "simple")}
    t0 = time.time() if do_profile else None

    # 1) 计算Total Score
    total_scores = compute_total_scores(
        archive, scores, omega, beta, tau, alpha, num_tasks, epsilon
    )
    if do_profile:
        try:
            jax.block_until_ready(total_scores)
        except Exception:
            pass
        prof["total_scores"] = time.time() - t0
        t0 = time.time()

    # 2) 概率归一化
    probs = total_scores / jnp.sum(total_scores)
    if do_profile:
        try:
            jax.block_until_ready(probs)
        except Exception:
            pass
        prof["probs_norm"] = time.time() - t0
        t0 = time.time()

    # 3) 抽样父母
    if use_matchmaker:
        # 3a) 抽样第一个父母
        p1_arr = jax.random.choice(k1, probs.size, shape=(1,), p=probs)
        parent_1_idx = int(p1_arr[0])
        if do_profile:
            prof["parent1_sample"] = time.time() - t0
            t0 = time.time()

        # 3b) 互补性矩阵与分数
        z = scores.sum(axis=0)
        z = jnp.where(z, z, 1) ** alpha
        fitness_matrix = scores.astype(jnp.float32) / z[None, :]
        match_score = jnp.maximum(
            0, fitness_matrix - fitness_matrix[parent_1_idx, :]
        ).sum(axis=1)
        if do_profile:
            try:
                jax.block_until_ready(match_score)
            except Exception:
                pass
            prof["complement_score"] = time.time() - t0
            t0 = time.time()

        # 3c) 互补性概率与第二个父母抽样
        match_probs = match_score / jnp.sum(match_score)
        p2_arr = jax.random.choice(k2, match_probs.size, shape=(1,), p=match_probs)
        parent_2_idx = int(p2_arr[0])
        if do_profile:
            prof["match_probs"] = time.time() - t0
    else:
        two_arr = jax.random.choice(k1, probs.size, shape=(2,), p=probs)
        parent_2_idx, parent_1_idx = int(two_arr[0]), int(two_arr[1])
        if do_profile:
            prof["two_sample"] = time.time() - t0

    if do_profile:
        global _last_select_profile
        _last_select_profile = prof
    
    # 返回父代参数以及其索引，便于记录谱系
    return archive[parent_1_idx], archive[parent_2_idx], int(parent_1_idx), int(parent_2_idx)


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
    epsilon: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        更新后的档案和得分矩阵 (archive, scores)
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
    row_selector = (
        jnp.arange(archive.shape[0], dtype=jnp.int32) == worst_ix
    ) & update_mask
    row_selector = row_selector[:, None]

    archive = jnp.where(row_selector, param[None, :], archive)
    scores = jnp.where(row_selector[: scores.shape[0], :], score[None, :], scores)

    return archive, scores


def get_last_select_profile():
    """
    获取最近一次 select 阶段的性能分析数据
    
    Returns:
        性能分析字典，包含各子阶段耗时
    """
    global _last_select_profile
    return _last_select_profile

