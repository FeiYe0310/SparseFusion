"""
Profiler Utilities for SparseFusion

This module contains profiling and timing utilities:
- Environment-based profiler control
- Fine-grained timing context managers
- PyTorch Profiler integration
- JAX Trace integration
"""

import os
import time
import torch
from contextlib import contextmanager, nullcontext
from typing import Optional, Dict, Any


def is_profiler_enabled(profiler_type: str = "torch") -> bool:
    """
    检查特定类型的profiler是否启用
    
    Args:
        profiler_type: "torch" 或 "jax"
    
    Returns:
        True if enabled, False otherwise
    """
    if profiler_type == "torch":
        return os.environ.get("TORCH_PROFILER", "0") == "1"
    elif profiler_type == "jax":
        return os.environ.get("JAX_PROFILER", "0") == "1"
    elif profiler_type == "time":
        return os.environ.get("TIME_PROFILE", "0") == "1"
    return False


def is_in_profile_range(iteration: int) -> bool:
    """
    检查当前迭代是否在profiler范围内
    
    Args:
        iteration: 当前迭代编号
    
    Returns:
        True if in range, False otherwise
    """
    prof_start = int(os.environ.get("PROF_ITER_START", "-1"))
    prof_end = int(os.environ.get("PROF_ITER_END", "-1"))
    
    if prof_start < 0 and prof_end < 0:
        return True  # No range specified, always profile
    
    if prof_start >= 0 and iteration < prof_start:
        return False
    
    if prof_end >= 0 and iteration > prof_end:
        return False
    
    return True


@contextmanager
def torch_profile_context(
    enabled: bool,
    activity_name: str,
    rank: int = 0,
):
    """
    PyTorch Profiler 上下文管理器
    
    Args:
        enabled: 是否启用profiler
        activity_name: 活动名称（用于record_function）
        rank: 分布式rank（仅rank 0记录）
    
    Yields:
        record_function对象或nullcontext
    """
    if enabled and rank == 0:
        with torch.autograd.profiler.record_function(activity_name) as rf:
            yield rf
    else:
        with nullcontext() as nc:
            yield nc


@contextmanager
def jax_trace_context(
    enabled: bool,
    trace_name: str,
    rank: int = 0,
):
    """
    JAX Trace 上下文管理器
    
    Args:
        enabled: 是否启用trace
        trace_name: trace名称
        rank: 分布式rank（仅rank 0记录）
    
    Yields:
        None
    """
    if enabled and rank == 0:
        try:
            import jax
            jax.profiler.trace(f"/tmp/jax-trace-{trace_name}")
            yield
            jax.profiler.stop_trace()
        except Exception as e:
            # Silently fail if JAX profiler is not available
            yield
    else:
        yield


class TimingContext:
    """
    细粒度计时上下文管理器
    
    用法:
        with TimingContext("my_operation", rank=0) as t:
            # do something
        print(f"Time: {t.elapsed:.3f}s")
    """
    
    def __init__(self, name: str, rank: int = 0, print_on_exit: bool = False):
        self.name = name
        self.rank = rank
        self.print_on_exit = print_on_exit
        self.start_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.time() - self.start_time
        
        if self.print_on_exit and self.rank == 0:
            print(f"[TimeProfile][{self.name}] {self.elapsed:.3f}s")
        
        return False


def print_profile_stats(
    stage: str,
    stats: Dict[str, Any],
    rank: int = 0,
    enabled: bool = True
):
    """
    打印性能统计信息
    
    Args:
        stage: 阶段名称（如 "SelectProfile", "EvalProfile"）
        stats: 统计字典 {key: value}
        rank: 分布式rank（仅rank 0打印）
        enabled: 是否启用打印
    """
    if not enabled or rank != 0:
        return
    
    stat_str = " | ".join([f"{k}={v:.3f}s" if isinstance(v, float) else f"{k}={v}" 
                           for k, v in stats.items()])
    print(f"[{stage}] {stat_str}")


def sync_and_time(func):
    """
    装饰器：同步CUDA并计时
    
    用法:
        @sync_and_time
        def my_function():
            # do something
    """
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"[Timer] {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

