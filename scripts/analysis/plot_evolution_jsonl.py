#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                # 跳过坏行
                continue
    return records


def normalize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 iteration 排序，并对重复 iteration 仅保留“最后一次”。
    若记录缺少 iteration，暂不纳入排序（这些记录通常不会用于绘图关键字段）。
    """
    last_by_iter: Dict[int, Dict[str, Any]] = {}
    for r in records:
        it = r.get("iteration")
        if isinstance(it, int):
            last_by_iter[it] = r  # 后出现的覆盖前面的
    if not last_by_iter:
        return records
    return [last_by_iter[it] for it in sorted(last_by_iter.keys())]


def moving_average(series: List[float], window: int) -> List[float]:
    if window <= 1:
        return series
    acc = []
    s = 0.0
    for i, v in enumerate(series):
        s += float(v)
        if i >= window:
            s -= float(series[i - window])
        acc.append(s / min(i + 1, window))
    return acc


def extract_child_matrix(records: List[Dict[str, Any]]) -> (List[int], List[List[float]]):
    """返回 (iterations, children_per_step_scores)。
    兼容多种字段名：child_scores / fitness_vector / child_total_scores。
    若均不存在，则返回空矩阵。
    """
    iters: List[int] = []
    matrix: List[List[float]] = []
    for r in records:
        it = int(r.get("iteration", len(iters) * 10))
        vec = None
        for key in ("child_scores", "fitness_vector", "child_total_scores"):
            if key in r and isinstance(r[key], list):
                vec = r[key]
                break
        if vec is None:
            # 有些日志只存 child_mean 等，跳过该步子代明细
            continue
        iters.append(it)
        # 确保为 float 列表
        matrix.append([float(x) for x in vec])
    return iters, matrix


def extract_archive_matrix(records: List[Dict[str, Any]]) -> (List[int], List[List[float]]):
    """返回 (iterations, archive_fitness per step)。
    期望每条记录中含有 archive_fitness_vals: [pop_size]。
    若字段缺失则返回空矩阵。
    """
    iters: List[int] = []
    matrix: List[List[float]] = []
    for r in records:
        vec = r.get("archive_fitness_vals")
        if isinstance(vec, list) and vec:
            it = int(r.get("iteration", len(iters) * 10))
            iters.append(it)
            matrix.append([float(x) for x in vec])
    return iters, matrix


def plot_file(path: str, outdir: str, smooth: int, only_archive_fitness: bool = False) -> str:
    records = read_jsonl(path)
    # 归一：排序 + 去重（保留同一 iteration 的最后一次）
    records = normalize_records(records)
    if not records:
        raise RuntimeError(f"Empty or invalid JSONL: {path}")

    # 输出文件名

    base = os.path.basename(path)
    png_name = f"{base}.png"
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, png_name)

    plt.figure(figsize=(11, 6))

    if only_archive_fitness:
        iters, arch_mat = extract_archive_matrix(records)
        if arch_mat:
            max_slots = max(len(row) for row in arch_mat)
            per_series: List[List[float]] = [[] for _ in range(max_slots)]
            per_iters: List[List[int]] = [[] for _ in range(max_slots)]
            for step_idx, vec in enumerate(arch_mat):
                for slot in range(len(vec)):
                    per_series[slot].append(vec[slot])
                    per_iters[slot].append(iters[step_idx])
            for slot in range(max_slots):
                if not per_series[slot]:
                    continue
                y = per_series[slot]
                x = per_iters[slot]
                if smooth and smooth > 1:
                    y = moving_average(y, smooth)
                plt.plot(x, y, alpha=0.35, linewidth=0.9)
    else:
        # 默认：画子代曲线 + 叠加均值/最大值
        iters, child_matrix = extract_child_matrix(records)
        if child_matrix:
            max_children = max(len(row) for row in child_matrix)
            per_child_series: List[List[float]] = [[] for _ in range(max_children)]
            per_child_iters: List[List[int]] = [[] for _ in range(max_children)]
            for step_idx, vec in enumerate(child_matrix):
                for child_idx in range(len(vec)):
                    per_child_series[child_idx].append(vec[child_idx])
                    per_child_iters[child_idx].append(iters[step_idx])
            for child_idx in range(max_children):
                if not per_child_series[child_idx]:
                    continue
                y = per_child_series[child_idx]
                x = per_child_iters[child_idx]
                if smooth and smooth > 1:
                    y = moving_average(y, smooth)
                plt.plot(x, y, alpha=0.35, linewidth=0.9)

    # 叠加均值/最大值（若存在）
    if not only_archive_fitness:
        xs = [int(r.get("iteration", i)) for i, r in enumerate(records)]
        child_mean = [float(r.get("child_mean")) for r in records if "child_mean" in r]
        if len(child_mean) == len(xs):
            y = moving_average(child_mean, smooth) if smooth and smooth > 1 else child_mean
            plt.plot(xs, y, color="black", linewidth=2.0, label="child_mean")

        arch_mean = [float(r.get("archive_total_mean")) for r in records if "archive_total_mean" in r]
        if len(arch_mean) == len(xs):
            y = moving_average(arch_mean, smooth) if smooth and smooth > 1 else arch_mean
            plt.plot(xs, y, color="tab:blue", linewidth=2.0, label="archive_total_mean")

        arch_max = [float(r.get("archive_total_max")) for r in records if "archive_total_max" in r]
        if len(arch_max) == len(xs):
            y = moving_average(arch_max, smooth) if smooth and smooth > 1 else arch_max
            plt.plot(xs, y, color="tab:red", linewidth=2.0, label="archive_total_max")

    plt.xlabel("iteration")
    plt.ylabel("score")
    plt.title(base)
    plt.grid(True, alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="一个或多个 JSONL 文件")
    parser.add_argument("--outdir", default="results/fitness_logs/plots", help="输出目录")
    parser.add_argument("--smooth", type=int, default=1, help="滑动平均窗口，>1 启用平滑")
    parser.add_argument("--only_archive_fitness", action="store_true", help="仅绘制全部 archive_fitness 曲线")
    args = parser.parse_args()

    outs = []
    for p in args.inputs:
        outs.append(plot_file(p, args.outdir, args.smooth, args.only_archive_fitness))
    print("Saved:")
    for o in outs:
        print(o)


if __name__ == "__main__":
    main()


