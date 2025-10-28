import argparse
import glob as _glob
import json
import os
from typing import List, Tuple, Dict, Any

import numpy as np

# 非交互环境使用无头后端
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # 跳过坏行
                continue


def extract_series(records: List[Dict[str, Any]], metric: str, iter_key: str = "iteration") -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    fallback_iter = 0
    for rec in records:
        # 迭代步数
        it = rec.get(iter_key)
        if isinstance(it, int):
            x = it
            fallback_iter = it
        else:
            # 退化：若不存在 iteration，则按 10 步累加
            fallback_iter = fallback_iter + 10
            x = fallback_iter

        # 指标取值
        val = rec.get(metric)
        if val is None:
            # 尝试常见别名
            aliases = {
                "archive_total_max": ["archive_total_max", "archive_total_score_max"],
                "archive_total_mean": ["archive_total_mean", "archive_total_score_mean"],
                "archive_fitness_max": ["archive_fitness_max"],
                "archive_fitness_mean": ["archive_fitness_mean"],
                "child_mean": ["child_mean", "child_score_mean"],
                "child_sparsity": ["child_sparsity"],
            }.get(metric, [])
            for k in aliases:
                if rec.get(k) is not None:
                    val = rec.get(k)
                    break

        if isinstance(val, (int, float)):
            xs.append(int(x))
            ys.append(float(val))

    return xs, ys


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) == 0:
        return values
    arr = np.asarray(values, dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    # 头部对齐（用前几个原始值填充）
    head = list(arr[: window - 1])
    return head + list(smoothed)


def main():
    parser = argparse.ArgumentParser(description="Plot evolution curves from JSONL logs")
    parser.add_argument("--inputs", nargs="*", help="Explicit list of JSONL files")
    parser.add_argument("--glob", dest="pattern", default=None, help="Glob pattern for JSONL files")
    parser.add_argument("--metrics", nargs="+", default=["archive_total_max"],
                        help="Metrics to plot per file (e.g., archive_total_max archive_fitness_max child_mean)")
    parser.add_argument("--out_dir", default="results/fitness_logs/plots", help="Output directory for plots")
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window size (>=1)")
    args = parser.parse_args()

    files: List[str] = []
    if args.pattern:
        files.extend(sorted(_glob.glob(args.pattern)))
    if args.inputs:
        files.extend([p for p in args.inputs if os.path.exists(p)])
    files = list(dict.fromkeys(files))  # 去重，保持次序

    if not files:
        raise SystemExit("No input JSONL files found. Use --glob or --inputs.")

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取全部记录
    file_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for fp in files:
        file_to_records[fp] = list(load_jsonl(fp))

    # 针对每个 metric 画一张图，图中包含多条曲线（每个文件一条）
    for metric in args.metrics:
        plt.figure(figsize=(10, 6))
        plotted = 0

        for fp, recs in file_to_records.items():
            xs, ys = extract_series(recs, metric)
            if not xs or not ys:
                continue
            if args.smooth and args.smooth > 1:
                ys = moving_average(ys, args.smooth)
            label = os.path.basename(fp).replace(".jsonl", "")
            plt.plot(xs, ys, label=label, linewidth=1.6)
            plotted += 1

        if plotted == 0:
            plt.close()
            continue

        plt.xlabel("iteration")
        plt.ylabel(metric)
        plt.title(f"Evolution: {metric}")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(fontsize=8)

        out_path = os.path.join(args.out_dir, f"plot_{metric}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    # 导出 CSV（可选）：每个文件导出其主指标（第一个 metric）
    if args.metrics:
        primary = args.metrics[0]
        for fp, recs in file_to_records.items():
            xs, ys = extract_series(recs, primary)
            if not xs or not ys:
                continue
            csv_path = os.path.join(args.out_dir, os.path.basename(fp).replace(".jsonl", f"_{primary}.csv"))
            with open(csv_path, "w", encoding="utf-8") as w:
                w.write("iteration,value\n")
                for a, b in zip(xs, ys):
                    w.write(f"{a},{b}\n")


if __name__ == "__main__":
    main()




import glob as _glob
import json
import os
from typing import List, Tuple, Dict, Any

import numpy as np

# 非交互环境使用无头后端
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # 跳过坏行
                continue


def extract_series(records: List[Dict[str, Any]], metric: str, iter_key: str = "iteration") -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    fallback_iter = 0
    for rec in records:
        # 迭代步数
        it = rec.get(iter_key)
        if isinstance(it, int):
            x = it
            fallback_iter = it
        else:
            # 退化：若不存在 iteration，则按 10 步累加
            fallback_iter = fallback_iter + 10
            x = fallback_iter

        # 指标取值
        val = rec.get(metric)
        if val is None:
            # 尝试常见别名
            aliases = {
                "archive_total_max": ["archive_total_max", "archive_total_score_max"],
                "archive_total_mean": ["archive_total_mean", "archive_total_score_mean"],
                "archive_fitness_max": ["archive_fitness_max"],
                "archive_fitness_mean": ["archive_fitness_mean"],
                "child_mean": ["child_mean", "child_score_mean"],
                "child_sparsity": ["child_sparsity"],
            }.get(metric, [])
            for k in aliases:
                if rec.get(k) is not None:
                    val = rec.get(k)
                    break

        if isinstance(val, (int, float)):
            xs.append(int(x))
            ys.append(float(val))

    return xs, ys


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) == 0:
        return values
    arr = np.asarray(values, dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    # 头部对齐（用前几个原始值填充）
    head = list(arr[: window - 1])
    return head + list(smoothed)


def main():
    parser = argparse.ArgumentParser(description="Plot evolution curves from JSONL logs")
    parser.add_argument("--inputs", nargs="*", help="Explicit list of JSONL files")
    parser.add_argument("--glob", dest="pattern", default=None, help="Glob pattern for JSONL files")
    parser.add_argument("--metrics", nargs="+", default=["archive_total_max"],
                        help="Metrics to plot per file (e.g., archive_total_max archive_fitness_max child_mean)")
    parser.add_argument("--out_dir", default="results/fitness_logs/plots", help="Output directory for plots")
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window size (>=1)")
    args = parser.parse_args()

    files: List[str] = []
    if args.pattern:
        files.extend(sorted(_glob.glob(args.pattern)))
    if args.inputs:
        files.extend([p for p in args.inputs if os.path.exists(p)])
    files = list(dict.fromkeys(files))  # 去重，保持次序

    if not files:
        raise SystemExit("No input JSONL files found. Use --glob or --inputs.")

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取全部记录
    file_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for fp in files:
        file_to_records[fp] = list(load_jsonl(fp))

    # 针对每个 metric 画一张图，图中包含多条曲线（每个文件一条）
    for metric in args.metrics:
        plt.figure(figsize=(10, 6))
        plotted = 0

        for fp, recs in file_to_records.items():
            xs, ys = extract_series(recs, metric)
            if not xs or not ys:
                continue
            if args.smooth and args.smooth > 1:
                ys = moving_average(ys, args.smooth)
            label = os.path.basename(fp).replace(".jsonl", "")
            plt.plot(xs, ys, label=label, linewidth=1.6)
            plotted += 1

        if plotted == 0:
            plt.close()
            continue

        plt.xlabel("iteration")
        plt.ylabel(metric)
        plt.title(f"Evolution: {metric}")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(fontsize=8)

        out_path = os.path.join(args.out_dir, f"plot_{metric}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    # 导出 CSV（可选）：每个文件导出其主指标（第一个 metric）
    if args.metrics:
        primary = args.metrics[0]
        for fp, recs in file_to_records.items():
            xs, ys = extract_series(recs, primary)
            if not xs or not ys:
                continue
            csv_path = os.path.join(args.out_dir, os.path.basename(fp).replace(".jsonl", f"_{primary}.csv"))
            with open(csv_path, "w", encoding="utf-8") as w:
                w.write("iteration,value\n")
                for a, b in zip(xs, ys):
                    w.write(f"{a},{b}\n")


if __name__ == "__main__":
    main()





