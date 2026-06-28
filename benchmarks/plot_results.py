"""从 runs/<run>/metrics.jsonl 读取指标，画对比图。

输出到 runs/<run>/plots/：
- 上下文 token 随轮次增长 (baseline vs xirang)
- 显存占用随轮次
- 端到端延迟随轮次
- 压缩前后 input token 对比

用法:
    python -m benchmarks.plot_results --runs baseline_run xirang_run --out runs/compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_run(metrics_dir: str, run: str) -> list[dict]:
    path = os.path.join(metrics_dir, run, "metrics.jsonl")
    if not os.path.exists(path):
        print(f"[warn] {path} not found")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(records: list[dict]) -> dict:
    by_turn: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_turn[r["turn"]].append(r)
    turns = sorted(by_turn)

    def avg(t, key, cast=float):
        vals = [cast(r.get(key) or 0) for r in by_turn[t]]
        return sum(vals) / len(vals) if vals else 0.0

    def avg_nested(t, *path, cast=float):
        vals = []
        for r in by_turn[t]:
            cur = r
            for p in path:
                cur = cur.get(p, {}) if isinstance(cur, dict) else {}
            vals.append(cast(cur or 0))
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "turns": turns,
        "input_tokens": [avg(t, "estimated_input_tokens") for t in turns],
        "gpu_pct": [avg_nested(t, "gpu", "after", "mem_used_pct") for t in turns],
        "e2e_ms": [avg_nested(t, "latency", "e2e_ms") for t in turns],
        "success_rate": [sum(1 for r in by_turn[t] if r["success"]) / len(by_turn[t]) for t in turns],
        "mode": records[0]["mode"] if records else "?",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names under runs/")
    ap.add_argument("--metrics-dir", default="runs")
    ap.add_argument("--out", default="runs/compare")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[error] matplotlib not installed: pip install matplotlib")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    summaries = {}
    for run in args.runs:
        recs = load_run(args.metrics_dir, run)
        if recs:
            summaries[run] = summarize(recs)

    if not summaries:
        print("[error] no data to plot")
        sys.exit(1)

    def _plot(key, ylabel, fname):
        plt.figure(figsize=(7, 4))
        for run, s in summaries.items():
            plt.plot(s["turns"], s[key], marker="o", label=f"{run} ({s['mode']})")
        plt.xlabel("turn")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs turn")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(args.out, fname)
        plt.savefig(p, dpi=120)
        plt.close()
        print(f"  saved {p}")

    _plot("input_tokens", "estimated input tokens", "input_tokens.png")
    _plot("gpu_pct", "GPU mem used %", "gpu_mem.png")
    _plot("e2e_ms", "end-to-end latency (ms)", "latency.png")
    _plot("success_rate", "success rate", "success_rate.png")

    # 文字汇总
    summary_path = os.path.join(args.out, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for run, s in summaries.items():
            avg_gpu = sum(s["gpu_pct"]) / len(s["gpu_pct"]) if s["gpu_pct"] else 0
            avg_lat = sum(s["e2e_ms"]) / len(s["e2e_ms"]) if s["e2e_ms"] else 0
            avg_sr = sum(s["success_rate"]) / len(s["success_rate"]) if s["success_rate"] else 0
            f.write(
                f"{run} ({s['mode']}): turns={len(s['turns'])} "
                f"avg_gpu={avg_gpu:.1f}% avg_latency={avg_lat:.0f}ms "
                f"avg_success={avg_sr:.2%}\n"
            )
    print(f"  saved {summary_path}")


if __name__ == "__main__":
    main()
