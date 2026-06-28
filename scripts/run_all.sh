#!/usr/bin/env bash
# 端到端跑一遍：baseline vs xirang，再出图。
# 前置: vLLM 后端已起 (./scripts/start_vllm.sh)，XiRang proxy 已起 (./scripts/start_xirang.sh)
set -euo pipefail

WORKLOAD="${1:-benchmarks/workloads/long_tool_call.jsonl}"
MODEL="${2:-Qwen/Qwen2.5-1.5B-Instruct}"

echo "=== [1/3] baseline (直连 vLLM) ==="
python -m benchmarks.run_baseline --workload "$WORKLOAD" --model "$MODEL" --run baseline_run

echo "=== [2/3] xirang (经 proxy) ==="
python -m benchmarks.run_xirang --workload "$WORKLOAD" --model "$MODEL" --run xirang_run

echo "=== [3/3] 出图 ==="
python -m benchmarks.plot_results --runs baseline_run xirang_run --out runs/compare

echo "done. 见 runs/compare/"
