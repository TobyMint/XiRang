#!/usr/bin/env bash
# 启动 vLLM OpenAI-compatible 后端。
# 用法: ./scripts/start_vllm.sh [model]
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${VLLM_PORT:-8001}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.6}"

echo "[start_vllm] model=$MODEL port=$PORT gpu_mem_util=$GPU_MEM_UTIL"

# 首次运行需安装: pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --trust-remote-code
