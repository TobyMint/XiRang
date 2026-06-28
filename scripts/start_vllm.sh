#!/usr/bin/env bash
# 启动 vLLM OpenAI-compatible 后端 (vLLM 0.10.2)。
#
# 模型: /data/models/Qwen3-4B (本地，不下载)
# 运行前需安装 vllm==0.10.2:
#   pip install vllm==0.10.2          # 预编译 wheel（推荐）
#   # 或从 third_party/vllm 源码安装（魔改时）:
#   # pip install -e third_party/vllm
#
# 用法: ./scripts/start_vllm.sh [model_path]
set -euo pipefail

MODEL="${1:-/data/models/Qwen3-4B}"
PORT="${VLLM_PORT:-8001}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.6}"
# Qwen3 为 thinking 模型，开启 thinking 模式
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

echo "[start_vllm] vllm=$(python -c 'import vllm,sys;sys.stdout.write(vllm.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "[start_vllm] model=$MODEL port=$PORT gpu_mem_util=$GPU_MEM_UTIL max_model_len=$MAX_MODEL_LEN"

if ! python -c 'import vllm' 2>/dev/null; then
    echo "[error] vllm 未安装。请运行: pip install vllm==0.10.2"
    exit 1
fi

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code
