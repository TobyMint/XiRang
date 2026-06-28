#!/usr/bin/env bash
# 启动 vLLM OpenAI-compatible 后端 (vLLM 0.10.2, xirang conda 环境)。
#
# 模型: /data/models/Qwen3-4B (本地)
# 前置: 已用 ./scripts/build_vllm.sh 在 xirang 环境编译安装 vllm==0.10.2
#
# 用法: ./scripts/start_vllm.sh [model_path]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate xirang

MODEL="${1:-/data/models/Qwen3-4B}"
PORT="${VLLM_PORT:-8001}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.6}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

echo "[start_vllm] env=xirang vllm=$(python -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "[start_vllm] model=$MODEL port=$PORT gpu_mem_util=$GPU_MEM_UTIL max_model_len=$MAX_MODEL_LEN"

if ! python -c 'import vllm' 2>/dev/null; then
    echo "[error] vllm 未安装。请运行: ./scripts/build_vllm.sh"
    exit 1
fi

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code
