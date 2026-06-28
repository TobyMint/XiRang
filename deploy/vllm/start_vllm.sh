#!/usr/bin/env bash
# 启动 vLLM OpenAI-compatible 后端 (vLLM 0.10.2, xirang conda 环境)。
#
# vLLM 跑在物理机(不容器化)，直接用 GPU + 复用已 editable 编译的 third_party/vllm。
# 模型: /data/models/Qwen3-4B (本地，served-model-name=Qwen3-4B)
# OpenClaw 容器经 host.docker.internal 连本服务 :8001。
#
# 用法:
#   ./deploy/vllm/start_vllm.sh                 # 默认卡3
#   CUDA_VISIBLE_DEVICES=3 ./deploy/vllm/start_vllm.sh
#   VLLM_GPU_MEM_UTIL=0.85 ./deploy/vllm/start_vllm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate xirang
cd "$ROOT"

# 默认卡3(实验用卡)。GPU0/1 常被占，卡3 空闲。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

MODEL="${VLLM_MODEL:-/data/models/Qwen3-4B}"
SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3-4B}"
PORT="${VLLM_PORT:-8001}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

echo "[start_vllm] env=xirang  vllm=$(python -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "[start_vllm] GPU=CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[start_vllm] model=$MODEL served-name=$SERVED_NAME port=$PORT gpu_mem_util=$GPU_MEM_UTIL max_model_len=$MAX_MODEL_LEN"

if ! python -c 'import vllm' 2>/dev/null; then
    echo "[error] vllm 未安装。请先运行: ./deploy/vllm/build_vllm.sh"
    exit 1
fi

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code
