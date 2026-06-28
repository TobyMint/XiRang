#!/usr/bin/env bash
# 在 xirang conda 环境 (python3.12, torch 2.8.0+cu126) 中以 editable 方式编译安装 vLLM 0.10.2。
# 一次性操作。之后用 ./deploy/vllm/start_vllm.sh 启动。
#
# 用法: ./deploy/vllm/build_vllm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate xirang
cd "$ROOT"

# CUDA 12.6 toolkit (匹配 torch cu126)
export CUDA_HOME="/data/spack/opt/spack/linux-ubuntu20.04-cascadelake/gcc-9.4.0/cuda-12.6.2-h3ia6zze3fxl6mbbb6xy3t6a652fbh3e"
export PATH="$CUDA_HOME/bin:$PATH"
# 限制并行避免内存峰值过高 (48核/251G 机器用 24 较稳)
export MAX_JOBS="${MAX_JOBS:-24}"
export VLLM_TARGET_DEVICE=cuda

echo "[build_vllm] env=xirang torch=$(python -c 'import torch;print(torch.__version__)')"
echo "[build_vllm] CUDA_HOME=$CUDA_HOME  nvcc=$(nvcc --version | tail -1)"
echo "[build_vllm] MAX_JOBS=$MAX_JOBS"
echo "[build_vllm] pip install -e third_party/vllm --no-build-isolation  (约 30-40 分钟)"

cd "$ROOT/third_party/vllm"
pip install -e . --no-build-isolation

# vLLM 0.10.2 声明 transformers>=4.55.2 但过松；transformers 5.x 有破坏性改动
# (ProcessorMixin)，需钉在 4.x。
pip install -q "transformers>=4.55.2,<5.0"
echo "[build_vllm] done. 验证: python -c 'import vllm;print(vllm.__version__)'"
