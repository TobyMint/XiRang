#!/usr/bin/env bash
# 启动 XiRang proxy (xirang conda 环境)。
# 环境变量可覆盖后端地址等 (见 xirang/config.py)。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate xirang
cd "$ROOT"

export XIRANG_BACKEND="${XIRANG_BACKEND:-http://127.0.0.1:8001}"
export XIRANG_PORT="${XIRANG_PORT:-8000}"

echo "[start_xirang] env=xirang backend=$XIRANG_BACKEND listen=:$XIRANG_PORT"
python -m xirang.proxy.server
