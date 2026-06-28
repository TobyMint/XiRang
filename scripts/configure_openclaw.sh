#!/usr/bin/env bash
# 把 OpenClaw 配置切换为 baseline (直连 vLLM) 或 xirang (经 XiRang proxy)。
# 用法: ./scripts/configure_openclaw.sh baseline|xirang
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-xirang}"

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm use v24.18.0 >/dev/null 2>&1 || true
export VLLM_API_KEY="${VLLM_API_KEY:-vllm-local}"

case "$MODE" in
  baseline) PATCH="$ROOT/configs/openclaw-vllm-baseline.json5" ;;
  xirang)   PATCH="$ROOT/configs/openclaw-vllm-xirang.json5" ;;
  *) echo "usage: $0 baseline|xirang"; exit 1 ;;
esac

echo "[configure_openclaw] mode=$MODE patch=$PATCH"
cd "$ROOT/third_party/openclaw"
./node_modules/.bin/openclaw config patch --file "$PATCH"
echo "[configure_openclaw] done. baseUrl:"
./node_modules/.bin/openclaw config get models.providers.vllm.baseUrl
