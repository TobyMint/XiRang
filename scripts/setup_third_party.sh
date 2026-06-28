#!/usr/bin/env bash
# 按固定版本 clone 外部依赖到 third_party/。
# 可复现：版本钉死在此脚本。后续接入 OpenClaw 时在此追加一段即可。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p third_party

clone_at() {
    local name="$1" url="$2" ref="$3"
    local dest="third_party/$name"
    if [ -d "$dest/.git" ]; then
        echo "[setup] $name 已存在，跳过 ($dest)"
        return 0
    fi
    echo "[setup] clone $name @ $ref -> $dest"
    git clone --depth 1 --branch "$ref" "$url" "$dest"
}

# ---- vLLM v0.10.2 ----
clone_at vllm https://github.com/vllm-project/vllm.git v0.10.2

# ---- OpenClaw v2026.6.10 (Agent 框架，TypeScript/pnpm monorepo) ----
clone_at openclaw https://github.com/openclaw/openclaw.git v2026.6.10

echo "[setup] done. third_party/:"
ls -1 third_party
