# OpenClaw 容器化部署

OpenClaw 以容器方式运行，vLLM 留在物理机。OpenClaw 容器复用物理机已 build 的 `third_party/openclaw`（挂载），不重新 install/build；经 `host.docker.internal` 连物理机 vLLM。

## 架构

```
物理机 (GPU3)
└─ vLLM :8001  (conda env `xirang`, Qwen3-4B)   ← 复用现成编译，不容器化

docker 容器 (xirang/openclaw:local)
├─ 挂载 third_party/openclaw  (源码+node_modules+dist，物理机已 build)
├─ 挂载 openclaw.container.json5  (容器专用配置，baseUrl=host.docker.internal:8001)
├─ OPENCLAW_CONFIG_PATH=/app/openclaw.container.json5
└─ extra_hosts: host.docker.internal -> host-gateway
   │  vllm provider (openai-completions)
   ▼
物理机 vLLM :8001 -> Qwen3-4B
```

容器**不需要 GPU**——它只是 HTTP 客户端连物理机 vLLM。

## 前置

1. 物理机已 `pnpm install + pnpm build` 好 `third_party/openclaw`
2. 物理机 vLLM 已起（卡3，:8001）：
   ```bash
   conda activate xirang
   CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
     --model /data/models/Qwen3-4B --served-model-name Qwen3-4B \
     --port 8001 --gpu-memory-utilization 0.85 --max-model-len 8192 --trust-remote-code
   ```

## 使用

```bash
cd deploy/openclaw

# 构建(首次)
docker compose build

# 一次性推理
docker compose run --rm openclaw infer model run --local \
  --model vllm/Qwen3-4B --prompt "你好" --thinking off

# 交互式 shell
docker compose run --rm openclaw bash

# 查看模型列表
docker compose run --rm openclaw models list --provider vllm
```

## 文件

| 文件 | 作用 |
|---|---|
| `Dockerfile` | Node 24 + pnpm 11.2.2 轻量镜像（不含 openclaw 产物，靠挂载） |
| `docker-compose.yml` | 声明式服务：挂载、网络、配置、环境变量 |
| `openclaw.container.json5` | 容器专用配置（baseUrl 指向 host.docker.internal） |

## 切换 baseline / 经 XiRang

当前容器配置直连 vLLM（baseline）。后续若加回 XiRang proxy 中间层，只需改 `openclaw.container.json5` 的 `baseUrl` 指向 XiRang proxy 端口（proxy 也需监听 `0.0.0.0` 或 host 网络）。

## 设计权衡

- **挂载而非打包 build 产物**：镜像小(~300MB)、构建快；代价是绑定物理机 `third_party/openclaw` 产物。当前开发阶段优先迭代速度，可接受。后续要分发可改为 Dockerfile 内 `pnpm install + build`。
- **独立配置文件**：用 `OPENCLAW_CONFIG_PATH` 指向容器专用配置，不污染物理机 `~/.openclaw`。
- **vLLM 不容器化**：复用物理机已编译的 vLLM（editable install，改源码即生效），避免重新构建 vLLM 镜像的巨大成本。
