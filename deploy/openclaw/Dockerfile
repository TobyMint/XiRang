# OpenClaw 运行镜像 (轻量：只装 Node 24，复用物理机已 build 的 third_party/openclaw)
# 策略：node_modules + dist 由 docker-compose 以挂载方式注入，镜像不打包它们。
# 因此镜像小、构建快；代价是绑定物理机已 build 的产物(符合当前阶段需求)。

FROM node:24-bookworm-slim

# pnpm 11.2.2 (openclaw package.json 钉死版本；corepack 激活)
RUN corepack enable && corepack prepare pnpm@11.2.2 --activate

# 基础工具(部分 npm 原生模块 postinstall 可能需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/openclaw

# openclaw.mjs / package.json / node_modules / dist 由 compose 挂载进来
ENV VLLM_API_KEY=vllm-local

# 默认起一个健康可探活的常驻进程；实际命令由 compose 覆盖
CMD ["node", "--version"]
