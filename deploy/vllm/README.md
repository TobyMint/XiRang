# vLLM 物理机部署

vLLM 跑在物理机（**不容器化**），直接用 GPU，复用已 editable 编译的 `third_party/vllm`。OpenClaw 容器经 `host.docker.internal` 连本服务。

## 为什么 vLLM 不容器化

- 复用物理机已编译的 vLLM editable 安装（改源码即生效，便于魔改 KV 管理）
- 避免重建 vLLM 镜像的巨大成本（含 CUDA 算子编译）
- OpenClaw 只需 HTTP 连 vLLM，vLLM 在不在容器里对它透明

## 一次性编译安装（仅首次）

```bash
./deploy/vllm/build_vllm.sh
```

在 `xirang` conda 环境（python3.12 / torch2.8.0+cu126 / CUDA12.6）以 editable 方式编译 vLLM 0.10.2，约 30-40 分钟。

## 启动

```bash
# 默认卡3
./deploy/vllm/start_vllm.sh

# 自定义
CUDA_VISIBLE_DEVICES=3 VLLM_GPU_MEM_UTIL=0.85 ./deploy/vllm/start_vllm.sh
```

默认参数：

| 参数 | 默认值 | 环境变量 |
|---|---|---|
| GPU | 卡3 | `CUDA_VISIBLE_DEVICES` |
| 模型 | `/data/models/Qwen3-4B` | `VLLM_MODEL` |
| served-model-name | `Qwen3-4B` | `VLLM_SERVED_NAME` |
| 端口 | `8001` | `VLLM_PORT` |
| gpu-memory-utilization | `0.85` | `VLLM_GPU_MEM_UTIL` |
| max-model-len | `8192` | `VLLM_MAX_MODEL_LEN` |

## 验证

```bash
curl http://127.0.0.1:8001/health          # -> 200
curl http://127.0.0.1:8001/v1/models       # -> Qwen3-4B
```

## 与 OpenClaw 容器对接

vLLM 监听 `0.0.0.0:8001`（vLLM 默认），OpenClaw 容器配置 `baseUrl=http://host.docker.internal:8001/v1` 即可连上。完整流程见 `deploy/openclaw/README.md`。

## 停止

```bash
# 找到 vllm 进程 pid 后精确 kill (勿用 pkill -f "Qwen3-4B"，会误杀命令自身)
pgrep -f "vllm.entrypoints.openai.api_server --model /data/models/Qwen3-4B"   # 取 pid
kill <pid>
```
