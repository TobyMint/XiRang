# XiRang

> 显存感知的 Agent 内存管理中间层：位于 Agent 框架与 vLLM 之间的 HTTP proxy。

第三届研究生操作系统大赛 · 研究创新赛道 · 高校赛题《面向智能体的内存管理系统设计与实现》。

## 是什么

XiRang v0 是一个 **HTTP proxy**，拦截 OpenAI 兼容的 `/v1/chat/completions` 请求，在转发给 vLLM 之前完成内存优化（prompt 重写、工具结果压缩、thinking 删除、超预算外部化），并记录 GPU/延迟指标用于可复现 benchmark。

**v0 不魔改 vLLM**。先证明：

- *naive Agent + naive vLLM*：多轮 / 工具调用下显存持续上涨、延迟变高、甚至 OOM；
- *加 XiRang 后*：任务成功率基本不变，但显存峰值 ↓、KV Cache 占用 ↓、TTFT ↓。

```
Agent 框架 ──HTTP──▶ XiRang proxy ──HTTP──▶ vLLM
                     (内存优化+指标)
```

## 快速开始

```bash
# 1. 装依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 跑测试
pytest -q

# 3. 无 GPU 演示（验证压缩效果，不需要 vLLM）
python -m benchmarks.run_xirang \
  --workload benchmarks/workloads/long_tool_call.jsonl --run xirang_demo --mock
```

### 真实后端（需 GPU + vLLM）

```bash
# 拉取外部依赖源码（vLLM 等）到 third_party/，可复现
./scripts/setup_third_party.sh
# 安装 vLLM 0.10.2（预编译 wheel）
pip install vllm==0.10.2

# 启动 vLLM（默认用本地 /data/models/Qwen3-4B）
./scripts/start_vllm.sh            # 终端1
./scripts/start_xirang.sh          # 终端2
./scripts/run_all.sh benchmarks/workloads/long_tool_call.jsonl  # 终端3
# 结果: runs/compare/
```

## 目录结构

```
XiRang/
├── README.md
├── requirements.txt / pyproject.toml
├── docs/
│   ├── design.md                 # 架构与模块设计
│   ├── benchmark.md              # benchmark 方案
│   └── deployment-openeuler.md   # openEuler 部署
├── xirang/
│   ├── config.py                 # 集中配置（环境变量可覆盖）
│   ├── proxy/
│   │   ├── server.py             # FastAPI proxy 主入口
│   │   ├── openai_adapter.py     # 透传 vLLM（流式/非流式）
│   │   └── request_rewriter.py   # 编排：lifecycle→budget→compression→外部化
│   ├── memory/
│   │   ├── segment.py            # 上下文分段 + token 估算
│   │   ├── lifecycle.py          # 会话级 KV 增长 / 前缀指纹
│   │   ├── budget.py             # token 软预算与淘汰决策
│   │   └── compression.py        # thinking 删除 / 工具结果截断 / system 去重
│   ├── tools/
│   │   ├── tool_store.py         # 工具结果落盘 + 占位符
│   │   └── lazy_loader.py        # 按需加载
│   └── metrics/
│       ├── gpu_monitor.py        # pynvml 显存采样（无 GPU 自动降级）
│       ├── latency.py            # TTFT / e2e / 吞吐
│       └── logger.py             # JSONL 指标日志
├── benchmarks/
│   ├── workloads/*.jsonl         # 长 tool 调用 / 多轮代码 Agent / 分支推理
│   ├── run_baseline.py           # 直连 vLLM
│   ├── run_xirang.py             # 经 proxy（含 --mock 演示）
│   └── plot_results.py           # 出图对比
├── scripts/
│   ├── setup_third_party.sh      # 拉取外部仓库 (vllm/openclaw) 到 third_party/
│   ├── start_vllm.sh / start_xirang.sh / run_all.sh
├── third_party/                  # 外部依赖仓库（gitignored，见 third_party/README.md）
│   └── vllm/                     # vLLM v0.10.2 源码（由 setup_third_party.sh 拉取）
└── tests/
    ├── test_rewriter.py / test_lifecycle.py / test_budget.py
```

> 模型统一放在 `/data/models/`，当前用 `/data/models/Qwen3-4B`，不纳入仓库。

## v0 优化手段

| 手段 | 模块 | 解决问题 |
|---|---|---|
| 工具结果 head-tail 截断 | `memory/compression.py` | 大规模工具中间数据进 KV Cache |
| 工具结果外部化 + 占位符 | `tools/tool_store.py` | 超预算时落盘、按需加载 |
| thinking 块删除 | `memory/compression.py` | `<think>` 残留进历史上下文 |
| system prompt 去重 | `memory/compression.py` | 重复的工具描述/规则 |
| 稳定前缀指纹 | `memory/lifecycle.py` | 量化 prefix cache 复用机会 |
| 上下文预算与淘汰 | `memory/budget.py` | 软预算下渐进压缩 |

## 配置

全部环境变量可覆盖（见 `xirang/config.py`）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `XIRANG_BACKEND` | `http://127.0.0.1:8001` | vLLM 后端地址 |
| `XIRANG_PORT` | `8000` | proxy 监听端口 |
| `XIRANG_ENABLED` | `1` | 是否启用内存优化（`0` = 透传） |
| `XIRANG_TOOL_COMPRESS_CHARS` | `800` | 工具结果超此字符才压缩 |
| `XIRANG_CONTEXT_BUDGET` | `6000` | 上下文 token 软预算 |
| `XIRANG_GPU_DEVICE` | `0` | 监控的 GPU 设备号 |

## 后续路线

- 接入 vLLM 内部：真正的 KV block 复用/淘汰/分层；
- 分支推理 KV CoW 共享（v0 仅记录分支指纹）；
- 显存-主存-外部存储分层迁移；
- 异构 NPU（DTK/CANN）适配，接口已隔离。

## 相关资料

- [vLLM 文档](https://docs.vllm.ai/en/stable/index.html) · [vLLM 源码](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) · [MiniCPM3-4B](https://www.modelscope.cn/models/OpenBMB/MiniCPM3-4B)
- [LangChain](https://github.com/langchain-ai/langchain) · [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
