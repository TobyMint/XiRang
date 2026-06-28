# XiRang 设计文档

## 1. 定位

XiRang v0 是一个**显存感知的 Agent 内存管理中间层**，位于 Agent 框架与 vLLM (OpenAI-compatible) 之间：

```
Agent 框架 (多轮/工具调用/分支)
        │  OpenAI-compatible HTTP
        ▼
   ┌──────────┐
   │  XiRang  │  ← 本项目 v0：HTTP proxy + 内存优化 + 指标记录
   └──────────┘
        │  OpenAI-compatible HTTP
        ▼
   vLLM (naive)  ← v0 不魔改
```

**v0 不魔改 vLLM**，而是做一个 HTTP proxy，拦截 `/v1/chat/completions`，在转发前完成内存优化。目标：先证明 *naive Agent + naive vLLM* 会显存持续上涨、延迟变高、甚至 OOM；加上 XiRang 后任务成功率基本不变，但显存峰值、KV Cache 占用、TTFT 下降。

## 2. v0 优化手段（全部在 proxy 侧，确定/近无损）

| 手段 | 模块 | 解决的问题 |
|---|---|---|
| 工具结果 head-tail 截断 | `memory/compression.py` | 工具调用产生的大规模中间数据进 KV Cache |
| 工具结果外部化 + 占位符 | `tools/tool_store.py` | 超预算时把老的工具结果落盘，按需加载 |
| thinking 块删除 | `memory/compression.py` | `<think>...</think>` 残留进历史上下文 |
| system prompt 去重 | `memory/compression.py` | 重复拼接的工具描述/规则 |
| 稳定前缀指纹 | `memory/lifecycle.py` | 量化 prefix cache 复用机会 |
| 上下文预算与淘汰决策 | `memory/budget.py` | 软预算下渐进压缩，保证成功率 |

> 这些是"先证明问题存在 + 先拿到收益"的最小集。后续接入 vLLM 内部后，会演化成真正的 KV block 复用/淘汰/分层（CoW 分支共享、显存-主存分层迁移）。

## 3. 模块职责

```
xirang/
├── config.py            集中配置（环境变量可覆盖）
├── proxy/
│   ├── server.py        FastAPI proxy，拦截 /v1/chat/completions，转发+记录
│   ├── openai_adapter.py  纯 HTTP/SSE 透传到 vLLM
│   └── request_rewriter.py 编排：lifecycle→budget→compression→外部化
├── memory/
│   ├── segment.py       上下文分段 + token 估算（启发式，不依赖分词器）
│   ├── lifecycle.py     会话级 KV 增长/前缀指纹记录
│   ├── budget.py        token 软预算与淘汰决策
│   └── compression.py   thinking 删除 / 工具结果截断 / system 去重
├── tools/
│   ├── tool_store.py    工具结果落盘 + 占位符引用
│   └── lazy_loader.py   按需取回工具结果
└── metrics/
    ├── gpu_monitor.py   pynvml 显存/利用率采样（无 GPU 自动降级）
    ├── latency.py       TTFT / e2e / 吞吐计时
    └── logger.py        JSONL 指标日志
```

## 4. 请求流转

1. Agent 发 `POST /v1/chat/completions`（带完整历史，模拟无状态 Agent）。
2. `server.py` 取出 messages，按 `x-xirang-session` header 归会话。
3. `RequestRewriter.rewrite`：
   - `lifecycle.observe`：记录本轮 naive 上下文 token 数与稳定前缀指纹；
   - `compression.compress_messages`：thinking 删除、工具结果截断、system 去重；
   - `budget.decide`：超预算时把最老的 tool 结果 `_externalize_tools` 到 `ToolStore`，上下文留占位符。
4. `openai_adapter` 把重写后的 payload 透传给 vLLM（流式/非流式）。
5. `metrics` 记录 GPU before/after、TTFT/e2e、压缩统计、lifecycle → `runs/<run>/metrics.jsonl`。

## 5. 为什么这样设计能拿分

- **应用效果 (40%)**：proxy 侧压缩直接降 input token → KV Cache 占用↓ → 显存峰值↓、TTFT↓。`run_baseline` vs `run_xirang` + `plot_results` 给出可复现曲线。
- **功能完整性 (30%)**：完整跑通一个支持多轮+工具调用+分支的 Agent 推理路径，集成内存优化；多任务由多 session 体现。
- **代码规范 (20%)** / **文档 (10%)**：分层清晰、有测试、有部署与 benchmark 文档。

## 6. v0 不做的事（后续路线）

- 不改 vLLM 内部 block allocator / prefix cache（v0 只在 proxy 侧量化复用机会）；
- 不做真正的 KV CoW 分支共享（仅记录分支指纹，证明可共享）；
- 不做显存-主存分层迁移（仅预留接口）；
- 不做异构 NPU 适配（仅 CUDA，预留扩展点）。
