# Benchmark 方案

## 目标

可复现地证明：

1. **baseline**（naive Agent + naive vLLM，直连）：多轮/工具调用下，上下文持续增长 → 显存占用上涨、e2e 延迟上升、长任务可能 OOM。
2. **xirang**（经 XiRang proxy）：任务成功率基本不变，但 input token ↓ → 显存峰值 ↓、KV Cache 占用 ↓、TTFT ↓。

## 工作负载

`benchmarks/workloads/*.jsonl`，每行一个任务，格式：

```json
{"id":"...", "turns":[ [msg...], [msg...], ... ]}
```

`turns[i]` 是第 i 轮发给 API 的**完整 messages 数组**（模拟无状态 Agent 重发历史，正是 KV Cache 浪费的来源）。

| 文件 | 场景 | 暴露的问题 |
|---|---|---|
| `long_tool_call.jsonl` | 超大工具输出（shell/log） | 工具中间数据进 KV Cache 膨胀 |
| `multi_turn_code_agent.jsonl` | 多轮代码 Agent + 重复 system | system 冗余 + 累积工具 diff |
| `branch_reasoning.jsonl` | 多路径分支推理 | 分支共享前缀、thinking 残留 |

## 运行

### 真实后端（需要 GPU + vLLM）

```bash
# 终端 1：起 vLLM
./scripts/start_vllm.sh Qwen/Qwen2.5-1.5B-Instruct
# 终端 2：起 XiRang proxy
./scripts/start_xirang.sh
# 终端 3：跑 baseline + xirang + 出图
./scripts/run_all.sh benchmarks/workloads/long_tool_call.jsonl
```

### 无 GPU 演示模式（验证压缩效果，不需要 vLLM）

```bash
# 仅演示 XiRang 压缩前后 input token 对比，不发真实请求
python -m benchmarks.run_xirang \
  --workload benchmarks/workloads/long_tool_call.jsonl \
  --run xirang_demo --mock
```

## 指标

每次请求记录到 `runs/<run>/metrics.jsonl`：

- `estimated_input_tokens`：proxy 侧估算的 input token（xirang 为压缩后，baseline 为原始）
- `latency`：`ttft_ms` / `e2e_ms` / `throughput_tok_s`
- `gpu`：before/after `mem_used_mib` / `mem_used_pct` / `gpu_util_pct`
- `compression`：`chars_before/after`、`tools_compressed`、`thinking_removed`、`externalized`、`naive_input_tokens`
- `lifecycle`：`turn` / `total_tokens` / `stable_prefix_tokens` / `prefix_fingerprint`
- `success` / `error`：任务成功率

## 出图

```bash
python -m benchmarks.plot_results --runs baseline_run xirang_run --out runs/compare
```

输出 `runs/compare/`：`input_tokens.png`、`gpu_mem.png`、`latency.png`、`success_rate.png`、`summary.txt`。

## 复现性约束

- 优化前后**同一硬件**、同一模型、同一工作负载；
- 显存利用率等 vLLM 启动参数固定（`start_vllm.sh`）；
- 指标全量落 JSONL，可重跑出图。
