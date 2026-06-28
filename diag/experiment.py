"""对照实验：易变 metadata 前置 vs 后置 对 vLLM prefix cache 命中率的影响。

模拟 Agent 多轮请求。每轮有一个"易变字段"(时间戳/turn 计数器)。
- 场景 A (前置, naive): 易变字段放在 system prompt 里 (前缀区) → 每轮前缀变化 → 命中率低
- 场景 B (后置, 重构): 易变字段放在最后一条 user message 里 → 前缀稳定 → 命中率高

直接打 vLLM /v1/chat/completions，用 /metrics 的 prefix_cache_hits/queries 做差测量。
产出对比图 diag/captures/prefix_experiment.png。

用法: python diag/experiment.py [--turns 8]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

VLLM = "http://127.0.0.1:8001"

# 模拟 OpenClaw 风格的稳定大前缀 (system + tools)。够长才能体现 prefix 价值。
STABLE_SYSTEM = (
    "You are a personal assistant running inside an agent framework.\n"
    "## Tooling\n"
    "Available tools: read, write, edit, exec, process, web_search, web_fetch, browser, "
    "canvas, nodes, cron, message, gateway, memory_get, memory_search, session_status.\n"
    "## Tool Call Style\n"
    "Routine low-risk calls: no narration. Narrate only for complex or sensitive steps.\n"
    "## Execution Bias\n"
    "Actionable request: act in this turn. Continue until done or genuinely blocked.\n"
    "Mutable facts need live checks: files, git, clocks, versions, services, processes.\n"
    "## Safety\n"
    "No independent goals. Safety/oversight over completion. Obey stop/pause/audit.\n"
    "## Workspace\n"
    "Treat this directory as the single global workspace for file operations.\n"
    "## Documentation\n"
    "Reply with concise steps. Use first-class tools; do not ask user to run CLI.\n"
) * 6  # 拉长到 ~3k chars，模拟真实大前缀


def metrics() -> tuple[float, float]:
    with urllib.request.urlopen(f"{VLLM}/metrics", timeout=10) as r:
        text = r.read().decode()
    q = h = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:prefix_cache_queries_total{"):
            q = float(line.split()[-1])
        elif line.startswith("vllm:prefix_cache_hits_total{"):
            h = float(line.split()[-1])
    return q, h


def send(messages: list[dict]) -> float:
    body = json.dumps({
        "model": "Qwen3-4B",
        "messages": messages,
        "max_tokens": 8,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{VLLM}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()
    return time.time() - t0


def run_scenario(name: str, build_messages, turns: int) -> dict:
    q0, h0 = metrics()
    latencies = []
    for t in range(turns):
        msgs = build_messages(t)
        latencies.append(send(msgs))
    time.sleep(0.5)
    q1, h1 = metrics()
    queries = q1 - q0
    hits = h1 - h0
    rate = (hits / queries) if queries > 0 else 0.0
    return {
        "scenario": name,
        "turns": turns,
        "prefix_cache_queries": int(queries),
        "prefix_cache_hits": int(hits),
        "hit_rate": round(rate, 3),
        "latencies_ms": [round(l * 1000, 1) for l in latencies],
        "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=8)
    args = ap.parse_args()

    print(f"稳定前缀长度: {len(STABLE_SYSTEM)} chars\n")

    # 场景 A: 易变字段(模拟时间戳/turn)放在 system 最前面 (前置, naive)
    # 模拟 OpenCode/Kimi 真实问题: 时间放在 tools 描述里, 而 tools 渲染在 prompt 最前
    # → 前缀从第 0 个 token 就分叉 → 命中率趋零
    def front(t):
        volatile = f"## Runtime\nturn={t} | timestamp=2026-06-28T19:{30+t:02d}:00 | request_id=req{t:04d}\n"
        return [{"role": "system", "content": volatile + STABLE_SYSTEM},
                {"role": "user", "content": f"step {t}: do the task"}]

    # 场景 B: 易变字段放在最后一条 user 消息里 (后置, 重构) — 前缀完全稳定
    def back(t):
        volatile = f"current turn={t}, timestamp=2026-06-28T19:{30+t:02d}:00, request_id=req{t:04d}"
        return [{"role": "system", "content": STABLE_SYSTEM},
                {"role": "user", "content": f"step {t}: do the task. (context: {volatile})"}]

    print("=== 场景 A: 易变字段前置 (naive) ===")
    a = run_scenario("front(naive)", front, args.turns)
    print(json.dumps(a, indent=2, ensure_ascii=False))
    print("\n=== 场景 B: 易变字段后置 (重构) ===")
    b = run_scenario("back(restructured)", back, args.turns)
    print(json.dumps(b, indent=2, ensure_ascii=False))

    # 出图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.bar(["front (naive)", "back (restructured)"], [a["hit_rate"], b["hit_rate"]], color=["#d62728", "#2ca02c"])
        ax1.set_ylabel("prefix cache hit rate"); ax1.set_ylim(0, 1); ax1.set_title("Prefix Cache Hit Rate")
        for i, v in enumerate([a["hit_rate"], b["hit_rate"]]):
            ax1.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
        ax2.plot(range(args.turns), a["latencies_ms"], "o-", label="front (naive)", color="#d62728")
        ax2.plot(range(args.turns), b["latencies_ms"], "o-", label="back (restructured)", color="#2ca02c")
        ax2.set_xlabel("turn"); ax2.set_ylabel("latency (ms)"); ax2.set_title("per-turn e2e latency"); ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        out = "diag/captures/prefix_experiment.png"
        plt.savefig(out, dpi=120); plt.close()
        print(f"\n图已保存: {out}")
    except ImportError:
        print("\n(matplotlib 未装，跳过出图)")

    print(f"\n结论: 前置命中率 {a['hit_rate']:.1%} → 后置 {b['hit_rate']:.1%}")


if __name__ == "__main__":
    main()
