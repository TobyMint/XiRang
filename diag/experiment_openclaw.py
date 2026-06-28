"""诚实实验: OpenClaw 真实 prompt 结构下 vLLM prefix cache 命中率。

OpenClaw 已把 Runtime(易变)后置到 system 末尾, 但 vLLM chat template 把
tools 渲染在 system 之后, 所以实际 prompt 结构是:
    [stable system] + [Runtime 易变] + [tools schema] + [user 历史]

本实验用真实抓到的 OpenClaw system prompt + 真实 tools, 模拟多轮:
- 场景 A (现状): Runtime 段含易变字段, 放 system 末尾 (OpenClaw 现状)
- 场景 B (改进): 把易变字段从 Runtime 移到 user 消息, system 完全稳定

测 vLLM prefix_cache_hits/queries, 量化"即使后置, Runtime 变化仍让 tools+历史失效"的损失。
"""

from __future__ import annotations

import json
import time
import urllib.request

VLLM = "http://127.0.0.1:8001"

# 真实抓到的 OpenClaw system prompt (30838 chars)
with open("diag/captures/real_system_template.txt", encoding="utf-8") as f:
    REAL_SYSTEM = f.read()

# 真实抓到的 28 个 tools
import glob
_cap = sorted(glob.glob("diag/captures/capture-*.jsonl"))[-1]
_r0 = json.loads(open(_cap).readline())
REAL_TOOLS = _r0["request"]["tools"]


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


def send(messages, tools):
    body = json.dumps({
        "model": "Qwen3-4B", "messages": messages, "tools": tools,
        "max_tokens": 8, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{VLLM}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()
    return time.time() - t0


def run(name, build_messages, turns):
    q0, h0 = metrics()
    lats = []
    for t in range(turns):
        lats.append(send(build_messages(t), REAL_TOOLS))
    time.sleep(0.5)
    q1, h1 = metrics()
    q, h = q1 - q0, h1 - h0
    return {
        "scenario": name, "turns": turns,
        "queries": int(q), "hits": int(h),
        "hit_rate": round(h / q, 3) if q else 0,
        "avg_lat_ms": round(sum(lats) / len(lats) * 1000, 1),
        "latencies": [round(l * 1000, 1) for l in lats],
    }


# system prompt 里 Runtime 段的真实前缀(用于定位)
RT_PREFIX = "## Runtime\nRuntime: agent="


def main():
    turns = 6
    print(f"真实 system prompt: {len(REAL_SYSTEM)} chars, tools: {len(REAL_TOOLS)} 个\n")

    # 场景 A: OpenClaw 现状 — Runtime 段含易变字段(turn/timestamp), 在 system 末尾
    def scenario_a(t):
        # 复制 system, 替换 Runtime 段里的易变部分
        sys = REAL_SYSTEM
        # Runtime 段在末尾, 含 thinking=off 等。这里在 Runtime 行后追加易变 turn 标记
        volatile = f"\n[DYNAMIC] turn={t} ts=2026-06-28T19:{30+t:02d}:00"
        # 插在 Runtime 段开头处(模拟 Runtime 本身每轮变)
        idx = sys.find(RT_PREFIX)
        sys_dyn = sys[:idx] + sys[idx:].replace(
            "Runtime: agent=", f"Runtime: turn={t} ts=T{30+t} | agent=", 1)
        return [{"role": "system", "content": sys_dyn},
                {"role": "user", "content": f"step {t}"}]

    # 场景 B: 改进 — Runtime 段完全静态(移除 turn/timestamp), 易变字段进 user 消息
    def scenario_b(t):
        return [{"role": "system", "content": REAL_SYSTEM},
                {"role": "user", "content": f"step {t} (context: turn={t} ts=T{30+t})"}]

    print("=== 场景 A: Runtime 含易变字段 (OpenClaw 现状) ===")
    a = run("A现状(Runtime易变)", scenario_a, turns)
    print(json.dumps(a, indent=2, ensure_ascii=False))
    print("\n=== 场景 B: 易变字段移入 user (改进) ===")
    b = run("B改进(易变入user)", scenario_b, turns)
    print(json.dumps(b, indent=2, ensure_ascii=False))

    # 出图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.bar(["A: Runtime volatile (current)", "B: volatile in user (proposed)"],
                [a["hit_rate"], b["hit_rate"]], color=["#d62728", "#2ca02c"])
        ax1.set_ylabel("prefix cache hit rate"); ax1.set_ylim(0, 1)
        ax1.set_title("OpenClaw real structure: vLLM prefix cache")
        for i, v in enumerate([a["hit_rate"], b["hit_rate"]]):
            ax1.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
        ax2.plot(range(turns), a["latencies"], "o-", label="A current", color="#d62728")
        ax2.plot(range(turns), b["latencies"], "o-", label="B proposed", color="#2ca02c")
        ax2.set_xlabel("turn"); ax2.set_ylabel("latency (ms)"); ax2.legend(); ax2.grid(alpha=0.3)
        ax2.set_title("per-turn latency")
        plt.tight_layout()
        out = "diag/captures/openclaw_real_experiment.png"
        plt.savefig(out, dpi=120); plt.close()
        print(f"\n图: {out}")
    except ImportError:
        pass
    print(f"\n结论: 现状 {a['hit_rate']:.1%} → 改进 {b['hit_rate']:.1%}")


if __name__ == "__main__":
    main()
