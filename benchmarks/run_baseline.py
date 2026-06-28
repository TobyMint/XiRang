"""基准测试驱动：直连 vLLM 后端（naive Agent + naive vLLM）。

不经过 XiRang，逐轮把完整上下文发给 vLLM，模拟常见无状态 Agent。
记录延迟、GPU、token 到 runs/<run>/metrics.jsonl (mode=baseline)。

用法:
    python -m benchmarks.run_baseline --workload benchmarks/workloads/long_tool_call.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid

import httpx

# 让 `import xirang...` 在直接运行脚本时可用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xirang.config import settings  # noqa: E402
from xirang.memory.budget import BudgetController  # noqa: E402
from xirang.memory.lifecycle import LifecycleManager  # noqa: E402
from xirang.metrics.gpu_monitor import GPUMonitor  # noqa: E402
from xirang.metrics.logger import MetricsLogger  # noqa: E402


def load_workload(path: str) -> list[dict]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


async def run_one(client: httpx.AsyncClient, task: dict, model: str, logger: MetricsLogger,
                  gpu: GPUMonitor, budget: BudgetController, lifecycle: LifecycleManager) -> None:
    session_id = task.get("id", uuid.uuid4().hex[:8])
    turns = task.get("turns", [])
    for turn_idx, messages in enumerate(turns):
        # 量化 naive 上下文增长（baseline 不做任何压缩）
        turn_rec = lifecycle.observe(session_id, messages, settings.chars_per_token)
        est_in = budget.estimate(messages)
        gpu_before = gpu.snapshot()
        t0 = time.time()
        success = True
        err = None
        out_tokens = 0
        try:
            payload = {"model": model, "messages": messages, "max_tokens": 64, "stream": False}
            resp = await client.post(f"{settings.backend_url}/v1/chat/completions", json=payload, timeout=120.0)
            if resp.status_code == 200:
                usage = resp.json().get("usage", {})
                out_tokens = usage.get("completion_tokens", 0)
            else:
                success = False
                err = f"status {resp.status_code}"
        except Exception as e:  # noqa: BLE001
            success = False
            err = str(e)
        e2e_ms = (time.time() - t0) * 1000
        gpu_after = gpu.snapshot()
        logger.log_request(
            run="baseline",
            mode="baseline",
            session_id=session_id,
            turn=turn_idx,
            estimated_input_tokens=est_in,
            output_tokens=out_tokens,
            latency={"ttft_ms": None, "e2e_ms": round(e2e_ms, 2), "throughput_tok_s": None},
            gpu={
                "before": {"mem_used_mib": round(gpu_before.mem_used_mib, 1),
                            "mem_total_mib": round(gpu_before.mem_total_mib, 1),
                            "mem_used_pct": round(gpu_before.mem_used_pct, 2)},
                "after": {"mem_used_mib": round(gpu_after.mem_used_mib, 1),
                          "mem_total_mib": round(gpu_after.mem_total_mib, 1),
                          "mem_used_pct": round(gpu_after.mem_used_pct, 2)},
            },
            compression={},
            lifecycle={"turn": turn_rec.turn, "total_tokens": turn_rec.total_tokens,
                       "stable_prefix_tokens": turn_rec.stable_prefix_tokens,
                       "prefix_fingerprint": turn_rec.prefix_fingerprint},
            success=success,
            error=err,
        )
        print(f"  [baseline] {session_id} turn {turn_idx}: in~{est_in}tok "
              f"e2e={e2e_ms:.0f}ms gpu={gpu_after.mem_used_pct:.1f}% "
              f"{'OK' if success else 'ERR:'+str(err)}")


async def main_async(args: argparse.Namespace) -> None:
    tasks = load_workload(args.workload)
    logger = MetricsLogger(args.run, settings.metrics_dir)
    gpu = GPUMonitor(settings.gpu_device)
    budget = BudgetController(settings.context_token_budget, settings.chars_per_token)
    lifecycle = LifecycleManager()

    print(f"[baseline] {len(tasks)} tasks, {args.model}, backend={settings.backend_url}")
    async with httpx.AsyncClient() as client:
        for task in tasks:
            await run_one(client, task, args.model, logger, gpu, budget, lifecycle)
    gpu.close()
    print(f"[baseline] done. metrics -> {logger.path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, help="path to a .jsonl workload")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--run", default="baseline_run")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
