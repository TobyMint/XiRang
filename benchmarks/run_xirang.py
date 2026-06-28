"""基准测试驱动：经过 XiRang proxy。

逐轮把完整上下文发给 XiRang proxy (默认 :8000)，由 proxy 做内存优化后转发 vLLM。
记录延迟、GPU、token、压缩统计到 runs/<run>/metrics.jsonl (mode=xirang)。

用法:
    python -m benchmarks.run_xirang --workload benchmarks/workloads/long_tool_call.jsonl
    # 演示模式（无需真实 vLLM）:
    python -m benchmarks.run_xirang --workload benchmarks/workloads/long_tool_call.jsonl --mock
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xirang.config import settings  # noqa: E402
from xirang.memory.budget import BudgetController  # noqa: E402
from xirang.memory.lifecycle import LifecycleManager  # noqa: E402
from xirang.metrics.gpu_monitor import GPUMonitor  # noqa: E402
from xirang.metrics.logger import MetricsLogger  # noqa: E402
from xirang.memory.compression import compress_messages  # noqa: E402


def load_workload(path: str) -> list[dict]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


async def run_one(client: httpx.AsyncClient, task: dict, model: str, logger: MetricsLogger,
                  gpu: GPUMonitor, budget: BudgetController, lifecycle: LifecycleManager,
                  endpoint: str, mock: bool) -> None:
    session_id = task.get("id", uuid.uuid4().hex[:8])
    turns = task.get("turns", [])
    for turn_idx, messages in enumerate(turns):
        # baseline 视角的 naive 上下文规模（用于对比）
        naive_tokens = budget.estimate(messages)
        turn_rec = lifecycle.observe(session_id, messages, settings.chars_per_token)
        gpu_before = gpu.snapshot()
        t0 = time.time()
        success = True
        err = None
        out_tokens = 0
        try:
            payload = {"model": model, "messages": messages, "max_tokens": 64, "stream": False,
                       "user": session_id}
            if mock:
                # 演示模式：不发请求，模拟一个回复并估算压缩效果
                compressed, cstats = compress_messages(messages)
                comp_tokens = budget.estimate(compressed)
                out_tokens = 8
                e2e_ms = 5.0
                gpu_after = gpu.snapshot()
                logger.log_request(
                    run="xirang",
                    mode="xirang",
                    session_id=session_id,
                    turn=turn_idx,
                    estimated_input_tokens=comp_tokens,
                    output_tokens=out_tokens,
                    latency={"ttft_ms": 1.0, "e2e_ms": e2e_ms, "throughput_tok_s": 1600.0},
                    gpu={"before": {"mem_used_pct": round(gpu_before.mem_used_pct, 2)},
                         "after": {"mem_used_pct": round(gpu_after.mem_used_pct, 2)}},
                    compression={"chars_before": cstats.chars_before,
                                 "chars_after": cstats.chars_after,
                                 "tools_compressed": cstats.tool_results_compressed,
                                 "thinking_removed": cstats.thinking_blocks_removed,
                                 "naive_input_tokens": naive_tokens,
                                 "compressed_input_tokens": comp_tokens},
                    lifecycle={"turn": turn_rec.turn, "total_tokens": turn_rec.total_tokens,
                               "stable_prefix_tokens": turn_rec.stable_prefix_tokens,
                               "prefix_fingerprint": turn_rec.prefix_fingerprint},
                    success=True,
                )
                print(f"  [xirang-mock] {session_id} t{turn_idx}: naive {naive_tokens}tok -> "
                      f"comp {comp_tokens}tok (Δ{naive_tokens - comp_tokens})")
                continue
            resp = await client.post(endpoint, json=payload, timeout=120.0,
                                     headers={"x-xirang-session": session_id})
            if resp.status_code == 200:
                usage = resp.json().get("usage", {})
                out_tokens = usage.get("completion_tokens", 0)
            else:
                success = False
                err = f"status {resp.status_code}: {resp.text[:200]}"
        except Exception as e:  # noqa: BLE001
            success = False
            err = str(e)
        e2e_ms = (time.time() - t0) * 1000
        gpu_after = gpu.snapshot()
        logger.log_request(
            run="xirang",
            mode="xirang",
            session_id=session_id,
            turn=turn_idx,
            estimated_input_tokens=naive_tokens,  # proxy 已记录压缩后值，这里记录请求侧
            output_tokens=out_tokens,
            latency={"ttft_ms": None, "e2e_ms": round(e2e_ms, 2), "throughput_tok_s": None},
            gpu={"before": {"mem_used_pct": round(gpu_before.mem_used_pct, 2)},
                 "after": {"mem_used_pct": round(gpu_after.mem_used_pct, 2)}},
            compression={"naive_input_tokens": naive_tokens},
            lifecycle={"turn": turn_rec.turn, "total_tokens": turn_rec.total_tokens,
                       "stable_prefix_tokens": turn_rec.stable_prefix_tokens,
                       "prefix_fingerprint": turn_rec.prefix_fingerprint},
            success=success,
            error=err,
        )
        print(f"  [xirang] {session_id} t{turn_idx}: in~{naive_tokens}tok "
              f"e2e={e2e_ms:.0f}ms gpu={gpu_after.mem_used_pct:.1f}% "
              f"{'OK' if success else 'ERR:'+str(err)}")


async def main_async(args: argparse.Namespace) -> None:
    tasks = load_workload(args.workload)
    logger = MetricsLogger(args.run, settings.metrics_dir)
    gpu = GPUMonitor(settings.gpu_device)
    budget = BudgetController(settings.context_token_budget, settings.chars_per_token)
    lifecycle = LifecycleManager()
    endpoint = args.endpoint
    print(f"[xirang] {len(tasks)} tasks, {args.model}, endpoint={endpoint}, mock={args.mock}")
    async with httpx.AsyncClient() as client:
        for task in tasks:
            await run_one(client, task, args.model, logger, gpu, budget, lifecycle, endpoint, args.mock)
    gpu.close()
    print(f"[xirang] done. metrics -> {logger.path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--run", default="xirang_run")
    ap.add_argument("--endpoint", default=f"http://{settings.host}:{settings.port}/v1/chat/completions")
    ap.add_argument("--mock", action="store_true", help="不连真实后端，仅演示压缩效果")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
