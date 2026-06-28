"""XiRang proxy server。

FastAPI 实现的 HTTP 代理，监听 OpenAI 兼容接口，拦截 /v1/chat/completions：
1. 解析请求 messages；
2. 经 RequestRewriter 做内存优化；
3. 转发到 vLLM 后端（流式/非流式）；
4. 记录 GPU / 延迟 / 压缩 / 生命周期指标。

运行：
    uvicorn xirang.proxy.server:app --host 0.0.0.0 --port 8000
或:
    python -m xirang.proxy.server
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import settings
from ..metrics.gpu_monitor import GPUMonitor
from ..metrics.latency import LatencyTracker
from ..metrics.logger import MetricsLogger
from .openai_adapter import OpenAIAdapter, count_output_tokens_from_stream, parse_completion_usage
from .request_rewriter import RequestRewriter

# 全局组件
rewriter: RequestRewriter | None = None
adapter: OpenAIAdapter | None = None
gpu_monitor: GPUMonitor | None = None
latency_tracker: LatencyTracker | None = None
metrics_logger: MetricsLogger | None = None
run_name: str = "default"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rewriter, adapter, gpu_monitor, latency_tracker, metrics_logger
    rewriter = RequestRewriter(settings)
    adapter = OpenAIAdapter(settings.backend_url)
    gpu_monitor = GPUMonitor(settings.gpu_device)
    latency_tracker = LatencyTracker()
    metrics_logger = MetricsLogger(run_name, settings.metrics_dir)
    print(
        f"[XiRang] proxy on {settings.host}:{settings.port} -> backend {settings.backend_url} "
        f"(gpu_monitor={'on' if gpu_monitor.available else 'off (no pynvml/GPU)'})"
    )
    yield
    gpu_monitor.close()


app = FastAPI(title="XiRang Proxy", version="0.1.0", lifespan=lifespan)


def _session_id(req_json: dict[str, Any], request: Request) -> str:
    # 优先用客户端自定义 header，否则按客户端 IP 兜底
    return request.headers.get("x-xirang-session") or req_json.get("user") or "default-session"


def _gpu_dict() -> dict[str, Any]:
    if gpu_monitor is None:
        return {}
    snap = gpu_monitor.snapshot()
    return {
        "mem_used_mib": round(snap.mem_used_mib, 1),
        "mem_total_mib": round(snap.mem_total_mib, 1),
        "mem_used_pct": round(snap.mem_used_pct, 2),
        "gpu_util_pct": round(snap.gpu_util_pct, 1),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": settings.backend_url,
        "gpu_monitor": gpu_monitor.available if gpu_monitor else False,
        "enabled": settings.enabled,
    }


@app.get("/v1/models")
async def list_models(request: Request):
    """透传 models 列表。"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{settings.backend_url}/v1/models", timeout=30.0)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req_json = await request.json()
    messages = req_json.get("messages", [])
    session_id = _session_id(req_json, request)
    request_id = uuid.uuid4().hex[:12]

    # ---- XiRang 重写 ----
    rewrite = rewriter.rewrite(messages, session_id) if settings.enabled else None
    if rewrite is not None:
        req_json["messages"] = rewrite.messages

    is_stream = bool(req_json.get("stream", False))
    headers = {"Content-Type": "application/json"}
    # 透传 Authorization
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth

    rec = latency_tracker.start(request_id)
    gpu_before = _gpu_dict()

    try:
        if is_stream:
            return await _handle_stream(req_json, headers, rec, rewrite, session_id, gpu_before)
        else:
            return await _handle_nonstream(req_json, headers, rec, rewrite, session_id, gpu_before)
    except Exception as e:  # noqa: BLE001
        latency_tracker.finish(rec, 0)
        _log(rec, rewrite, session_id, gpu_before, _gpu_dict(), success=False, error=str(e))
        return JSONResponse(content={"error": str(e)}, status_code=502)


async def _handle_nonstream(req_json, headers, rec, rewrite, session_id, gpu_before):
    async with httpx.AsyncClient() as client:
        resp, _ = await adapter.forward(client, "/v1/chat/completions", req_json, headers)
    latency_tracker.finish(rec, 0)
    rec.t_first_token = rec.t_start  # 非流式无 TTFT 概念，记为 0
    if resp.status_code == 200:
        data = resp.json()
        usage = parse_completion_usage(data)
        rec.output_tokens = usage.get("completion_tokens", 0)
    _log(rec, rewrite, session_id, gpu_before, _gpu_dict())
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def _handle_stream(req_json, headers, rec, rewrite, session_id, gpu_before):
    async def gen():
        out_tokens = 0
        try:
            async with httpx.AsyncClient() as client:
                async for chunk in adapter.forward_stream(
                    client, "/v1/chat/completions", req_json, headers
                ):
                    if rec.t_first_token is None:
                        latency_tracker.mark_first_token(rec)
                    out_tokens += count_output_tokens_from_stream(chunk)
                    yield chunk
        finally:
            latency_tracker.finish(rec, out_tokens)
            _log(rec, rewrite, session_id, gpu_before, _gpu_dict())

    return StreamingResponse(gen(), media_type="text/event-stream")


def _log(rec, rewrite, session_id, gpu_before, gpu_after, success=True, error=None):
    metrics_logger.log_request(
        run=run_name,
        mode="xirang" if settings.enabled else "passthrough",
        session_id=session_id,
        turn=rewrite.lifecycle["turn"] if rewrite else 0,
        estimated_input_tokens=rewrite.budget_total_tokens if rewrite else 0,
        output_tokens=rec.output_tokens,
        latency=rec.to_dict(),
        gpu={"before": gpu_before, "after": gpu_after},
        compression={
            "chars_before": rewrite.stats.chars_before if rewrite else 0,
            "chars_after": rewrite.stats.chars_after if rewrite else 0,
            "tools_compressed": rewrite.stats.tool_results_compressed if rewrite else 0,
            "thinking_removed": rewrite.stats.thinking_blocks_removed if rewrite else 0,
            "externalized": rewrite.externalized_tools if rewrite else 0,
        },
        lifecycle=rewrite.lifecycle if rewrite else {},
        success=success,
        error=error,
    )


def main():
    import uvicorn

    uvicorn.run(
        "xirang.proxy.server:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
