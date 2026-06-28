"""vLLM 抓包 tap：透明记录 Agent ↔ vLLM 的所有 request/response。

部署在 OpenClaw 和 vLLM 之间：
    OpenClaw --(8002)--> tap --(8001)--> vLLM

记录每条请求的完整 body (messages/tools 等) 与响应 (含流式 SSE 重组)，
用于分析 Agent 如何组织 prompt、是否破坏 vLLM prefix caching。

运行:
    python diag/tap.py                          # 0.0.0.0:8002 -> 127.0.0.1:8001
    TAP_PORT=8002 VLLM_UPSTREAM=127.0.0.1:8001 python diag/tap.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

TAP_PORT = int(os.environ.get("TAP_PORT", "8002"))
VLLM_UPSTREAM = os.environ.get("VLLM_UPSTREAM", "http://127.0.0.1:8001")
LOG_DIR = os.environ.get("TAP_LOG_DIR", "diag/captures")
# 响应体日志截断（请求体不截断，分析需要完整 prompt）
RESP_TRUNC = int(os.environ.get("TAP_RESP_TRUNC", "4000"))

os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"capture-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")

app = FastAPI(title="XiRang vLLM Tap")
_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))


def _write(record: dict[str, Any]) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.on_event("shutdown")
async def _shutdown() -> None:
    await _client.aclose()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request):
    rid = uuid.uuid4().hex[:12]
    t0 = time.time()
    body = await request.body()
    url = f"{VLLM_UPSTREAM}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    # 解析请求体
    req_json: Any = None
    try:
        req_json = json.loads(body) if body else None
    except Exception:
        req_json = body.decode("utf-8", errors="replace")[:2000]

    fwd_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length", "transfer-encoding")}

    is_stream = isinstance(req_json, dict) and req_json.get("stream", False)

    if is_stream:
        return await _stream(rid, t0, url, body, fwd_headers, req_json)
    return await _nonstream(rid, t0, url, body, fwd_headers, req_json)


async def _nonstream(rid, t0, url, body, headers, req_json):
    resp = await _client.post(url, content=body, headers=headers) \
        if body else await _client.get(url, headers=headers)
    dt = (time.time() - t0) * 1000
    resp_text = resp.text
    try:
        resp_json = resp.json()
    except Exception:
        resp_json = None
    usage = (resp_json or {}).get("usage", {}) if isinstance(resp_json, dict) else {}
    _write({
        "id": rid, "ts": t0, "method": "POST", "path": url.split("/")[-1] or "/",
        "stream": False, "latency_ms": round(dt, 1), "status": resp.status_code,
        "request": req_json,
        "response": (resp_text[:RESP_TRUNC] if not resp_json else resp_json),
        "usage": usage,
    })
    return JSONResponse(content=resp.json() if resp_json else resp_text,
                        status_code=resp.status_code)


async def _stream(rid, t0, url, body, headers, req_json):
    async def gen():
        assembled = []          # 重组的 delta 文本
        first_token_ts = None
        out_tokens = 0
        status = 200
        usage = {}
        try:
            async with _client.stream("POST", url, content=body, headers=headers) as resp:
                status = resp.status_code
                async for chunk in resp.aiter_raw():
                    if first_token_ts is None:
                        first_token_ts = time.time()
                    text = chunk.decode("utf-8", errors="replace")
                    # 粗略提取 delta 文本与 usage
                    for line in text.splitlines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            continue
                        try:
                            obj = json.loads(data)
                        except Exception:
                            continue
                        for ch in obj.get("choices", []):
                            d = ch.get("delta", {}).get("content")
                            if d:
                                assembled.append(d)
                        if obj.get("usage"):
                            usage = obj["usage"]
                    yield chunk
        finally:
            dt = (time.time() - t0) * 1000
            ttft = ((first_token_ts - t0) * 1000) if first_token_ts else None
            out_tokens = usage.get("completion_tokens", len("".join(assembled)) // 3)
            _write({
                "id": rid, "ts": t0, "method": "POST", "path": "chat/completions",
                "stream": True, "latency_ms": round(dt, 1), "ttft_ms": round(ttft, 1) if ttft else None,
                "status": status, "request": req_json,
                "response_text": "".join(assembled)[:RESP_TRUNC],
                "usage": usage, "output_tokens": out_tokens,
            })

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print(f"[tap] {TAP_PORT} -> {VLLM_UPSTREAM}  log={LOG_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=TAP_PORT, log_level="warning")
