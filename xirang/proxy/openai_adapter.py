"""OpenAI 兼容请求转发适配。

负责把 proxy 收到的请求转发给 vLLM 后端，透传 stream / non-stream 两种模式。
只关心 HTTP 透传与 SSE 透传，不关心内存优化（那是 request_rewriter 的活）。
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx


class OpenAIAdapter:
    def __init__(self, backend_url: str, timeout: float = 120.0) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.timeout = timeout

    @property
    def chat_url(self) -> str:
        return f"{self.backend_url}/v1/chat/completions"

    @property
    def completions_url(self) -> str:
        return f"{self.backend_url}/v1/completions"

    async def forward(
        self, client: httpx.AsyncClient, path: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> tuple[httpx.Response, bool]:
        """转发非流式请求，返回 (response, is_stream)。"""
        is_stream = bool(payload.get("stream", False))
        url = f"{self.backend_url}{path}"
        resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
        return resp, is_stream

    async def forward_stream(
        self, client: httpx.AsyncClient, path: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[bytes]:
        """转发流式请求，逐块产出原始 SSE 字节。"""
        url = f"{self.backend_url}{path}"
        async with client.stream("POST", url, json=payload, headers=headers, timeout=self.timeout) as resp:
            async for chunk in resp.aiter_raw():
                yield chunk


def count_output_tokens_from_stream(chunk_bytes: bytes) -> int:
    """从一块 SSE 字节里粗略数 output token（按出现次数）。

    用于吞吐统计；非精确。vLLM 的 chunk 里含 `delta.content`，
    我们按字符 / chars_per_token 估算，或统计 data 行数。
    """
    if not chunk_bytes:
        return 0
    text = chunk_bytes.decode("utf-8", errors="ignore")
    n = 0
    for line in text.splitlines():
        if line.startswith("data:") and "[DONE]" not in line:
            n += 1
    return n


def parse_completion_usage(resp_json: dict[str, Any]) -> dict[str, int]:
    usage = resp_json.get("usage", {}) or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
