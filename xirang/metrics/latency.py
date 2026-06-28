"""延迟计时工具。

记录端到端延迟、TTFT（首 token 时间）、生成吞吐。
对 streaming 响应，TTFT = 第一个 SSE chunk 到达时间。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class LatencyRecord:
    request_id: str
    t_start: float
    t_first_token: float | None = None
    t_end: float | None = None
    output_tokens: int = 0

    @property
    def ttft_ms(self) -> float | None:
        if self.t_first_token is None:
            return None
        return (self.t_first_token - self.t_start) * 1000

    @property
    def e2e_ms(self) -> float | None:
        if self.t_end is None:
            return None
        return (self.t_end - self.t_start) * 1000

    @property
    def throughput_tok_s(self) -> float | None:
        if self.t_end is None or self.t_first_token is None or self.output_tokens <= 0:
            return None
        dt = self.t_end - self.t_first_token
        return self.output_tokens / dt if dt > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "ttft_ms": self.t_first_token and round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
            "e2e_ms": round(self.e2e_ms, 2) if self.e2e_ms is not None else None,
            "output_tokens": self.output_tokens,
            "throughput_tok_s": round(self.throughput_tok_s, 2) if self.throughput_tok_s is not None else None,
        }


class LatencyTracker:
    def __init__(self) -> None:
        self.records: list[LatencyRecord] = []

    def start(self, request_id: str) -> LatencyRecord:
        rec = LatencyRecord(request_id=request_id, t_start=time.time())
        self.records.append(rec)
        return rec

    def mark_first_token(self, rec: LatencyRecord) -> None:
        if rec.t_first_token is None:
            rec.t_first_token = time.time()

    def finish(self, rec: LatencyRecord, output_tokens: int = 0) -> None:
        rec.t_end = time.time()
        rec.output_tokens = output_tokens

    @contextmanager
    def measure(self, request_id: str):
        rec = self.start(request_id)
        try:
            yield rec
        finally:
            if rec.t_end is None:
                self.finish(rec)
