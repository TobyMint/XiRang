"""结构化指标日志。

把每次请求的：会话、轮次、token 估算、压缩统计、GPU 快照、延迟，
以 JSONL 形式追加写入 runs/<run_name>/metrics.jsonl，供 plot_results 出图。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


class MetricsLogger:
    def __init__(self, run_name: str, metrics_dir: str = "runs") -> None:
        self.run_name = run_name
        self.dir = os.path.join(metrics_dir, run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "metrics.jsonl")

    def log(self, record: dict[str, Any]) -> None:
        record = {"_ts": time.time(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_request(
        self,
        *,
        run: str,
        session_id: str,
        turn: int,
        mode: str,  # "baseline" | "xirang"
        estimated_input_tokens: int,
        output_tokens: int,
        latency: dict[str, Any],
        gpu: dict[str, Any],
        compression: dict[str, Any] | None = None,
        lifecycle: dict[str, Any] | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        self.log(
            {
                "run": run,
                "mode": mode,
                "session_id": session_id,
                "turn": turn,
                "estimated_input_tokens": estimated_input_tokens,
                "output_tokens": output_tokens,
                "latency": latency,
                "gpu": gpu,
                "compression": compression or {},
                "lifecycle": lifecycle or {},
                "success": success,
                "error": error,
            }
        )

    def read_all(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
