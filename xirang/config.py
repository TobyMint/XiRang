"""集中配置。所有可调参数放这里，避免散落各处。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # ---- proxy ----
    host: str = os.environ.get("XIRANG_HOST", "0.0.0.0")
    port: int = _env_int("XIRANG_PORT", 8000)
    # vLLM OpenAI-compatible 后端地址
    backend_url: str = os.environ.get("XIRANG_BACKEND", "http://127.0.0.1:8001")

    # ---- 内存优化开关 ----
    enabled: bool = os.environ.get("XIRANG_ENABLED", "1") != "0"
    compress_tool_results: bool = os.environ.get("XIRANG_COMPRESS_TOOL", "1") != "0"
    strip_thinking: bool = os.environ.get("XIRANG_STRIP_THINKING", "1") != "0"
    dedup_system_prompt: bool = os.environ.get("XIRANG_DEDUP_SYS", "1") != "0"

    # ---- 压缩阈值 ----
    # 工具结果超过这么多字符才压缩
    tool_compress_chars: int = _env_int("XIRANG_TOOL_COMPRESS_CHARS", 800)
    # 压缩后保留的头/尾字符数
    tool_keep_head: int = _env_int("XIRANG_TOOL_KEEP_HEAD", 300)
    tool_keep_tail: int = _env_int("XIRANG_TOOL_KEEP_TAIL", 300)

    # ---- token 估算 ----
    # 粗略估算：1 token ≈ chars_per_token 个字符
    chars_per_token: float = _env_float("XIRANG_CHARS_PER_TOKEN", 4.0)

    # ---- 显存预算 ----
    # 上下文 token 软上限，超过则触发更激进的压缩/淘汰
    context_token_budget: int = _env_int("XIRANG_CONTEXT_BUDGET", 6000)

    # ---- GPU 监控 ----
    gpu_device: int = _env_int("XIRANG_GPU_DEVICE", 0)
    gpu_sample_interval: float = _env_float("XIRANG_GPU_SAMPLE_INTERVAL", 0.2)

    # ---- 日志/指标 ----
    metrics_dir: str = os.environ.get("XIRANG_METRICS_DIR", "runs")

    # ---- 工具结果外部存储 ----
    tool_store_dir: str = os.environ.get("XIRANG_TOOL_STORE_DIR", ".xirang_tool_store")


settings = Settings()
