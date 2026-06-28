"""上下文压缩。

v0 的压缩策略是确定性的、不依赖模型的：
- 工具结果 / 超长 assistant 内容：保留头尾，中间用占位符替换 (head-tail summarization)；
- thinking 块 (`<think>...</think>`)：删除；
- system prompt：去重连续重复行。

这些是"无损或近无损"的体积优化，目的是先证明 KV Cache 占用与显存能降下来，
任务成功率基本不变。后续可替换为模型摘要式压缩。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .segment import SegmentKind

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r".*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class CompressionStats:
    chars_before: int = 0
    chars_after: int = 0
    tool_results_compressed: int = 0
    thinking_blocks_removed: int = 0
    system_lines_deduped: int = 0


def strip_thinking(text: str) -> tuple[str, int]:
    """删除 <think>...</think> 块。返回 (新文本, 删除块数)。

    处理未闭合的块：仅出现 <think> 开头时删除其后全部内容（流式残留）。
    """
    if not text:
        return text, 0
    blocks = _THINK_RE.findall(text)
    cleaned = _THINK_RE.sub("", text)
    # 残留的未闭合 think
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    removed = len(blocks) + (1 if _THINK_OPEN_RE.search(text) and not blocks else 0)
    return cleaned.strip(), removed


def compress_tool_result(
    text: str, keep_head: int = 300, keep_tail: int = 300
) -> str:
    """对超长工具结果做 head-tail 截断，中间用占位符。"""
    if not text or len(text) <= keep_head + keep_tail:
        return text
    head = text[:keep_head]
    tail = text[-keep_tail:]
    omitted = len(text) - keep_head - keep_tail
    return f"{head}\n...[XiRang: omitted {omitted} chars of tool output]...\n{tail}"


def dedup_system_prompt(text: str) -> tuple[str, int]:
    """删除 system prompt 中连续重复的行（常见于拼接的工具描述）。"""
    if not text:
        return text, 0
    lines = text.splitlines()
    out: list[str] = []
    removed = 0
    for line in lines:
        if out and out[-1] == line and line.strip():
            removed += 1
            continue
        out.append(line)
    return "\n".join(out), removed


def compress_message(
    message: dict[str, Any],
    stats: CompressionStats,
    *,
    compress_tool: bool = True,
    strip_think: bool = True,
    dedup_sys: bool = True,
    tool_keep_head: int = 300,
    tool_keep_tail: int = 300,
    tool_threshold: int = 800,
) -> dict[str, Any]:
    """就地压缩单条 message 的文本 content。返回新的 message dict。"""
    new_msg = dict(message)
    content = new_msg.get("content")
    if not isinstance(content, str):
        return new_msg  # 多模态/工具调用结构 v0 不动

    stats.chars_before += len(content)
    role = new_msg.get("role", "user")

    if strip_think and role == "assistant":
        content, n = strip_thinking(content)
        stats.thinking_blocks_removed += n

    if compress_tool and role == "tool" and len(content) > tool_threshold:
        content = compress_tool_result(content, tool_keep_head, tool_keep_tail)
        stats.tool_results_compressed += 1

    if dedup_sys and role == "system":
        content, n = dedup_system_prompt(content)
        stats.system_lines_deduped += n

    stats.chars_after += len(content)
    new_msg["content"] = content
    return new_msg


def compress_messages(
    messages: list[dict[str, Any]],
    *,
    compress_tool: bool = True,
    strip_think: bool = True,
    dedup_sys: bool = True,
    tool_keep_head: int = 300,
    tool_keep_tail: int = 300,
    tool_threshold: int = 800,
) -> tuple[list[dict[str, Any]], CompressionStats]:
    stats = CompressionStats()
    out = [
        compress_message(
            m,
            stats,
            compress_tool=compress_tool,
            strip_think=strip_think,
            dedup_sys=dedup_sys,
            tool_keep_head=tool_keep_head,
            tool_keep_tail=tool_keep_tail,
            tool_threshold=tool_threshold,
        )
        for m in messages
    ]
    return out, stats
