"""上下文分段 (segment)。

把一条 chat/completions 请求里的 messages 拆成有语义边界的段，便于：
- 识别可复用的稳定前缀 (system prompt / 固定工具描述)；
- 区分易变段 (多轮对话 / 工具结果)；
- 估算每段 token 占用，喂给 budget 做淘汰决策。

v0 只做基于 role 与内容的轻量分段，不依赖分词器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SegmentKind(str, Enum):
    SYSTEM = "system"          # 稳定前缀：system prompt / 工具描述
    USER = "user"              # 用户输入
    ASSISTANT = "assistant"    # 模型历史回复（可能含 thinking）
    TOOL = "tool"              # 工具调用结果（最易膨胀的段）


@dataclass
class Segment:
    kind: SegmentKind
    content: str
    # 估算 token 数
    tokens: int = 0
    # 该段是否被视为可复用的稳定前缀
    stable: bool = False
    # 原始 message 的引用，便于回写
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:  # 便于按字符处理
        return len(self.content)


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """粗略 token 估算：中文按 ~1.5 字/token，英文按 ~4 字符/token。

    这里用统一启发式，足以支撑 budget 决策；正式 benchmark 时可用真实分词器覆盖。
    """
    if not text:
        return 0
    # 简单区分：含较多非 ASCII 视为中文为主
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii > len(text) * 0.3:
        return max(1, int(len(text) / 1.5))
    return max(1, int(len(text) / chars_per_token))


def message_to_segment(message: dict[str, Any], chars_per_token: float = 4.0) -> Segment:
    role = message.get("role", "user")
    content = message.get("content") or ""
    if isinstance(content, list):  # 多模态/工具调用块，取文本拼起来
        content = " ".join(
            blk.get("text", "") for blk in content if isinstance(blk, dict)
        )
    kind = {
        "system": SegmentKind.SYSTEM,
        "assistant": SegmentKind.ASSISTANT,
        "tool": SegmentKind.TOOL,
    }.get(role, SegmentKind.USER)
    stable = kind == SegmentKind.SYSTEM
    return Segment(
        kind=kind,
        content=content,
        tokens=estimate_tokens(content, chars_per_token),
        stable=stable,
        meta={"role": role},
    )


def segment_messages(messages: list[dict[str, Any]], chars_per_token: float = 4.0) -> list[Segment]:
    return [message_to_segment(m, chars_per_token) for m in messages]


def total_tokens(segments: list[Segment]) -> int:
    return sum(s.tokens for s in segments)


def stable_prefix_tokens(segments: list[Segment]) -> int:
    """从开头起连续 stable 段的 token 数。"""
    n = 0
    for s in segments:
        if not s.stable:
            break
        n += s.tokens
    return n
