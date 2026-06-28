"""KV Cache 生命周期管理 (v0)。

v0 不魔改 vLLM，无法真正控制 vLLM 内部的 block allocator。这里的 lifecycle 负责：
- 在 proxy 侧按会话记录"稳定前缀"指纹与 token 数，量化"可被 prefix caching 复用的部分"；
- 追踪每个会话上下文随轮次的增长曲线（用于画图证明 naive 持续上涨）；
- 在 budget 触发时，给出"建议淘汰的段"列表（v0 仅记录建议，真正淘汰靠 compression）。

后续接入 vLLM 内部后，这里会演化成真正的 KV block 复用/淘汰/分层控制器。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .segment import Segment, segment_messages, stable_prefix_tokens, total_tokens


def _fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class TurnRecord:
    turn: int
    total_tokens: int
    stable_prefix_tokens: int
    prefix_fingerprint: str


@dataclass
class SessionState:
    session_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    # 已见过的稳定前缀指纹 -> 命中次数（即 vLLM prefix caching 可复用次数）
    prefix_hits: dict[str, int] = field(default_factory=dict)

    def record_turn(self, segments: list[Segment]) -> TurnRecord:
        stable = stable_prefix_tokens(segments)
        total = total_tokens(segments)
        prefix_text = "".join(s.content for s in segments if s.stable)
        fp = _fingerprint(prefix_text)
        rec = TurnRecord(
            turn=len(self.turns),
            total_tokens=total,
            stable_prefix_tokens=stable,
            prefix_fingerprint=fp,
        )
        self.turns.append(rec)
        if fp:
            self.prefix_hits[fp] = self.prefix_hits.get(fp, 0) + 1
        return rec

    @property
    def prefix_cache_hit_turns(self) -> int:
        """稳定前缀未变的轮次数 = 可被复用的轮次。"""
        return sum(v - 1 for v in self.prefix_hits.values() if v > 1)


class LifecycleManager:
    """按 session_id 维护会话级 KV 生命周期状态。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def observe(
        self, session_id: str, messages: list[dict[str, Any]], chars_per_token: float = 4.0
    ) -> TurnRecord:
        segs = segment_messages(messages, chars_per_token)
        return self.get(session_id).record_turn(segs)

    def all_sessions(self) -> dict[str, SessionState]:
        return self._sessions
