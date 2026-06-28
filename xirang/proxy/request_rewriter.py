"""请求重写器。

把进入的 OpenAI chat/completions 请求做内存优化后转发：
1. lifecycle 观察上下文（量化 KV 增长 / 前缀复用）；
2. budget 估算并决策；
3. compression 压缩（工具结果 head-tail、thinking 删除、system 去重）；
4. 超预算时把最老的 tool 结果外部化到 ToolStore（占位符替换）。

返回重写后的 messages 与统计信息，供 server 记录指标。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..memory.budget import BudgetController
from ..memory.compression import CompressionStats, compress_messages
from ..memory.lifecycle import LifecycleManager
from ..tools.tool_store import ToolStore

_PLACEHOLDER_RE = re.compile(r"\[XiRang-Tool:([^\]]+)\]")


@dataclass
class RewriteResult:
    messages: list[dict[str, Any]]
    stats: CompressionStats
    budget_total_tokens: int
    budget_over: bool
    externalized_tools: int = 0
    lifecycle: dict[str, Any] = field(default_factory=dict)


class RequestRewriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.budget = BudgetController(
            token_budget=settings.context_token_budget,
            chars_per_token=settings.chars_per_token,
        )
        self.lifecycle = LifecycleManager()
        self.tool_store = ToolStore(settings.tool_store_dir)

    def rewrite(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
    ) -> RewriteResult:
        settings = self.settings

        # 0) lifecycle 观察（在压缩前记录 naive 上下文规模）
        turn_rec = self.lifecycle.observe(session_id, messages, settings.chars_per_token)

        if not settings.enabled:
            return RewriteResult(
                messages=messages,
                stats=CompressionStats(),
                budget_total_tokens=self.budget.estimate(messages),
                budget_over=False,
                lifecycle={"turn": turn_rec.turn, "total_tokens": turn_rec.total_tokens,
                           "stable_prefix_tokens": turn_rec.stable_prefix_tokens,
                           "prefix_fingerprint": turn_rec.prefix_fingerprint},
            )

        # 1) 常规压缩
        messages, stats = compress_messages(
            messages,
            compress_tool=settings.compress_tool_results,
            strip_think=settings.strip_thinking,
            dedup_sys=settings.dedup_system_prompt,
            tool_keep_head=settings.tool_keep_head,
            tool_keep_tail=settings.tool_keep_tail,
            tool_threshold=settings.tool_compress_chars,
        )

        # 2) budget 决策：超预算则把最老的 tool 结果外部化
        decision = self.budget.decide(messages)
        externalized = 0
        if decision.over_budget and decision.drop_indices:
            messages = self._externalize_tools(messages, decision.drop_indices)
            externalized = len(decision.drop_indices)

        return RewriteResult(
            messages=messages,
            stats=stats,
            budget_total_tokens=decision.total_tokens,
            budget_over=decision.over_budget,
            externalized_tools=externalized,
            lifecycle={"turn": turn_rec.turn, "total_tokens": turn_rec.total_tokens,
                       "stable_prefix_tokens": turn_rec.stable_prefix_tokens,
                       "prefix_fingerprint": turn_rec.prefix_fingerprint},
        )

    def _externalize_tools(self, messages: list[dict], indices: list[int]) -> list[dict]:
        """把指定 tool message 的内容存到 ToolStore，上下文留占位符。"""
        out = [dict(m) for m in messages]
        for i in indices:
            m = out[i]
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            if not isinstance(content, str) or not content:
                continue
            # 已有占位符则跳过
            if _PLACEHOLDER_RE.search(content):
                continue
            tool_name = m.get("name") or m.get("tool_call_id") or "tool"
            rec = self.tool_store.put(str(tool_name), content)
            out[i] = {**m, "content": f"[XiRang: tool result externalized] {rec.placeholder}"}
        return out
