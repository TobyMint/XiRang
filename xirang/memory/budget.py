"""显存/上下文预算管理。

给定上下文 token 估算与软预算，决定是否触发更激进的压缩/淘汰。
v0 的策略：
- 未超预算：仅做常规压缩（工具结果 head-tail、thinking 删除）；
- 超预算：对最老的非稳定 tool 段进一步压缩（更小的 keep 头尾），
  若仍超，则丢弃最老的 tool 结果（替换为引用占位符，由 tool_store 兜底）。

这是"软"预算——保证任务成功率不下降的前提下尽量降占用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .segment import SegmentKind, segment_messages, total_tokens


@dataclass
class BudgetDecision:
    over_budget: bool
    total_tokens: int
    budget: int
    # 建议进一步压缩的 message 索引（最老的 tool 段）
    shrink_indices: list[int] = None  # type: ignore[assignment]
    # 建议丢弃的 message 索引
    drop_indices: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.shrink_indices is None:
            self.shrink_indices = []
        if self.drop_indices is None:
            self.drop_indices = []


class BudgetController:
    def __init__(self, token_budget: int = 6000, chars_per_token: float = 4.0) -> None:
        self.token_budget = token_budget
        self.chars_per_token = chars_per_token

    def estimate(self, messages: list[dict[str, Any]]) -> int:
        segs = segment_messages(messages, self.chars_per_token)
        return total_tokens(segs)

    def decide(self, messages: list[dict[str, Any]]) -> BudgetDecision:
        segs = segment_messages(messages, self.chars_per_token)
        total = total_tokens(segs)
        over = total > self.token_budget
        if not over:
            return BudgetDecision(False, total, self.token_budget)

        # 优先进一步压缩最老的 tool 段
        shrink = [
            i for i, s in enumerate(segs)
            if s.kind == SegmentKind.TOOL
        ]
        drop: list[int] = []
        # 若压缩后仍超预算，丢弃最老的 tool 段（保留最近一个）
        if shrink:
            projected = total
            for i in shrink[:-1]:
                # 粗略假设进一步压缩能砍掉 60% tool token
                seg = segs[i]
                projected -= int(seg.tokens * 0.6)
            if projected > self.token_budget:
                drop = shrink[:-1]  # 保留最新一个 tool 结果
        return BudgetDecision(True, total, self.token_budget, shrink, drop)
