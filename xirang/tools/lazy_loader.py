"""按需加载器。

当工具结果已被外部化到 ToolStore、上下文里只剩占位符时，
Agent 若需要回看完整结果，通过 lazy_loader 取回——而不是从一开始就把
全部中间数据塞进 KV Cache。

v0 提供显式 API；后续可对接 Agent 的"回忆"动作自动触发。
"""

from __future__ import annotations

from .tool_store import ToolStore


class LazyLoader:
    def __init__(self, store: ToolStore) -> None:
        self.store = store

    def load(self, tool_id: str) -> str | None:
        rec = self.store.get(tool_id)
        return rec.content if rec else None

    def load_into(self, messages: list[dict], tool_id: str, role: str = "user") -> list[dict]:
        """把某条工具结果作为一条新 message 注入上下文（按需加载）。"""
        content = self.load(tool_id)
        if content is None:
            return messages
        messages = list(messages)
        messages.append({"role": role, "content": f"[XiRang lazy-load {tool_id}]\n{content}"})
        return messages
