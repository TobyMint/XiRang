"""工具调用结果外部存储。

把超大的工具结果（如 shell 输出、文件内容、检索结果）落到磁盘，
在上下文里只留一个引用句柄 `[XiRang-Tool: <id>]`，避免整段进入 KV Cache。
lazy_loader 可在 Agent 需要时按 id 取回完整内容。

这是赛题"工具调用数据优化：结构化存储或按需加载"的直接实现。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolRecord:
    tool_id: str
    tool_name: str
    content: str
    created_at: float
    # 上下文中保留的引用占位符
    placeholder: str


class ToolStore:
    def __init__(self, store_dir: str) -> None:
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self._index: dict[str, ToolRecord] = {}

    def _path(self, tool_id: str) -> str:
        return os.path.join(self.store_dir, f"{tool_id}.json")

    def put(self, tool_name: str, content: str) -> ToolRecord:
        """存储一条工具结果，返回占位符引用。"""
        ts = time.time()
        digest = hashlib.sha1(
            f"{tool_name}:{ts}:{len(content)}".encode("utf-8")
        ).hexdigest()[:12]
        tool_id = f"{tool_name}_{digest}"
        placeholder = f"[XiRang-Tool:{tool_id}]"
        rec = ToolRecord(
            tool_id=tool_id,
            tool_name=tool_name,
            content=content,
            created_at=ts,
            placeholder=placeholder,
        )
        self._index[tool_id] = rec
        with open(self._path(tool_id), "w", encoding="utf-8") as f:
            json.dump(
                {"tool_id": tool_id, "tool_name": tool_name, "content": content, "created_at": ts},
                f,
                ensure_ascii=False,
            )
        return rec

    def get(self, tool_id: str) -> ToolRecord | None:
        if tool_id in self._index:
            return self._index[tool_id]
        path = self._path(tool_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        rec = ToolRecord(
            tool_id=data["tool_id"],
            tool_name=data["tool_name"],
            content=data["content"],
            created_at=data["created_at"],
            placeholder=f"[XiRang-Tool:{data['tool_id']}]",
        )
        self._index[tool_id] = rec
        return rec

    def resolve(self, text: str) -> str:
        """把文本里的占位符还原为完整工具结果（用于真正需要完整内容时）。"""
        import re

        def _repl(m: re.Match) -> str:
            rec = self.get(m.group(1))
            return rec.content if rec else m.group(0)

        return re.sub(r"\[XiRang-Tool:([^\]]+)\]", _repl, text)

    def stats(self) -> dict[str, int]:
        return {"stored": len(self._index)}
