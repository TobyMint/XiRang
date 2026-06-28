"""分析 tap 抓到的请求，诊断 prefix-cache 友好度。

核心问题：Agent 是否把"易变 metadata"(时间/chat_id/cwd 等) 放在 prompt 前部
(system / tools)，导致 vLLM prefix caching 命中率为 0？

方法：
1. 渲染每条请求的 prompt 文本 (system + tools + messages 按顺序拼接，模拟 vLLM chat template 的前缀结构)
2. 对相邻请求做 diff，定位第一个变化点的"相对位置"(前/中/后)
3. 统计变化是否集中在 system/tools (前缀区)
4. 用 difflib 找出易变片段，归类(时间戳/路径/id/计数器等)

用法: python diag/analyze.py diag/captures/capture-*.jsonl
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def render_prefix(req: dict) -> str:
    """模拟 vLLM chat template 的前缀拼接顺序：system + tools + messages。

    vLLM 默认模板大致为：[tools 描述] + system + user/assistant...，
    system 和 tools 都在前缀区。这里按 OpenAI 请求结构还原文本前缀。
    """
    r = req.get("request", {})
    if not isinstance(r, dict):
        return ""
    parts = []
    # tools 在多数模板里位于 system 之前或之中，都属前缀区
    tools = r.get("tools") or []
    if tools:
        parts.append("[TOOLS]\n" + json.dumps(tools, ensure_ascii=False))
    msgs = r.get("messages") or []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        parts.append(f"[{role}]\n{content}")
    return "\n".join(parts)


def first_diff_pos(a: str, b: str) -> float:
    """返回 a/b 第一个差异字符在 a 中的相对位置 [0,1]。相同返回 1.0。"""
    if a == b:
        return 1.0
    sm = SequenceMatcher(None, a, b, autojunk=False)
    i = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            i = i1
            break
        i = i2
    return i / max(len(a), 1)


def classify_fragment(frag: str) -> str:
    s = frag.strip()
    if not s:
        return "whitespace"
    if re.search(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}:\d{2}(:\d{2})?)\b", s):
        return "timestamp"
    if re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}\b", s) or re.search(r"\b[0-9a-f]{12,}\b", s):
        return "id/hash"
    if re.search(r"/(home|data|tmp|var|usr)/\S+", s) or re.search(r"[A-Za-z]:\\\\", s):
        return "path/cwd"
    if re.fullmatch(r"[\d.,]+\s*(MB|GB|KB|ms|%|bytes)?", s, re.I):
        return "number/metric"
    if re.search(r"\b\d+\b", s):
        return "counter/number"
    return "text"


def extract_changed_fragments(a: str, b: str) -> list[str]:
    """提取 a→b 之间变化的片段（取 b 中新增/变化的短串）。"""
    frags = []
    sm = SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            frags.append(b[j1:j2])
        elif tag == "delete":
            frags.append(a[i1:i2])
    # 切成小片，每片限长
    out = []
    for fr in frags:
        for piece in re.split(r"[\n,;{}\[\]]+", fr):
            p = piece.strip()
            if 1 <= len(p) <= 80:
                out.append(p)
    return out


def analyze(path: str) -> dict:
    recs = load(path)
    # 只看 chat/completions
    chats = [r for r in recs if r.get("path") == "chat/completions"
             and isinstance(r.get("request"), dict)]
    if len(chats) < 2:
        return {"file": path, "n_requests": len(chats),
                "note": "请求不足 2 条，无法做 diff 分析"}

    prefixes = [render_prefix(c) for c in chats]
    diff_positions = []
    changed_frags = []
    same_count = 0
    for i in range(1, len(prefixes)):
        if prefixes[i] == prefixes[i-1]:
            same_count += 1
            diff_positions.append(1.0)
            continue
        pos = first_diff_pos(prefixes[i-1], prefixes[i])
        diff_positions.append(pos)
        changed_frags.extend(extract_changed_fragments(prefixes[i-1], prefixes[i]))

    # 前缀区 = system+tools。估算每条请求前缀区长度（第一个非 system/tool message 之前）
    def prefix_region_len(req):
        r = req["request"]
        n = 0
        tools = r.get("tools") or []
        if tools:
            n += len(json.dumps(tools, ensure_ascii=False)) + 10
        for m in r.get("messages", []):
            if m.get("role") == "system":
                c = m.get("content", "")
                n += len(c if isinstance(c, str) else json.dumps(c))
            else:
                break
        return n

    region_lens = [prefix_region_len(c) for c in chats]
    avg_region = sum(region_lens) / len(region_lens) if region_lens else 0
    avg_prefix_len = sum(len(p) for p in prefixes) / len(prefixes)
    # 前缀区占整 prompt 比例
    region_ratio = avg_region / avg_prefix_len if avg_prefix_len else 0

    # 变化点落在前缀区的比例
    front_changes = sum(1 for p in diff_positions if p <= region_ratio + 0.02)
    front_ratio = front_changes / len(diff_positions) if diff_positions else 0

    frag_classes = Counter(classify_fragment(f) for f in changed_frags)
    top_frags = Counter(changed_frags).most_common(15)

    # system prompt 内容采样
    systems = []
    for c in chats:
        for m in c["request"].get("messages", []):
            if m.get("role") == "system":
                systems.append(m.get("content", ""))
                break

    return {
        "file": path,
        "n_requests": len(chats),
        "identical_consecutive": same_count,
        "avg_prompt_chars": round(avg_prefix_len),
        "avg_prefix_region_chars": round(avg_region),
        "prefix_region_ratio": round(region_ratio, 3),
        "diff_positions": [round(p, 3) for p in diff_positions],
        "front_change_ratio": round(front_ratio, 3),
        "volatile_fragment_classes": dict(frag_classes.most_common()),
        "top_changed_fragments": top_frags,
        "system_prompt_sample": (systems[0][:800] if systems else None),
        "n_with_tools": sum(1 for c in chats if c["request"].get("tools")),
    }


def main():
    paths = sys.argv[1:] or sorted(glob.glob("diag/captures/capture-*.jsonl"))
    if not paths:
        print("no capture files found"); sys.exit(1)
    for p in paths[-1:]:  # 只分析最新一份
        print(f"\n{'='*70}\n分析: {p}\n{'='*70}")
        r = analyze(p)
        for k, v in r.items():
            if k in ("diff_positions", "top_changed_fragments"):
                print(f"  {k}:")
                for x in v[:15] if isinstance(v, list) else v:
                    print(f"      {x}")
            elif k == "system_prompt_sample":
                print(f"  {k}:\n      " + str(v).replace("\n", "\n      ")[:600])
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
