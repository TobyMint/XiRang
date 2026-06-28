"""测试 request_rewriter：thinking 删除、工具结果压缩、system 去重。"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xirang.memory.compression import compress_messages, strip_thinking, compress_tool_result, dedup_system_prompt  # noqa: E402


def test_strip_thinking_removes_block():
    text = "before <think>secret reasoning here</think> after"
    out, n = strip_thinking(text)
    assert n == 1
    assert "secret reasoning" not in out
    assert "before" in out and "after" in out


def test_strip_thinking_unclosed():
    text = "answer <think>streaming tail"
    out, n = strip_thinking(text)
    assert n >= 1
    assert "streaming tail" not in out


def test_compress_tool_result_keeps_head_tail():
    big = "HEAD" + "x" * 2000 + "TAIL"
    out = compress_tool_result(big, keep_head=10, keep_tail=10)
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "omitted" in out
    assert len(out) < len(big)


def test_compress_tool_result_short_passthrough():
    short = "small"
    assert compress_tool_result(short) == short


def test_dedup_system_prompt():
    text = "line1\nline1\nline2\nline2\nline2\nline3"
    out, n = dedup_system_prompt(text)
    assert n == 3
    assert out.count("line1") == 1
    assert out.count("line2") == 1


def test_compress_messages_full_pipeline():
    messages = [
        {"role": "system", "content": "dup\ndup\nrule"},
        {"role": "assistant", "content": "<think>r</think>visible"},
        {"role": "tool", "name": "shell", "content": "H" + "y" * 2000 + "T"},
    ]
    out, stats = compress_messages(messages, tool_threshold=500, tool_keep_head=10, tool_keep_tail=10)
    assert stats.thinking_blocks_removed == 1
    assert stats.tool_results_compressed == 1
    assert stats.system_lines_deduped == 1
    assert "<think>" not in out[1]["content"]
    assert "visible" in out[1]["content"]
    assert out[2]["content"].startswith("H")
    assert out[2]["content"].endswith("T")
    assert len(out[2]["content"]) < 2000
