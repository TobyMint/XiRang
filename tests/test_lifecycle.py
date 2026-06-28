"""测试 lifecycle：会话级 KV 增长记录与稳定前缀指纹。"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xirang.memory.lifecycle import LifecycleManager  # noqa: E402


def test_turns_recorded_and_tokens_grow():
    mgr = LifecycleManager()
    msgs_t0 = [{"role": "system", "content": "you are an agent"},
               {"role": "user", "content": "hi"}]
    msgs_t1 = [{"role": "system", "content": "you are an agent"},
               {"role": "user", "content": "hi"},
               {"role": "assistant", "content": "hello"},
               {"role": "user", "content": "do step 2"}]
    r0 = mgr.observe("s1", msgs_t0)
    r1 = mgr.observe("s1", msgs_t1)
    assert r1.total_tokens > r0.total_tokens, "上下文应随轮次增长"
    assert r0.turn == 0 and r1.turn == 1


def test_stable_prefix_fingerprint_stable():
    mgr = LifecycleManager()
    sys_msg = [{"role": "system", "content": "fixed system prompt"}]
    # 两条不同对话，system 相同 -> 稳定前缀指纹应一致
    a = mgr.observe("sA", sys_msg + [{"role": "user", "content": "q1"}])
    b = mgr.observe("sB", sys_msg + [{"role": "user", "content": "q2"}])
    assert a.prefix_fingerprint == b.prefix_fingerprint
    assert a.stable_prefix_tokens == b.stable_prefix_tokens


def test_prefix_cache_hit_counted():
    mgr = LifecycleManager()
    sys_msg = [{"role": "system", "content": "fixed"}]
    for _ in range(3):
        mgr.observe("s", sys_msg + [{"role": "user", "content": "x"}])
    sess = mgr.get("s")
    # 同一 session 三轮，同一前缀指纹命中 3 次 -> 复用轮次 = 2
    assert sess.prefix_cache_hit_turns == 2
