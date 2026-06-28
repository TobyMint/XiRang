"""测试 budget：估算与超预算决策。"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xirang.memory.budget import BudgetController  # noqa: E402


def test_estimate_grows_with_content():
    ctrl = BudgetController(token_budget=1000, chars_per_token=4.0)
    small = ctrl.estimate([{"role": "user", "content": "hi"}])
    big = ctrl.estimate([{"role": "user", "content": "x" * 400}])
    assert big > small


def test_not_over_budget():
    ctrl = BudgetController(token_budget=10000, chars_per_token=4.0)
    dec = ctrl.decide([{"role": "user", "content": "small message"}])
    assert not dec.over_budget
    assert dec.drop_indices == []


def test_over_budget_drops_old_tools():
    ctrl = BudgetController(token_budget=50, chars_per_token=4.0)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "name": "t1", "content": "x" * 400},   # 老
        {"role": "tool", "name": "t2", "content": "y" * 400},   # 老
        {"role": "tool", "name": "t3", "content": "z" * 400},   # 最新，保留
    ]
    dec = ctrl.decide(messages)
    assert dec.over_budget
    # 保留最新一个 tool (index 3)，丢弃前面的
    assert 3 not in dec.drop_indices
    assert set(dec.drop_indices).issubset({1, 2})
