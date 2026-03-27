from __future__ import annotations

from rl_riskaware.evaluation.walkforward import Window, make_walkforward_windows


def make_walkforward_windows_from_rows(n_rows: int, train_len: int, test_len: int, step: int) -> list[Window]:
    return make_walkforward_windows(total_len=n_rows, train_len=train_len, test_len=test_len, step=step)
