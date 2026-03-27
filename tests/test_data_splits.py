from rl_riskaware.data.splits import make_walkforward_windows_from_rows


def test_make_walkforward_windows_from_rows() -> None:
    windows = make_walkforward_windows_from_rows(n_rows=50, train_len=20, test_len=10, step=10)
    assert len(windows) == 3
    assert windows[0].train_start == 0
    assert windows[2].test_end == 50

