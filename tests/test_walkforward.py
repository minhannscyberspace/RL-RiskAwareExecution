import numpy as np

from rl_riskaware.evaluation import make_walkforward_windows, run_window_benchmarks


def test_make_walkforward_windows_basic() -> None:
    windows = make_walkforward_windows(total_len=100, train_len=40, test_len=20, step=20)
    # windows: [0:40]->[40:60], [20:60]->[60:80], [40:80]->[80:100]
    assert len(windows) == 3
    assert windows[0].train_start == 0
    assert windows[0].test_end == 60
    assert windows[-1].test_end == 100


def test_run_window_benchmarks_returns_policies_per_window() -> None:
    total_len = 120
    prices = np.linspace(100.0, 101.0, total_len, dtype=np.float64)
    volumes = np.full((total_len,), 1_000_000.0, dtype=np.float64)
    windows = make_walkforward_windows(total_len=total_len, train_len=40, test_len=20, step=20)

    results = run_window_benchmarks(
        prices=prices,
        volumes=volumes,
        order_size=10_000.0,
        participation_cap=0.01,
        impact_eta=0.01,
        fixed_fee=0.0,
        terminal_penalty=0.1,
        windows=windows,
    )
    # 3 policies per window
    assert len(results) == len(windows) * 3
    assert {r.policy for r in results} == {"TWAP", "VWAP", "POV(70%)"}
