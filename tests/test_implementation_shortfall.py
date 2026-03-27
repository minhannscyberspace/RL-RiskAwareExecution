import numpy as np

from rl_riskaware.evaluation import completion_rate, implementation_shortfall, slippage_bps, volume_weighted_avg_exec_price


def test_implementation_shortfall_basic() -> None:
    arrival = 100.0
    exec_prices = np.asarray([100.0, 101.0, 99.0], dtype=np.float64)
    qty = np.asarray([10.0, 5.0, 5.0], dtype=np.float64)
    # (0*10) + (1*5) + (-1*5) = 0
    assert implementation_shortfall(arrival, exec_prices, qty) == 0.0


def test_completion_and_slippage_helpers() -> None:
    arrival = 100.0
    exec_prices = np.asarray([101.0, 99.0], dtype=np.float64)
    qty = np.asarray([5.0, 5.0], dtype=np.float64)

    assert completion_rate(10.0, qty) == 1.0
    assert volume_weighted_avg_exec_price(exec_prices, qty) == 100.0
    assert slippage_bps(arrival, exec_prices, qty) == 0.0

