import numpy as np

from rl_riskaware.baselines import build_vwap_schedule, pov_action


def test_build_vwap_schedule_proportional_to_volume() -> None:
    order_size = 100.0
    horizon = 4
    volumes = np.asarray([1.0, 1.0, 2.0, 6.0], dtype=np.float64)
    sched = build_vwap_schedule(order_size=order_size, volumes=volumes, horizon_steps=horizon)

    assert sched.fills.shape == (horizon,)
    assert float(np.sum(sched.fills)) == order_size
    # Higher volume step should get higher allocation
    assert sched.fills[3] > sched.fills[2] > sched.fills[0]


def test_build_vwap_schedule_fallback_when_zero_volume() -> None:
    sched = build_vwap_schedule(order_size=100.0, volumes=np.zeros((5,), dtype=np.float64), horizon_steps=5)
    assert float(np.sum(sched.fills)) == 100.0


def test_pov_action_clipped() -> None:
    assert pov_action(-1.0) == 0.0
    assert pov_action(0.5) == 0.5
    assert pov_action(2.0) == 1.0

