import numpy as np

from rl_riskaware.baselines import build_twap_schedule, twap_participation_action


def test_build_twap_schedule_sums_to_order_size() -> None:
    sched = build_twap_schedule(order_size=100.0, horizon_steps=4)
    assert sched.fills.shape == (4,)
    assert float(np.sum(sched.fills)) == 100.0


def test_twap_participation_action_bounds() -> None:
    a = twap_participation_action(target_fill=10.0, volume=100.0, participation_cap=0.1)
    assert 0.0 <= a <= 1.0
    assert twap_participation_action(target_fill=10.0, volume=0.0, participation_cap=0.1) == 0.0

