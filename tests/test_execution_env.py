import numpy as np

from rl_riskaware.env import ExecutionEnv, ExecutionEnvConfig


def test_execution_env_inventory_decreases_and_terminates() -> None:
    horizon = 5
    order_size = 100.0
    prices = np.full((horizon,), 100.0, dtype=np.float64)
    volumes = np.full((horizon,), 1000.0, dtype=np.float64)

    cfg = ExecutionEnvConfig(
        horizon_steps=horizon,
        order_size=order_size,
        participation_cap=0.1,
        impact_eta=0.0,
        fixed_fee=0.0,
        terminal_penalty=10.0,
    )
    env = ExecutionEnv(cfg=cfg, volumes=volumes, prices=prices)
    obs, _ = env.reset(seed=123)
    assert obs.shape == (3,)

    remaining = order_size
    for _ in range(horizon):
        out = env.step(1.0)  # max participation
        assert 0.0 <= float(out.info["fill"]) <= remaining
        remaining = float(out.info["remaining"])
        if out.terminated:
            break

    assert remaining <= order_size
    assert out.terminated

