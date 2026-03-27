import numpy as np

from rl_riskaware.agents import ExecutionGymEnv
from rl_riskaware.env import ExecutionEnvConfig


def test_sb3_env_adapter_reset_step_shapes() -> None:
    horizon = 5
    prices = np.linspace(100.0, 100.2, horizon, dtype=np.float64)
    volumes = np.full((horizon,), 1_000_000.0, dtype=np.float64)
    cfg = ExecutionEnvConfig(
        horizon_steps=horizon,
        order_size=1000.0,
        participation_cap=0.01,
        impact_eta=0.01,
        fixed_fee=0.0,
        terminal_penalty=1.0,
    )

    env = ExecutionGymEnv(cfg=cfg, volumes=volumes, prices=prices)
    obs, info = env.reset(seed=0)
    assert obs.shape == (3,)
    assert isinstance(info, dict)

    action = np.asarray([0.5], dtype=np.float32)
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2.shape == (3,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "fill" in info2
