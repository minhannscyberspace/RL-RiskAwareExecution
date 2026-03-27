import numpy as np

from rl_riskaware.agents.policy_eval import run_actor_episode
from rl_riskaware.agents.sb3_env import ExecutionGymEnv
from rl_riskaware.env import ExecutionEnvConfig


def test_run_actor_episode_returns_metrics() -> None:
    horizon = 6
    prices = np.linspace(100.0, 100.3, horizon, dtype=np.float64)
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
    metrics = run_actor_episode(
        env_reset=lambda seed: env.reset(seed=seed),
        env_step=lambda action: env.step(np.asarray([action], dtype=np.float32)),
        actor=lambda _obs: 0.5,
        horizon_steps=horizon,
        order_size=cfg.order_size,
        arrival_price=float(prices[0]),
        seed=0,
    )
    assert "is" in metrics
    assert 0.0 <= metrics["completion"] <= 1.0
