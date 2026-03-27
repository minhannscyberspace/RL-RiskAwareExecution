from __future__ import annotations

import numpy as np

from rl_riskaware.baselines import build_twap_schedule, twap_participation_action
from rl_riskaware.env import ExecutionEnv, ExecutionEnvConfig
from rl_riskaware.evaluation import implementation_shortfall


def main() -> None:
    horizon = 10
    order_size = 1000.0

    prices = np.linspace(100.0, 100.5, horizon, dtype=np.float64)
    volumes = np.full((horizon,), 1_000_000.0, dtype=np.float64)

    cfg = ExecutionEnvConfig(
        horizon_steps=horizon,
        order_size=order_size,
        participation_cap=0.01,
        impact_eta=0.05,
        fixed_fee=0.0001,
        terminal_penalty=1.0,
    )
    env = ExecutionEnv(cfg=cfg, volumes=volumes, prices=prices)
    obs, _ = env.reset(seed=0)

    sched = build_twap_schedule(order_size=order_size, horizon_steps=horizon)
    exec_prices: list[float] = []
    exec_qty: list[float] = []

    for t in range(horizon):
        action = twap_participation_action(
            target_fill=float(sched.fills[t]),
            volume=float(volumes[t]),
            participation_cap=cfg.participation_cap,
        )
        out = env.step(action)
        exec_prices.append(float(out.info["exec_price"]))
        exec_qty.append(float(out.info["fill"]))
        obs = out.observation
        if out.terminated or out.truncated:
            break

    is_val = implementation_shortfall(arrival_price=float(prices[0]), exec_prices=np.asarray(exec_prices), exec_qty=np.asarray(exec_qty))
    total_fill = float(np.sum(exec_qty))
    print(f"Smoke run: total_fill={total_fill:.2f} IS={is_val:.6f} last_obs={obs}")


if __name__ == "__main__":
    main()

