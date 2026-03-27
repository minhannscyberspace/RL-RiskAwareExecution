from __future__ import annotations

from typing import Callable

import numpy as np

from rl_riskaware.evaluation import completion_rate, implementation_shortfall, slippage_bps, volume_weighted_avg_exec_price


def run_actor_episode(
    env_reset: Callable[[int | None], tuple[np.ndarray, dict[str, object]]],
    env_step: Callable[[float], tuple[np.ndarray, float, bool, bool, dict[str, object]]],
    actor: Callable[[np.ndarray], float],
    horizon_steps: int,
    order_size: float,
    arrival_price: float,
    seed: int | None = None,
) -> dict[str, float]:
    obs, _ = env_reset(seed)
    exec_prices: list[float] = []
    exec_qty: list[float] = []
    for _ in range(horizon_steps):
        action = float(actor(obs))
        obs, _, terminated, truncated, info = env_step(action)
        exec_prices.append(float(info["exec_price"]))
        exec_qty.append(float(info["fill"]))
        if terminated or truncated:
            break

    px = np.asarray(exec_prices, dtype=np.float64)
    qty = np.asarray(exec_qty, dtype=np.float64)
    return {
        "total_fill": float(np.sum(qty)),
        "completion": completion_rate(order_size, qty),
        "is": implementation_shortfall(arrival_price, px, qty),
        "avg_exec_price": volume_weighted_avg_exec_price(px, qty),
        "slippage_bps": slippage_bps(arrival_price, px, qty),
    }
