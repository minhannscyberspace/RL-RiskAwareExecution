from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_riskaware.baselines import build_twap_schedule, build_vwap_schedule, pov_action, twap_participation_action
from rl_riskaware.env import ExecutionEnv, ExecutionEnvConfig
from rl_riskaware.evaluation import completion_rate, implementation_shortfall, slippage_bps, volume_weighted_avg_exec_price


@dataclass(frozen=True)
class RunResult:
    name: str
    total_fill: float
    completion: float
    is_value: float
    avg_exec_price: float
    slip_bps: float


def run_policy(
    name: str,
    env: ExecutionEnv,
    prices: np.ndarray,
    volumes: np.ndarray,
    cfg: ExecutionEnvConfig,
    actions: list[float],
) -> RunResult:
    env.reset(seed=0)
    exec_prices: list[float] = []
    exec_qty: list[float] = []

    for t in range(cfg.horizon_steps):
        out = env.step(actions[t])
        exec_prices.append(float(out.info["exec_price"]))
        exec_qty.append(float(out.info["fill"]))
        if out.terminated or out.truncated:
            break

    exec_prices_arr = np.asarray(exec_prices, dtype=np.float64)
    exec_qty_arr = np.asarray(exec_qty, dtype=np.float64)

    total_fill = float(np.sum(exec_qty_arr))
    comp = completion_rate(cfg.order_size, exec_qty_arr)
    is_val = implementation_shortfall(float(prices[0]), exec_prices_arr, exec_qty_arr)
    avg_px = volume_weighted_avg_exec_price(exec_prices_arr, exec_qty_arr)
    slip = slippage_bps(float(prices[0]), exec_prices_arr, exec_qty_arr)
    return RunResult(name=name, total_fill=total_fill, completion=comp, is_value=is_val, avg_exec_price=avg_px, slip_bps=slip)


def main() -> None:
    horizon = 20
    order_size = 50_000.0

    prices = np.linspace(100.0, 100.8, horizon, dtype=np.float64)
    volumes = (1_000_000.0 + 200_000.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, horizon))).astype(np.float64)

    cfg = ExecutionEnvConfig(
        horizon_steps=horizon,
        order_size=order_size,
        participation_cap=0.01,
        impact_eta=0.05,
        fixed_fee=0.0001,
        terminal_penalty=1.0,
    )

    def make_env() -> ExecutionEnv:
        return ExecutionEnv(cfg=cfg, volumes=volumes, prices=prices)

    twap = build_twap_schedule(order_size=order_size, horizon_steps=horizon)
    twap_actions = [
        twap_participation_action(float(twap.fills[t]), float(volumes[t]), cfg.participation_cap) for t in range(horizon)
    ]

    vwap = build_vwap_schedule(order_size=order_size, volumes=volumes, horizon_steps=horizon)
    vwap_actions = [
        twap_participation_action(float(vwap.fills[t]), float(volumes[t]), cfg.participation_cap) for t in range(horizon)
    ]

    pov_actions = [pov_action(0.7) for _ in range(horizon)]

    results = [
        run_policy("TWAP", make_env(), prices, volumes, cfg, twap_actions),
        run_policy("VWAP", make_env(), prices, volumes, cfg, vwap_actions),
        run_policy("POV(70%)", make_env(), prices, volumes, cfg, pov_actions),
    ]

    print("name,total_fill,completion,is,avg_exec_price,slippage_bps")
    for r in results:
        print(f"{r.name},{r.total_fill:.2f},{r.completion:.4f},{r.is_value:.6f},{r.avg_exec_price:.6f},{r.slip_bps:.3f}")


if __name__ == "__main__":
    main()

