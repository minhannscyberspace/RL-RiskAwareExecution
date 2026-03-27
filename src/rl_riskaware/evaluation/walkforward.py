from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rl_riskaware.baselines import build_twap_schedule, build_vwap_schedule, pov_action, twap_participation_action
from rl_riskaware.env import ExecutionEnv, ExecutionEnvConfig
from rl_riskaware.evaluation.implementation_shortfall import (
    completion_rate,
    implementation_shortfall,
    slippage_bps,
    volume_weighted_avg_exec_price,
)


@dataclass(frozen=True)
class Window:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class WindowResult:
    window_id: int
    policy: str
    is_value: float
    completion: float
    avg_exec_price: float
    slippage_bps_value: float


def make_walkforward_windows(total_len: int, train_len: int, test_len: int, step: int) -> list[Window]:
    if total_len <= 0 or train_len <= 0 or test_len <= 0 or step <= 0:
        raise ValueError("total_len/train_len/test_len/step must be > 0")
    windows: list[Window] = []
    train_start = 0
    while True:
        train_end = train_start + train_len
        test_start = train_end
        test_end = test_start + test_len
        if test_end > total_len:
            break
        windows.append(
            Window(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end)
        )
        train_start += step
    return windows


def run_window_benchmarks(
    prices: npt.NDArray[np.float64],
    volumes: npt.NDArray[np.float64],
    order_size: float,
    participation_cap: float,
    impact_eta: float,
    fixed_fee: float,
    terminal_penalty: float,
    windows: list[Window],
) -> list[WindowResult]:
    results: list[WindowResult] = []
    for i, w in enumerate(windows):
        p_test = prices[w.test_start : w.test_end]
        v_test = volumes[w.test_start : w.test_end]
        horizon = len(p_test)

        cfg = ExecutionEnvConfig(
            horizon_steps=horizon,
            order_size=order_size,
            participation_cap=participation_cap,
            impact_eta=impact_eta,
            fixed_fee=fixed_fee,
            terminal_penalty=terminal_penalty,
        )

        actions_by_policy = _build_actions(v_test, cfg)
        for policy_name, actions in actions_by_policy.items():
            env = ExecutionEnv(cfg=cfg, volumes=v_test, prices=p_test)
            _, _ = env.reset(seed=0)

            exec_prices: list[float] = []
            exec_qty: list[float] = []
            for t in range(horizon):
                out = env.step(actions[t])
                exec_prices.append(float(out.info["exec_price"]))
                exec_qty.append(float(out.info["fill"]))
                if out.terminated or out.truncated:
                    break

            exec_prices_arr = np.asarray(exec_prices, dtype=np.float64)
            exec_qty_arr = np.asarray(exec_qty, dtype=np.float64)
            results.append(
                WindowResult(
                    window_id=i,
                    policy=policy_name,
                    is_value=implementation_shortfall(float(p_test[0]), exec_prices_arr, exec_qty_arr),
                    completion=completion_rate(order_size, exec_qty_arr),
                    avg_exec_price=volume_weighted_avg_exec_price(exec_prices_arr, exec_qty_arr),
                    slippage_bps_value=slippage_bps(float(p_test[0]), exec_prices_arr, exec_qty_arr),
                )
            )
    return results


def _build_actions(volumes: npt.NDArray[np.float64], cfg: ExecutionEnvConfig) -> dict[str, list[float]]:
    horizon = len(volumes)
    twap = build_twap_schedule(order_size=cfg.order_size, horizon_steps=horizon)
    vwap = build_vwap_schedule(order_size=cfg.order_size, volumes=volumes, horizon_steps=horizon)
    return {
        "TWAP": [
            twap_participation_action(float(twap.fills[t]), float(volumes[t]), cfg.participation_cap)
            for t in range(horizon)
        ],
        "VWAP": [
            twap_participation_action(float(vwap.fills[t]), float(volumes[t]), cfg.participation_cap)
            for t in range(horizon)
        ],
        "POV(70%)": [pov_action(0.7) for _ in range(horizon)],
    }
