from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rl_riskaware.types import ArrayF, StepOutput


@dataclass(frozen=True)
class ExecutionEnvConfig:
    horizon_steps: int
    order_size: float
    participation_cap: float  # max fraction of available volume per step in [0, 1]
    impact_eta: float  # impact coefficient (reduced-form)
    fixed_fee: float  # per-share fee / slippage term
    terminal_penalty: float  # penalty per unfilled share at horizon
    reward_scale: float = 1_000.0  # scales raw cost/penalties to RL-friendly reward magnitude
    idle_penalty: float = 0.0  # optional per-step penalty when no shares are executed


@dataclass
class ExecutionEnvState:
    t: int
    remaining: float
    arrival_price: float
    last_price: float
    rng: np.random.Generator


class ExecutionEnv:
    """
    Minimal Gym-style environment for optimal execution.

    Action: participation rate in [0, 1] (will be clipped), scaled by participation_cap.
    This Phase-1 implementation is intentionally simple: it is a correctness anchor for
    inventory accounting + cost/impact plumbing + termination logic.
    """

    def __init__(self, cfg: ExecutionEnvConfig, volumes: ArrayF, prices: ArrayF) -> None:
        if cfg.horizon_steps <= 0:
            raise ValueError("horizon_steps must be > 0")
        if cfg.order_size <= 0:
            raise ValueError("order_size must be > 0")
        if not (0.0 <= cfg.participation_cap <= 1.0):
            raise ValueError("participation_cap must be in [0, 1]")
        if volumes.ndim != 1 or prices.ndim != 1:
            raise ValueError("volumes and prices must be 1D arrays")
        if len(volumes) < cfg.horizon_steps or len(prices) < cfg.horizon_steps:
            raise ValueError("volumes/prices length must cover horizon_steps")
        if np.any(volumes < 0):
            raise ValueError("volumes must be non-negative")
        if cfg.reward_scale <= 0:
            raise ValueError("reward_scale must be > 0")

        self._cfg = cfg
        self._volumes = volumes.astype(np.float64, copy=False)
        self._prices = prices.astype(np.float64, copy=False)
        self._state: ExecutionEnvState | None = None

    def reset(self, seed: int | None = None) -> tuple[ArrayF, dict[str, object]]:
        rng = np.random.default_rng(seed)
        arrival = float(self._prices[0])
        self._state = ExecutionEnvState(
            t=0, remaining=float(self._cfg.order_size), arrival_price=arrival, last_price=arrival, rng=rng
        )
        return self._obs(), {}

    def step(self, action: float) -> StepOutput:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        cfg = self._cfg
        s = self._state

        a = float(np.clip(action, 0.0, 1.0))
        vol = float(self._volumes[s.t])
        max_fill = cfg.participation_cap * a * vol
        fill = min(s.remaining, max_fill)

        price = float(self._prices[s.t])
        impact = cfg.impact_eta * (fill / max(vol, 1e-12)) if fill > 0.0 else 0.0
        exec_price = price + impact

        step_cost = (exec_price - s.arrival_price) * fill + cfg.fixed_fee * fill
        raw_reward = -float(step_cost)
        if fill <= 0.0 and s.remaining > 0.0:
            raw_reward -= float(cfg.idle_penalty)

        s.remaining -= fill
        s.last_price = price
        s.t += 1

        terminated = (s.t >= cfg.horizon_steps) or (s.remaining <= 0.0)
        truncated = False

        if terminated and s.remaining > 0.0:
            raw_reward -= float(cfg.terminal_penalty * s.remaining)
        reward = raw_reward / float(cfg.reward_scale)

        info: dict[str, object] = {
            "t": s.t,
            "fill": fill,
            "remaining": s.remaining,
            "price": price,
            "exec_price": exec_price,
            "step_cost": step_cost,
            "raw_reward": raw_reward,
            "reward_scale": cfg.reward_scale,
        }

        obs = self._obs()
        return StepOutput(observation=obs, reward=reward, terminated=terminated, truncated=truncated, info=info)

    def _obs(self) -> ArrayF:
        if self._state is None:
            raise RuntimeError("Call reset() before requesting observation.")
        s = self._state
        cfg = self._cfg
        t_norm = s.t / max(cfg.horizon_steps, 1)
        rem_norm = s.remaining / cfg.order_size
        price = float(self._prices[min(s.t, cfg.horizon_steps - 1)])
        return np.asarray([t_norm, rem_norm, price], dtype=np.float64)

