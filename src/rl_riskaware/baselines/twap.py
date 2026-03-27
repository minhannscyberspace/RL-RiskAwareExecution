from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class TWAPSchedule:
    fills: npt.NDArray[np.float64]  # shares to execute each step


def build_twap_schedule(order_size: float, horizon_steps: int) -> TWAPSchedule:
    if order_size <= 0:
        raise ValueError("order_size must be > 0")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be > 0")

    per = order_size / horizon_steps
    fills = np.full((horizon_steps,), per, dtype=np.float64)
    fills[-1] = order_size - float(np.sum(fills[:-1]))
    return TWAPSchedule(fills=fills)


def twap_participation_action(target_fill: float, volume: float, participation_cap: float) -> float:
    """
    Convert a target share fill into an action in [0, 1], given the env semantics:
    fill = min(remaining, participation_cap * action * volume)
    """
    if volume <= 0.0 or participation_cap <= 0.0:
        return 0.0
    a = target_fill / (participation_cap * volume)
    return float(np.clip(a, 0.0, 1.0))

