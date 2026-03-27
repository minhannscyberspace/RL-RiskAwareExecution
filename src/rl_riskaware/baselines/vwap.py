from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class VWAPSchedule:
    fills: npt.NDArray[np.float64]


def build_vwap_schedule(order_size: float, volumes: npt.NDArray[np.float64], horizon_steps: int) -> VWAPSchedule:
    """
    VWAP schedule: allocate shares proportional to volume over the horizon.
    """
    if order_size <= 0:
        raise ValueError("order_size must be > 0")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be > 0")
    if volumes.ndim != 1:
        raise ValueError("volumes must be 1D")
    if len(volumes) < horizon_steps:
        raise ValueError("volumes length must cover horizon_steps")
    if np.any(volumes[:horizon_steps] < 0):
        raise ValueError("volumes must be non-negative")

    vol = volumes[:horizon_steps].astype(np.float64, copy=False)
    total = float(np.sum(vol))
    if total <= 0.0:
        # fallback: no volume info -> TWAP-like equal allocation
        fills = np.full((horizon_steps,), order_size / horizon_steps, dtype=np.float64)
        fills[-1] = order_size - float(np.sum(fills[:-1]))
        return VWAPSchedule(fills=fills)

    weights = vol / total
    fills = (weights * order_size).astype(np.float64, copy=False)
    # Fix rounding so total equals order_size exactly.
    fills[-1] = order_size - float(np.sum(fills[:-1]))
    return VWAPSchedule(fills=fills)

