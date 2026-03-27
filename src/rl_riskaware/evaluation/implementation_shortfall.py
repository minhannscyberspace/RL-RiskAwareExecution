from __future__ import annotations

import numpy as np
import numpy.typing as npt


def implementation_shortfall(arrival_price: float, exec_prices: npt.NDArray[np.float64], exec_qty: npt.NDArray[np.float64]) -> float:
    """
    Compute (signed) implementation shortfall vs arrival:
      IS = sum((p_exec - p_arrival) * q_exec)
    where q_exec are filled shares. Positive IS means worse (paid more).
    """
    if exec_prices.shape != exec_qty.shape:
        raise ValueError("exec_prices and exec_qty must have same shape")
    if exec_prices.ndim != 1:
        raise ValueError("exec_prices and exec_qty must be 1D arrays")
    return float(np.sum((exec_prices - float(arrival_price)) * exec_qty))


def completion_rate(order_size: float, exec_qty: npt.NDArray[np.float64]) -> float:
    if order_size <= 0:
        raise ValueError("order_size must be > 0")
    return float(np.sum(exec_qty) / order_size)


def volume_weighted_avg_exec_price(exec_prices: npt.NDArray[np.float64], exec_qty: npt.NDArray[np.float64]) -> float:
    if exec_prices.shape != exec_qty.shape:
        raise ValueError("exec_prices and exec_qty must have same shape")
    total = float(np.sum(exec_qty))
    if total <= 0.0:
        return float("nan")
    return float(np.sum(exec_prices * exec_qty) / total)


def slippage_bps(arrival_price: float, exec_prices: npt.NDArray[np.float64], exec_qty: npt.NDArray[np.float64]) -> float:
    """
    Volume-weighted slippage in basis points vs arrival price.
    Positive is worse (paid more vs arrival).
    """
    vwap = volume_weighted_avg_exec_price(exec_prices, exec_qty)
    if not np.isfinite(vwap) or arrival_price == 0.0:
        return float("nan")
    return float((vwap - float(arrival_price)) / float(arrival_price) * 1e4)

