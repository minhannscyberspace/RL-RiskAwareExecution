from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketData:
    df: pd.DataFrame


REQUIRED_COLUMNS = ("timestamp", "close", "volume")


def load_market_csv(path: str) -> MarketData:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise")
    out = out.sort_values("timestamp").reset_index(drop=True)

    if out["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps are not allowed")
    if (out["volume"] < 0).any():
        raise ValueError("volume must be non-negative")
    if out["close"].isna().any() or out["volume"].isna().any():
        raise ValueError("close/volume cannot contain NaN")

    return MarketData(df=out)
