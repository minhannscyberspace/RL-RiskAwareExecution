from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureOutput:
    df: pd.DataFrame


def build_lag_safe_features(df: pd.DataFrame) -> FeatureOutput:
    required = {"timestamp", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    out["ret_1"] = out["close"].pct_change()
    out["vol_chg_1"] = out["volume"].pct_change()
    out["ret_5_std"] = out["ret_1"].rolling(window=5, min_periods=5).std()

    # Shift all predictive features by one step to prevent lookahead leakage.
    for col in ("ret_1", "vol_chg_1", "ret_5_std"):
        out[col] = out[col].shift(1)

    out = out.dropna().reset_index(drop=True)
    return FeatureOutput(df=out)


def prices_volumes_from_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df["close"].to_numpy(dtype=np.float64), df["volume"].to_numpy(dtype=np.float64)
