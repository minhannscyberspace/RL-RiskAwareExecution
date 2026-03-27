from __future__ import annotations

import numpy as np


def aggregate_policy_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_policy: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        p = str(r["policy"])
        by_policy.setdefault(p, []).append(r)

    out: list[dict[str, object]] = []
    metrics = ("completion", "is", "avg_exec_price", "slippage_bps")
    for policy, vals in by_policy.items():
        row: dict[str, object] = {"policy": policy, "n_windows": len(vals)}
        for m in metrics:
            arr = np.asarray([float(v[m]) for v in vals], dtype=np.float64)
            row[f"{m}_mean"] = float(np.nanmean(arr))
            row[f"{m}_median"] = float(np.nanmedian(arr))
        out.append(row)
    return out
