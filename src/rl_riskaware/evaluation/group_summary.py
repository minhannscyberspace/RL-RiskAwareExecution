from __future__ import annotations

import numpy as np


def aggregate_rows_by_keys(
    rows: list[dict[str, object]],
    group_keys: tuple[str, ...],
    metrics: tuple[str, ...] = ("completion", "is", "avg_exec_price", "slippage_bps"),
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[k]) for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for key_vals, vals in grouped.items():
        rec: dict[str, object] = {k: v for k, v in zip(group_keys, key_vals)}
        rec["n_rows"] = len(vals)
        for m in metrics:
            arr = np.asarray([float(v[m]) for v in vals], dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                rec[f"{m}_mean"] = float("nan")
                rec[f"{m}_median"] = float("nan")
            else:
                rec[f"{m}_mean"] = float(np.mean(finite))
                rec[f"{m}_median"] = float(np.median(finite))
        out.append(rec)
    return out
