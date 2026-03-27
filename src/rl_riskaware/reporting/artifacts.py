from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import csv
import json

import yaml


def make_run_dir(base_dir: str | Path, prefix: str) -> Path:
    base = Path(base_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base.mkdir(parents=True, exist_ok=True)
    for i in range(1000):
        suffix = "" if i == 0 else f"_{i:03d}"
        run_dir = base / f"{prefix}_{ts}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate unique run directory")


def write_config_snapshot(run_dir: str | Path, config: dict[str, object]) -> Path:
    out = Path(run_dir) / "config_snapshot.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=True)
    return out


def write_metadata_json(run_dir: str | Path, metadata: dict[str, object]) -> Path:
    out = Path(run_dir) / "metadata.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return out


def write_results_csv(run_dir: str | Path, filename: str, rows: list[dict[str, object]]) -> Path:
    if len(rows) == 0:
        raise ValueError("rows must not be empty")
    out = Path(run_dir) / filename
    fieldnames = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out
