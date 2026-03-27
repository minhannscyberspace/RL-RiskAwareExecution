from pathlib import Path

import pandas as pd

from rl_riskaware.reporting import build_eval_report


def test_build_eval_report(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval_dir"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"window": 0, "policy": "PPO", "completion": 1.0, "is": -10.0, "avg_exec_price": 100.0, "slippage_bps": -1.0},
            {"window": 0, "policy": "TWAP", "completion": 1.0, "is": 10.0, "avg_exec_price": 101.0, "slippage_bps": 1.0},
        ]
    ).to_csv(eval_dir / "results.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy": "PPO",
                "n_windows": 1,
                "completion_mean": 1.0,
                "completion_median": 1.0,
                "is_mean": -10.0,
                "is_median": -10.0,
                "avg_exec_price_mean": 100.0,
                "avg_exec_price_median": 100.0,
                "slippage_bps_mean": -1.0,
                "slippage_bps_median": -1.0,
            }
        ]
    ).to_csv(eval_dir / "summary.csv", index=False)

    out = build_eval_report(eval_dir)
    assert Path(out["report_md"]).exists()
    assert Path(out["plots_dir"]).exists()
    assert (Path(out["plots_dir"]) / "completion_by_policy.png").exists()
