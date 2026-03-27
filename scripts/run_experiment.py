from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from rl_riskaware.config import load_yaml_config
from rl_riskaware.reporting import make_run_dir, write_config_snapshot, write_metadata_json


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _latest_run_dir(base_dir: str | Path, prefix: str) -> Path:
    base = Path(base_dir)
    matches = sorted([p for p in base.glob(f"{prefix}_*") if p.is_dir()], key=lambda p: p.name)
    if not matches:
        raise RuntimeError(f"No run directories found for prefix: {prefix}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot experiment runner: train -> eval -> snapshot.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    exp = cfg.get("experiment", {})
    data = cfg.get("data", {})
    train = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    if not isinstance(exp, dict) or not isinstance(data, dict) or not isinstance(train, dict) or not isinstance(eval_cfg, dict):
        raise ValueError("experiment/data/train/eval must be mapping objects")

    reports_dir = str(exp.get("reports_dir", "reports"))
    algo = str(train.get("algo", "ppo")).lower()
    if algo != "ppo":
        raise ValueError("run_experiment.py currently supports train.algo=ppo only")

    train_cmd = [
        sys.executable,
        "scripts/train_ppo.py",
        "--seed",
        str(train.get("seed", 42)),
        "--total-timesteps",
        str(train.get("total_timesteps", 200)),
        "--series-len",
        str(data.get("series_len", 120)),
        "--horizon",
        str(data.get("horizon", 10)),
        "--order-size",
        str(train.get("order_size", 30_000.0)),
        "--reports-dir",
        reports_dir,
        "--data-path",
        str(data.get("data_path", "")),
        "--terminal-penalty",
        str(train.get("terminal_penalty", 20.0)),
        "--reward-scale",
        str(train.get("reward_scale", 500.0)),
        "--idle-penalty",
        str(train.get("idle_penalty", 2.0)),
        "--learning-rate",
        str(train.get("learning_rate", 1e-4)),
        "--ent-coef",
        str(train.get("ent_coef", 0.03)),
    ]
    _run(train_cmd)

    train_dir = _latest_run_dir(reports_dir, "train_ppo")
    model_path = str(train_dir / "ppo_model")

    eval_cmd = [
        sys.executable,
        "scripts/eval_ppo_vs_benchmarks.py",
        "--algo",
        "ppo",
        "--model-path",
        model_path,
        "--reports-dir",
        reports_dir,
        "--data-path",
        str(data.get("data_path", "")),
        "--series-len",
        str(data.get("series_len", 120)),
        "--train-len",
        str(eval_cfg.get("train_len", 8)),
        "--test-len",
        str(eval_cfg.get("test_len", 4)),
        "--step",
        str(eval_cfg.get("step", 4)),
        "--order-size",
        str(train.get("order_size", 30_000.0)),
        "--terminal-penalty",
        str(train.get("terminal_penalty", 20.0)),
        "--reward-scale",
        str(train.get("reward_scale", 500.0)),
        "--idle-penalty",
        str(train.get("idle_penalty", 2.0)),
    ]
    _run(eval_cmd)

    eval_dir = _latest_run_dir(reports_dir, "eval_ppo_vs_benchmarks")
    report_cmd = [sys.executable, "scripts/generate_eval_report.py", "--eval-dir", str(eval_dir)]
    _run(report_cmd)

    run_dir = make_run_dir(reports_dir, "pipeline")
    write_config_snapshot(run_dir, cfg)
    write_metadata_json(
        run_dir,
        {
            "experiment_name": str(exp.get("name", "default_pipeline")),
            "algo": "ppo",
            "train_dir": str(train_dir),
            "eval_dir": str(eval_dir),
            "eval_report_md": str(eval_dir / "report.md"),
            "eval_report_html": str(eval_dir / "report.html"),
            "model_path": model_path + ".zip",
        },
    )
    _update_latest_pointer(reports_dir, run_dir)

    print(f"Pipeline complete. Pipeline artifact dir: {run_dir}")
    print(f"Train artifacts: {train_dir}")
    print(f"Eval artifacts: {eval_dir}")


def _update_latest_pointer(reports_dir: str | Path, pipeline_dir: Path) -> None:
    base = Path(reports_dir)
    latest_dir = base / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    latest_file = latest_dir / "LATEST_PIPELINE.txt"
    latest_file.write_text(str(pipeline_dir), encoding="utf-8")
    snapshot_meta = pipeline_dir / "metadata.json"
    if snapshot_meta.exists():
        shutil.copy2(snapshot_meta, latest_dir / "LATEST_METADATA.json")


if __name__ == "__main__":
    main()
