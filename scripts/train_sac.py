from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from rl_riskaware.agents import ExecutionGymEnv
from rl_riskaware.env import ExecutionEnvConfig
from rl_riskaware.reporting import make_run_dir, write_config_snapshot, write_metadata_json


def _build_synthetic_series(length: int) -> tuple[np.ndarray, np.ndarray]:
    prices = np.linspace(100.0, 102.0, length, dtype=np.float64) + 0.2 * np.sin(np.linspace(0.0, 12.0 * np.pi, length))
    volumes = (900_000.0 + 250_000.0 * (1.0 + np.sin(np.linspace(0.0, 10.0 * np.pi, length)))).astype(np.float64)
    return prices, volumes


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC on RL-RiskAware ExecutionGymEnv.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=5000)
    parser.add_argument("--series-len", type=int, default=320)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--order-size", type=float, default=30_000.0)
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--terminal-penalty", type=float, default=20.0)
    parser.add_argument("--reward-scale", type=float, default=500.0)
    parser.add_argument("--idle-penalty", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prices, volumes = _build_synthetic_series(args.series_len)
    cfg = ExecutionEnvConfig(
        horizon_steps=args.horizon,
        order_size=args.order_size,
        participation_cap=0.01,
        impact_eta=0.05,
        fixed_fee=0.0001,
        terminal_penalty=args.terminal_penalty,
        reward_scale=args.reward_scale,
        idle_penalty=args.idle_penalty,
    )
    env = ExecutionGymEnv(cfg=cfg, volumes=volumes[: args.horizon], prices=prices[: args.horizon])

    model = SAC(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=0,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device="cpu",
    )
    model.learn(total_timesteps=args.total_timesteps)

    run_dir = make_run_dir(args.reports_dir, "train_sac")
    model_path = Path(run_dir) / "sac_model"
    model.save(str(model_path))

    config = {
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "series_len": args.series_len,
        "horizon": args.horizon,
        "order_size": args.order_size,
        "env_config": {
            "participation_cap": cfg.participation_cap,
            "impact_eta": cfg.impact_eta,
            "fixed_fee": cfg.fixed_fee,
            "terminal_penalty": cfg.terminal_penalty,
            "reward_scale": cfg.reward_scale,
            "idle_penalty": cfg.idle_penalty,
        },
        "sac": {"learning_rate": args.learning_rate, "batch_size": args.batch_size},
    }
    write_config_snapshot(run_dir, config)
    write_metadata_json(
        run_dir,
        {"model_path": str(model_path) + ".zip", "algorithm": "SAC", "notes": "Synthetic data training for Phase 2D"},
    )
    print(f"Training complete. Artifacts written to: {run_dir}")
    print(f"Model: {model_path}.zip")


if __name__ == "__main__":
    main()
