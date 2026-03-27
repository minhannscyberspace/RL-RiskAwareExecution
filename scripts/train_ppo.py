from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from rl_riskaware.agents import ExecutionGymEnv
from rl_riskaware.config import load_yaml_config
from rl_riskaware.data import load_market_csv
from rl_riskaware.env import ExecutionEnvConfig
from rl_riskaware.features import build_lag_safe_features, prices_volumes_from_features
from rl_riskaware.reporting import make_run_dir, write_config_snapshot, write_metadata_json


def _build_synthetic_series(length: int) -> tuple[np.ndarray, np.ndarray]:
    prices = np.linspace(100.0, 102.0, length, dtype=np.float64) + 0.2 * np.sin(np.linspace(0.0, 12.0 * np.pi, length))
    volumes = (900_000.0 + 250_000.0 * (1.0 + np.sin(np.linspace(0.0, 10.0 * np.pi, length)))).astype(np.float64)
    return prices, volumes


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, remaining = pre.parse_known_args()
    file_cfg = load_yaml_config(pre_args.config) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Train PPO on RL-RiskAware ExecutionGymEnv.")
    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument("--seed", type=int, default=int(file_cfg.get("seed", 42)))
    parser.add_argument("--total-timesteps", type=int, default=int(file_cfg.get("total_timesteps", 3000)))
    parser.add_argument("--series-len", type=int, default=int(file_cfg.get("series_len", 320)))
    parser.add_argument("--horizon", type=int, default=int(file_cfg.get("horizon", 40)))
    parser.add_argument("--order-size", type=float, default=float(file_cfg.get("order_size", 30_000.0)))
    parser.add_argument("--reports-dir", type=str, default=str(file_cfg.get("reports_dir", "reports")))
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(file_cfg.get("data_path", "")),
        help="Optional CSV with timestamp,close,volume",
    )
    parser.add_argument("--terminal-penalty", type=float, default=float(file_cfg.get("terminal_penalty", 20.0)))
    parser.add_argument("--reward-scale", type=float, default=float(file_cfg.get("reward_scale", 500.0)))
    parser.add_argument("--idle-penalty", type=float, default=float(file_cfg.get("idle_penalty", 2.0)))
    parser.add_argument("--learning-rate", type=float, default=float(file_cfg.get("learning_rate", 1e-4)))
    parser.add_argument("--ent-coef", type=float, default=float(file_cfg.get("ent_coef", 0.03)))
    args = parser.parse_args(remaining)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.data_path:
        market = load_market_csv(args.data_path)
        feats = build_lag_safe_features(market.df)
        prices, volumes = prices_volumes_from_features(feats.df)
    else:
        prices, volumes = _build_synthetic_series(args.series_len)

    if len(prices) < args.horizon:
        raise ValueError("Not enough rows for chosen horizon after feature processing")
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

    rollout_steps = min(128, max(16, args.total_timesteps))
    batch_size = min(64, rollout_steps)
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=0,
        n_steps=rollout_steps,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        device="cpu",
    )
    model.learn(total_timesteps=args.total_timesteps)

    run_dir = make_run_dir(args.reports_dir, "train_ppo")
    model_path = Path(run_dir) / "ppo_model"
    model.save(str(model_path))

    config = {
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "series_len": args.series_len,
        "config_path": args.config if args.config else "",
        "data_path": args.data_path if args.data_path else "synthetic",
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
        "ppo": {"learning_rate": args.learning_rate, "ent_coef": args.ent_coef, "n_steps": rollout_steps, "batch_size": batch_size},
    }
    write_config_snapshot(run_dir, config)
    write_metadata_json(
        run_dir,
        {"model_path": str(model_path) + ".zip", "algorithm": "PPO", "notes": "Synthetic data training for Phase 2B"},
    )
    print(f"Training complete. Artifacts written to: {run_dir}")
    print(f"Model: {model_path}.zip")


if __name__ == "__main__":
    main()
