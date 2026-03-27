from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from rl_riskaware.agents import ExecutionGymEnv
from rl_riskaware.agents.policy_eval import run_actor_episode
from rl_riskaware.baselines import build_twap_schedule, build_vwap_schedule, pov_action, twap_participation_action
from rl_riskaware.config import load_yaml_config
from rl_riskaware.data import load_market_csv
from rl_riskaware.env import ExecutionEnvConfig
from rl_riskaware.evaluation import aggregate_policy_rows, make_walkforward_windows
from rl_riskaware.features import build_lag_safe_features, prices_volumes_from_features
from rl_riskaware.reporting import make_run_dir, write_config_snapshot, write_results_csv


def _build_synthetic_series(length: int) -> tuple[np.ndarray, np.ndarray]:
    prices = np.linspace(100.0, 102.0, length, dtype=np.float64) + 0.2 * np.sin(np.linspace(0.0, 10.0 * np.pi, length))
    volumes = (900_000.0 + 250_000.0 * (1.0 + np.sin(np.linspace(0.0, 8.0 * np.pi, length)))).astype(np.float64)
    return prices, volumes


def _eval_baseline(policy_name: str, cfg: ExecutionEnvConfig, prices: np.ndarray, volumes: np.ndarray) -> dict[str, float]:
    horizon = cfg.horizon_steps
    if policy_name == "TWAP":
        sched = build_twap_schedule(order_size=cfg.order_size, horizon_steps=horizon)
        actions = [
            twap_participation_action(float(sched.fills[t]), float(volumes[t]), cfg.participation_cap) for t in range(horizon)
        ]
    elif policy_name == "VWAP":
        sched = build_vwap_schedule(order_size=cfg.order_size, volumes=volumes, horizon_steps=horizon)
        actions = [
            twap_participation_action(float(sched.fills[t]), float(volumes[t]), cfg.participation_cap) for t in range(horizon)
        ]
    elif policy_name == "POV(70%)":
        actions = [pov_action(0.7) for _ in range(horizon)]
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    env = ExecutionGymEnv(cfg=cfg, volumes=volumes, prices=prices)
    idx = {"t": 0}

    def actor(_obs: np.ndarray) -> float:
        a = actions[idx["t"]]
        idx["t"] += 1
        return a

    return run_actor_episode(
        env_reset=lambda seed: env.reset(seed=seed),
        env_step=lambda action: env.step(np.asarray([action], dtype=np.float32)),
        actor=actor,
        horizon_steps=horizon,
        order_size=cfg.order_size,
        arrival_price=float(prices[0]),
        seed=0,
    )


def _load_model(algo: str, model_path: str):
    if algo == "ppo":
        return PPO.load(model_path), "PPO"
    if algo == "sac":
        return SAC.load(model_path), "SAC"
    raise ValueError(f"Unsupported algo: {algo}")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, remaining = pre.parse_known_args()
    file_cfg = load_yaml_config(pre_args.config) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Evaluate trained RL model against TWAP/VWAP/POV on walk-forward windows.")
    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(file_cfg.get("model_path", "")),
        required=("model_path" not in file_cfg),
        help="Path to RL model .zip (base name accepted)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=str(file_cfg.get("algo", "ppo")),
        choices=["ppo", "sac"],
        help="RL algorithm model type",
    )
    parser.add_argument("--reports-dir", type=str, default=str(file_cfg.get("reports_dir", "reports")))
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(file_cfg.get("data_path", "")),
        help="Optional CSV with timestamp,close,volume",
    )
    parser.add_argument("--series-len", type=int, default=int(file_cfg.get("series_len", 240)))
    parser.add_argument("--train-len", type=int, default=int(file_cfg.get("train_len", 80)))
    parser.add_argument("--test-len", type=int, default=int(file_cfg.get("test_len", 40)))
    parser.add_argument("--step", type=int, default=int(file_cfg.get("step", 40)))
    parser.add_argument("--order-size", type=float, default=float(file_cfg.get("order_size", 30_000.0)))
    parser.add_argument("--terminal-penalty", type=float, default=float(file_cfg.get("terminal_penalty", 20.0)))
    parser.add_argument("--reward-scale", type=float, default=float(file_cfg.get("reward_scale", 500.0)))
    parser.add_argument("--idle-penalty", type=float, default=float(file_cfg.get("idle_penalty", 2.0)))
    args = parser.parse_args(remaining)

    if args.data_path:
        market = load_market_csv(args.data_path)
        feats = build_lag_safe_features(market.df)
        prices, volumes = prices_volumes_from_features(feats.df)
        total_len = len(prices)
    else:
        prices, volumes = _build_synthetic_series(args.series_len)
        total_len = args.series_len
    windows = make_walkforward_windows(total_len, args.train_len, args.test_len, args.step)
    model, model_policy_name = _load_model(args.algo, args.model_path)

    rows: list[dict[str, object]] = []
    for wi, w in enumerate(windows):
        p_test = prices[w.test_start : w.test_end]
        v_test = volumes[w.test_start : w.test_end]
        cfg = ExecutionEnvConfig(
            horizon_steps=len(p_test),
            order_size=args.order_size,
            participation_cap=0.01,
            impact_eta=0.05,
            fixed_fee=0.0001,
            terminal_penalty=args.terminal_penalty,
            reward_scale=args.reward_scale,
            idle_penalty=args.idle_penalty,
        )

        ppo_env = ExecutionGymEnv(cfg=cfg, volumes=v_test, prices=p_test)
        ppo_metrics = run_actor_episode(
            env_reset=lambda seed: ppo_env.reset(seed=seed),
            env_step=lambda action: ppo_env.step(np.asarray([action], dtype=np.float32)),
            actor=lambda obs: float(model.predict(obs, deterministic=True)[0][0]),
            horizon_steps=cfg.horizon_steps,
            order_size=cfg.order_size,
            arrival_price=float(p_test[0]),
            seed=0,
        )
        rows.append({"window": wi, "policy": model_policy_name, **ppo_metrics})

        for policy in ("TWAP", "VWAP", "POV(70%)"):
            rows.append({"window": wi, "policy": policy, **_eval_baseline(policy, cfg, p_test, v_test)})

    run_dir = make_run_dir(args.reports_dir, f"eval_{args.algo}_vs_benchmarks")
    write_results_csv(run_dir, "results.csv", rows)
    summary_rows = aggregate_policy_rows(rows)
    write_results_csv(run_dir, "summary.csv", summary_rows)
    write_config_snapshot(
        run_dir,
        {
            "model_path": str(Path(args.model_path)),
            "algo": args.algo,
            "config_path": args.config if args.config else "",
            "series_len": args.series_len,
            "data_path": args.data_path if args.data_path else "synthetic",
            "train_len": args.train_len,
            "test_len": args.test_len,
            "step": args.step,
            "order_size": args.order_size,
            "windows": len(windows),
            "terminal_penalty": args.terminal_penalty,
            "reward_scale": args.reward_scale,
            "idle_penalty": args.idle_penalty,
        },
    )

    print(f"Evaluation complete. Artifacts written to: {run_dir}")
    print(f"Results CSV: {run_dir / 'results.csv'}")
    print(f"Summary CSV: {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
