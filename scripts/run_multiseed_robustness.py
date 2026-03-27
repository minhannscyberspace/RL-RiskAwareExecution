from __future__ import annotations

import argparse
import random

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC

from rl_riskaware.agents import ExecutionGymEnv
from rl_riskaware.agents.policy_eval import run_actor_episode
from rl_riskaware.baselines import build_twap_schedule, build_vwap_schedule, pov_action, twap_participation_action
from rl_riskaware.env import ExecutionEnvConfig
from rl_riskaware.evaluation import aggregate_rows_by_keys, make_walkforward_windows
from rl_riskaware.reporting import make_run_dir, write_config_snapshot, write_results_csv


class ActionShapedEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Training-only wrapper that injects a minimum action floor and Gaussian noise.
    This reduces collapse into zero-participation policies in harder regimes.
    """

    metadata = {"render_modes": []}

    def __init__(self, base_env: ExecutionGymEnv, action_floor: float, action_noise_std: float, seed: int) -> None:
        super().__init__()
        self._base = base_env
        self._action_floor = float(np.clip(action_floor, 0.0, 1.0))
        self._action_noise_std = max(0.0, float(action_noise_std))
        self._rng = np.random.default_rng(seed)
        self.action_space = self._base.action_space
        self.observation_space = self._base.observation_space

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        return self._base.reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        a = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        if self._action_noise_std > 0.0:
            a += float(self._rng.normal(0.0, self._action_noise_std))
        a = float(np.clip(max(a, self._action_floor), 0.0, 1.0))
        return self._base.step(np.asarray([a], dtype=np.float32))


def _ppo_behavior_clone_warmstart(
    model: PPO,
    obs_dim: int,
    target_action: float,
    epochs: int,
    batch_size: int = 256,
) -> None:
    """
    Supervised warm start: push PPO policy mean action toward a simple expert
    action level before RL updates. This helps avoid degenerate all-zero actions.
    """
    if epochs <= 0:
        return
    target = float(np.clip(target_action, 0.0, 1.0))
    device = model.device
    n_samples = max(batch_size, 1024)
    obs_np = np.random.uniform(low=0.0, high=1.0, size=(n_samples, obs_dim)).astype(np.float32)
    target_np = np.full((n_samples, 1), target, dtype=np.float32)
    obs_t = torch.as_tensor(obs_np, device=device)
    tgt_t = torch.as_tensor(target_np, device=device)

    for _ in range(epochs):
        idx = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            sl = idx[start : start + batch_size]
            batch_obs = obs_t[sl]
            batch_tgt = tgt_t[sl]
            dist = model.policy.get_distribution(batch_obs)
            mean_actions = dist.distribution.mean
            loss = torch.nn.functional.mse_loss(mean_actions, batch_tgt)
            model.policy.optimizer.zero_grad()
            loss.backward()
            model.policy.optimizer.step()


def _build_synthetic_series(length: int, regime: str) -> tuple[np.ndarray, np.ndarray]:
    x_price = np.linspace(0.0, 10.0 * np.pi, length)
    x_vol = np.linspace(0.0, 8.0 * np.pi, length)
    trend = np.linspace(100.0, 102.0, length, dtype=np.float64)

    if regime == "calm":
        prices = trend + 0.08 * np.sin(x_price)
        volumes = (1_000_000.0 + 100_000.0 * (1.0 + np.sin(x_vol))).astype(np.float64)
    elif regime == "volatile":
        prices = trend + 0.35 * np.sin(x_price) + 0.10 * np.sin(2.5 * x_price)
        volumes = (850_000.0 + 350_000.0 * (1.0 + np.sin(x_vol))).astype(np.float64)
    elif regime == "trending":
        prices = np.linspace(100.0, 103.5, length, dtype=np.float64) + 0.12 * np.sin(0.6 * x_price)
        volumes = (900_000.0 + 220_000.0 * (1.0 + np.sin(0.8 * x_vol))).astype(np.float64)
    else:
        raise ValueError(f"Unknown regime: {regime}")
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
        raise ValueError(policy_name)

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


def _train_model(
    algo: str,
    seed: int,
    cfg: ExecutionEnvConfig,
    prices: np.ndarray,
    volumes: np.ndarray,
    total_timesteps: int,
    use_curriculum: bool,
    regime: str,
    curriculum_warmup_frac: float,
    curriculum_volatile_boost: float,
    ppo_learning_rate: float,
    ppo_ent_coef: float,
    ppo_volatile_recovery: bool,
    ppo_action_floor: float,
    ppo_action_noise_std: float,
    ppo_action_floor_end: float,
    ppo_action_noise_std_end: float,
    ppo_bc_warmstart_epochs: int,
    ppo_bc_target_action: float,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base_env = ExecutionGymEnv(cfg=cfg, volumes=volumes[: cfg.horizon_steps], prices=prices[: cfg.horizon_steps])
    if algo == "ppo" and regime == "volatile":
        env: gym.Env[np.ndarray, np.ndarray] = ActionShapedEnv(
            base_env=base_env,
            action_floor=ppo_action_floor_end,
            action_noise_std=ppo_action_noise_std_end,
            seed=seed,
        )
    else:
        env = base_env
    if algo == "ppo":
        rollout_steps = min(128, max(16, total_timesteps))
        batch_size = min(64, rollout_steps)
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=0,
            n_steps=rollout_steps,
            batch_size=batch_size,
            learning_rate=ppo_learning_rate,
            ent_coef=ppo_ent_coef,
            device="cpu",
        )
        if regime == "volatile" and ppo_bc_warmstart_epochs > 0:
            _ppo_behavior_clone_warmstart(
                model=model,
                obs_dim=int(base_env.observation_space.shape[0]),
                target_action=ppo_bc_target_action,
                epochs=ppo_bc_warmstart_epochs,
            )
    elif algo == "sac":
        model = SAC("MlpPolicy", env, seed=seed, verbose=0, learning_rate=3e-4, batch_size=128, device="cpu")
    else:
        raise ValueError(algo)
    if use_curriculum:
        boost = curriculum_volatile_boost if regime == "volatile" else 1.0
        if algo == "ppo" and regime == "volatile" and ppo_volatile_recovery:
            # PPO-specific staged warmup to avoid collapse to all-zero participation in volatile series.
            stage1 = ExecutionEnvConfig(
                horizon_steps=cfg.horizon_steps,
                order_size=cfg.order_size,
                participation_cap=cfg.participation_cap,
                impact_eta=cfg.impact_eta * 0.2,
                fixed_fee=cfg.fixed_fee * 0.2,
                terminal_penalty=cfg.terminal_penalty * 3.0,
                reward_scale=max(1.0, cfg.reward_scale * 0.25),
                idle_penalty=cfg.idle_penalty * 3.0,
            )
            stage2 = ExecutionEnvConfig(
                horizon_steps=cfg.horizon_steps,
                order_size=cfg.order_size,
                participation_cap=cfg.participation_cap,
                impact_eta=cfg.impact_eta * 0.6,
                fixed_fee=cfg.fixed_fee * 0.6,
                terminal_penalty=cfg.terminal_penalty * 2.0,
                reward_scale=max(1.0, cfg.reward_scale * 0.5),
                idle_penalty=cfg.idle_penalty * 2.0,
            )
            s1_env = ActionShapedEnv(
                base_env=ExecutionGymEnv(cfg=stage1, volumes=volumes[: cfg.horizon_steps], prices=prices[: cfg.horizon_steps]),
                action_floor=ppo_action_floor,
                action_noise_std=ppo_action_noise_std,
                seed=seed + 101,
            )
            s2_env = ActionShapedEnv(
                base_env=ExecutionGymEnv(cfg=stage2, volumes=volumes[: cfg.horizon_steps], prices=prices[: cfg.horizon_steps]),
                action_floor=max(0.0, 0.5 * (ppo_action_floor + ppo_action_floor_end)),
                action_noise_std=max(0.0, 0.5 * (ppo_action_noise_std + ppo_action_noise_std_end)),
                seed=seed + 201,
            )
            stage_steps = max(40, int(total_timesteps * curriculum_warmup_frac * 0.5))
            model.set_env(s1_env)
            model.learn(total_timesteps=stage_steps)
            model.set_env(s2_env)
            model.learn(total_timesteps=stage_steps)
            model.set_env(env)
            model.learn(total_timesteps=max(1, total_timesteps - (2 * stage_steps)))
        else:
            warm_cfg = ExecutionEnvConfig(
                horizon_steps=cfg.horizon_steps,
                order_size=cfg.order_size,
                participation_cap=cfg.participation_cap,
                impact_eta=cfg.impact_eta,
                fixed_fee=cfg.fixed_fee,
                terminal_penalty=cfg.terminal_penalty * 1.5 * boost,
                reward_scale=max(1.0, cfg.reward_scale * 0.5 / boost),
                idle_penalty=cfg.idle_penalty * 1.5 * boost,
            )
            warm_env_base = ExecutionGymEnv(cfg=warm_cfg, volumes=volumes[: cfg.horizon_steps], prices=prices[: cfg.horizon_steps])
            warm_env = ActionShapedEnv(
                base_env=warm_env_base,
                action_floor=ppo_action_floor if algo == "ppo" and regime == "volatile" else 0.0,
                action_noise_std=ppo_action_noise_std if algo == "ppo" and regime == "volatile" else 0.0,
                seed=seed + 301,
            )
            model.set_env(warm_env)
            warm_steps = max(50, int(total_timesteps * curriculum_warmup_frac))
            model.learn(total_timesteps=warm_steps)
            model.set_env(env)
            model.learn(total_timesteps=max(1, total_timesteps - warm_steps))
    else:
        model.learn(total_timesteps=total_timesteps)
    return model


def _effective_timesteps(
    base_timesteps: int,
    algo: str,
    regime: str,
    ppo_mult: float,
    sac_mult: float,
    volatile_mult: float,
) -> int:
    if algo == "ppo":
        algo_mult = ppo_mult
    elif algo == "sac":
        algo_mult = sac_mult
    else:
        raise ValueError(algo)
    regime_mult = volatile_mult if regime == "volatile" else 1.0
    return max(1, int(base_timesteps * algo_mult * regime_mult))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed robustness for PPO/SAC vs benchmark policies.")
    parser.add_argument("--seeds", type=str, default="11,22")
    parser.add_argument("--algos", type=str, default="ppo,sac")
    parser.add_argument("--timesteps", type=int, default=400)
    parser.add_argument("--series-len", type=int, default=120)
    parser.add_argument("--train-len", type=int, default=40)
    parser.add_argument("--test-len", type=int, default=20)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--regimes", type=str, default="calm,volatile,trending")
    parser.add_argument("--curriculum", action="store_true", help="Enable warmup curriculum before main training")
    parser.add_argument("--curriculum-warmup-frac", type=float, default=0.4)
    parser.add_argument("--curriculum-volatile-boost", type=float, default=1.75)
    parser.add_argument("--ppo-timesteps-mult", type=float, default=1.0)
    parser.add_argument("--sac-timesteps-mult", type=float, default=1.5)
    parser.add_argument("--volatile-timesteps-mult", type=float, default=2.0)
    parser.add_argument("--ppo-learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-ent-coef", type=float, default=0.03)
    parser.add_argument("--ppo-volatile-recovery", action="store_true")
    parser.add_argument("--ppo-action-floor", type=float, default=0.03)
    parser.add_argument("--ppo-action-noise-std", type=float, default=0.05)
    parser.add_argument("--ppo-action-floor-end", type=float, default=0.005)
    parser.add_argument("--ppo-action-noise-std-end", type=float, default=0.01)
    parser.add_argument("--ppo-bc-warmstart-epochs", type=int, default=20)
    parser.add_argument("--ppo-bc-target-action", type=float, default=0.7)
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    algos = [x.strip().lower() for x in args.algos.split(",") if x.strip()]
    regimes = [x.strip().lower() for x in args.regimes.split(",") if x.strip()]
    windows = make_walkforward_windows(args.series_len, args.train_len, args.test_len, args.step)
    cfg = ExecutionEnvConfig(
        horizon_steps=args.test_len,
        order_size=30_000.0,
        participation_cap=0.01,
        impact_eta=0.05,
        fixed_fee=0.0001,
        terminal_penalty=20.0,
        reward_scale=500.0,
        idle_penalty=2.0,
    )

    rows: list[dict[str, object]] = []
    for regime in regimes:
        prices, volumes = _build_synthetic_series(args.series_len, regime)
        for algo in algos:
            for seed in seeds:
                eff_steps = _effective_timesteps(
                    base_timesteps=args.timesteps,
                    algo=algo,
                    regime=regime,
                    ppo_mult=args.ppo_timesteps_mult,
                    sac_mult=args.sac_timesteps_mult,
                    volatile_mult=args.volatile_timesteps_mult,
                )
                model = _train_model(
                    algo,
                    seed,
                    cfg,
                    prices,
                    volumes,
                    eff_steps,
                    use_curriculum=args.curriculum,
                    regime=regime,
                    curriculum_warmup_frac=args.curriculum_warmup_frac,
                    curriculum_volatile_boost=args.curriculum_volatile_boost,
                    ppo_learning_rate=args.ppo_learning_rate,
                    ppo_ent_coef=args.ppo_ent_coef,
                    ppo_volatile_recovery=args.ppo_volatile_recovery,
                    ppo_action_floor=args.ppo_action_floor,
                    ppo_action_noise_std=args.ppo_action_noise_std,
                    ppo_action_floor_end=args.ppo_action_floor_end,
                    ppo_action_noise_std_end=args.ppo_action_noise_std_end,
                    ppo_bc_warmstart_epochs=args.ppo_bc_warmstart_epochs,
                    ppo_bc_target_action=args.ppo_bc_target_action,
                )
                model_policy = algo.upper()
                for wi, w in enumerate(windows):
                    p_test = prices[w.test_start : w.test_end]
                    v_test = volumes[w.test_start : w.test_end]
                    w_cfg = ExecutionEnvConfig(
                        horizon_steps=len(p_test),
                        order_size=cfg.order_size,
                        participation_cap=cfg.participation_cap,
                        impact_eta=cfg.impact_eta,
                        fixed_fee=cfg.fixed_fee,
                        terminal_penalty=cfg.terminal_penalty,
                        reward_scale=cfg.reward_scale,
                        idle_penalty=cfg.idle_penalty,
                    )
                    rl_env = ExecutionGymEnv(cfg=w_cfg, volumes=v_test, prices=p_test)
                    rl_metrics = run_actor_episode(
                        env_reset=lambda s: rl_env.reset(seed=s),
                        env_step=lambda action: rl_env.step(np.asarray([action], dtype=np.float32)),
                        actor=lambda obs: float(model.predict(obs, deterministic=True)[0][0]),
                        horizon_steps=w_cfg.horizon_steps,
                        order_size=w_cfg.order_size,
                        arrival_price=float(p_test[0]),
                        seed=seed,
                    )
                    rows.append(
                        {"regime": regime, "algo": algo, "seed": seed, "window": wi, "policy": model_policy, **rl_metrics}
                    )
                    for baseline in ("TWAP", "VWAP", "POV(70%)"):
                        rows.append(
                            {
                                "regime": regime,
                                "algo": algo,
                                "seed": seed,
                                "window": wi,
                                "policy": baseline,
                                **_eval_baseline(baseline, w_cfg, p_test, v_test),
                            }
                        )

    run_dir = make_run_dir(args.reports_dir, "robust_multiseed")
    write_results_csv(run_dir, "results.csv", rows)
    summary_policy = aggregate_rows_by_keys(rows, group_keys=("regime", "algo", "policy"))
    write_results_csv(run_dir, "summary_by_algo_policy.csv", summary_policy)
    summary_policy_seed = aggregate_rows_by_keys(rows, group_keys=("regime", "algo", "seed", "policy"))
    write_results_csv(run_dir, "summary_by_algo_seed_policy.csv", summary_policy_seed)
    summary_regime_policy = aggregate_rows_by_keys(rows, group_keys=("regime", "policy"))
    write_results_csv(run_dir, "summary_by_regime_policy.csv", summary_regime_policy)
    write_config_snapshot(
        run_dir,
        {
            "seeds": seeds,
            "algos": algos,
            "regimes": regimes,
            "timesteps": args.timesteps,
            "ppo_timesteps_mult": args.ppo_timesteps_mult,
            "sac_timesteps_mult": args.sac_timesteps_mult,
            "volatile_timesteps_mult": args.volatile_timesteps_mult,
            "series_len": args.series_len,
            "train_len": args.train_len,
            "test_len": args.test_len,
            "step": args.step,
            "windows": len(windows),
            "curriculum": args.curriculum,
            "curriculum_warmup_frac": args.curriculum_warmup_frac,
            "curriculum_volatile_boost": args.curriculum_volatile_boost,
            "ppo_learning_rate": args.ppo_learning_rate,
            "ppo_ent_coef": args.ppo_ent_coef,
            "ppo_volatile_recovery": args.ppo_volatile_recovery,
            "ppo_action_floor": args.ppo_action_floor,
            "ppo_action_noise_std": args.ppo_action_noise_std,
            "ppo_action_floor_end": args.ppo_action_floor_end,
            "ppo_action_noise_std_end": args.ppo_action_noise_std_end,
            "ppo_bc_warmstart_epochs": args.ppo_bc_warmstart_epochs,
            "ppo_bc_target_action": args.ppo_bc_target_action,
        },
    )
    print(f"Robustness run complete. Artifacts: {run_dir}")
    print(f"Summary: {run_dir / 'summary_by_algo_policy.csv'}")


if __name__ == "__main__":
    main()
