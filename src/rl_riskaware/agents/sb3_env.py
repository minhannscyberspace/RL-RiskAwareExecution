from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl_riskaware.env import ExecutionEnv, ExecutionEnvConfig


class ExecutionGymEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, cfg: ExecutionEnvConfig, volumes: np.ndarray, prices: np.ndarray) -> None:
        super().__init__()
        self._core = ExecutionEnv(cfg=cfg, volumes=volumes, prices=prices)
        self.action_space = gym.spaces.Box(low=np.array([0.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32))
        # [t_norm, remaining_norm, price]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -np.inf], dtype=np.float32),
            high=np.array([1.0, 1.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[np.ndarray, dict]:
        del options
        obs, info = self._core.reset(seed=seed)
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        out = self._core.step(a)
        return out.observation.astype(np.float32), float(out.reward), bool(out.terminated), bool(out.truncated), out.info
