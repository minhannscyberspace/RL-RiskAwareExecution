from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


ArrayF = npt.NDArray[np.float64]


@dataclass(frozen=True)
class StepOutput:
    observation: ArrayF
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class Env(Protocol):
    def reset(self, seed: int | None = None) -> tuple[ArrayF, dict[str, object]]: ...

    def step(self, action: float) -> StepOutput: ...
