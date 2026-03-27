from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class POVPolicy:
    participation: float  # constant participation rate in [0, 1]


def pov_action(participation: float) -> float:
    return float(np.clip(participation, 0.0, 1.0))

