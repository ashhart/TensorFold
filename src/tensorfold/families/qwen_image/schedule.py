"""Qwen-Image-2.1 sigma schedule: rectified-flow Euler with a resolution-dependent shift."""

from __future__ import annotations

import math

import numpy as np

BASE_SHIFT, MAX_SHIFT = 0.5, 0.9
BASE_TOKENS, MAX_TOKENS = 256, 8192
TERMINAL = 0.02


def shift_for(width: int, height: int) -> float:
    """The exponent of the time shift for an image of ``width`` x ``height`` pixels."""

    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_TOKENS - BASE_TOKENS)
    return slope * (width * height / 256) + BASE_SHIFT - slope * BASE_TOKENS


def sigmas(steps: int, width: int, height: int, nodes: tuple[float, ...] | None = None) -> np.ndarray:
    """``steps + 1`` noise levels from 1 down to 0.

    ``nodes`` replaces the even spacing with fixed raw levels (a distilled adapter's training nodes); those take
    the shift but not the terminal stretch.
    """

    if nodes is None:
        if steps < 1:
            raise ValueError("at least one step is needed")
        raw = np.linspace(1.0, 1.0 / steps, steps, dtype=np.float32)
    else:
        raw = np.asarray(nodes, dtype=np.float32)
        if raw.ndim != 1 or len(raw) < 1 or np.any(raw <= 0) or np.any(raw > 1) or np.any(np.diff(raw) >= 0):
            raise ValueError("nodes must fall from at most 1 towards 0, exclusive of 0")
    scale = np.float32(math.exp(shift_for(width, height)))
    shifted = scale / (scale + (1.0 / raw - 1.0))
    if nodes is None:
        rest = 1.0 - shifted
        shifted = 1.0 - rest / (rest[-1] / (1.0 - TERMINAL))
    return np.concatenate([shifted, np.zeros(1, dtype=np.float32)]).astype(np.float32)
