"""H3 sigma schedules: rectified-flow Euler with a sigma shift, one schedule per modality."""

# Adapted from minimax-h3-mlx (Apache-2.0, https://github.com/mrbizarro/minimax-h3-mlx, revision 7919020).

from __future__ import annotations

import numpy as np

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def _linspace(points: int) -> np.ndarray:
    """``linspace(1, 0, points)`` in float32 as torch computes it: a float32 step, symmetric halves, one rounding.

    Duplicate sigmas are dropped below, so a one-ulp difference could change the number of forwards.
    """

    step = float(np.float32(-1.0 / np.float32(points - 1)))
    half = points // 2
    index = np.arange(points, dtype=np.float64)
    out = np.empty(points, dtype=np.float64)
    out[:half] = 1.0 + step * index[:half]
    out[half:] = -step * (points - 1 - index[half:])
    return out.astype(np.float32)


class Schedule:
    """``sigma' = s sigma / (1 + (s - 1) sigma)`` over ``linspace(1, 0, points)``; N sigmas give N - 1 forwards."""

    def __init__(self, shift: float, points: int, subset: tuple[int, ...] | None = None):
        if points < 2:
            raise ValueError(f"a schedule needs at least 2 points, got {points}")
        base = _linspace(points)
        shifted = (np.float32(shift) * base) / (np.float32(1.0) + np.float32(shift - 1.0) * base)
        values: list[float] = []
        for value in shifted.tolist():  # the shift crowds the grid near 1 and can collide in float32
            if not values or value != values[-1]:
                values.append(value)
        if subset is not None:
            # step-distilled adapters publish their ladder as points of a longer grid
            if subset[0] != 0 or subset[-1] != len(values) - 1 or list(subset) != sorted(set(subset)):
                raise ValueError(f"subset {subset} must be sorted, unique and span 0..{len(values) - 1}")
            values = [values[i] for i in subset]
        self.sigmas = np.asarray(values, dtype=np.float32)
        self.timesteps = (np.float32(1.0) - self.sigmas[:-1]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.timesteps)

    def step(self, index: int, velocity, sample):
        """One Euler step. The model predicts a data-ward velocity: ``x0 = x_t + (1 - t) v``."""

        import mlx.core as mx

        # the sigma for x0 comes from the timestep the model saw, the Euler ratio from the sigma grid; below
        # 0.5 the float32 round trip 1 - (1 - sigma) is not exact and the reference keeps the two apart
        sigma_seen = float(np.float32(1.0) - self.timesteps[index])
        ratio = self.sigmas[index + 1] / self.sigmas[index]
        clean = sample.astype(mx.float32) + sigma_seen * velocity.astype(mx.float32)
        return float(ratio) * sample.astype(mx.float32) + float(np.float32(1.0) - ratio) * clean


def parse_subset(spec: str | None) -> tuple[int, tuple[int, ...]] | None:
    """``"50:0,16,33,49"`` to ``(50, (0, 16, 33, 49))``."""

    if not spec:
        return None
    grid, _, tail = spec.partition(":")
    return int(grid), tuple(int(part) for part in tail.split(",") if part.strip())
