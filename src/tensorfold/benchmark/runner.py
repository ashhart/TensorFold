"""Run the fixed public benchmark fixtures against a serving model."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from .client import stream_request
from .protocol import DEFAULT_REPETITIONS, DEFAULT_TEMPERATURES, DEFAULT_TOKENS, FIXTURES, SEED_BASE


def run_suite(
    base_url: str,
    model_id: str,
    *,
    tokens: int = DEFAULT_TOKENS,
    repetitions: int = DEFAULT_REPETITIONS,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    serial: bool = False,
    timeout: float = 600,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """One warmup per cell followed by every measured attempt.

    Failed warmups use repeat=-1, are retained, and prevent complete-cell
    summaries. Cancellation returns every completed attempt plus the cancelled
    request so a caller can save an incomplete receipt.
    """

    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 2 <= tokens <= 8192:
        raise ValueError("tokens must be an integer between 2 and 8192")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or not 1 <= repetitions <= 20:
        raise ValueError("repetitions must be an integer between 1 and 20")
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    temperatures = tuple(temperatures)
    if not temperatures or len(temperatures) > 4 or len(set(temperatures)) != len(temperatures):
        raise ValueError("temperatures must contain between 1 and 4 distinct values")
    if any(
        isinstance(value, bool) or not isinstance(value, (float, int))
        or not math.isfinite(value) or not 0 <= value <= 2 for value in temperatures
    ):
        raise ValueError("temperatures must be finite numbers between 0 and 2")
    samples = []
    for temperature in temperatures:
        for fixture in FIXTURES:
            warmup = stream_request(
                base_url, model_id, fixture, tokens=tokens, temperature=temperature,
                seed=SEED_BASE, repeat=-1, serial=serial, timeout=timeout,
            )
            if progress is not None:
                progress(warmup)
            if warmup["status"] not in {"ok", "early_eos", "unmeasured"}:
                samples.append(warmup)
                if warmup["error_code"] == "cancelled":
                    return samples
            for repeat in range(repetitions):
                sample = stream_request(
                    base_url, model_id, fixture, tokens=tokens, temperature=temperature,
                    seed=SEED_BASE + repeat, repeat=repeat, serial=serial, timeout=timeout,
                )
                samples.append(sample)
                if progress is not None:
                    progress(sample)
                if sample["error_code"] == "cancelled":
                    return samples
    return samples
