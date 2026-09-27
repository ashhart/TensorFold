"""Admit concurrent streams only when projected memory, including every live stream's longest reply, fits the budget."""

from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
from typing import Any, Callable, Sequence

# a stream is priced this many tokens past its length (KV caches grow in steps of 256 positions)
_SLACK_TOKENS = 256
# the longest prompt chunk a prefill runs at a time (the engine's step)
_CHUNK = 2048
# positions a decode spare buffer grows by at a time (``alternating_kv.AlternatingKVCache.grow``)
_SPARE_STEP = 2048


def ram_bytes() -> int:
    import os

    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def used_elsewhere(own: int) -> int:
    """Bytes the rest of the system holds: RAM less the free share ``memory_pressure`` reports, less ``own``."""

    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=10).stdout
        free = int(re.search(r"free percentage:\s*(\d+)", out).group(1))
    except Exception:  # noqa: BLE001 - no reading: assume the rest of the machine is idle
        return 0
    return max(0, int(ram_bytes() * (100 - free) / 100) - int(own))


@dataclass
class StreamMemory:
    """Model stream growth beyond bounded draft context, prefill memory as a * chunk + b * chunk * context, and the widest shared round."""

    short_tokens: int
    short: int
    long_tokens: int
    long: int
    per_token: float
    prefill_a: float
    prefill_b: float
    round_bytes: int

    def stream_bytes(self, tokens: int) -> int:
        t = int(tokens) + _SLACK_TOKENS
        if t <= self.short_tokens:
            return int(self.short)
        if t <= self.long_tokens:
            return int(self.short + (self.long - self.short) * (t - self.short_tokens)
                       / (self.long_tokens - self.short_tokens))
        return int(self.long + self.per_token * (t - self.long_tokens))

    def prefill_bytes(self, tokens: int) -> int:
        chunk = min(_CHUNK, max(1, int(tokens)))
        return int(self.prefill_a * chunk + self.prefill_b * chunk * int(tokens))


class Admission:
    """Budget current memory, live and new streams at their longest replies, and the larger of prefill and shared-round working memory."""

    def __init__(self, budget: int, memory: StreamMemory, used: Callable[[], int] | None = None) -> None:
        self.budget = int(budget)
        self.memory = memory
        self.used = used or _mlx_used
        self.refused = 0

    def projected(self, prompt: int, longest: int, live: Sequence[tuple[int, int]]) -> int:
        """Project the new stream from ``prompt`` to ``longest`` tokens and every live stream from its current to longest length."""

        grow = sum(max(0, most - now) for now, most in live) * self.memory.per_token
        work = max(self.memory.round_bytes, self.memory.prefill_bytes(prompt))
        return int(self.used() + grow + self.memory.stream_bytes(longest) + work)

    def admits(self, prompt: int, longest: int, live: Sequence[tuple[int, int]]) -> bool:
        ok = self.projected(prompt, longest, live) <= self.budget
        self.refused += 0 if ok else 1
        return ok

    def fitting(self, tokens: int) -> int:
        """How many streams of ``tokens`` tokens fit the budget beside what is resident now (at most 64)."""

        room = self.budget - self.used() - max(self.memory.round_bytes, self.memory.prefill_bytes(tokens))
        each = self.memory.stream_bytes(tokens)
        return 64 if each <= 0 else max(0, min(64, int(room // each)))


def _mlx_used() -> int:
    import mlx.core as mx

    return int(mx.get_active_memory() + mx.get_cache_memory())


def _kv_bytes(cache: list[Any]) -> tuple[float, float]:
    """(bytes a position of every KV cache, bytes a position of the spare buffers decoding adds)."""

    kv = spare = 0.0
    for item in cache:
        keys, values = getattr(item, "keys", None), getattr(item, "values", None)
        if getattr(keys, "ndim", 0) == 4 and getattr(values, "ndim", 0) == 4 and int(keys.shape[2]):
            each = (keys.nbytes + values.nbytes) / int(keys.shape[2])
            kv += each
            if hasattr(item, "spare_keys"):
                spare += each
    return kv, spare


def measure(engine: Any, probe: tuple[int, int, int] = (64, _CHUNK + 64, 2 * _CHUNK + 64)) -> StreamMemory:
    """Probe cache growth beyond bounded draft context and peak prefill memory, then account for the shared-round working set."""

    import mlx.core as mx

    from tensorfold.engine.family_common import cache_arrays

    sizes, peaks, held = [], [], []
    # Replace retained forward state before probing so measured growth belongs only to the probe caches.
    mx.eval(*cache_arrays(engine.prefill_prefix([1000 + i for i in range(probe[0])], cache=None, cached_tokens=0)))
    for n in probe:
        tokens = [1000 + i for i in range(n)]
        mx.eval(mx.zeros((1,)))
        before = mx.get_active_memory()
        mx.reset_peak_memory()
        cache = engine.prefill_prefix(tokens, cache=None, cached_tokens=0)
        mx.eval(*cache_arrays(cache))
        after = mx.get_active_memory()
        sizes.append(max(0, after - before))
        peaks.append(max(0, mx.get_peak_memory() - after))
        held.append(cache)
    (n1, n2, n3), (s1, s2, s3), (_, t2, t3) = probe, sizes, peaks
    # Floor growth at per-position cache bytes plus spares because stepped allocation can hide growth; price the next spare step up front.
    kv, spare = _kv_bytes(held[-1])
    per_token = max((s3 - s2) / (n3 - n2), kv + spare)
    s1, s2 = s1 + spare * _SPARE_STEP, s2 + spare * _SPARE_STEP
    # Fit t = a c + b c L for c-row chunks at context length L; both probes finish with full chunks.
    b = max(0.0, (t3 - t2) / (_CHUNK * (n3 - n2)))
    a = max(0.0, t2 / _CHUNK - b * (n2 - 64))
    del held
    mx.clear_cache()
    round_bytes = int(getattr(engine, "round_working_set", lambda: 0)())
    return StreamMemory(n1, s1, n2, max(s1, s2), per_token, a, b, round_bytes)


__all__ = ["Admission", "StreamMemory", "measure", "ram_bytes", "used_elsewhere"]
