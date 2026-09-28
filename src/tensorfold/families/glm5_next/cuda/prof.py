"""TF_GLM_PROFILE=1: time (with device syncs) and memory of each block kind of every prompt chunk, printed after each prefill; never inside decode or capture."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import torch

ENABLED = os.environ.get("TF_GLM_PROFILE", "0") == "1"
active = False                     # set only around prefill chunks
totals: dict[str, float] = {}
peaks: dict[str, int] = {}         # the most memory a block allocated on top of what was live before it


@contextmanager
def timed(name: str):
    if not (ENABLED and active):
        yield
        return
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    yield
    torch.cuda.synchronize()
    totals[name] = totals.get(name, 0.0) + time.perf_counter() - t
    peaks[name] = max(peaks.get(name, 0), torch.cuda.max_memory_allocated() - before)


def report(tokens: int) -> None:
    if not ENABLED or not totals:
        return
    total = sum(v for k, v in totals.items() if ":" not in k)          # "dsa: x" parts are inside "dsa (total)"
    parts = ", ".join(f"{k} {v:.2f}s ({100 * v / total:.0f}%)" for k, v in sorted(totals.items(), key=lambda kv: -kv[1]))
    print(f"[tensorfold] prefill profile, {tokens} tokens in {total:.1f}s timed ({tokens / total:.0f} tok/s): {parts}",
          flush=True)
    big = ", ".join(f"{k} {v / 2**30:.2f}" for k, v in sorted(peaks.items(), key=lambda kv: -kv[1])[:6])
    print(f"[tensorfold] prefill memory (GiB): allocated {torch.cuda.memory_allocated() / 2**30:.1f}, reserved "
          f"{torch.cuda.memory_reserved() / 2**30:.1f}; most a block added: {big}", flush=True)
    totals.clear()
    peaks.clear()
