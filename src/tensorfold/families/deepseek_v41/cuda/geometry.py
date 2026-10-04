"""Admission geometry: the device bytes a rank's caches and scratch take at a cache-slot capacity."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping

from tensorfold.cuda.capacity import GIB, Geometry

from ..config import Config
from . import MAX_ROWS, PREFILL_ROWS, buffers

MINIMUM_SLOTS = 4096
CACHE_ENV = "TF_DSV41_CACHE_GIB"
CACHE_GIB = 3.0                  # kept snapshots of other conversations' prompts, unless CACHE_ENV says otherwise


def rope_bytes(cfg: Config, capacity: int) -> int:
    """fp32 cos and sin tables of both RoPE kinds (local, YaRN), one row of qk_rope_head_dim / 2 a slot."""

    return 2 * 2 * capacity * (cfg.qk_rope_head_dim // 2) * 4


def dsv41_geometry(cfg: Config, world: int, reserve: int = MAX_ROWS) -> Geometry:
    """State, RoPE tables, a decode window's and a prompt chunk's buffers, exactly as the engine allocates them."""

    def bytes_at(slots: int) -> int:
        slots = int(slots)
        state = buffers.State(cfg, slots, "meta").nbytes()
        decode = buffers.bytes(cfg, world, MAX_ROWS, slots)
        prompt = buffers.bytes(cfg, world, PREFILL_ROWS, slots, prefill=True)
        return state + rope_bytes(cfg, slots) + decode + prompt

    return Geometry(bytes_at, reserve, MINIMUM_SLOTS)


def cache_wanted(environ: Mapping[str, str] | None = None) -> int:
    """Bytes CACHE_ENV asks for kept snapshots; ValueError naming the variable for a negative or non-numeric value."""

    value = (os.environ if environ is None else environ).get(CACHE_ENV, "").strip()
    try:
        gib = float(value) if value else CACHE_GIB
    except ValueError:
        gib = math.nan
    if not (math.isfinite(gib) and gib >= 0):
        raise ValueError(f"{CACHE_ENV}={value}: a number of GiB, 0 or more")
    return int(gib * GIB)


def kept_bytes(plan: Mapping[str, int], wanted: int) -> int:
    """What this rank gives kept snapshots after admission: ``wanted``, up to the budget the admitted window leaves."""

    return max(0, min(wanted, plan["budget_bytes"] - plan["total_bytes_estimate"]))
