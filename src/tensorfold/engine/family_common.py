"""Small pieces the family rounds share: cache arrays, spare buffers, and the rounds' diagnostic switches."""

from __future__ import annotations

import os
from typing import Any

# TF_FAMILY_PROFILE logs mean host time per round phase.
_PROFILE = os.environ.get("TF_FAMILY_PROFILE", "") == "1"
# TF_FAMILY_ROUND_LOG=path appends one line a round: position kind rows kept ms (kind: head, copy, forced, none)
_ROUND_LOG = os.environ.get("TF_FAMILY_ROUND_LOG", "")


def drop_spares(cache: list[Any]) -> list[Any]:
    """``alternating_kv.drop_spares`` (imported when used: this module loads without MLX)."""

    from tensorfold.engine.alternating_kv import drop_spares as drop

    return drop(cache)


def cache_arrays(cache: list[Any]) -> list[Any]:
    """Every array a cache list holds (a KV cache nothing was written to yet has none)."""

    arrays: list[Any] = []
    for item in cache:
        if getattr(item, "keys", 0) is None:
            continue
        state = item.state
        if isinstance(state, (list, tuple)):
            arrays.extend(a for a in state if a is not None and hasattr(a, "shape"))
        elif state is not None and hasattr(state, "shape"):
            arrays.append(state)
    return arrays
