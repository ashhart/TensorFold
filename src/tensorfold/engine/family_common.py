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


def _arrays_in(value: Any) -> list[Any]:
    """The arrays in ``value``, through nested lists and tuples (mlx-lm 0.32 states add offsets and nested lists)."""

    if isinstance(value, (list, tuple)):
        return [a for v in value for a in _arrays_in(v)]
    return [value] if value is not None and hasattr(value, "shape") else []


def cache_arrays(cache: list[Any]) -> list[Any]:
    """Every array a cache list holds (a KV cache nothing was written to yet has none)."""

    return [a for item in cache if getattr(item, "keys", 0) is not None for a in _arrays_in(item.state)]


def cache_contents(item: Any) -> list[Any]:
    """One layer's cached values, the same under mlx-lm 0.31 and 0.32."""

    # KV rows stop at the offset. Under 0.32, state is the whole buffer plus that offset.
    if getattr(item, "keys", 0) is None:
        return []
    rows = getattr(item, "keys_and_values", None)
    return _arrays_in(rows() if callable(rows) else item.state)
