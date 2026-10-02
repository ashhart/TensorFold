"""Threadgroup sizes a GPU takes: an M2's pipelines can cap a kernel below 1024 threads (896, 512 by its registers),
and MLX reports that only when the launch is evaluated."""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx

_widths: dict[Any, int] = {}


def widest(key: Any, sizes: tuple[int, ...], launch: Callable[[int], Any]) -> tuple[int, Any]:
    """The first of ``sizes`` (widest first) whose launch runs here, and that launch's outputs; 0 and None if none.

    The first call for a key evaluates its launch (one wait); later calls only launch at the size found."""

    size = _widths.get(key)
    if size is not None:
        return size, (launch(size) if size else None)
    for size in sizes:
        try:
            out = launch(size)
            mx.eval(out)
        except (RuntimeError, ValueError):
            continue
        _widths[key] = size
        return size, out
    _widths[key] = 0
    return 0, None
