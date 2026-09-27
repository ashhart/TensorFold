"""Each mx.fast.metal_kernel call's Metal signature by name and template: MLX 0.31 rebuilt a kernel on a new one."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import mlx.core as mx


def signature(call: dict[str, Any]) -> tuple:
    inputs = [a if isinstance(a, mx.array) else mx.array(a) for a in call["inputs"]]   # MLX converts scalars too
    return (tuple((str(a.dtype), a.ndim == 0, a.size < 8) for a in inputs),
            tuple(str(d) for d in call["output_dtypes"]))


@contextmanager
def recording() -> Iterator[dict[tuple[str, str], set]]:
    """{(kernel name, template): {signatures}} of the calls to kernels made inside the block."""

    seen: dict[tuple[str, str], set] = {}
    real = mx.fast.metal_kernel

    def make(**spec: Any) -> Any:
        kernel = real(**spec)

        def call(**kw: Any) -> Any:
            seen.setdefault((spec["name"], repr(kw.get("template"))), set()).add(signature(kw))
            return kernel(**kw)
        return call

    mx.fast.metal_kernel = make
    try:
        yield seen
    finally:
        mx.fast.metal_kernel = real


def changed(seen: dict[tuple[str, str], set]) -> str:
    """The kernels called with more than one signature, for an assertion message ("" when none)."""

    return "; ".join(f"{name} {tmpl}: {sorted(sigs)}" for (name, tmpl), sigs in seen.items() if len(sigs) > 1)
