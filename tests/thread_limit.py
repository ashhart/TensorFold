"""Emulate a GPU whose pipelines take at most ``limit`` threads a threadgroup, raising MLX's own error above it."""

from __future__ import annotations

import re
from typing import Any

import mlx.core as mx

MESSAGE = "Thread group size ({}) is greater than  the maximum allowed threads per threadgroup ({})."
largest: dict[str, int] = {}          # each kernel's largest launch, by name
reserved: dict[str, int] = {}         # sizes kernels reserve in their headers, by name
_RESERVED = re.compile(r"max_total_threads_per_threadgroup\((\d+)\)")


def install(limit: int) -> None:
    """Wrap mx.fast.metal_kernel before any kernel is built (modules cache theirs); a reserved size is the limit."""

    from tensorfold.kernels import threads

    threads.probing = True                    # an emulated M1/M2: fit() probes, as it does there
    real = mx.fast.metal_kernel

    def make(*args: Any, **spec: Any) -> Any:
        kernel = real(*args, **spec)
        name = spec.get("name") or args[0]
        found = _RESERVED.search(spec.get("header", ""))
        cap = int(found.group(1)) if found else limit
        if found:
            reserved[name] = cap

        def call(*cargs: Any, **kw: Any) -> Any:
            tg = kw["threadgroup"]
            size = int(tg[0]) * int(tg[1]) * int(tg[2])
            if cap and size > cap:
                raise ValueError(MESSAGE.format(size, cap))
            largest[name] = max(largest.get(name, 0), size)
            return kernel(*cargs, **kw)

        return call

    mx.fast.metal_kernel = make
