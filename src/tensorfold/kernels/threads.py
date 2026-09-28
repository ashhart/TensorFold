"""Keep each Metal launch within its pipeline's thread limit, which on M1 and M2 falls as its registers rise."""

from __future__ import annotations

import re
from typing import Any, Callable, Hashable, Iterable, Sequence

import mlx.core as mx

# threads a threadgroup every pipeline takes on every Apple GPU, at the most registers a thread can hold
SAFE = 256
_LIMIT = re.compile(r"maximum allowed threads per threadgroup \((\d+)\)")
_TRACING = "function transformations"
_fitted: dict[Hashable, int] = {}
# keys first launched inside mx.compile, where no probe can run: they took the largest size up to SAFE
guessed: set[Hashable] = set()


def _probes() -> bool:
    """Whether pipelines here can take fewer threads than a launch asks: M1 and M2 (M3 on give every one 1024)."""

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    found = re.match(r"applegpu_g(\d+)", str(info.get("architecture", "")))
    return not found or int(found.group(1)) < 15


# fit() probes only where limits vary (tests emulating an M1/M2 set it)
probing = _probes()


def limit_in(err: BaseException) -> int | None:
    """The limit MLX names when a launch passes its pipeline's threads per threadgroup, else None."""

    found = _LIMIT.search(str(err)) if isinstance(err, ValueError) else None
    return int(found.group(1)) if found else None


def chip() -> str:
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return str(info.get("device_name") or info.get("architecture") or "this GPU")


def fitted(key: Hashable) -> int | None:
    return _fitted.get(key)


def reserve(size: int) -> str:
    """The end of a kernel's header that makes every GPU's pipeline take ``size`` threads (M1/M2 cut registers)."""

    # MLX writes the header straight before `[[kernel]] void`, so a kernel using this passes no template args
    return f"\n[[max_total_threads_per_threadgroup({int(size)})]]\n"


def fit(key: tuple, sizes: Iterable[int], launch: Callable[[int], Any], inputs: Sequence[Any] = ()) -> Any:
    """launch(size) at the largest size key's pipeline takes here (key[0] names the kernel); a new key probes alone."""

    size = _fitted.get(key)
    if size is not None:
        return launch(size)
    options = sorted({int(s) for s in sizes}, reverse=True)
    if not probing:
        _fitted[key] = options[0]
        return launch(options[0])
    cap = None
    for size in options:
        if cap is not None and size > cap:
            continue
        try:
            if size > SAFE:
                mx.eval(*[a for a in inputs if isinstance(a, mx.array)])   # a throwing launch leaves nothing pending
            out = launch(size)
            if size > SAFE:
                mx.eval(out)
        except ValueError as err:
            if _TRACING in str(err):                  # inside mx.compile: no launch can run alone here
                guessed.add(key)
                return launch(next((s for s in options if s <= SAFE), options[-1]))
            cap = limit_in(err)
            if cap is None:
                raise
            continue
        _fitted[key] = size
        return out
    raise RuntimeError(f"[tensorfold] {chip()} allows {cap} threads a threadgroup for the {key[0]} kernel, which needs "
                       f"{options[-1]}: run `tensorfold update`, and if it still fails report this line at "
                       "https://github.com/ashhart/TensorFold/issues")


__all__ = ["SAFE", "chip", "fit", "fitted", "guessed", "limit_in", "probing", "reserve"]
