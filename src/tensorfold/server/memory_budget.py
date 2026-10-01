"""MLX limits and prompt admission from cache sizes."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Mapping, Sequence

GIB = 1024**3
MEMORY_FRACTION = 0.70
LIMIT_ENV = "TENSORFOLD_MEMORY_LIMIT_GB"
# the process's memory outside MLX's buffers and Metal's late returns
PROCESS_BYTES = 3 * GIB
# probe peaks move run to run, so the worst of PROBE_REPEATS sizes the chunk and window
PROBE_REPEATS = 3


def physical_memory_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def process_footprint() -> int | None:
    """This process's physical footprint as macOS counts it (Metal buffers included); None where it can't be read."""

    import ctypes

    class Usage(ctypes.Structure):      # rusage_info_v4 up to ri_phys_footprint, then the rest unread
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("counters", ctypes.c_uint64 * 7), ("footprint", ctypes.c_uint64),
                    ("rest", ctypes.c_uint64 * 27)]

    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        info = Usage()
        if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
            return None
    except (OSError, AttributeError):
        return None
    return int(info.footprint)


def model_fraction(package: Any, ram: int | None = None) -> float:
    """The share of RAM a family's model may use on this Mac: its stated allowance, else ``MEMORY_FRACTION``."""

    hook = getattr(package, "memory_fraction", None)
    allowance = hook(physical_memory_bytes() if ram is None else int(ram)) if callable(hook) else None
    return float(allowance) if allowance else MEMORY_FRACTION


def memory_limit_bytes(mx: Any, *, fraction: float = MEMORY_FRACTION,
                       environ: Mapping[str, str] | None = None, physical_bytes: int | None = None) -> int:
    """The explicit process budget or the model's share of RAM, capped by physical memory and the GPU's working set."""

    ram = physical_memory_bytes() if physical_bytes is None else int(physical_bytes)
    if ram <= 0:
        raise ValueError("physical memory must be positive")
    limit = int(fraction * ram)
    value = (os.environ if environ is None else environ).get(LIMIT_ENV)
    if value is not None:
        try:
            gib = float(value)
        except ValueError:
            raise ValueError(f"{LIMIT_ENV} must be a positive number in GiB") from None
        if not math.isfinite(gib) or gib <= 0:
            raise ValueError(f"{LIMIT_ENV} must be a positive number in GiB")
        limit = max(1, int(min(gib, ram / GIB) * GIB))
    return min(limit, budget_ceiling(mx, ram))


def budget_ceiling(mx: Any, physical_bytes: int | None = None) -> int:
    """The largest budget this Mac takes: physical RAM, capped by the GPU's recommended working set."""

    ram = physical_memory_bytes() if physical_bytes is None else int(physical_bytes)
    device_info = getattr(mx, "device_info", None) or getattr(getattr(mx, "metal", None), "device_info", None)
    recommended = int(device_info().get("max_recommended_working_set_size", 0)) if device_info else 0
    return min(ram, recommended) if recommended > 0 else ram


def raise_hint(need: int, ceiling: int) -> str:
    """How to give the process ``need`` bytes with the environment variable, or "" when this Mac can't."""

    if need >= ceiling:
        return ""
    return (f"Raise the budget past {need / GIB:.1f} GiB with {LIMIT_ENV} (this Mac takes up to {ceiling / GIB:.1f}; "
            "the default leaves the rest of RAM to other apps), or serve it")


def configure_mlx(mx: Any, cache_limit_bytes: int, *, reserve_bytes: int = PROCESS_BYTES, **kwargs: Any) -> int:
    """The process's memory budget, applied before weights load; MLX's buffers get it less ``reserve_bytes``."""

    budget = memory_limit_bytes(mx, **kwargs)
    cache = int(cache_limit_bytes)
    if cache < 0:
        raise ValueError("MLX cache limit must be nonnegative")
    limit = max(1, budget - int(reserve_bytes))
    mx.set_memory_limit(limit)
    mx.set_cache_limit(min(cache, limit))
    return budget


def _array_bytes(value: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    if isinstance(value, (list, tuple)):
        return sum(_array_bytes(v, seen) for v in value)
    if id(value) in seen:
        return 0
    seen.add(id(value))
    return max(0, int(getattr(value, "nbytes", 0) or 0))


def cache_nbytes(cache: Sequence[Any]) -> int:
    """Include allocated capacity and auxiliary arrays, which a cache's state views can omit."""

    total = 0
    for item in cache:
        total += _held_bytes(item)
    return total


def _held_bytes(item: Any) -> int:
    stored = _array_bytes(list(vars(item).values())) if hasattr(item, "__dict__") else 0
    return max(stored, _array_bytes(getattr(item, "state", None)), int(getattr(item, "nbytes", 0) or 0))


@dataclass(frozen=True)
class CacheMemory:
    fixed_bytes: int
    bytes_per_token: int
    step: int = 256
    entry_bytes_per_token: int = 0      # the largest growing entry's; 0: unknown, the whole cache grows at once

    def __post_init__(self) -> None:
        if min(self.fixed_bytes, self.bytes_per_token, self.entry_bytes_per_token) < 0 or self.step < 1:
            raise ValueError("cache sizes must be nonnegative and step positive")

    @classmethod
    def from_cache(cls, cache: Sequence[Any]) -> "CacheMemory":
        """Size a populated probe cache; bounded timelines stay fixed, alternating KV reserves both buffers."""

        fixed = per_token = entry = 0
        step = 256
        for item in cache:
            growth = getattr(item, "memory_growth", None)
            if callable(growth):                  # a cache that states its own (fixed bytes, bytes a token)
                item_fixed, item_per_token = growth()
                fixed, per_token = fixed + int(item_fixed), per_token + int(item_per_token)
                step = max(step, int(getattr(item, "step", 256)))
                continue
            held = _held_bytes(item)
            keys, values = getattr(item, "keys", None), getattr(item, "values", None)
            # an empty KV cache (keys and values both unset) has no size yet; an entry without values is not KV
            if keys is None and values is None and hasattr(item, "keys") and hasattr(item, "values"):
                raise ValueError("size KV memory from a populated probe cache")
            if getattr(keys, "ndim", 0) != 4 or getattr(values, "ndim", 0) != 4:
                fixed += held
                continue
            positions = int(keys.shape[2])
            if positions < 1 or int(values.shape[2]) != positions:
                raise ValueError("KV arrays must contain the same positive number of positions")
            main = _array_bytes(keys) + _array_bytes(values)
            each = -(-main // positions)
            spare = _array_bytes(getattr(item, "spare_keys", None))
            spare += _array_bytes(getattr(item, "spare_values", None))
            extra = max(0, held - main - spare)
            valid = max(1, min(positions, int(getattr(item, "offset", positions))))
            auxiliary = -(-extra // valid)
            capacity = int(getattr(item, "max_size", 0) or 0)
            if capacity:
                fixed += max(held, (each + auxiliary) * capacity)
                continue
            alternating = hasattr(item, "spare_keys")
            per_token += each * (2 if alternating else 1) + auxiliary
            entry = max(entry, each + auxiliary)
            step = max(step, int(getattr(item, "step", 256)),
                       int(getattr(item, "grow", 256)) if alternating else 256)
        return cls(fixed, per_token, step, entry)

    def cache_bytes(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("tokens must be nonnegative")
        positions = -(-int(tokens) // self.step) * self.step
        return self.fixed_bytes + positions * self.bytes_per_token

    def growth_bytes(self, tokens: int, in_flight: int = 2) -> int:
        """A prompt chunk's growth: each entry's old buffer lives until its copy lands, a few entries at a time."""

        positions = -(-int(tokens) // self.step) * self.step
        each = self.bytes_per_token if not self.entry_bytes_per_token else min(
            self.bytes_per_token, in_flight * self.entry_bytes_per_token)
        return self.fixed_bytes + positions * each


def needed_bytes(memory: CacheMemory, tokens: int, *, resident_bytes: int, working_bytes: int = 0,
                 cache_copies: int = 1, reserve_tokens: int = 0) -> int:
    """Projected bytes; resident excludes the new cache, work includes prompt temporaries and KV growth."""

    if min(tokens, resident_bytes, working_bytes, reserve_tokens) < 0 or cache_copies < 1:
        raise ValueError("memory sizes and tokens must be nonnegative, cache copies positive")
    return int(resident_bytes + working_bytes + cache_copies * memory.cache_bytes(tokens + reserve_tokens))


def fits(memory: CacheMemory, tokens: int, *, budget_bytes: int, **kwargs: Any) -> bool:
    return needed_bytes(memory, tokens, **kwargs) <= int(budget_bytes)


def largest_context(memory: CacheMemory, window_tokens: int, *, budget_bytes: int, **kwargs: Any) -> int:
    """Largest prompt in the full model window, leaving ``reserve_tokens`` for its reply."""

    if window_tokens < 0:
        raise ValueError("window tokens must be nonnegative")
    if not fits(memory, 0, budget_bytes=budget_bytes, **kwargs):
        return 0
    lo, hi = 0, max(0, int(window_tokens) - int(kwargs.get("reserve_tokens", 0)))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(memory, mid, budget_bytes=budget_bytes, **kwargs):
            lo = mid
        else:
            hi = mid - 1
    return lo


__all__ = ["PROBE_REPEATS", "PROCESS_BYTES", "CacheMemory", "budget_ceiling", "cache_nbytes", "configure_mlx", "fits",
           "largest_context", "memory_limit_bytes", "model_fraction", "needed_bytes", "process_footprint",
           "raise_hint"]
