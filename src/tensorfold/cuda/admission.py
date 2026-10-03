"""Whether a request may start: ``MemoryGate``'s room and, on a unified-memory GPU, host memory above a floor."""
# On a GB10 the GPU allocates host memory: MemFree, idle allocator blocks and clean unmapped page cache are its room.

from __future__ import annotations

from collections.abc import Callable

from .memory_gate import MemoryGate

GIB = 1 << 30
KEYS = ("MemTotal", "MemFree", "MemAvailable", "Dirty", "Writeback", "Mapped")


def read_meminfo(path: str = "/proc/meminfo") -> dict[str, int]:
    """Bytes of the ``/proc/meminfo`` fields admission reads; {} where there is none."""

    out = {}
    try:
        with open(path) as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if key in KEYS and parts:
                    out[key] = int(parts[0]) * (1024 if parts[1:] == ["kB"] else 1)
    except (OSError, ValueError):
        return {}
    return out


def page_cache_credit(mi: dict[str, int], keep: int = 2 * GIB) -> int:
    """Page cache an allocation can take now: MemAvailable less MemFree, less dirty pages and max(Mapped, ``keep``)."""

    if "MemAvailable" not in mi or "MemFree" not in mi:
        return 0
    extra = mi["MemAvailable"] - mi["MemFree"] - mi.get("Dirty", 0) - mi.get("Writeback", 0)
    return max(0, extra - max(mi.get("Mapped", 0), int(keep)))


class Admission:
    """``fits(need)``: the gate has room and, on unified memory, ``need`` leaves ``hard_floor`` bytes usable."""

    def __init__(
        self,
        gate: MemoryGate | None = None,
        *,
        floor: int = 5 * GIB,
        hard_floor: int = 4 * GIB,
        meminfo: Callable[[], dict] = read_meminfo,
        unified: bool = False,
        idle: Callable[[], int] | None = None,
        trim: Callable[[], None] | None = None,
        keep: int = 2 * GIB,
    ) -> None:
        if hard_floor > floor:
            raise ValueError("the hard floor is above the floor")
        self.gate, self.floor, self.hard_floor = gate, int(floor), int(hard_floor)
        self.meminfo, self.unified, self.idle, self.trim_fn, self.keep = meminfo, bool(unified), idle, trim, int(keep)
        self.waits = self.trims = 0

    def usable(self) -> int | None:
        """Bytes the GPU can still allocate from host memory; None on a discrete GPU or without meminfo."""

        if not self.unified:
            return None
        mi = self.meminfo()
        if "MemFree" not in mi:
            return None
        idle = int(self.idle()) if self.idle is not None else 0
        return mi["MemFree"] + idle + page_cache_credit(mi, self.keep)

    def fits(self, need: int) -> bool:
        ok = self.gate is None or self.gate.fits(need)
        usable = self.usable() if ok else None
        ok = ok and (usable is None or usable - int(need) >= self.hard_floor)
        self.waits += not ok
        return ok

    def low(self) -> bool:
        """Under the floor now: requests still start, and the server should say memory is tight."""

        usable = self.usable()
        return usable is not None and usable < self.floor

    def why(self, need: int) -> str:
        """The line a waiting request logs."""

        parts = []
        if self.gate is not None and not self.gate.fits(need):
            parts.append(f"stream caches hold {self.gate.held / GIB:.2f} of {self.gate.room / GIB:.2f} GiB")
        usable = self.usable()
        if usable is not None and usable - int(need) < self.hard_floor:
            parts.append(
                f"{usable / GIB:.2f} GiB usable less {int(need) / GIB:.2f} GiB is under the "
                f"{self.hard_floor / GIB:.2f} GiB floor"
            )
        return "waiting for memory: " + ("; ".join(parts) or "room was short a moment ago")

    def trim(self) -> int:
        """Give the allocator's idle blocks back to the host (they count as usable already); returns the bytes."""

        if self.trim_fn is None or self.idle is None:
            return 0
        before = int(self.idle())
        if before <= 0:
            return 0
        self.trim_fn()
        self.trims += 1
        return before


def for_torch(torch, gate: MemoryGate | None = None, **kw) -> Admission:
    """An ``Admission`` reading this process's CUDA allocator (unified when the device says it is integrated)."""

    from .capacity import unified

    def idle() -> int:
        return int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated())

    return Admission(gate, unified=unified(torch), idle=idle, trim=torch.cuda.empty_cache, **kw)


__all__ = ["Admission", "for_torch", "page_cache_credit", "read_meminfo"]
