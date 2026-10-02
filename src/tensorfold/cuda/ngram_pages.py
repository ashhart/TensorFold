"""Page-aligned n-gram pinning and advice without changing mapped data or gathered rows."""

from __future__ import annotations

import ctypes
import functools
import mmap
import os


PAGE = mmap.PAGESIZE
RUN_BYTES = 1 << 30


def span(address: int, size: int) -> tuple[int, int]:
    if size <= 0:
        return 0, 0
    return address // PAGE * PAGE, -(-(address + size) // PAGE) * PAGE


def merge(ranges) -> list[tuple[int, int]]:
    out = []
    for begin, end in sorted(ranges):
        if begin >= end:
            continue
        if out and begin <= out[-1][1]:
            out[-1] = (out[-1][0], max(end, out[-1][1]))
        else:
            out.append((begin, end))
    return out


def without(begin: int, end: int, pinned) -> list[tuple[int, int]]:
    out, cursor = [], begin
    for left, right in pinned:
        if right <= cursor:
            continue
        if left >= end:
            break
        if left > cursor:
            out.append((cursor, min(left, end)))
        cursor = max(cursor, right)
    if cursor < end:
        out.append((cursor, end))
    return out


def arrays(table):
    return list(table.values) if hasattr(table, "values") else list(table.words) + list(table.scales) + list(table.biases)


def lock_bytes(table) -> int:
    return sum(end - begin for begin, end in merge(span(int(a.ctypes.data), int(a.nbytes)) for a in arrays(table)))


@functools.lru_cache(maxsize=1)
def libc():
    if os.name == "nt":
        return None
    try:
        api = ctypes.CDLL(None, use_errno=True)
        api.mlock.argtypes = api.munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        return api
    except (AttributeError, OSError):
        return None


class Pins:
    def __init__(self) -> None:
        self.ranges: list[tuple[int, int]] = []
        self.error: int | None = None

    @property
    def nbytes(self) -> int:
        return sum(end - begin for begin, end in self.ranges)

    def all(self, words) -> bool:
        before = list(self.ranges)
        needed = sum(end - begin for begin, end in merge(span(int(a.ctypes.data), int(a.nbytes)) for a in words))
        self.runs(words, needed - self.nbytes)
        if self.nbytes == needed:
            return True
        api = libc()
        retained = []
        if api is not None:
            for begin, end in self.ranges:
                for left, right in without(begin, end, before):
                    if api.munlock(left, right - left) != 0:
                        retained.append((left, right))
                        self.error = ctypes.get_errno()
        self.ranges = merge([*before, *retained])
        return False

    def runs(self, words, budget: int, run_bytes: int = RUN_BYTES) -> int:
        api = libc()
        if api is None or budget <= 0:
            return 0
        added = 0
        for array in words:
            if not array.flags.c_contiguous or array.strides[0] <= 0:
                raise ValueError("n-gram pinning requires contiguous positive-stride rows")
            step = max(1, run_bytes // array.strides[0])
            for row in range(0, array.shape[0], step):
                run = array[row:row + step]
                begin, end = span(int(run.ctypes.data), int(run.nbytes))
                missing = without(begin, end, self.ranges)
                cost = sum(right - left for left, right in missing)
                if cost > budget - added:
                    return added
                done = []
                for left, right in missing:
                    if api.mlock(left, right - left) != 0:
                        self.error = ctypes.get_errno()
                        retained = []
                        for past, last in done:
                            if api.munlock(past, last - past) != 0:
                                retained.append((past, last))
                        self.ranges = merge([*self.ranges, *retained])
                        return added + sum(last - past for past, last in retained)
                    done.append((left, right))
                self.ranges = merge([*self.ranges, *done])
                added += cost
        return added
