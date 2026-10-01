"""The native Engram row reader (C++, pread from a thread pool)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def reader():
    from tensorfold.cuda.build import load

    return load("tensorfold_dsv41_rowread_v4", [str(Path(__file__).with_name("rowread.cpp"))],
                extra_cflags=["-O3"])
