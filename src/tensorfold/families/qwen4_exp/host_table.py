"""Host n-gram shards for CUDA and for Metal past GPU memory: memory-mapped here, or read from disk by SSDTable."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

from tensorfold.families.qwen4_exp.ssd_table import SSDTable

_PARTS = ("weight", "scales", "biases")


class HostTable:
    """Keep n-gram shards memory-mapped on the host; gather copies only requested rows, never whole tables to the GPU."""

    def __init__(self, files: list[tuple[Path, dict, dict, dict]]) -> None:
        self.words, self.scales, self.biases, starts = [], [], [], [0]
        maps: dict = {}
        fidx, wbase, sbase, bbase = [], [], [], []
        for path, hw, hs, hb in files:
            self.words.append(_memmap(path, hw, np.uint32))
            self.scales.append(_memmap(path, hs, np.uint16))
            self.biases.append(_memmap(path, hb, np.uint16))
            starts.append(starts[-1] + self.words[-1].shape[0])
            if path not in maps:
                with open(path, "rb") as f:
                    data = 8 + struct.unpack("<Q", f.read(8))[0]
                maps[path] = (len(maps), np.memmap(path, dtype=np.uint8, mode="r"), data)
            index, _, data = maps[path]
            fidx.append(index)
            wbase.append(data + hw["data_offsets"][0])
            sbase.append(data + hs["data_offsets"][0])
            bbase.append(data + hb["data_offsets"][0])
        self.starts = np.array(starts, dtype=np.int64)
        self.rows = int(self.starts[-1])
        # byte views of the files, so a gather is one fancy index per file and component, not per shard
        self.files = [m for _, m, _ in sorted(maps.values(), key=lambda t: t[0])]
        self.fidx = np.array(fidx, dtype=np.int64)
        self.wbase, self.sbase, self.bbase = (np.array(x, dtype=np.int64) for x in (wbase, sbase, bbase))
        self.wrow = self.words[0].shape[1] * 4
        self.grow = self.scales[0].shape[1] * 2

    def gather(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows ``ids`` (global) -> words [n, W] uint32, scales and biases [n, G] (bf16 bits as uint16)."""

        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        n = len(flat)
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        local = flat - self.starts[shard]
        where = self.fidx[shard]
        wo = self.wbase[shard] + local * self.wrow
        so = self.sbase[shard] + local * self.grow
        bo = self.bbase[shard] + local * self.grow
        w = np.empty((n, self.wrow), dtype=np.uint8)
        sc = np.empty((n, self.grow), dtype=np.uint8)
        bi = np.empty((n, self.grow), dtype=np.uint8)
        aw, ag = np.arange(self.wrow), np.arange(self.grow)
        for f in np.unique(where):
            at = np.nonzero(where == f)[0]
            mm = self.files[f]
            w[at] = mm[wo[at, None] + aw]
            sc[at] = mm[so[at, None] + ag]
            bi[at] = mm[bo[at, None] + ag]
        return w.view(np.uint32), sc.view(np.uint16), bi.view(np.uint16)

    def lock(self) -> bool:
        """Pin every shard's pages (mlock); False, with nothing locked, where the memory-lock limit forbids it."""

        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.mlock.argtypes = libc.munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        done = []
        for arr in self.words + self.scales + self.biases:
            at, size = arr.ctypes.data, arr.nbytes
            if libc.mlock(at, size) != 0:
                for a, n in done:
                    libc.munlock(a, n)
                return False
            done.append((at, size))
        return True

    def prefetch(self, workers: int = 8) -> float:
        """Read every shard once so the lookups hit the page cache (seconds taken); the pages stay evictable."""

        import time
        from concurrent.futures import ThreadPoolExecutor

        def touch(arr) -> None:
            flat = arr.reshape(-1).view(np.uint8)
            step = 64 << 20
            for i in range(0, flat.size, step):
                np.asarray(flat[i:i + step]).sum(dtype=np.uint64)

        t0 = time.time()
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(touch, self.words + self.scales + self.biases))
        return time.time() - t0


class ReadAhead:
    """A host table whose rows for a coming lookup are read on a thread while the GPU works: the same bytes, sooner."""

    depth = 2                           # lookups read ahead at most (the next prompt chunk, and the one after)

    def __init__(self, table: Any) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.table = table
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="ngram-read-ahead")
        self._ahead: dict[bytes, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name == "table":
            raise AttributeError(name)
        return getattr(self.table, name)

    def read_ahead(self, ids: np.ndarray) -> None:
        """Start reading rows ``ids`` for a lookup of the same ids to take."""

        key = _key(ids)
        if key not in self._ahead:
            while len(self._ahead) >= self.depth:
                self._ahead.pop(next(iter(self._ahead)))
            self._ahead[key] = self._pool.submit(self.table.gather, ids)

    def gather(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ahead = self._ahead.pop(_key(ids), None)
        return ahead.result() if ahead is not None else self.table.gather(ids)


def _key(ids: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(ids, dtype=np.int64).reshape(-1)).tobytes()


def _memmap(path: Path, entry: dict, dtype) -> np.ndarray:
    with open(path, "rb") as f:
        header = struct.unpack("<Q", f.read(8))[0]
    begin, end = entry["data_offsets"]
    shape = tuple(entry["shape"])
    return np.memmap(path, dtype=dtype, mode="r", offset=8 + header + begin, shape=shape)


def read_header(path: Path) -> dict:
    """A safetensors file's JSON header, refusing a truncated or malformed one."""

    with open(path, "rb") as f:
        head = f.read(8)
        n = struct.unpack("<Q", head)[0] if len(head) == 8 else -1
        if not 0 <= n <= min(100 << 20, os.fstat(f.fileno()).st_size - 8):
            raise ValueError(f"{Path(path).name}: truncated or invalid safetensors header")
        header = json.loads(f.read(n))
    if not isinstance(header, dict):
        raise ValueError(f"{Path(path).name}: the safetensors header is not a JSON object")
    return header


def from_checkpoint(model_dir: Path, name: str, count: int, *, ssd: bool = False) -> HostTable | SSDTable:
    """Shards ``{name}.shard_{i}``, i < count, each in one file: memory-mapped, or with ``ssd`` read at each lookup."""

    headers = {path: read_header(path) for path in sorted(Path(model_dir).glob("model*.safetensors"))}
    files = []
    for i in range(count):
        key = f"{name}.shard_{i}"
        found = [(path, h) for path, h in headers.items() if any(f"{key}.{part}" in h for part in _PARTS)]
        if len(found) != 1 or not all(f"{key}.{part}" in found[0][1] for part in _PARTS):
            raise ValueError(f"{key}: expected its weight, scales and biases together in one checkpoint file")
        path, h = found[0]
        files.append((path, *(h[f"{key}.{part}"] for part in _PARTS)))
    return SSDTable(files) if ssd else HostTable(files)
