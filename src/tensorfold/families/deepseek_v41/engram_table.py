"""Engram rows read from the checkpoint's own files with buffered pread: never resident, never mmap'd, never pinned."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import weakref
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tensorfold.families.qwen4_exp.ssd_table import _fill, _no_cache

WORKERS = 32                # reads in flight at once: os.pread releases the GIL
PAGE = 4096
_DTYPES = {"weight": "F8_E4M3", "scale": "F8_E8M0"}


@dataclass(frozen=True)
class Table:
    """One Engram layer's table: row ``i`` is at ``weight_abs + wrow*i`` and ``scale_abs + srow*i`` in ``path``."""

    layer: int
    path: Path
    size: int
    weight_abs: int
    scale_abs: int
    rows: int
    wrow: int
    srow: int
    bounds: tuple[int, ...]     # first row of each hash column, then ``rows``

    @classmethod
    def parse(cls, layer: int, path: Path, header: dict, data: int, size: int, buckets: Sequence[int]) -> Table:
        """The table of ``layers.{layer}.engram.embed`` from a safetensors header whose tensor bytes start at ``data``."""

        spans = []
        for part, dtype in _DTYPES.items():
            name = f"layers.{layer}.engram.embed.{part}"
            entry = header.get(name)
            if not isinstance(entry, dict) or entry.get("dtype") != dtype:
                raise ValueError(f"{path.name}: {name} must be a {dtype} tensor")
            shape, span = entry.get("shape"), entry.get("data_offsets")
            if not (isinstance(shape, list) and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape)):
                raise ValueError(f"{path.name}: {name} shape {shape} is not [rows, columns]")
            if not (isinstance(span, list) and len(span) == 2 and all(type(n) is int for n in span) and span[0] >= 0
                    and span[1] - span[0] == shape[0] * shape[1]):
                raise ValueError(f"{path.name}: {name} bytes {span} disagree with its shape")
            if data + span[1] > size:
                raise ValueError(f"{path.name}: {name} ends at byte {data + span[1]}, past the file's {size}")
            spans.append((shape, data + span[0]))
        (rows, wrow), weight_abs = spans[0]
        (srows, srow), scale_abs = spans[1]
        if srows != rows or wrow != 32 * srow:
            raise ValueError(f"{path.name}: layer {layer}'s Engram scales are not one per 32 bytes of each row")
        bounds = tuple(int(n) for n in np.cumsum([0, *buckets]))
        if bounds[-1] != rows or min(buckets, default=0) <= 0:
            raise ValueError(f"{path.name}: layer {layer}'s hash buckets cover {bounds[-1]} rows, the table {rows}")
        return cls(layer, path, size, weight_abs, scale_abs, rows, wrow, srow, bounds)


def _header(path: Path) -> tuple[bytes, int, int]:
    """(header bytes, absolute data start, file size) of one safetensors file."""

    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        head = f.read(8)
        length = struct.unpack("<Q", head)[0] if len(head) == 8 else size
        if 8 + length > size:
            raise ValueError(f"{path.name}: truncated safetensors header")
        return f.read(length), 8 + length, size


@dataclass(frozen=True)
class Layout:
    """The Engram tables in layer order; a global row is ``starts[i]`` plus a row of table ``i``."""

    tables: tuple[Table, ...]
    shards: tuple[tuple[int, bytes], ...]       # (file size, header sha256) of each Engram file
    starts: tuple[int, ...]

    @classmethod
    def read(cls, model_dir: str | Path, layer_ids: Sequence[int], buckets: Sequence[Sequence[int]]) -> Layout:
        """Tables from the index and shard headers, byte ranges checked; ``buckets[i]`` are layer i's primes."""

        model_dir = Path(model_dir)
        weight_map = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        if len(buckets) != len(layer_ids):
            raise ValueError(f"{len(buckets)} bucket lists for {len(layer_ids)} Engram layers")
        tables, shards, seen = [], [], {}
        for layer, sizes in zip(layer_ids, buckets):
            files = {weight_map.get(f"layers.{layer}.engram.embed.{part}") for part in _DTYPES}
            if None in files or len(files) != 1:
                raise ValueError(f"layer {layer}'s Engram weight and scale must sit in one listed file")
            path = (model_dir / files.pop()).resolve()
            if path not in seen:
                raw, data, size = _header(path)
                seen[path] = (json.loads(raw), data, size)
                shards.append((size, hashlib.sha256(raw).digest()))
            header, data, size = seen[path]
            tables.append(Table.parse(layer, path, header, data, size, sizes))
        starts = tuple(int(n) for n in np.cumsum([0, *(t.rows for t in tables)]))
        return cls(tuple(tables), tuple(shards), starts)

    def digest(self) -> int:
        """sha256 over each Engram file's (size, header sha256), as a signed int64 for the ranks' agreement vector."""

        h = hashlib.sha256()
        for size, sha in self.shards:
            h.update(struct.pack("<Q", size) + sha)
        return int.from_bytes(h.digest()[:8], "big", signed=True)

    def table(self, layer: int) -> Table:
        for t in self.tables:
            if t.layer == layer:
                return t
        raise ValueError(f"layer {layer} has no Engram table here")


def _release(fds: list[int], pools: tuple[ThreadPoolExecutor, ...]) -> None:
    for pool in pools:
        pool.shutdown(wait=True)
    while fds:
        os.close(fds.pop())


class Reader:
    """Rows by global id into caller buffers: deduplicated, batched per worker, read through the page cache."""

    def __init__(self, layout: Layout, workers: int = WORKERS) -> None:
        self.layout = layout
        self.workers = workers
        self._fds: list[int] = []
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="engram-read")
        self._async = ThreadPoolExecutor(1, thread_name_prefix="engram-gather")   # gathers wait on _pool
        self._closer = weakref.finalize(self, _release, self._fds, (self._async, self._pool))
        if len({(t.wrow, t.srow) for t in layout.tables}) != 1:
            raise ValueError("the Engram tables differ in row width")
        self.wrow, self.srow = layout.tables[0].wrow, layout.tables[0].srow
        opened: dict[Path, int] = {}
        try:
            for t in layout.tables:
                if t.path not in opened:
                    fd = os.open(t.path, os.O_RDONLY)
                    self._fds.append(fd)
                    _no_cache(fd)
                    if os.fstat(fd).st_size != t.size:
                        raise OSError(f"{t.path.name} changed size since its layout was read")
                    opened[t.path] = len(self._fds) - 1
        except BaseException:
            self.close()
            raise
        self._starts = np.array(layout.starts, dtype=np.int64)
        self._file = np.array([opened[t.path] for t in layout.tables], dtype=np.int64)
        self._fd_of = np.array(self._fds, dtype=np.int64)
        self._base = np.array([(t.weight_abs, t.scale_abs) for t in layout.tables], dtype=np.int64)

    def _locate(self, ids: np.ndarray, unique: bool) -> tuple:
        """(file indices, weight offsets, scale offsets, inverse) of each id, or of the distinct ids when ``unique``."""

        if not self._closer.alive:
            raise ValueError("the Engram reader is closed")
        ids = np.asarray(ids)
        if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu"):
            raise TypeError(f"Engram row ids must be a 1-D integer array, not {ids.dtype} {ids.shape}")
        if ids.size and (ids.min() < 0 or ids.max() >= self._starts[-1]):
            raise ValueError(f"Engram row ids must lie in [0, {self._starts[-1]})")
        ids = ids.astype(np.int64)
        inverse = None
        if unique:
            ids, inverse = np.unique(ids, return_inverse=True)
        table = np.searchsorted(self._starts, ids, side="right") - 1
        local = ids - self._starts[table]
        base = self._base[table]
        return self._file[table], base[:, 0] + local * self.wrow, base[:, 1] + local * self.srow, inverse

    def gather(self, ids: np.ndarray, out_w: np.ndarray, out_s: np.ndarray) -> None:
        """Rows ``ids`` (int [n], global) into ``out_w`` (u8 [n, wrow]) and ``out_s`` (u8 [n, srow]); views allowed."""

        files, woff, soff, inverse = self._locate(ids, unique=True)
        for out, width in ((out_w, self.wrow), (out_s, self.srow)):
            if out.dtype != np.uint8 or out.shape != (inverse.size, width):
                raise ValueError(f"{inverse.size} Engram rows go into a uint8 [n, {width}] buffer, not "
                                 f"{out.dtype} {out.shape}")
        fds = self._fd_of[files]
        rows_w = np.empty((fds.size, self.wrow), dtype=np.uint8)
        rows_s = np.empty((fds.size, self.srow), dtype=np.uint8)
        reads = []
        for offsets, width, rows in ((woff, self.wrow, rows_w), (soff, self.srow, rows_s)):
            view = memoryview(rows.reshape(-1))
            reads += zip(fds.tolist(), offsets.tolist(), [width] * fds.size, [view] * fds.size,
                         range(0, fds.size * width, width))
        batches = min(self.workers, len(reads))
        if batches > 1:
            list(self._pool.map(_fill, [reads[i::batches] for i in range(batches)]))
        else:
            _fill(reads)
        np.take(rows_w, inverse, axis=0, out=out_w)
        np.take(rows_s, inverse, axis=0, out=out_s)

    def gather_async(self, ids: np.ndarray, out_w: np.ndarray, out_s: np.ndarray) -> Future:
        """``gather`` on a background thread; the caller keeps the buffers untouched until the Future is done."""

        if not self._closer.alive:
            raise ValueError("the Engram reader is closed")
        return self._async.submit(self.gather, ids, out_w, out_s)

    def advise(self, ids: np.ndarray) -> None:
        """Ask the kernel to start reading the 4 KiB pages of rows ``ids`` (POSIX_FADV_WILLNEED, merged spans)."""

        if not hasattr(os, "posix_fadvise"):
            return
        files, woff, soff, _ = self._locate(ids, unique=False)
        for f, fd in enumerate(self._fds):
            mine = files == f
            begin = np.concatenate([woff[mine], soff[mine]])
            if not begin.size:
                continue
            end = np.concatenate([woff[mine] + self.wrow, soff[mine] + self.srow])
            order = np.argsort(begin)
            begin = begin[order] // PAGE * PAGE
            end = -(-np.maximum.accumulate(end[order]) // PAGE) * PAGE
            first = np.flatnonzero(np.r_[True, begin[1:] > end[:-1]])
            for lo, hi in zip(begin[first].tolist(), end[np.r_[first[1:] - 1, begin.size - 1]].tolist()):
                os.posix_fadvise(fd, lo, hi - lo, os.POSIX_FADV_WILLNEED)

    def close(self) -> None:
        """Close the files and the read threads (also done at exit); a later gather raises."""

        self._closer()
