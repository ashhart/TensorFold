"""Read n-gram (PLE) rows from the checkpoint's files at each lookup, holding no table in memory."""

from __future__ import annotations

import os
import struct
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WORKERS = 16                # reads in flight at once: os.pread releases the GIL
MAX_READ = 1 << 20          # bytes one read of adjacent rows may cover
_KINDS = (("weight", "U32", 4), ("scales", "BF16", 2), ("biases", "BF16", 2))


def _no_cache(fd: int) -> None:
    """Disable caching on macOS; Linux FADV_RANDOM disables readahead but keeps buffered I/O."""

    if sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)       # 48: F_NOCACHE in <sys/fcntl.h>
    elif hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)


def _release(fds: list[int], pool: ThreadPoolExecutor) -> None:
    pool.shutdown(wait=True)
    while fds:
        os.close(fds.pop())


def _span(entry: object, kind: tuple[str, str, int], data: int, size: int, name: str) -> tuple[int, int, int]:
    """(rows, bytes a row, file offset of row 0) of one tensor; refuses a dtype, shape or range it can't read."""

    part, dtype, item = kind
    allowed = ("U32", "I32") if kind == _KINDS[0] else (dtype,)  # affine words are packed bits
    if not isinstance(entry, dict) or entry.get("dtype") not in allowed:
        expected = " or ".join(allowed)
        raise ValueError(f"{name}: the n-gram {part} must be a {expected} tensor")
    shape, span = entry.get("shape"), entry.get("data_offsets")
    if not (isinstance(shape, list) and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape)):
        raise ValueError(f"{name}: the n-gram {part} shape {shape} is not [rows, columns]")
    if not (isinstance(span, list) and len(span) == 2 and all(type(n) is int for n in span) and span[0] >= 0
            and span[1] - span[0] == shape[0] * shape[1] * item and data + span[1] <= size):
        raise ValueError(f"{name}: the n-gram {part} bytes {span} disagree with its shape or pass the file's end")
    return shape[0], shape[1] * item, data + span[0]


def _fill(reads: list[tuple[int, int, int, memoryview, int]]) -> None:
    """Each (fd, offset, size, out, at) read into ``out[at:]``, finishing short reads; end of file means it changed."""

    for fd, offset, size, out, at in reads:
        done = 0
        while done < size:
            data = os.pread(fd, size - done, offset + done)
            if not data:
                raise OSError(f"short read of the n-gram tables at byte {offset + done}: the checkpoint changed")
            out[at + done:at + done + len(data)] = data
            done += len(data)


class SSDRows:
    """Validated tensor rows read at each gather: deduplicated, coalesced and parallel, without memory maps."""

    def __init__(self, files: list[list[tuple[Path, dict]]], kinds: tuple[tuple[str, str, int], ...],
                 *, nocache: bool = True) -> None:
        self._fds: list[int] = []
        self._inline_rows = 0
        self._pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="ple-ssd")
        self._closer = weakref.finalize(self, _release, self._fds, self._pool)
        try:
            self._layout(files, kinds, nocache)
        except BaseException:
            self.close()
            raise

    def _layout(self, files: list[list[tuple[Path, dict]]], kinds: tuple[tuple[str, str, int], ...],
                nocache: bool) -> None:
        if not files:
            raise ValueError("the n-gram table has no shards")
        opened: dict[Path, tuple[int, int, int]] = {}
        starts, indices, bases, widths = [0], [], [], None
        total = 0
        for components in files:
            if len(components) != len(kinds):
                raise ValueError("an n-gram shard has missing tensor components")
            spans, shard_indices = [], []
            for (path, entry), kind in zip(components, kinds, strict=True):
                path = Path(path)
                if path not in opened:
                    fd = os.open(path, os.O_RDONLY)
                    self._fds.append(fd)
                    if nocache:
                        _no_cache(fd)
                    size, head = os.fstat(fd).st_size, os.pread(fd, 8, 0)
                    data = 8 + struct.unpack("<Q", head)[0] if len(head) == 8 else size + 1
                    if data > size:
                        raise ValueError(f"{path.name}: truncated safetensors header")
                    opened[path] = (len(self._fds) - 1, data, size)
                index, data, size = opened[path]
                spans.append(_span(entry, kind, data, size, path.name))
                shard_indices.append(index)
            rows = spans[0][0]
            row_widths = tuple(span[1] for span in spans)
            self._validate(rows, spans, Path(components[0][0]).name)
            if widths not in (None, row_widths):
                raise ValueError(f"{Path(components[0][0]).name}: the n-gram shards differ in row width")
            widths = row_widths
            total += rows * sum(widths)
            starts.append(starts[-1] + rows)
            indices.append(shard_indices)
            bases.append([span[2] for span in spans])
        self.starts = np.array(starts, dtype=np.int64)
        self.rows = int(self.starts[-1])
        self.fidx = np.array(indices, dtype=np.int64)     # [shard, component]: file containing its rows
        self.bases = np.array(bases, dtype=np.int64)      # [shard, component]: file offset of row 0
        self.widths = widths
        self.nbytes = total
        self._fd_of = np.array(self._fds, dtype=np.int64)

    def _validate(self, rows: int, spans: list[tuple[int, int, int]], name: str) -> None:
        """Refuse tensor components with different row counts."""

        if any(span[0] != rows for span in spans):
            raise ValueError(f"{name}: the n-gram components differ in row count")

    def gather(self, ids: np.ndarray) -> tuple[np.ndarray, ...]:
        """Rows ``ids`` (global) -> one [n, row bytes] uint8 array per tensor component."""

        if not self._closer.alive:
            raise ValueError("the n-gram table is closed")
        flat = np.asarray(ids).reshape(-1)
        if flat.size and flat.dtype.kind not in "iu":
            raise TypeError(f"n-gram row ids must be integers, not {flat.dtype}")
        if flat.size and (flat.min() < 0 or flat.max() >= self.rows):
            raise ValueError(f"n-gram row ids must lie in [0, {self.rows})")
        unique, inverse = np.unique(flat.astype(np.int64), return_inverse=True)
        shard = np.searchsorted(self.starts, unique, side="right") - 1
        local = unique - self.starts[shard]
        outs, reads = [], []
        for part, width in enumerate(self.widths):
            outs.append(np.empty((unique.size, width), dtype=np.uint8))
            reads += self._reads(self.fidx[shard, part], self.bases[shard, part] + local * width, width, outs[-1])
        batches = min(WORKERS, len(reads))
        if batches > 1 and unique.size > self._inline_rows:
            list(self._pool.map(_fill, [reads[i::batches] for i in range(batches)]))
        else:
            _fill(reads)
        return tuple(out[inverse] for out in outs)

    def _reads(self, where: np.ndarray, offsets: np.ndarray, width: int, out: np.ndarray) -> list[tuple]:
        """One read per run of rows adjacent in one file, each at most MAX_READ bytes, into ``out``'s rows."""

        n = offsets.size
        if not n:
            return []
        cut = np.ones(n, dtype=bool)
        cut[1:] = (where[1:] != where[:-1]) | (offsets[1:] != offsets[:-1] + width)
        first = np.flatnonzero(cut)
        cut |= (np.arange(n) - first[np.cumsum(cut) - 1]) % max(1, MAX_READ // width) == 0
        first = np.flatnonzero(cut)
        rows = np.diff(first, append=n)
        view = memoryview(out.reshape(-1))
        runs = zip(self._fd_of[where[first]].tolist(), offsets[first].tolist(), (rows * width).tolist(),
                   (first * width).tolist(), strict=True)
        return [(fd, offset + start, min(MAX_READ, size - start), view, at + start)
                for fd, offset, size, at in runs for start in range(0, size, MAX_READ)]

    def prefetch(self, workers: int = 8) -> float:
        """Nothing to warm: rows are read at each lookup, so this reads nothing (0 seconds)."""

        return 0.0

    def close(self) -> None:
        """Close the checkpoint's files and the read threads (also done at exit); a later gather raises."""

        self._closer()


class SSDTable(SSDRows):
    """HostTable's affine rows read from files at each gather, with the same words, scales and biases."""

    def __init__(self, files: list[tuple[Path, dict, dict, dict]], *, nocache: bool = True) -> None:
        super().__init__([[(path, entry) for entry in entries] for path, *entries in files], _KINDS,
                         nocache=nocache)
        self.wrow, self.grow, _ = self.widths

    def _validate(self, rows: int, spans: list[tuple[int, int, int]], name: str) -> None:
        """Require affine 4-bit rows with scales and biases every 32 values."""

        (_, wrow, _), (srows, grow, _), (brows, brow, _) = spans
        if not rows == srows == brows or wrow != 8 * grow or brow != grow:
            raise ValueError(f"{name}: an n-gram shard is not 4-bit rows with a scale and bias every 32")

    def gather(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows ``ids`` -> uint32 words and uint16 bf16 scale and bias bits."""

        words, scales, biases = super().gather(ids)
        return words.view(np.uint32), scales.view(np.uint16), biases.view(np.uint16)
