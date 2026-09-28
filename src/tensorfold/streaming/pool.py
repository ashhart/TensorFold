"""A GPU-resident pool of expert slots, filled from the checkpoint by a host thread that answers each layer's routing."""

from __future__ import annotations

import os
import queue
import struct
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.streaming.checkpoint import headers

PARTS = ("weight", "scales", "biases")
_DTYPES = {"U32": mx.uint32, "BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32}
TIMEOUT_MS = 60_000          # a layer's loads finish long before this; past it the forward fails


@dataclass
class Source:
    """Where expert 0 of one tensor starts in a checkpoint file, and the bytes of one expert."""

    path: Path
    offset: int
    nbytes: int


@dataclass
class Slots:
    """One projection's pool: weight, scales and biases for every slot, in the resident stacks' layout."""

    weight: mx.array
    scales: mx.array
    biases: mx.array


def sources(model_dir: Path, names: dict[tuple, str]) -> tuple[dict, dict]:
    """Sources and each (proj, part)'s (shape, dtype) for keys (layer, proj, part) of stacks or (..., expert)."""

    shards = headers(model_dir)
    found, shapes = {}, {}
    for key, name in names.items():
        hits = [(p, base, h[name]) for p, (base, h) in shards.items() if name in h]
        if len(hits) != 1:
            raise ValueError(f"expert tensor {name}: expected in exactly one checkpoint file, found {len(hits)}")
        path, base, entry = hits[0]
        shape, (begin, end) = entry["shape"], entry["data_offsets"]
        stacked = len(key) == 3
        found[key] = Source(path, base + begin, (end - begin) // shape[0] if stacked else end - begin)
        spec = (tuple(shape[1:] if stacked else shape), entry["dtype"])
        if shapes.setdefault((key[1], key[2]), spec) != spec:
            raise ValueError(f"expert tensor {name}: shape or dtype differs between layers or experts")
    return found, shapes


def expert_nbytes(found: dict, layer: int) -> int:
    """Bytes of one expert of ``layer``: every projection's weight, scales and biases."""

    return sum(Streamer.source(found, layer, key[1], key[2], 0)[0].nbytes
               for key in found if key[0] == layer and (len(key) == 3 or key[3] == 0))


def _uncached(path: Path) -> int:
    """A read-only fd whose reads bypass the page cache (macOS F_NOCACHE) or skip readahead (Linux): the pool caches."""

    fd = os.open(path, os.O_RDONLY)
    if sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)       # 48: F_NOCACHE in <sys/fcntl.h>
    elif hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    return fd


class Streamer:
    """Slots [0, E): one layer's window (expert e in slot e) for prompt chunks; [E, S): an LRU for decode rows."""

    def __init__(self, found: dict, shapes: dict, *, layers: int, experts: int, top_k: int, slots: int,
                 box_rows: int, hostsync: Any, workers: int = 8) -> None:
        if slots < 2 * experts:
            raise ValueError(f"the expert pool needs a layer's {experts} window slots plus as many for decode, "
                             f"got {slots}")
        self.hs, self.found, self.layers, self.experts, self.top_k = hostsync, found, layers, experts, top_k
        arrays = []
        self.pool: dict[str, Slots] = {}
        for proj in sorted({key[1] for key in found}):
            parts = [mx.zeros((slots, *shapes[(proj, part)][0]), dtype=_DTYPES[shapes[(proj, part)][1]])
                     for part in PARTS]
            self.pool[proj] = Slots(*parts)
            arrays += parts
        self.slot_of = mx.zeros((layers * experts,), dtype=mx.int32)
        self.box = mx.zeros((max(box_rows * top_k, experts, 8),), dtype=mx.uint32)
        self.layer_ids = [mx.array([i] + [0] * 7, dtype=mx.int32) for i in range(layers)]
        mx.eval(*arrays, self.slot_of, self.box, *self.layer_ids)
        self.fds = {path: _uncached(path) for path in {s.path for s in found.values()}}
        self.channel = hostsync.Channel()
        self.value = 0
        self.window: list[int] = [-1] * experts                         # the layer whose expert sits in slot e
        self.lru: OrderedDict[tuple[int, int], int] = OrderedDict()     # (layer, expert) -> slot, oldest first
        self.free = list(range(slots - 1, experts - 1, -1))
        self.hits = self.misses = self.bytes_read = 0
        self.error: BaseException | None = None
        self._readers = ThreadPoolExecutor(workers, thread_name_prefix="expert-read")
        self._jobs: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._serve, name="expert-stream", daemon=True)
        self._thread.start()

    def marks(self, kind: str, layer: int, rows: int) -> tuple[int, int]:
        """(signal, ready) for one MoE call: "rows" reads rows x top_k ids from the box, "window" E presence flags."""

        if self.error is not None:
            raise RuntimeError("expert streaming stopped") from self.error
        signal, ready = self.value + 1, self.value + 2
        self.value = ready
        self._jobs.put((kind, layer, rows, signal, ready))
        return signal, ready

    def hold(self, token: mx.array, kind: str, layer: int, rows: int) -> mx.array:
        """``token`` once the GPU has signalled this call to the host and the host has answered it (graph marks)."""

        signal, ready = self.marks(kind, layer, rows)
        return self.hs.gpu_wait(self.hs.gpu_signal(token, self.channel, signal), self.channel, ready)

    def window_views(self) -> dict[str, Slots]:
        """Each projection's window slots [0, E) as stacks of the model's shape: expert e of the loaded layer."""

        return {proj: Slots(*(getattr(s, p)[:self.experts] for p in PARTS)) for proj, s in self.pool.items()}

    def _serve(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            kind, layer, rows, signal, ready = job
            try:
                if not self.channel.wait(signal, TIMEOUT_MS):
                    raise TimeoutError(f"no routing from the GPU for layer {layer}")
                if kind == "rows":
                    ids = np.frombuffer(self.hs.peek(self.box, 0, 4 * rows * self.top_k), dtype=np.uint32)
                    self._load(layer, sorted({int(e) for e in ids}))
                else:
                    flags = np.frombuffer(self.hs.peek(self.box, 0, 4 * self.experts), dtype=np.uint32)
                    self._fill_window(layer, np.flatnonzero(flags).tolist())
            except BaseException as error:        # the GPU still gets its release; the next call then raises
                self.error = error
            self.channel.signal(ready)

    def _load(self, layer: int, ids: list[int]) -> None:
        """Every id resident in the LRU for ``layer``, then its slot written into the table."""

        need, table = [], []
        for e in ids:
            if not 0 <= e < self.experts:
                raise ValueError(f"routed expert {e} is outside the model")
            slot = self.lru.pop((layer, e), None)
            if slot is None:
                slot = self.free.pop() if self.free else self._evict({(layer, i) for i in ids})
                need.append((e, slot))
                self.misses += 1
            else:
                self.hits += 1
            self.lru[(layer, e)] = slot
            table.append((e, slot))
        self._read_all(layer, need)
        for e, slot in table:
            self.hs.write_into(self.slot_of, 4 * (layer * self.experts + e), struct.pack("<i", slot))

    def _fill_window(self, layer: int, ids: list[int]) -> None:
        need = [(e, e) for e in ids if self.window[e] != layer]
        self._read_all(layer, need)
        for e, _ in need:
            self.window[e] = layer

    def _evict(self, keep: set) -> int:
        for key in self.lru:
            if key not in keep:
                return self.lru.pop(key)
        raise MemoryError("the expert pool is smaller than one call's experts")

    def _read_all(self, layer: int, need: list[tuple[int, int]]) -> None:
        reads = [(proj, part, e, slot) for e, slot in need for proj in self.pool for part in PARTS]
        list(self._readers.map(lambda item: self._read(layer, *item), reads))

    @staticmethod
    def source(found: dict, layer: int, proj: str, part: str, expert: int) -> tuple[Source, int]:
        """The tensor holding ``expert`` and the file offset of its bytes."""

        one = found.get((layer, proj, part, expert))
        if one is not None:
            return one, one.offset
        stack = found[(layer, proj, part)]
        return stack, stack.offset + expert * stack.nbytes

    def _read(self, layer: int, proj: str, part: str, expert: int, slot: int) -> None:
        src, offset = self.source(self.found, layer, proj, part, expert)
        self.hs.pread_into(getattr(self.pool[proj], part), slot * src.nbytes, self.fds[src.path], offset, src.nbytes)
        self.bytes_read += src.nbytes

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=5)
        self._readers.shutdown(wait=True)
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


__all__ = ["PARTS", "Slots", "Source", "Streamer", "expert_nbytes", "sources"]
