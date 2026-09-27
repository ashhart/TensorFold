"""Memory-map n-gram shards on the host for CUDA and for Metal checkpoints that exceed GPU memory, reading only requested rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class HostTable:
    """Keep n-gram shards memory-mapped on the host; gather copies only requested rows, never whole tables to the GPU."""

    def __init__(self, files: list[tuple[Path, dict, dict, dict]]) -> None:
        import struct

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


def _memmap(path: Path, entry: dict, dtype) -> np.ndarray:
    import struct

    with open(path, "rb") as f:
        header = struct.unpack("<Q", f.read(8))[0]
    begin, end = entry["data_offsets"]
    shape = tuple(entry["shape"])
    return np.memmap(path, dtype=dtype, mode="r", offset=8 + header + begin, shape=shape)


def read_header(path: Path) -> dict:
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def from_checkpoint(model_dir: Path, name: str, count: int) -> HostTable:
    """The table of shards ``{name}.shard_{i}``, i < count (each shard's words, scales and biases in one file)."""

    files, headers = [], {}
    for path in sorted(Path(model_dir).glob("model*.safetensors")):
        headers[path] = read_header(path)
    for i in range(count):
        key = f"{name}.shard_{i}"
        path = next(p for p, h in headers.items() if key + ".weight" in h)
        h = headers[path]
        files.append((path, h[key + ".weight"], h[key + ".scales"], h[key + ".biases"]))
    return HostTable(files)
