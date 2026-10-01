"""Host n-gram shards for CUDA and for Metal past GPU memory: memory-mapped here, or read from disk by SSDTable."""

from __future__ import annotations

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tensorfold.families.qwen4_exp.ssd_table import SSDTable

_PARTS = ("weight", "scales", "biases")
# a prompt chunk's gather copies big row runs on worker threads (GIL released); bytes stay the same as single-threaded
GATHER_THREADS = 16
GATHER_SPLIT = 512


def ngrams_on_host(model_dir: Path, ssd: bool = False) -> bool:
    """Host n-gram tables when read from SSD, else past the GPU working-set threshold (TF_NGRAM_HOST=0/1 overrides)."""

    flag = os.environ.get("TF_NGRAM_HOST", "")
    if ssd:
        if flag == "0":
            raise ValueError("--ple-on-ssd reads the n-gram tables on the host: unset TF_NGRAM_HOST=0")
        return True
    if flag in ("0", "1"):
        return flag == "1"
    import mlx.core as mx

    size = sum(p.stat().st_size for p in Path(model_dir).glob("model*.safetensors"))
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return size > 0.75 * int(info["max_recommended_working_set_size"])


def windows_lock_pages(arrays, kernel32=None):
    """Best effort page pinning on Windows: VirtualLock answers False when it refuses, and nothing stays pinned."""

    import ctypes

    try:
        api = kernel32 if kernel32 is not None else ctypes.WinDLL("kernel32", use_last_error=True)
        pinned = []
        for array in arrays:
            address, size = ctypes.c_void_p(array.ctypes.data), ctypes.c_size_t(array.nbytes)
            if not api.VirtualLock(address, size):
                for past, past_size in pinned:
                    api.VirtualUnlock(past, past_size)
                return False
            pinned.append((address, size))
        return True
    except (AttributeError, OSError):      # no kernel32 here either: the tables simply stay unpinned
        return False


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
                view = np.memmap(path, dtype=np.uint8, mode="r")
                _random_access(view)          # gathers read through this view: a fault reads its page, not those around
                maps[path] = (len(maps), view, data)
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
        self.nbytes = sum(a.nbytes for a in self.words + self.scales + self.biases)
        # threads start with the first threaded gather
        self._pool = ThreadPoolExecutor(GATHER_THREADS, thread_name_prefix="ngram-gather")

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

        def copy(job) -> None:
            mm, at = job
            w[at] = mm[wo[at, None] + aw]
            sc[at] = mm[so[at, None] + ag]
            bi[at] = mm[bo[at, None] + ag]

        if GATHER_THREADS > 1 and n >= 2 * GATHER_SPLIT:          # a prompt chunk: copy on threads
            jobs = []
            for f in np.unique(where):
                at = np.nonzero(where == f)[0]
                parts = max(1, min(GATHER_THREADS, len(at) // GATHER_SPLIT))
                jobs += [(self.files[f], piece) for piece in np.array_split(at, parts)]
            list(self._pool.map(copy, jobs))
        else:
            for f in np.unique(where):
                copy((self.files[f], np.nonzero(where == f)[0]))
        return w.view(np.uint32), sc.view(np.uint16), bi.view(np.uint16)

    def lock(self) -> bool:
        """Pin every shard's pages (mlock); False, with nothing locked, where the memory-lock limit forbids it."""

        import ctypes

        if os.name == "nt":
            return windows_lock_pages(self.words + self.scales + self.biases)
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

        return _prefetch(self.words + self.scales + self.biases, workers)


class BF16Table:
    """bf16 n-gram shards (the NVFP4 checkpoint's): memory-mapped, gathered a lookup at a time as bf16 bits."""

    bits = 16

    def __init__(self, files: list[tuple[Path, dict]]) -> None:
        self.values, starts = [], [0]
        for path, weight in files:
            if not isinstance(weight, dict) or weight.get("dtype") != "BF16":
                raise ValueError(f"{Path(path).name}: the n-gram weights must be BF16 tensors")
            self.values.append(_memmap(path, weight, np.uint16))
            if self.values[-1].shape[1] != self.values[0].shape[1]:
                raise ValueError(f"{Path(path).name}: the n-gram shards differ in row width")
            starts.append(starts[-1] + self.values[-1].shape[0])
        self.starts = np.array(starts, dtype=np.int64)
        self.rows = int(self.starts[-1])
        self.width = int(self.values[0].shape[1])       # bf16 values a row (the engine's ``dh``)
        self.wrow = self.width * 2                      # bytes a row
        self.nbytes = sum(a.nbytes for a in self.values)
        self._pool = ThreadPoolExecutor(GATHER_THREADS, thread_name_prefix="ngram-gather")

    def _where(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each id's shard and row in it, after checking the ids lie in the table."""

        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        if flat.size and (flat.min() < 0 or flat.max() >= self.rows):
            raise ValueError(f"n-gram row ids must lie in [0, {self.rows})")
        shard = np.searchsorted(self.starts, flat, side="right") - 1
        return shard, flat - self.starts[shard]

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Rows ``ids`` (global) -> [n, W] uint16 (the bf16 bits of W values)."""

        shard, local = self._where(ids)
        out = np.empty((shard.size, self.width), dtype=np.uint16)

        def copy(f: int, at: np.ndarray) -> None:
            out[at] = self.values[f][local[at]]

        _copy_rows(self._pool, shard, copy)
        return out

    def lock(self) -> bool:
        """Pin every shard's pages (mlock); False, with nothing locked, where the memory-lock limit forbids it."""

        import ctypes

        if os.name == "nt":
            return windows_lock_pages(self.values)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mlock.argtypes = libc.munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        done = []
        for arr in self.values:
            at, size = arr.ctypes.data, arr.nbytes
            if libc.mlock(at, size) != 0:
                for a, n in done:
                    libc.munlock(a, n)
                return False
            done.append((at, size))
        return True

    def prefetch(self, workers: int = 8) -> float:
        """Read every shard once so the lookups hit the page cache (seconds taken); the pages stay evictable."""

        return _prefetch(self.values, workers)


class FP8Table(BF16Table):
    """e4m3 n-gram shards with one table scale: lookups give bf16(e4m3 x scale) through a 256-entry table."""

    def __init__(self, files: list[tuple[Path, dict]], scale: float) -> None:
        from tensorfold.cuda.nvfp4.format import e4m3

        self.values, starts = [], [0]
        for path, weight in files:
            if not isinstance(weight, dict) or weight.get("dtype") != "F8_E4M3":
                raise ValueError(f"{Path(path).name}: the n-gram weights must be F8_E4M3 tensors")
            self.values.append(_memmap(path, weight, np.uint8))
            if self.values[-1].shape[1] != self.values[0].shape[1]:
                raise ValueError(f"{Path(path).name}: the n-gram shards differ in row width")
            starts.append(starts[-1] + self.values[-1].shape[0])
        self.starts = np.array(starts, dtype=np.int64)
        self.rows = int(self.starts[-1])
        self.width = int(self.values[0].shape[1])
        self.wrow = self.width
        self.nbytes = sum(a.nbytes for a in self.values)
        self._pool = ThreadPoolExecutor(GATHER_THREADS, thread_name_prefix="ngram-gather")
        f32 = (e4m3(np.arange(256)) * np.float32(scale)).astype(np.float32).view(np.uint32).astype(np.uint64)
        self.lut = ((f32 + 0x7FFF + ((f32 >> 16) & 1)) >> 16).astype(np.uint16)   # round to nearest even
        self.lut[(np.arange(256) & 0x7F) == 0x7F] = 0x7FC0                         # e4m3's NaN codes stay NaN

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Rows ``ids`` (global) -> [n, W] uint16 (bf16 bits of e4m3 x scale)."""

        return self.lut[super().gather(ids)]


class NVFP4Table(BF16Table):
    """NVFP4 n-gram shards (e2m1 codes, e4m3 a 16 values, one fp32 table scale): lookups give bf16(code x scale x g)."""

    def __init__(self, files: list[tuple[Path, dict, dict]], scale: float) -> None:
        from tensorfold.cuda.nvfp4.format import E2M1, e4m3

        self.values, self.scales, starts = [], [], [0]
        for path, weight, block in files:
            if weight.get("dtype") != "U8" or block.get("dtype") != "F8_E4M3":
                raise ValueError(f"{Path(path).name}: NVFP4 n-gram shards are U8 codes with F8_E4M3 scales")
            self.values.append(_memmap(path, weight, np.uint8))
            self.scales.append(_memmap(path, block, np.uint8))
            if self.values[-1].shape[1] != self.values[0].shape[1] or \
                    self.scales[-1].shape[1] * 8 != self.values[-1].shape[1]:
                raise ValueError(f"{Path(path).name}: the n-gram shards differ in row width")
            starts.append(starts[-1] + self.values[-1].shape[0])
        self.starts = np.array(starts, dtype=np.int64)
        self.rows = int(self.starts[-1])
        self.width = int(self.values[0].shape[1]) * 2
        self.wrow = self.width // 2
        self.nbytes = sum(a.nbytes for a in self.values + self.scales)
        self.e2m1, self.e4m3, self.g = E2M1, e4m3(np.arange(256)), np.float32(scale)
        self._pool = ThreadPoolExecutor(GATHER_THREADS, thread_name_prefix="ngram-gather")

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """Rows ``ids`` (global) -> [n, W] uint16: bf16 bits of the fp32 code x block scale x table scale, rounded once."""

        shard, local = self._where(ids)
        codes = np.empty((shard.size, self.width // 2), dtype=np.uint8)
        blocks = np.empty((shard.size, self.width // 16), dtype=np.uint8)

        def copy(f: int, at: np.ndarray) -> None:
            codes[at], blocks[at] = self.values[f][local[at]], self.scales[f][local[at]]

        _copy_rows(self._pool, shard, copy)
        nib = np.stack([codes & 0xF, codes >> 4], -1).reshape(shard.size, self.width)
        v = (self.e2m1[nib] * np.repeat(self.e4m3[blocks], 16, axis=1)).astype(np.float32) * self.g
        f32 = v.astype(np.float32).view(np.uint32).astype(np.uint64)
        return ((f32 + 0x7FFF + ((f32 >> 16) & 1)) >> 16).astype(np.uint16)

    def lock(self) -> bool:
        return False

    def prefetch(self, workers: int = 8) -> float:
        return _prefetch(self.values + self.scales, workers)


def shard_keys(name: str, count: int, names) -> list[str]:
    """Resolve the flat and nested shard spellings used by MLX checkpoints."""

    return [next((key for key in (f"{name}.shard_{i}", f"{name}.shards.{i}")
                  if key + ".weight" in names), f"{name}.shard_{i}") for i in range(count)]


def open_table(model_dir: Path, shards: list[tuple[str, str]], scale, *, ssd: bool = False):
    """The n-gram table in its shards' layout (MLX 4-bit, bf16, FP8, NVFP4); ``scale(name)`` reads a table scale."""

    headers: dict[str, dict] = {}
    kinds: dict[str, list] = {"mlx": [], "bf16": [], "fp8": [], "nvfp4": []}
    for shard, key in shards:
        if shard not in headers:
            headers[shard] = read_header(model_dir / shard)
        h, path = headers[shard], model_dir / shard
        if key + ".scales" in h:
            kinds["mlx"].append((path, h[key + ".weight"], h[key + ".scales"], h[key + ".biases"]))
        elif h[key + ".weight"].get("dtype") == "F8_E4M3":
            kinds["fp8"].append((path, h[key + ".weight"]))
        elif key + ".weight_scale" in h:
            kinds["nvfp4"].append((path, h[key + ".weight"], h[key + ".weight_scale"]))
        else:
            kinds["bf16"].append((path, h[key + ".weight"]))
    used = [k for k, v in kinds.items() if v]
    if len(used) != 1:
        raise ValueError(f"the n-gram shards mix layouts: {', '.join(used)}")
    files = kinds[used[0]]
    if used[0] == "nvfp4":
        return NVFP4Table(files, scale("weight_scale_2"))
    if used[0] == "fp8":
        return FP8Table(files, scale("weight_scale"))
    table = BF16Table(files) if used[0] == "bf16" else SSDTable(files) if ssd else HostTable(files)
    table.weight_scale = float(scale("weight_scale"))
    return table


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


def _copy_rows(pool: ThreadPoolExecutor, shard: np.ndarray, copy) -> None:
    """``copy(f, at)`` for each shard's rows; a prompt chunk's (2 GATHER_SPLIT rows or more) split over the pool."""

    if GATHER_THREADS > 1 and shard.size >= 2 * GATHER_SPLIT:
        jobs = []
        for f in np.unique(shard):
            at = np.nonzero(shard == f)[0]
            parts = max(1, min(GATHER_THREADS, len(at) // GATHER_SPLIT))
            jobs += [(f, piece) for piece in np.array_split(at, parts)]
        list(pool.map(lambda job: copy(*job), jobs))
    else:
        for f in np.unique(shard):
            copy(f, np.nonzero(shard == f)[0])


def _key(ids: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(ids, dtype=np.int64).reshape(-1)).tobytes()


def _memmap(path: Path, entry: dict, dtype) -> np.ndarray:
    with open(path, "rb") as f:
        header = struct.unpack("<Q", f.read(8))[0]
    begin, end = entry["data_offsets"]
    shape = tuple(entry["shape"])
    array = np.memmap(path, dtype=dtype, mode="r", offset=8 + header + begin, shape=shape)
    _random_access(array)
    return array


def _random_access(array: np.ndarray) -> None:
    """Advise random access on a table's mapping (read-ahead only evicts useful pages); best effort."""

    try:
        import mmap as _mmap

        array._mmap.madvise(_mmap.MADV_RANDOM)          # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        pass


PREFETCH_READ = 16 << 20        # bytes a prefetch read: a page fault under MADV_RANDOM reads one page, a read the span


def _prefetch(arrays: list[np.ndarray], workers: int = 8) -> float:
    """Read every array's file bytes once, in PREFETCH_READ spans over ``workers`` threads, so lookups hit the page cache (seconds taken); the pages stay evictable."""

    import threading
    import time

    spans = [(arr, at) for arr in arrays for at in range(0, arr.nbytes, PREFETCH_READ)]
    local = threading.local()

    def read(span) -> None:
        arr, at = span
        n = min(PREFETCH_READ, arr.nbytes - at)
        path, offset = getattr(arr, "filename", None), getattr(arr, "offset", None)
        if path is None or offset is None:             # not a file's map: fault its pages in
            np.asarray(arr.reshape(-1).view(np.uint8)[at:at + n]).sum(dtype=np.uint64)
            return
        if getattr(local, "buf", None) is None:
            local.buf = memoryview(bytearray(PREFETCH_READ))
        with open(path, "rb", buffering=0) as f:     # portable (Linux and macOS): seek, then one read into the buffer
            f.seek(offset + at)
            f.readinto(local.buf[:n])

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(read, spans))
    return time.time() - t0


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
    for key in shard_keys(name, count, {key for h in headers.values() for key in h}):
        found = [(path, h) for path, h in headers.items() if any(f"{key}.{part}" in h for part in _PARTS)]
        if len(found) != 1 or not all(f"{key}.{part}" in found[0][1] for part in _PARTS):
            raise ValueError(f"{key}: expected its weight, scales and biases together in one checkpoint file")
        path, h = found[0]
        files.append((path, *(h[f"{key}.{part}"] for part in _PARTS)))
    return SSDTable(files) if ssd else HostTable(files)
