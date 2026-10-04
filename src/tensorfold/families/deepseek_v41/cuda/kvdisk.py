# SPDX-License-Identifier: MIT
# Portions Copyright (c) 2026 Jay Leaton (jayleaton/deepseek-v41-tensorfold-spark, engine/serving/sessdisk.py),
# MIT License; adapted for TensorFold's DeepSeek-V4.1 shared-pool kept prompts.
# Adapted for tensorfold dsv41-cuda: kept states (``multi.Kept``) instead of session pages; the file laid out in a
# pinned stage exactly as on disk (no whole-file mmap copy), a SHA-256 per 8 MiB chunk, a background writer whose
# index changes all happen on the protocol thread (both ranks' indexes stay equal), streamed entries larger than the
# stage, a prefix digest and short tail instead of the ids in RAM.
"""The NVMe tier of DeepSeek-V4.1's kept prompts (``TF_DSV41_DISK``): a kept state evicted from the shared pool is
spilled to local NVMe and restored into free pool rows and a free bank entry when a later prompt resumes from it.

One file an entry: ``<root>/<compat hash[:16]>/rank<r>/<key hex>.tfk`` =

    "TFDSKEPT" | u64 header bytes | header JSON | zero pad to 4 KiB | segments, each at a 4 KiB-aligned offset

segments: ``ids`` (int32), then the engine's ``kept_views`` (raw bytes of every kv source's compressed entries and
indexer keys, then the bank entry's window rings). The header names each segment's offset, length and a SHA-256 per
8 MiB chunk, plus a SHA-256 of itself (``hsum``); a restore checks every chunk and ``key == H(ids)``.

- Writes go to ``<key>.tmp``, then fsync, ``os.replace`` and an fsync of the directory: a crash leaves no half entry
  (``reconcile`` deletes ``.tmp``). O_DIRECT where the filesystem allows it (the stage is page aligned and laid out
  as the file), else buffered with the written range dropped from the page cache.
- The index (an LRU ``OrderedDict``) changes only on the calling (protocol) thread: ``spill`` inserts and trims at
  enqueue time, ``touch``, ``delete``, ``retain``. The writer thread only writes files and sets an entry's state, so
  two ranks making the same calls in the same order keep equal indexes whatever their disks' timing.
- The compat ident names the directory: a different build never reads another's entries.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:                                  # host tests: numpy views only
    torch = None

MAGIC = b"TFDSKEPT"
FORMAT = 1
ALIGN = 4096                 # O_DIRECT granularity: offsets, lengths and buffer addresses
CHUNK = 8 << 20              # a SHA-256 per this many bytes of a segment
WRITE = 64 << 20             # bytes a pwritev at most
BLOCK = 256                  # the prefix digest covers whole blocks of this many tokens
PENDING, OK, FAILED = "pending", "ok", "failed"

# operational knobs that change no cached byte (everything else under TF_ names the compat directory)
SKIP_KNOBS = ("TF_PORT", "TF_API_KEY", "TF_RANK", "TF_DSV41_MEMLOG", "TF_MULTI_PROF", "TF_ROUND_PROF", "TF_FILL_SHARE",
              "TF_DSV41_KEPT_ENTRIES", "TF_DSV41_POOL_TOKENS", "TF_DSV41_POOL_GIB", "TF_DSV41_GROW_AHEAD",
              "TF_DSV41_FIXED_GIB", "TF_DSV41_RESERVE_GIB", "TF_CARVEOUT", "TF_COMM", "TF_MULTI_CHECK",
              "TF_DSV41_LAUNCH_T0", "TF_HEALTH", "TF_STALL_S", "TF_DSV41_TOOL_GRAMMAR", "TF_DSV41_PREPARED",
              "TF_DSV41_CALIB", "TF_DSV41_WARM", "TF_DSV41_ENGRAM_DIR", "TF_REVISION")


def _up(n: int) -> int:
    return -(-int(n) // ALIGN) * ALIGN


def compat_hash(ident: dict) -> str:
    return hashlib.sha256(json.dumps(dict(ident, format=FORMAT), sort_keys=True, default=str).encode()).hexdigest()


def knobs_from_env(prefix: str = "TF_", skip: Sequence[str] = SKIP_KNOBS) -> dict:
    """The knobs that may change bytes: every ``prefix`` variable but the operational ones and TF_DSV41_DISK*."""

    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith(prefix) and k not in skip and not k.startswith("TF_DSV41_DISK")}


def ids_digest(ids) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(ids, dtype=np.int32).tobytes(), digest_size=16).digest()


def chain(p) -> list[bytes]:
    """``chain(p)[j]`` = ``ids_digest(p[:BLOCK * (j + 1)])`` for every whole block of ``p``."""

    a = np.ascontiguousarray(p, dtype=np.int32)
    h = hashlib.blake2b(digest_size=16)
    out = []
    for j in range(len(a) // BLOCK):
        h.update(a[j * BLOCK:(j + 1) * BLOCK].tobytes())
        out.append(h.copy().digest())
    return out


def common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if len(diff) else n


def key_ints(key: bytes) -> tuple[int, int]:
    """A 16-byte key as two signed int64 (protocol words)."""

    return (int.from_bytes(key[:8], "little", signed=True), int.from_bytes(key[8:], "little", signed=True))


def ints_key(hi: int, lo: int) -> bytes:
    return int(hi).to_bytes(8, "little", signed=True) + int(lo).to_bytes(8, "little", signed=True)


@dataclass(eq=False)
class Ent:
    """An indexed entry: its length, window starts, prefix digest of its first ``B`` tokens and the rest (``tail``),
    data bytes (budget) and write state."""

    n: int
    vs: int
    vd: int
    B: int
    pre: bytes
    tail: np.ndarray
    size: int
    state: str = OK


class _File:
    """A file opened O_DIRECT when the filesystem allows it (after sessdisk's ``_File``)."""

    def __init__(self, path: Path, write: bool, direct: bool) -> None:
        flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC) if write else os.O_RDONLY
        self.direct = False
        if direct and hasattr(os, "O_DIRECT"):
            try:
                self.fd = os.open(path, flags | os.O_DIRECT, 0o644)
                self.direct = True
                return
            except OSError:
                pass
        self.fd = os.open(path, flags, 0o644)

    def close(self, drop: bool = False) -> None:
        if drop and not self.direct and hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        os.close(self.fd)


def _sync() -> None:
    """Every device stream idle (prefill side streams and graph replays write the arena and the bank too)."""

    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _flat(v):
    """A view as flat uint8 bytes (no copy: the views are contiguous)."""

    if isinstance(v, np.ndarray):
        if not v.flags.c_contiguous:
            raise ValueError("a kept view must be contiguous")
        return v.reshape(-1).view(np.uint8)
    if not v.is_contiguous():
        raise ValueError("a kept view must be contiguous")
    return v.reshape(-1).view(torch.uint8)


def _shape(v) -> list[int]:
    return [int(d) for d in v.shape]


def _nbytes(v) -> int:
    return int(v.nbytes) if isinstance(v, np.ndarray) else int(v.numel() * v.element_size())


def _alloc_stage(nbytes: int):
    """(torch tensor or None, numpy uint8 array, keepalive) of ``nbytes`` page-aligned bytes: pinned host memory when
    CUDA is there (asynchronous copies), else an anonymous mapping."""

    if torch is not None and torch.cuda.is_available():
        t = torch.empty((nbytes + ALIGN,), dtype=torch.uint8, pin_memory=True)
        skip = (-t.data_ptr()) % ALIGN
        t = t[skip:skip + nbytes]
        return t, t.numpy(), None
    m = mmap.mmap(-1, nbytes)
    return None, np.frombuffer(m, dtype=np.uint8), m


class KeptDisk:
    """Kept prompts on NVMe, one rank's (see the module docstring)."""

    def __init__(self, root: str | Path, rank: int, *, budget_gib: float = 128.0, min_tokens: int = 2048,
                 stage_mib: int = 1024, direct: bool = True, threads: int | None = None, quiet: bool = False) -> None:
        self.root = Path(root)
        self.rank = int(rank)
        self.budget = int(float(budget_gib) * 2 ** 30)
        self.min_tokens = int(min_tokens)
        self.direct = direct
        self.quiet = quiet
        self.stage_bytes = _up(max(CHUNK, int(stage_mib) << 20))
        self.stage_t, self.stage, self._keep = _alloc_stage(self.stage_bytes)
        self.dir: Path | None = None
        self.index: OrderedDict[bytes, Ent] = OrderedDict()     # least recently used first
        self._cv = threading.Condition()
        self._jobs: deque = deque()                  # writes queued, oldest first (the writer takes them in order)
        self._regions: deque = deque()               # stage [start, end) held by queued writes, the same order
        self._pool = ThreadPoolExecutor(max(2, threads or (os.cpu_count() or 2) - 1), thread_name_prefix="kvdisk")
        self._writer = threading.Thread(target=self._write_loop, name="kvdisk-writer", daemon=True)
        self._writer.start()
        self.stats = {"writes": 0, "written_mb": 0.0, "write_s": 0.0, "reads": 0, "read_mb": 0.0, "bad": 0,
                      "trimmed": 0, "failed_writes": 0}

    # -- setup ------------------------------------------------------------------------------------------------------
    def attach(self, ident: dict) -> int:
        """Use the directory of this build's ``ident`` and index its entries (``reconcile``); returns their count."""

        h = compat_hash(ident)
        self.dir = self.root / h[:16] / f"rank{self.rank}"
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir.parent / f"compat.json.{os.getpid()}.{self.rank}"
        tmp.write_text(json.dumps(dict(ident, format=FORMAT), sort_keys=True, indent=1, default=str))
        os.replace(tmp, self.dir.parent / "compat.json")
        return self.reconcile()

    def describe(self) -> str:
        used = sum(e.size for e in self.index.values())
        return (f"kept prompts on NVMe at {self.dir}: {len(self.index)} entries, {used / 2 ** 30:.2f} of "
                f"{self.budget / 2 ** 30:.0f} GiB (states of {self.min_tokens}+ tokens; "
                f"{self.stage_bytes >> 20} MiB stage)")

    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(f"[kvdisk r{self.rank}] {msg}", flush=True)

    # -- keys and lookup ----------------------------------------------------------------------------------------------
    @staticmethod
    def key(ids) -> bytes:
        a = np.ascontiguousarray(ids, dtype=np.int32)
        h = hashlib.blake2b(b"dsv41-kept" + struct.pack("<Q", len(a)), digest_size=16)
        h.update(a.tobytes())
        return h.digest()

    def path(self, key: bytes) -> Path:
        assert self.dir is not None, "attach first"
        return self.dir / f"{key.hex()}.tfk"

    def keys(self) -> list[bytes]:
        """The indexed keys, least recently used first."""

        return list(self.index)

    def find(self, prompt, resumable: Callable[[int, int, int, int], bool], *, span: int,
             least: int = 0, align: int = 2048) -> tuple[int, bytes, Ent] | None:
        """(m, key, entry) of the entry ``prompt`` resumes from with the most tokens (ties: the most recent): its
        first m tokens shared (at least one left to prefill), ``m >= least``, ``resumable(m, len(prompt), vs, vd)``
        and the entry fitting a ``span``-row extent; None when none does."""

        p = np.asarray(prompt, dtype=np.int64)
        if not self.index or len(p) < 2:
            return None
        need = max((e.B for e in self.index.values() if e.state != FAILED), default=0)
        cp = chain(p[:need]) if need else []
        best = None
        for key, e in self.index.items():
            if e.state == FAILED or -(-e.n // align) * align > span:
                continue
            if e.B and (len(p) <= e.B or cp[e.B // BLOCK - 1] != e.pre):
                continue
            m = min(e.B + common_prefix(e.tail, p[e.B:]), len(p) - 1, e.n)
            if m < max(least, 1) or not resumable(m, len(p), e.vs, e.vd):
                continue
            if best is None or m >= best[0]:
                best = (m, key, e)
        return best

    # -- the index (protocol thread only) -------------------------------------------------------------------------------
    def touch(self, key: bytes) -> None:
        if key in self.index:
            self.index.move_to_end(key)

    def delete(self, key: bytes) -> None:
        e = self.index.pop(key, None)
        if e is not None:
            with self._cv:
                while e.state == PENDING:
                    self._cv.wait()
        if self.dir is None:
            return
        for p in (self.path(key), self.path(key).with_suffix(".tmp")):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                self._log(f"could not delete {p.name}: {exc}")

    def retain(self, keys: Sequence[bytes]) -> None:
        """Keep only ``keys`` (the entries both ranks hold), in their order (rank 0's: both indexes the same)."""

        want = [k for k in keys if k in self.index]
        keep = set(want)
        for k in [k for k in self.index if k not in keep]:
            self.delete(k)
        for k in want:
            self.index.move_to_end(k)
        self._trim()

    def _trim(self) -> None:
        """Least recently used entries out while the total passes the budget (the newest one always fits)."""

        while len(self.index) > 1 and sum(e.size for e in self.index.values()) > self.budget:
            self.delete(next(iter(self.index)))
            self.stats["trimmed"] += 1

    # -- layout ---------------------------------------------------------------------------------------------------
    def _layout(self, key: bytes, n: int, vs: int, vd: int, ids: np.ndarray, views: list) -> tuple[dict, list, int]:
        """(header with placeholder hashes, [(name, source, offset, nbytes)], data bytes): offsets from the first data
        byte, each 4 KiB aligned; the header's length is final (fixed-width hashes)."""

        segs, off = [], 0
        for name, v in [("ids", ids), *views]:
            nb = _nbytes(v)
            segs.append((name, v, off, nb))
            off = _up(off + nb)
        B = vs // BLOCK * BLOCK
        head = {"format": FORMAT, "key": key.hex(), "rank": self.rank, "n": int(n), "vs": int(vs), "vd": int(vd),
                "B": B, "pre": ids_digest(ids[:B]).hex(), "tail": [int(t) for t in ids[B:n]],
                "segments": [{"name": name, "shape": _shape(v), "offset": o, "nbytes": nb,
                              "sha": ["0" * 64] * max(1, -(-nb // CHUNK))} for name, v, o, nb in segs]}
        return head, segs, off

    @staticmethod
    def _chunks(head: dict) -> list[tuple[int, int, int, int]]:
        """(segment, chunk, data offset, bytes) of every chunk, in file order."""

        out = []
        for i, s in enumerate(head["segments"]):
            for j in range(len(s["sha"])):
                a = j * CHUNK
                out.append((i, j, s["offset"] + a, min(CHUNK, s["nbytes"] - a)))
        return out

    def _batches(self, chunks: list) -> list[list]:
        """Consecutive chunks grouped so each group's span fits the stage."""

        out, cur = [], []
        for c in chunks:
            if cur and _up(c[2] + c[3]) - cur[0][2] > self.stage_bytes:
                out.append(cur)
                cur = []
            cur.append(c)
        if cur:
            out.append(cur)
        return out

    @staticmethod
    def _header_bytes(head: dict) -> bytes:
        body = {k: v for k, v in head.items() if k != "hsum"}
        h = dict(body, hsum=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest())
        hj = json.dumps(h, sort_keys=True).encode()
        raw = MAGIC + struct.pack("<Q", len(hj)) + hj
        return raw + b"\0" * (_up(len(raw)) - len(raw))

    def _read_header(self, path: Path) -> tuple[dict, int, int]:
        """(header, data base, data bytes); ValueError when it is not a whole, self-consistent header."""

        with open(path, "rb") as fh:
            first = fh.read(16)
            if len(first) < 16 or first[:8] != MAGIC:
                raise ValueError("not a kept-prompt entry")
            (n,) = struct.unpack("<Q", first[8:16])
            if n > 1 << 26:
                raise ValueError("header too large")
            raw = fh.read(n)
        head = json.loads(raw)
        body = {k: v for k, v in head.items() if k != "hsum"}
        if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != head.get("hsum"):
            raise ValueError("header checksum")
        if head.get("format") != FORMAT:
            raise ValueError(f"format {head.get('format')}")
        span = max(s["offset"] + _up(s["nbytes"]) for s in head["segments"])
        return head, _up(16 + n), span

    def _ent(self, head: dict, span: int, state: str) -> Ent:
        return Ent(int(head["n"]), int(head["vs"]), int(head["vd"]), int(head["B"]), bytes.fromhex(head["pre"]),
                   np.asarray(head["tail"], dtype=np.int64), span, state)

    # -- the stage: a ring of regions held by queued writes ---------------------------------------------------------
    def _reserve(self, size: int) -> int:
        """A stage offset with ``size`` free bytes after the newest queued region or at the start (waits for the
        writer to free the oldest ones)."""

        with self._cv:
            while self._regions:
                head, tail = self._regions[-1][1], self._regions[0][0]
                if self._regions[-1][0] >= tail:     # not wrapped: free after the newest and before the oldest
                    if self.stage_bytes - head >= size:
                        return head
                    if tail >= size:
                        return 0
                elif tail - head >= size:            # wrapped: free between the newest and the oldest
                    return head
                self._cv.wait()
            return 0

    # -- spill ------------------------------------------------------------------------------------------------------
    def spill(self, key: bytes, n: int, vs: int, vd: int, ids, views) -> bool:
        """Spill kept state ``key`` (``ids``; ``views``: the engine's ``kept_views`` or a callable giving them) to
        disk: indexed now (pending), trimmed to the budget, its bytes copied off the device before this returns (the
        rows and bank entry are free to reuse then), the file written in the background. An already indexed key is
        only touched. Returns whether it was queued; never raises (a failure leaves a failed entry, a miss later)."""

        if self.dir is None:
            return False
        if key in self.index:
            self.touch(key)
            return False
        try:
            a = np.ascontiguousarray(np.asarray(ids)[:n], dtype=np.int32)
            views = list(views() if callable(views) else views)
            head, segs, span = self._layout(key, n, vs, vd, a, views)
        except Exception as exc:                     # noqa: BLE001  (a host-side refusal: nothing indexed)
            self._log(f"spill of {n} tokens refused: {exc}")
            return False
        if span > self.budget:
            return False
        ent = self._ent(head, span, PENDING)
        self.index[key] = ent
        self._trim()
        try:
            if span <= self.stage_bytes:
                start = self._reserve(span)
                _sync()                              # every write into those rows and the bank entry done
                self._to_stage(segs, start, 0, None)
                _sync()
                with self._cv:
                    self._regions.append((start, start + span))
                    self._jobs.append((key, ent, head, start, span))
                    self._cv.notify_all()
            else:                                    # larger than the stage: streamed through it, now
                self.drain()
                self._write_big(key, ent, head, segs, span)
        except Exception as exc:                     # noqa: BLE001
            with self._cv:
                ent.state = FAILED
                self._cv.notify_all()
            self.stats["failed_writes"] += 1
            self._log(f"spill of {n} tokens failed: {exc}")
        return True

    def _to_stage(self, segs: list, start: int, lo: int, chunks: list | None) -> None:
        """Copy segment bytes into the stage at ``start + offset - lo``: whole segments (``chunks`` None, padding
        zeroed) or just ``chunks``."""

        def put(v, a: int, b: int, at: int) -> None:
            if isinstance(v, np.ndarray):
                self.stage[at:at + b - a] = _flat(v)[a:b]
            elif self.stage_t is not None:
                self.stage_t[at:at + b - a].copy_(_flat(v)[a:b], non_blocking=True)
            else:
                self.stage[at:at + b - a] = _flat(v)[a:b].cpu().numpy()

        if chunks is None:
            for name, v, off, nb in segs:
                put(v, 0, nb, start + off - lo)
                pad = _up(off + nb) - (off + nb)
                if pad:
                    self.stage[start + off + nb - lo:start + off + nb - lo + pad] = 0
            return
        for i, j, off, nb in chunks:
            _, v, soff, snb = segs[i]
            a = j * CHUNK
            put(v, a, a + nb, start + off - lo)
            end = off + nb
            if end == soff + snb and _up(end) > end:
                self.stage[start + end - lo:start + _up(end) - lo] = 0

    def _hash(self, buf: np.ndarray, chunks: list, lo: int) -> list[str]:
        return list(self._pool.map(lambda c: hashlib.sha256(memoryview(buf[c[2] - lo:c[2] - lo + c[3]])).hexdigest(),
                                   chunks))

    def _pwrite(self, f: _File, buf: np.ndarray, at: int) -> None:
        mv = memoryview(buf)
        done = 0
        while done < len(mv):
            k = min(WRITE, len(mv) - done)
            wrote = os.pwritev(f.fd, [mv[done:done + k]], at + done)
            if wrote <= 0:
                raise OSError(f"short write ({wrote} of {k})")
            done += wrote

    def _finish_file(self, f: _File, tmp: Path, key: bytes, head: dict) -> None:
        raw = self._header_bytes(head)
        hb = mmap.mmap(-1, len(raw))                 # page aligned (O_DIRECT)
        try:
            hb.write(raw)
            self._pwrite(f, np.frombuffer(hb, dtype=np.uint8), 0)
            os.fsync(f.fd)
        except BaseException:
            f.close()
            raise
        finally:
            hb.close()
        f.close(drop=True)
        os.replace(tmp, self.path(key))
        try:
            d = os.open(self.dir, os.O_RDONLY)
            try:
                os.fsync(d)
            finally:
                os.close(d)
        except OSError:
            pass

    def _base(self, head: dict) -> int:
        return len(self._header_bytes(head))

    def _write_loop(self) -> None:
        while True:
            with self._cv:
                while not self._jobs:
                    self._cv.wait()
                key, ent, head, start, span = self._jobs[0]
            t0 = time.perf_counter()
            try:
                buf = self.stage[start:start + span]
                chunks = self._chunks(head)
                for (i, j, _, _), sha in zip(chunks, self._hash(buf, chunks, 0)):
                    head["segments"][i]["sha"][j] = sha
                base = self._base(head)
                tmp = self.path(key).with_suffix(".tmp")
                f = _File(tmp, True, self.direct)
                try:
                    self._pwrite(f, buf, base)
                except BaseException:
                    f.close()
                    raise
                self._finish_file(f, tmp, key, head)
                state = OK
                self.stats["writes"] += 1
                self.stats["written_mb"] += span / 2 ** 20
            except Exception as exc:                 # noqa: BLE001  (the entry stays indexed, failed: a miss later)
                state = FAILED
                self.stats["failed_writes"] += 1
                self._log(f"write of {key.hex()[:12]} failed: {exc}")
            self.stats["write_s"] += time.perf_counter() - t0
            with self._cv:
                ent.state = state
                self._jobs.popleft()
                self._regions.popleft()
                self._cv.notify_all()

    def _write_big(self, key: bytes, ent: Ent, head: dict, segs: list, span: int) -> None:
        """An entry larger than the stage: copied, hashed and written batch by batch (the writer is idle)."""

        base = self._base(head)
        tmp = self.path(key).with_suffix(".tmp")
        f = _File(tmp, True, self.direct)
        try:
            for batch in self._batches(self._chunks(head)):
                lo, hi = batch[0][2], _up(batch[-1][2] + batch[-1][3])
                _sync()
                self._to_stage(segs, 0, lo, batch)
                _sync()
                buf = self.stage[:hi - lo]
                for (i, j, _, _), sha in zip(batch, self._hash(buf, batch, lo)):
                    head["segments"][i]["sha"][j] = sha
                self._pwrite(f, buf, base + lo)
        except BaseException:
            f.close()
            raise
        self._finish_file(f, tmp, key, head)
        with self._cv:
            ent.state = OK
            self._cv.notify_all()
        self.stats["writes"] += 1
        self.stats["written_mb"] += span / 2 ** 20

    def drain(self) -> None:
        """Every queued write finished."""

        with self._cv:
            while self._jobs:
                self._cv.wait()

    # -- restore ----------------------------------------------------------------------------------------------------
    def restore(self, key: bytes, n: int, vs: int, vd: int, views) -> np.ndarray | None:
        """Entry ``key``'s bytes into ``views`` (the engine's ``kept_views`` of free pool rows and a free bank entry,
        or a callable giving them), every chunk checked; its ids (int64), or None when it is missing, damaged or not
        the state asked for (the caller agrees with the other rank, then deletes). Never raises."""

        try:
            if self.dir is None or key not in self.index:
                return None
            self.drain()                             # (the stage is shared with the writer)
            if self.index[key].state != OK:
                return None
            views = list(views() if callable(views) else views)
            path = self.path(key)
            head, base, span = self._read_header(path)
            if (head["key"] != key.hex() or head["rank"] != self.rank or
                    (head["n"], head["vs"], head["vd"]) != (n, vs, vd)):
                raise ValueError("not the entry asked for")
            want = [("ids", [n], n * 4), *[(name, _shape(v), _nbytes(v)) for name, v in views]]
            got = [(s["name"], s["shape"], s["nbytes"]) for s in head["segments"]]
            if want != got:
                raise ValueError("its segments do not match this engine's views")
            if os.path.getsize(path) < base + span:
                raise ValueError("truncated")
            ids = np.empty((n,), dtype=np.int32)
            dst = [ids, *[v for _, v in views]]
            f = _File(path, False, self.direct)
            try:
                for batch in self._batches(self._chunks(head)):
                    lo, hi = batch[0][2], _up(batch[-1][2] + batch[-1][3])
                    buf = self.stage[:hi - lo]
                    self._pread(f, buf, base + lo)
                    sums = self._hash(buf, batch, lo)
                    for (i, j, off, nb), sha in zip(batch, sums):
                        if sha != head["segments"][i]["sha"][j]:
                            raise ValueError(f"segment {head['segments'][i]['name']} chunk {j}: checksum mismatch")
                    for i, j, off, nb in batch:
                        self._from_stage(dst[i], j * CHUNK, buf[off - lo:off - lo + nb], off - lo, nb)
                    _sync()                          # before the stage is read into again
            finally:
                f.close(drop=True)
            if self.key(ids) != key:
                raise ValueError("ids do not match the key")
            self.stats["reads"] += 1
            self.stats["read_mb"] += span / 2 ** 20
            return ids.astype(np.int64)
        except Exception as exc:                     # noqa: BLE001  (a miss on both ranks after the agreement)
            self.stats["bad"] += 1
            self._log(f"restore of {key.hex()[:12]} failed: {exc}")
            try:
                _sync()
            except Exception as err:                 # noqa: BLE001  (the device's own error surfaces next)
                self._log(f"device sync after the failed restore: {err}")
            return None

    def _from_stage(self, v, a: int, src: np.ndarray, at: int, nb: int) -> None:
        if isinstance(v, np.ndarray):
            _flat(v)[a:a + nb] = src
        elif self.stage_t is not None:
            _flat(v)[a:a + nb].copy_(self.stage_t[at:at + nb], non_blocking=True)
        else:
            _flat(v)[a:a + nb].copy_(torch.from_numpy(src))

    def _pread(self, f: _File, buf: np.ndarray, at: int) -> None:
        """``buf`` from file offset ``at``: up to four aligned parts read in parallel."""

        total = len(buf)
        parts = min(4, max(1, total // CHUNK))
        step = _up(-(-total // parts))
        spans = [(o, min(step, total - o)) for o in range(0, total, step)]

        def one(span) -> None:
            o, k = span
            mv = memoryview(buf)[o:o + k]
            got = 0
            while got < k:
                r = os.preadv(f.fd, [mv[got:]], at + o + got)
                if r <= 0:
                    raise ValueError(f"short read ({got} of {k})")
                got += r

        list(self._pool.map(one, spans))

    # -- startup ----------------------------------------------------------------------------------------------------
    def reconcile(self) -> int:
        """Index the directory's entries (least recently modified first), deleting temporary, foreign and damaged
        ones (headers only: chunks are checked when restored), then trim. Returns the count indexed."""

        assert self.dir is not None
        self.drain()
        self.index.clear()
        found = []
        for p in self.dir.iterdir():
            if p.suffix == ".tmp" or (p.suffix != ".tfk" and p.is_file()):
                p.unlink(missing_ok=True)
                continue
            if p.suffix != ".tfk":
                continue
            try:
                head, base, span = self._read_header(p)
                if head["key"] != p.stem or head["rank"] != self.rank:
                    raise ValueError("key / rank")
                if os.path.getsize(p) < base + span:
                    raise ValueError("truncated")
                if head["segments"][0]["name"] != "ids" or head["segments"][0]["nbytes"] != 4 * head["n"]:
                    raise ValueError("ids")
                found.append((p.stat().st_mtime_ns, p.stem, head, span))
            except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
                self.stats["bad"] += 1
                p.unlink(missing_ok=True)
        for _, stem, head, span in sorted(found, key=lambda x: (x[0], x[1])):
            self.index[bytes.fromhex(stem)] = self._ent(head, span, OK)
        self._trim()
        return len(self.index)

    def close(self) -> None:
        self.drain()
        self._pool.shutdown(wait=False)


def intersect(d: KeptDisk, gather: Callable[[list[int]], list[list[int]]]) -> int:
    """Both ranks at startup (after ``attach``): keep only the entries both hold, in rank 0's order, so the two
    indexes are equal (``gather``: the engine's all-gather of equal-length int lists). Returns the count kept."""

    mine = [w for key in d.keys() for w in key_ints(key)]  # noqa: SIM118  (a method: LRU order)
    counts = [row[0] for row in gather([len(mine)])]
    rows = gather(mine + [0] * (max(counts) - len(mine))) if max(counts) else [[], []]
    keys = [[ints_key(r[i], r[i + 1]) for i in range(0, c, 2)] for r, c in zip(rows, counts)]
    held = set(keys[1])
    d.retain([k for k in keys[0] if k in held])
    return len(d.index)


def settings() -> dict:
    """The disk tier's settings from the environment (``TF_DSV41_DISK`` unset or empty: off)."""

    return {"root": os.environ.get("TF_DSV41_DISK", "").strip(),
            "gib": float(os.environ.get("TF_DSV41_DISK_GIB") or 128),
            "min": int(os.environ.get("TF_DSV41_DISK_MIN") or 2048),
            "stage_mib": int(os.environ.get("TF_DSV41_DISK_STAGE_MIB") or 1024)}


def agreement_words() -> list[int]:
    """What both ranks must agree on (index contents and memory): on, budget MiB, minimum tokens, stage MiB."""

    s = settings()
    return [int(bool(s["root"])), int(s["gib"] * 1024), s["min"], s["stage_mib"]]


def from_env(rank: int) -> KeptDisk | None:
    """``TF_DSV41_DISK=<dir>`` (unset: off), ``TF_DSV41_DISK_GIB`` (128), ``TF_DSV41_DISK_MIN`` (2048 tokens: shorter
    states are not written), ``TF_DSV41_DISK_STAGE_MIB`` (1024: the pinned staging buffer). Not attached yet."""

    s = settings()
    if not s["root"]:
        return None
    return KeptDisk(s["root"], rank, budget_gib=s["gib"], min_tokens=s["min"], stage_mib=s["stage_mib"])


def _code_digest() -> str:
    """SHA-256 over the installed tensorfold package's sources (every module the prompt path imports)."""

    import tensorfold

    top = Path(tensorfold.__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(top.rglob("*")):
        if p.suffix in (".py", ".cu", ".cuh", ".cpp", ".h") and p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(top)).encode() + b"\0")
            h.update(p.read_bytes())
    return h.hexdigest()


def compat_ident(e, model_dir: str | Path, engram_dir: str | Path) -> dict:
    """What decides whether an entry's bytes mean the same to this build (after sessdisk's ``compat_ident``): the
    cache layout, the model and Engram files, the libraries and device, every byte-relevant knob and the code."""

    from tensorfold.cuda import precision, prompt_precision

    from . import kernels as K
    from . import serial as S
    from .pool import ALIGN as POOL_ALIGN

    def files(d) -> list:
        d = Path(d)
        return sorted((p.name, p.stat().st_size) for p in d.glob("*.safetensors")) if d.is_dir() else []

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""

    c = e.c
    model_dir = Path(model_dir)

    def fmt(t) -> list:                                  # a cache's storage format, self-describing
        if isinstance(t, K.QRows):
            return [type(t).__name__, t.dim, list(t.planes), *(getattr(t, a, None) for a in ("plain", "group", "scale"))]
        return [str(t.dtype), int(t.shape[1])]

    src = min(c.kv_source_layer_ids)
    ident = {
        "format": FORMAT,
        "layout": {"DRING": S.DRING, "WINDOW_ROWS": S.WINDOW_ROWS, "MAX_ROWS": S.MAX_ROWS, "RING": S.RING,
                   "PROMPT_ROWS": S.PROMPT_ROWS, "ALIGN": POOL_ALIGN,
                   "kv_format": {"mode": S.KV_MODE, "comp": fmt(e.big.comp[src]), "ik": fmt(e.big.ik[src])},
                   "rings": [[list(t.shape[1:]), str(t.dtype)] for t in e._rings()],
                   "ratios": [int(r) for r in c.layer_ratios], "sources": sorted(int(s) for s in c.kv_source_layer_ids)},
        "drafts": e.drafter is not None, "tail_min": e.tail_min, "deep_reach": e.deep_reach,
        "prompt_fp8": bool(prompt_precision.fp8()), "precision": precision.mode(),
        "torch": torch.__version__ if torch is not None else None, "cuda": getattr(torch.version, "cuda", None),
        "device": [torch.cuda.get_device_name(), list(torch.cuda.get_device_capability())],
        "config": sha(model_dir / "config.json"), "tokenizer": sha(model_dir / "tokenizer.json"),
        "weights": files(model_dir), "engram": files(engram_dir),
        "code": _code_digest(), "knobs": knobs_from_env(),
    }
    try:
        import triton

        ident["triton"] = triton.__version__
    except ImportError:
        ident["triton"] = None
    return ident


__all__ = ["Ent", "KeptDisk", "agreement_words", "chain", "compat_hash", "compat_ident", "from_env", "ints_key",
           "key_ints", "knobs_from_env", "settings"]
