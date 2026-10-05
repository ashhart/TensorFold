"""The spill tier's I/O: pinned staging buffers, writer threads and reads, O_DIRECT where the file system takes it."""

from __future__ import annotations

import errno
import os
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch

from tensorfold.cuda.spill_format import ALIGN, _blocks, _crc_placeholder, _Crcs, _up

if TYPE_CHECKING:
    from tensorfold.cuda.spill import Entry, SpillConfig

CHUNK = 64 << 20                       # bytes a staging buffer

_PINNED: dict[str, list[torch.Tensor]] = {}     # staging allocated before the engine sizes its caches


def _pinned(n: int) -> list[torch.Tensor]:
    pin = torch.cuda.is_available()
    return [torch.empty(CHUNK, dtype=torch.uint8, pin_memory=pin) for _ in range(n)]


def preallocate(cfg: SpillConfig | None) -> None:
    """Pin the staging buffers (two a writer, two for reads) before the engine sizes its caches from what is left."""

    if cfg is None or _PINNED:
        return
    _PINNED["write"] = _pinned(2 * cfg.writers)
    _PINNED["read"] = _pinned(2)


class Job:
    """One entry on its way to disk: ``copied`` once its bytes left the device, ``done`` once on disk or given up."""

    def __init__(self, entry: Entry, path: str, header: bytes, items: list, src_event, seq: int = 0) -> None:
        self.entry = entry
        self.key = entry.key
        self.path = path
        self.partial = f"{path}.{seq}.partial"      # its own: a later write of the same prompt never shares it
        self.nbytes = entry.size
        self.header = header
        self.items = items
        self.src_event = src_event
        self.copied = threading.Event()
        self.done = threading.Event()
        self.ok = False


class LoadJob:
    """A read: the file, its header and the new device tensors its bytes go to (``event``: their copies queued)."""

    def __init__(self, entry: Entry, path: str, header: dict, data_at: int, dests: dict) -> None:
        self.entry = entry
        self.path = path
        self.header = header
        self.data_at = data_at
        self.dests = dests             # name -> device tensor the bytes go to
        self.event = None
        self.after = None              # the caller's stream event the copies wait for before writing
        self.t0 = time.perf_counter()
        self.meta = header.get("__metadata__", {}) or {}


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % ALIGN == 0


class SpillIO:
    """``SpillStore``'s writer and reader threads: rows through pinned buffers to its files and back to the device."""

    # -- writing -------------------------------------------------------------------------------------------------------
    def _start(self) -> None:
        if self._threads:
            return
        bufs = _PINNED.get("write") or _pinned(2 * self.cfg.writers)
        if any(not _aligned(b) for b in bufs):
            self.direct = False
        for i in range(self.cfg.writers):
            t = threading.Thread(target=self._writer, args=(bufs[2 * i:2 * i + 2],),
                                 name=f"spill-writer-{self.rank}-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def drain(self, timeout: float | None = None) -> bool:
        """Wait until every queued write is on disk (a shutdown). True when the writes finished in time."""

        end = None if timeout is None else time.monotonic() + timeout
        while True:
            with self.lock:
                jobs = list(self.pending.values())
            if not jobs:
                break
            left = None if end is None else max(0.0, end - time.monotonic())
            if not jobs[0].done.wait(left) and left is not None and left <= 0:
                return False
        self._sync_dirs(force=True)
        return True

    def _sync_dirs(self, force: bool = False) -> None:
        with self._dir_lock:
            now = time.monotonic()
            if self._dirty and (force or now - self._last_dir_sync >= 1.0):
                _fsync_dir(self.dir)
                self._dirty = False
                self._last_dir_sync = now

    def _open_write(self, path: str) -> tuple[int, bool]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if self.direct:
            try:
                return os.open(path, flags | os.O_DIRECT, 0o600), True
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
                self.direct = False
        return os.open(path, flags, 0o600), False

    @staticmethod
    def _write_all(fd: int, mv: memoryview) -> None:
        while len(mv):
            k = os.write(fd, mv)
            mv = mv[k:]

    @staticmethod
    def _pwrite_all(fd: int, mv: memoryview, at: int) -> None:
        while len(mv):
            k = os.pwrite(fd, mv, at)
            mv, at = mv[k:], at + k

    def _write(self, job: Job, bufs: list[torch.Tensor], stream, hasher: ThreadPoolExecutor) -> bool:
        """One entry through two staging buffers: one buffer's copy off the device overlaps the other's write."""

        path = job.partial
        fd, direct = self._open_write(path)
        total = 0
        try:
            if stream is not None and job.src_event is not None:
                stream.wait_event(job.src_event)        # the rows are final where the engine recorded
            cur, fill, ev = 0, 0, [None, None]
            hdr = np.frombuffer(job.header, dtype=np.uint8)
            bufs[cur][:len(hdr)].copy_(torch.from_numpy(hdr.copy()))
            fill = len(hdr)
            pending = None                              # (buffer index, bytes) copied, not yet written
            crcs, skip = _Crcs(), [len(hdr)]            # the data's CRCs (the header in the first buffer is not data)

            def flush(i: int, n: int) -> None:
                nonlocal total
                if ev[i] is not None:
                    ev[i].synchronize()
                    ev[i] = None
                k = _up(n) if direct else n
                mv = memoryview(bufs[i].numpy())
                hashed = hasher.submit(crcs.feed, mv[skip[0]:n])     # beside the write, which reads the same bytes
                skip[0] = 0
                self._write_all(fd, mv[:k])
                hashed.result()
                total += n

            for _, t in job.items:
                v = t.reshape(-1).view(torch.uint8)
                s, size = 0, v.numel()
                while s < size:
                    k = min(CHUNK - fill, size - s)
                    if stream is not None and v.is_cuda:
                        with torch.cuda.stream(stream):
                            bufs[cur][fill:fill + k].copy_(v[s:s + k], non_blocking=True)
                    else:
                        bufs[cur][fill:fill + k].copy_(v[s:s + k])
                    fill += k
                    s += k
                    if fill == CHUNK:
                        if stream is not None:
                            e = torch.cuda.Event()
                            e.record(stream)
                            ev[cur] = e
                        if pending is not None:
                            flush(*pending)
                        pending = (cur, CHUNK)
                        cur ^= 1
                        if ev[cur] is not None:
                            ev[cur].synchronize()
                            ev[cur] = None
                        fill = 0
            if stream is not None and fill:
                e = torch.cuda.Event()
                e.record(stream)
                ev[cur] = e
            for i in (0, 1):
                if ev[i] is not None:
                    ev[i].synchronize()
            job.copied.set()                            # every byte is in host memory: the rows may move now
            if pending is not None:
                flush(*pending)
            if fill:
                flush(cur, fill)
            slot = b'"crc32":"' + _crc_placeholder(crcs.nbytes).encode() + b'"'
            final = job.header.replace(slot, b'"crc32":"' + crcs.hex().encode() + b'"')
            if job.header.count(slot) != 1 or len(final) != len(job.header):
                raise RuntimeError("the header's CRC field is missing or changed length")
            head = np.frombuffer(final, dtype=np.uint8)
            bufs[0][:len(head)].copy_(torch.from_numpy(head.copy()))
            self._pwrite_all(fd, memoryview(bufs[0].numpy())[:len(head)], 0)
            os.ftruncate(fd, len(job.header) + sum(t.numel() * t.element_size() for _, t in job.items))
            os.fsync(fd)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
            os.close(fd)
            fd = -1
            with self.lock:                             # still this prompt's entry: its file (one dropped meanwhile,
                current = self.index.get(job.key) is job.entry       # by the cap or a failed read, leaves none)
                if current:
                    os.replace(path, job.path)
            if not current:
                os.remove(path)
                return False
            self._own(job.path)
            with self._dir_lock:
                self._dirty = True
            self._sync_dirs()
            with self.lock:
                self.stats.written_bytes += total
            return True
        except Exception as exc:  # noqa: BLE001 - only this prompt is not written
            self._log(exc)
            job.copied.set()
            try:
                if fd >= 0:
                    os.close(fd)
            except OSError:
                pass
            try:
                os.remove(path)
            except OSError:
                pass
            return False

    def _writer(self, bufs: list[torch.Tensor]) -> None:
        stream = torch.cuda.Stream(device=self.device) if torch.cuda.is_available() else None
        hasher = ThreadPoolExecutor(1, thread_name_prefix=f"spill-crc-w{self.rank}")
        while True:
            job = self.jobs.get()
            t0 = time.perf_counter()
            try:
                ok = self._write(job, bufs, stream, hasher)
            except Exception as exc:  # noqa: BLE001 - the writer never dies
                self._log(exc)
                ok = False
            job.ok = ok
            self._entry_done(job, ok)
            job.items = None                            # its tensors go now, not when the next write arrives
            job.copied.set()
            job.done.set()
            with self.lock:
                self.stats.write_s += time.perf_counter() - t0
            job = None

    def _log(self, exc: BaseException) -> None:
        with self.lock:
            self.stats.write_failed += 1
        print(f"[tensorfold] spill rank {self.rank}: write failed: {exc!r}", flush=True)

    def release(self, job: Job) -> None:
        """Wait until the rows ``job`` reads are off the device, before the engine moves or frees them."""

        if job.copied.is_set():
            return
        w0 = time.perf_counter()
        job.copied.wait()
        self.stats.copy_wait_s += time.perf_counter() - w0
        self.stats.copy_waits += 1

    # -- reading -------------------------------------------------------------------------------------------------------
    def _read(self, lj: LoadJob) -> None:
        """A file into ``lj.dests`` on this thread: a chunk's copies and CRCs run beside the next chunk's read."""

        if self._reader is None:                       # pinned buffers, a CUDA stream and CRC threads, made once
            self._reader = (_PINNED.get("read") or _pinned(2),
                            torch.cuda.Stream(device=self.device) if torch.cuda.is_available() else None,
                            ThreadPoolExecutor(4, thread_name_prefix=f"spill-crc-r{self.rank}"))
        pins, stream, pool = self._reader
        direct = self.direct and all(_aligned(p) for p in pins) and lj.data_at % ALIGN == 0
        flags = os.O_RDONLY
        fd = -1
        if direct:
            try:
                fd = os.open(lj.path, flags | os.O_DIRECT)
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
                direct = False
        if fd < 0:
            fd = os.open(lj.path, flags)
        try:
            if stream is not None and lj.after is not None:
                stream.wait_event(lj.after)
            table = sorted(((k, v) for k, v in lj.header.items() if k != "__metadata__"),
                           key=lambda kv: kv[1]["data_offsets"][0])
            end = max([v["data_offsets"][1] for _, v in table] or [0])
            if os.fstat(fd).st_size != lj.data_at + end:
                raise EOFError(f"{lj.path}: the file is shorter or longer than its header says")
            views = [(v["data_offsets"][0], v["data_offsets"][1], lj.dests[k].reshape(-1).view(torch.uint8))
                     for k, v in table]
            ev = [None, None]
            hashing: list = [[], []]                    # a pin's block hashes in flight (they read the pin)
            sums: list = []
            flip, s, ti = 0, 0, 0
            while s < end:
                k = min(CHUNK, end - s)
                pin = pins[flip]
                if ev[flip] is not None:
                    ev[flip].synchronize()
                for h in hashing[flip]:
                    h.result()
                want = _up(k) if direct else k
                mv = memoryview(pin.numpy())[:want]
                got = (os.preadv(fd, [mv], lj.data_at + s) if hasattr(os, "preadv")
                       else os.pread(fd, want, lj.data_at + s))
                if isinstance(got, bytes):
                    mv[:len(got)] = got
                    got = len(got)
                if got < k:
                    raise EOFError(f"{lj.path}: short read")
                hashing[flip] = [pool.submit(zlib.crc32, mv[a:b]) for a, b in _blocks(k)]
                sums += hashing[flip]
                lo, hi = s, s + k
                while ti < len(views) and views[ti][1] <= lo:
                    ti += 1
                j = ti
                ctx = torch.cuda.stream(stream) if stream is not None else None
                if ctx is not None:
                    ctx.__enter__()
                try:
                    while j < len(views) and views[j][0] < hi:
                        a, b, v = views[j]
                        x0, x1 = max(a, lo), min(b, hi)
                        if x1 > x0:
                            v[x0 - a:x1 - a].copy_(pin[x0 - lo:x1 - lo], non_blocking=stream is not None)
                        j += 1
                finally:
                    if ctx is not None:
                        ctx.__exit__(None, None, None)
                if stream is not None:
                    e = torch.cuda.Event()
                    e.record(stream)
                    ev[flip] = e
                flip ^= 1
                s = hi
                self.stats.loaded_bytes += k
            if stream is not None:
                e = torch.cuda.Event()
                e.record(stream)
                lj.event = e
                e.synchronize()
            stored = [int(c, 16) for c in str(lj.meta.get("crc32", "")).split(",") if c]
            seen = [h.result() for h in sums]
            if seen != stored:
                bad = next((i for i, (a, b) in enumerate(zip(seen, stored)) if a != b), min(len(seen), len(stored)))
                self.stats.crc_failed += 1
                raise ValueError(f"{lj.path}: block {bad} does not match its CRC-32")
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
        except BaseException:
            if stream is not None:                      # copies still queued write tensors the caller may free now
                stream.synchronize()
            raise
        finally:
            os.close(fd)
