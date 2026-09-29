"""Checkpoint bytes read with O_DIRECT in 64 MiB spans: to the GPU through two pinned pieces, or into host memory."""

from __future__ import annotations

import errno
import json
import os
import struct
from pathlib import Path

import torch

PIECE = 64 << 20         # bytes a direct read fills: the size of each pinned staging piece
ALIGN = 4096             # O_DIRECT's file offset, length and buffer alignment

DTYPES = {"BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "U16": torch.uint16, "I16": torch.int16,
          "U32": torch.uint32, "I32": torch.int32, "U64": torch.uint64, "I64": torch.int64, "F16": torch.float16,
          "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64, "F8_E4M3": torch.float8_e4m3fn,
          "F8_E5M2": torch.float8_e5m2}


def _up(x: int) -> int:
    return -(-x // ALIGN) * ALIGN


class Reader:
    """Byte ranges of files as uint8 tensors on a device; ``direct`` is False once O_DIRECT has been refused."""

    def __init__(self) -> None:
        self.direct = hasattr(os, "O_DIRECT")
        self.staging: list[list] = []         # [pinned piece, event of its last copy]
        self.turn = 0

    def read(self, path: str | Path, offset: int, n: int, device: str | torch.device = "cpu", *,
             pinned: bool = False) -> torch.Tensor:
        """Bytes [offset, offset + n) of ``path`` as a new uint8 tensor on ``device``; ``pinned``: in page-locked host
        memory, for a caller's own asynchronous upload (a direct read only; the buffered fallback's are pageable)."""

        cuda = torch.device(device).type == "cuda"
        if n > 0 and self.direct:
            try:
                return self._to_device(path, offset, n, device) if cuda else self._to_host(path, offset, n, pinned)
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
                self.direct = False               # the file system refuses O_DIRECT, on the open or on a read
        raw = self._buffered(path, offset, n)
        return raw.to(device) if cuda else raw

    def _buffered(self, path, offset: int, n: int) -> torch.Tensor:
        raw = torch.empty((n,), dtype=torch.uint8)
        view = memoryview(raw.numpy())
        with open(path, "rb", buffering=0) as f:
            f.seek(offset)
            at = 0
            while at < n:
                got = f.readinto(view[at:at + PIECE])
                if not got:
                    raise IOError(f"short read of {path} at {offset + at}")
                at += got
        return raw

    @staticmethod
    def _fill(fd: int, view: memoryview, lo: int, need: int, path) -> None:
        """Read the aligned span at ``lo`` into ``view`` until ``need`` bytes arrived (the span may run past EOF)."""

        got = 0
        while got < need:
            k = os.preadv(fd, [view[got:]], lo + got)
            if k <= 0:
                raise IOError(f"short read of {path} at {lo + got}")
            got += k

    def _to_host(self, path, offset: int, n: int, pinned: bool = False) -> torch.Tensor:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            size = os.fstat(fd).st_size
            if offset + n > size:                 # a truncated file: the loop below would never reach n bytes
                raise IOError(f"short read of {path}: bytes {offset}-{offset + n} past its end ({size})")
            lo, end = offset // ALIGN * ALIGN, _up(size)
            hi = min(_up(offset + n), end)
            block = torch.empty((hi - lo + ALIGN,), dtype=torch.uint8, pin_memory=pinned)
            lead = -block.data_ptr() % ALIGN      # host blocks need not be page-aligned: align the span here
            view = memoryview(block[lead:lead + hi - lo].numpy())
            skip = offset - lo
            at = 0
            while at < skip + n:                  # one read of at most PIECE bytes at a time
                take = min(PIECE, hi - lo - at)
                self._fill(fd, view[at:at + take], lo + at, min(take, skip + n - at), path)
                at += take
        finally:
            os.close(fd)
        out = block[lead + skip:lead + skip + n]
        if pinned or (lead + skip) % 8 == 0:      # pinned: uploaded as bytes, and a clone would not be pinned
            return out
        return out.clone()                        # views as any dtype need an aligned start

    def _to_device(self, path, offset: int, n: int, device) -> torch.Tensor:
        if not self.staging:
            for _ in range(2):                    # pinned blocks need not be page-aligned: align the pieces here
                block = torch.empty((PIECE + 3 * ALIGN,), dtype=torch.uint8, pin_memory=True)
                lead = -block.data_ptr() % ALIGN
                self.staging.append([block[lead:lead + PIECE + 2 * ALIGN], None])
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            size = os.fstat(fd).st_size
            if offset + n > size:
                raise IOError(f"short read of {path}: bytes {offset}-{offset + n} past its end ({size})")
            end = _up(size)
            out = torch.empty((n,), dtype=torch.uint8, device=device)
            stream = torch.cuda.current_stream(out.device)   # the copies' stream, whichever device is current
            at = 0
            while at < n:
                take = min(PIECE, n - at)
                slot = self.staging[self.turn]
                self.turn ^= 1
                piece, copied = slot
                if copied is not None:
                    copied.synchronize()          # the piece's previous copy has finished
                lo = (offset + at) // ALIGN * ALIGN
                hi = min(_up(offset + at + take), end)
                skip = offset + at - lo
                self._fill(fd, memoryview(piece.numpy())[:hi - lo], lo, skip + take, path)
                out[at:at + take].copy_(piece[skip:skip + take], non_blocking=True)
                slot[1] = torch.cuda.Event()
                slot[1].record(stream)
                at += take
            return out
        finally:
            os.close(fd)


def read_header(path: str | Path) -> tuple[int, dict]:
    """(offset of the data, header) of a safetensors file."""

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


class SafeTensors:
    """Tensors by name from safetensors files (a later file's name wins), each read whole through one ``Reader``."""

    def __init__(self, files, reader: Reader | None = None) -> None:
        self.reader = reader or Reader()
        self.where: dict[str, tuple[Path, int, int, str, list[int]]] = {}   # name -> (file, begin, bytes, dtype, shape)
        for path in files:
            base, header = read_header(path)
            for name, e in header.items():
                if name != "__metadata__":
                    begin, end = e["data_offsets"]
                    self.where[name] = (Path(path), base + begin, end - begin, e["dtype"], list(e["shape"]))

    def keys(self) -> list[str]:
        return list(self.where)

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def get(self, name: str, device: str | torch.device = "cpu") -> torch.Tensor:
        path, begin, n, dtype, shape = self.where[name]
        if dtype not in DTYPES:
            raise ValueError(f"{name}: safetensors dtype {dtype} is not supported")
        return self.reader.read(path, begin, n, device).view(DTYPES[dtype]).reshape(shape)


class ReadAhead:
    """Tensors read ahead on ``threads`` threads: neighbouring byte ranges of a file come in one direct read of up to
    ``run`` bytes (ranges ``gap`` or fewer bytes apart), cut into the tensors. With a CUDA device, each read goes up in
    one asynchronous copy on a stream of its own and is cut there, so ``take`` returns device tensors whose reads and
    uploads overlapped the caller's work. Many reads in flight keep the SSD's queue full: one at a time leaves it idle
    between requests."""

    def __init__(self, reader: Reader | None = None, threads: int = 8, run: int = 128 << 20,
                 gap: int = 1 << 20) -> None:
        self.reader = reader or Reader()
        self.threads, self.run, self.gap = threads, run, gap
        self.ahead: dict = {}                              # key -> the read's future: (upload event or None, tensors)
        self.pool = None
        self.stream = None

    def queue(self, items, device=None, cut=None) -> None:
        """Start reading ``items``, each (key, path, first byte, end byte, meta), those not queued yet. ``cut(raw,
        meta)`` makes the tensor from its bytes (uint8, on the host or the device, a view of the shared read, so it
        must copy); by default the bytes' own copy."""

        from concurrent.futures import ThreadPoolExecutor

        cut = cut or (lambda raw, meta: raw.clone())
        if device is not None and torch.device(device).type != "cuda":
            device = None
        if self.pool is None:
            self.pool = ThreadPoolExecutor(self.threads, thread_name_prefix="read-ahead")
        if device is not None and self.stream is None:
            self.stream = torch.cuda.Stream(torch.device(device))
        by_path: dict[str, list] = {}
        for item in items:
            if item[0] not in self.ahead:
                by_path.setdefault(str(item[1]), []).append(item)
        for path, group in by_path.items():
            group.sort(key=lambda item: item[2])
            run: list = []
            hi = 0
            for item in group + [None]:
                if run and (item is None or item[2] - hi > self.gap or item[3] - run[0][2] > self.run):
                    future = self.pool.submit(self._read, path, run[0][2], hi, run, device, cut)
                    for queued in run:
                        self.ahead[queued[0]] = future
                    run = []
                if item is not None:
                    hi = max(hi, item[3]) if run else item[3]
                    run.append(item)

    def take(self, key):
        """``key``'s tensor, once read (on a device, the caller's stream waits for its upload); None if not queued."""

        future = self.ahead.pop(key, None)
        if future is None:
            return None
        uploaded, tensors = future.result()
        out = tensors.pop(key)
        if uploaded is not None:
            stream = torch.cuda.current_stream(out.device)
            stream.wait_event(uploaded)
            out.record_stream(stream)
        return out

    def drop(self, keys) -> None:
        """Forget queued tensors nobody will take, so their copies are freed once read."""

        for key in keys:
            future = self.ahead.pop(key, None)
            if future is not None:
                future.add_done_callback(lambda f, k=key: f.cancelled() or f.exception() or f.result()[1].pop(k, None))

    def close(self) -> None:
        """Cancel the reads not started, wait for the rest, and give the uploads' pinned buffers back."""

        self.ahead.clear()
        if self.pool is not None:
            self.pool.shutdown(cancel_futures=True)
            self.pool = None
        if self.stream is not None:
            self.stream.synchronize()
            getattr(torch._C, "_host_emptyCache", lambda: None)()
            self.stream = None

    def _read(self, path: str, lo: int, hi: int, run: list, device, cut) -> tuple:
        if device is None:
            raw = self.reader.read(path, lo, hi - lo)
            return None, {key: cut(raw[b - lo:e - lo], meta) for key, _, b, e, meta in run}
        host = self.reader.read(path, lo, hi - lo, pinned=True)
        with torch.cuda.device(torch.device(device)), torch.cuda.stream(self.stream):
            raw = host.to(device, non_blocking=True)
            out = {key: cut(raw[b - lo:e - lo], meta) for key, _, b, e, meta in run}
            uploaded = torch.cuda.Event()
            uploaded.record(self.stream)
        return uploaded, out


def in_background(job, futures: list) -> None:
    """Start ``job`` on a thread of its own and append its future to ``futures`` (``wait_all`` waits for them)."""

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(1, thread_name_prefix="background-read")
    futures.append(pool.submit(job))
    pool.shutdown(wait=False)                        # the thread ends with its one job


def wait_all(futures: list) -> None:
    """Wait for every future (none is left running), then raise the first one's error, if any."""

    errors = [future.exception() for future in futures]
    futures.clear()
    for error in errors:
        if error is not None:
            raise error


__all__ = ["ALIGN", "DTYPES", "PIECE", "ReadAhead", "Reader", "SafeTensors", "in_background", "read_header", "wait_all"]
