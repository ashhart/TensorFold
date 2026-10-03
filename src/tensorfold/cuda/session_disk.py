"""Prompt-end states on local disk (``--snapshot-dir``, ``--spill-gib``): one checked file an entry, across restarts."""
# A file: "TFSTATE1", the header's length, a JSON header, then 4 KiB-aligned segments, each with its SHA-256.

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .sessions import compat_hash, strict_prefix

MAGIC = b"TFSTATE1"
ALIGN = 4096
CHUNK = 8 << 20  # bytes one aligned staging write or read moves


def _up(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


class _File:
    """A descriptor opened O_DIRECT where the file system allows it (tmpfs does not)."""

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

    def drop_cache(self) -> None:
        """A buffered descriptor's pages out of the page cache (written back first)."""

        if not self.direct and hasattr(os, "posix_fadvise"):
            try:
                os.fdatasync(self.fd)
                os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass

    def close(self) -> None:
        os.close(self.fd)


class DiskTier:
    """A ``sessions.Tier`` under ``root/<compat hash>/rank<r>``: another build's entries live elsewhere, never read."""

    def __init__(
        self, root: str | Path, compat: dict, *, limit: int, rank: int = 0, min_tokens: int = 1, direct: bool = True
    ) -> None:
        self.dir = Path(root) / compat_hash(compat)[:16] / f"rank{rank}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "compat.json").write_text(json.dumps(compat, sort_keys=True, default=str))
        self.limit, self.min_tokens, self.direct = int(limit), int(min_tokens), bool(direct)
        self.index: dict[str, tuple[list[int], int, int]] = {}  # key -> (ids, file bytes, last use)
        self.by_len: dict[int, set[str]] = {}
        self.clock = 0
        self.lock = threading.Lock()
        self.reconcile()

    @property
    def used(self) -> int:
        return sum(size for _, size, _ in self.index.values())

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.tfs"

    def keys(self) -> list[str]:
        return sorted(self.index)

    def retain(self, keys: Sequence[str]) -> None:
        """Keep only ``keys``: every rank calls it with the keys all ranks hold, so a resume never misses on one."""

        keep = set(keys)
        for key in [k for k in self.index if k not in keep]:
            self.drop(key)

    def lengths(self) -> set[int]:
        with self.lock:
            return set(self.by_len)

    def _indexed(self, key: str, ids: list[int], size: int) -> None:
        self.index[key] = (ids, size, self.clock)
        self.by_len.setdefault(len(ids), set()).add(key)

    def has(self, key: str) -> bool:
        with self.lock:
            return key in self.index

    def find(self, prompt: Sequence[int]) -> tuple[str, int] | None:
        best = None
        for key, (ids, _, _) in self.index.items():
            if strict_prefix(ids, prompt) and (best is None or len(ids) > best[1]):
                best = (key, len(ids))
        return best

    def put(self, key: str, ids: Sequence[int], arrays: dict[str, np.ndarray], *, owned: bool = False) -> bool:
        segments = [("ids", np.asarray(ids, dtype=np.int32))] + sorted(arrays.items())
        table, at = [], 0
        for name, a in segments:
            raw = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            table.append(
                {
                    "name": name,
                    "dtype": a.dtype.str,
                    "shape": list(a.shape),
                    "offset": at,
                    "nbytes": int(raw.size),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            at = _up(at + raw.size)
        head = json.dumps({"key": key, "segments": table}).encode()
        lead = MAGIC + struct.pack("<Q", len(head)) + head
        base = _up(len(lead))
        total = base + at
        if len(ids) < self.min_tokens or total > self.limit:
            return False
        tmp = self.dir / f"{key}.tmp"
        f = _File(tmp, True, self.direct)
        try:
            stage = mmap.mmap(-1, CHUNK)  # page-aligned, as O_DIRECT needs
            try:
                self._write(f, stage, 0, lead)
                for (_, a), t in zip(segments, table):
                    self._write(f, stage, base + t["offset"], np.ascontiguousarray(a).view(np.uint8).reshape(-1))
            finally:
                stage.close()
            os.ftruncate(f.fd, total)
            f.drop_cache()
        finally:
            f.close()
        os.replace(tmp, self.path(key))
        with self.lock:
            self.clock += 1
            self._indexed(key, [int(t) for t in ids], total)
        self._trim(key)
        return True

    @staticmethod
    def _write(f: _File, stage: mmap.mmap, offset: int, data) -> None:
        view = memoryview(data).cast("B")
        done = 0
        while done < len(view):
            k = min(CHUNK, len(view) - done)
            stage.seek(0)
            stage.write(view[done : done + k])
            span = _up(k) if f.direct else k
            if span > k:
                stage.write(b"\0" * (span - k))
            out = memoryview(stage)[:span]
            try:
                wrote = os.pwritev(f.fd, [out], offset + done)
            finally:
                out.release()
            if wrote != span:
                raise OSError(f"short write ({wrote} of {span} bytes)")
            done += k

    def get(self, key: str) -> tuple[list[int], dict[str, np.ndarray]]:
        """(ids, arrays) with every segment checked; ValueError, and the file deleted, when it is damaged."""

        path = self.path(key)
        try:
            head, base = self._header(path)
            raw = self._read(path)
            out = {}
            for t in head["segments"]:
                a = raw[base + t["offset"] : base + t["offset"] + t["nbytes"]]
                if len(a) != t["nbytes"] or hashlib.sha256(a).hexdigest() != t["sha256"]:
                    raise ValueError(f"segment {t['name']} does not match its checksum")
                out[t["name"]] = a.view(np.dtype(t["dtype"])).reshape(t["shape"])
        except (OSError, ValueError, KeyError) as exc:
            self.drop(key)
            raise ValueError(f"prompt state {key}: {exc}") from None
        with self.lock:
            if key in self.index:
                self.clock += 1
                ids, size, _ = self.index[key]
                self.index[key] = (ids, size, self.clock)
        return [int(t) for t in out.pop("ids").tolist()], out

    def _read(self, path: Path) -> np.ndarray:
        size = os.path.getsize(path)
        f = _File(path, False, self.direct)
        try:
            span = _up(size) or ALIGN
            block = mmap.mmap(-1, span)
            try:
                view, got = memoryview(block), 0
                while got < size:
                    part = view[got : got + min(CHUNK, span - got)]
                    n = os.preadv(f.fd, [part], got)
                    part.release()
                    if n <= 0:
                        break
                    got += n
                view.release()
                if got < size:
                    raise ValueError(f"short read ({got} of {size} bytes)")
                return np.frombuffer(block, dtype=np.uint8, count=size).copy()
            finally:
                block.close()
        finally:
            f.drop_cache()
            f.close()

    @staticmethod
    def _header(path: Path) -> tuple[dict, int]:
        with open(path, "rb") as fh:
            lead = fh.read(16)
            if lead[:8] != MAGIC:
                raise ValueError("not a prompt state file")
            (n,) = struct.unpack("<Q", lead[8:16])
            return json.loads(fh.read(n)), _up(16 + n)

    def drop(self, key: str) -> None:
        with self.lock:
            gone = self.index.pop(key, None)
            if gone is not None:
                keys = self.by_len.get(len(gone[0]))
                if keys is not None:
                    keys.discard(key)
                    if not keys:
                        del self.by_len[len(gone[0])]
        try:
            self.path(key).unlink()
        except FileNotFoundError:
            pass

    def _trim(self, keep: str) -> None:
        """Least recently used entries out past the limit (never the one just written)."""

        while self.used > self.limit and len(self.index) > 1:
            self.drop(min((k for k in self.index if k != keep), key=lambda k: self.index[k][2]))

    def reconcile(self) -> int:
        """Index the directory's whole entries, oldest first, and delete temporary or unreadable files."""

        self.index.clear()
        self.by_len.clear()
        found = []
        for p in self.dir.iterdir():
            if p.suffix == ".tmp":
                p.unlink()
                continue
            if p.suffix != ".tfs":
                continue
            try:
                head, base = self._header(p)
                ids_t = next(t for t in head["segments"] if t["name"] == "ids")
                end = base + max(t["offset"] + t["nbytes"] for t in head["segments"])
                if head["key"] != p.stem or p.stat().st_size < end:
                    raise ValueError("truncated or renamed")
                with open(p, "rb") as fh:
                    fh.seek(base + ids_t["offset"])
                    raw = fh.read(ids_t["nbytes"])
                if hashlib.sha256(raw).hexdigest() != ids_t["sha256"]:
                    raise ValueError("ids do not match their checksum")
                found.append(
                    (p.stat().st_mtime_ns, p.stem, np.frombuffer(raw, dtype=np.int32).tolist(), p.stat().st_size)
                )
            except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError):
                p.unlink()
        for _, key, ids, size in sorted(found):
            self.clock += 1
            self._indexed(key, ids, size)
        while self.used > self.limit and self.index:
            self.drop(min(self.index, key=lambda k: self.index[k][2]))
        return len(self.index)


__all__ = ["DiskTier"]
