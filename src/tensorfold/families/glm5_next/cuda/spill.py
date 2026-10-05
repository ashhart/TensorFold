"""The CUDA spill tier: kept prompt states (``Snapshot``) written to a per-rank disk directory when evicted,
read back on a matching prompt's resume, and saved at clean shutdown. Rank 0 decides hits as it does for
in-memory ones; each rank writes and reads its own ``{key}.rank{r}.safetensors``, both ranks deciding alike.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import struct
import threading
import time
from typing import Sequence

import torch

from tensorfold.engine.prefix_snapshots import FORMAT, PARTIAL, read_metadata, remove_stale_partials, snapshot_key

# torch dtype -> safetensors dtype name; buffers are read back with ``torch.frombuffer``, so bf16 and the
# float8 formats need no numpy equivalent
DTYPES = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16", torch.int64: "I64",
          torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
          torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2"}
NAMES = {name: dtype for dtype, name in DTYPES.items()}


def rank_path(directory: Path, key: str, rank: int) -> Path:
    return Path(directory) / f"{key}.rank{rank}.safetensors"


def _pack(tensor: torch.Tensor, arrays: dict[str, tuple[str, list[int], bytes]], name: str) -> None:
    """One contiguous tensor's dtype name, shape and raw little-endian bytes under ``name``."""

    flat = tensor.detach().cpu().contiguous()
    arrays[name] = (DTYPES[flat.dtype], list(flat.shape), flat.view(torch.uint8).numpy().tobytes())


class SpillJob:
    """The fields of a dying snapshot, captured at enqueue time (the engine clears the object's own)."""

    def __init__(self, snap) -> None:
        self.ids, self.rec, self.conv, self.pending = list(snap.ids), snap.rec, snap.conv, snap.pending
        self.mtp_len, self.drafter_end = snap.mtp_len, snap.drafter_end
        self.rows, self.drafter_rows = snap.rows, snap.drafter_rows


def write(directory: Path, model_id: str, rank: int, job, *, limit_bytes: int = 0) -> Path | None:
    """One snapshot's tensors to ``{key}.rank{rank}.safetensors`` (partial first, renamed when whole)."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = snapshot_key(model_id, job.ids)
    target = rank_path(directory, key, rank)
    if target.exists():
        os.utime(target)
        return None
    arrays: dict[str, tuple[str, list[int], bytes]] = {}
    _pack(job.rec, arrays, "rec")
    _pack(job.conv, arrays, "conv")
    if job.pending is not None:
        _pack(job.pending, arrays, "pending")
    for i, row in enumerate(job.rows or ()):
        _pack(row, arrays, f"rows.{i}")
    for i, row in enumerate(job.drafter_rows or ()):
        _pack(row, arrays, f"drafter_rows.{i}")
    meta = {"format": str(FORMAT), "model": model_id, "tokens": json.dumps([int(t) for t in job.ids]),
            "state": json.dumps({"mtp_len": int(job.mtp_len), "drafter_end": int(job.drafter_end),
                                 "rows": len(job.rows or ()), "drafter_rows": len(job.drafter_rows or ())}),
            "saved": str(time.time())}
    at = 0
    header = {}
    for name, (dtype, shape, data) in arrays.items():   # the header and the data share this one order
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(data)]}
        at += len(data)
    blob = json.dumps({**header, "__metadata__": meta}, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    partial = target.with_name(f"{key}.{os.getpid()}{PARTIAL}")
    try:
        with open(partial, "wb") as handle:
            handle.write(struct.pack("<Q", len(blob)))
            handle.write(blob)
            for _, (_, _, data) in arrays.items():   # the order the header's data_offsets were computed in
                handle.write(data)
        partial.rename(target)
    except BaseException:
        partial.unlink(missing_ok=True)   # a full disk must not leave these bytes outside every byte budget
        raise
    if limit_bytes > 0:
        prune(directory, model_id, rank, limit_bytes)
    return target


def prune(directory: Path, model_id: str, rank: int, limit_bytes: int) -> int:
    """This rank's oldest files of this model until the directory stays under ``limit_bytes``; bytes freed."""

    directory = Path(directory)
    if not directory.is_dir():
        return 0
    mine = []
    for path in directory.glob(f"*.rank{rank}.safetensors"):
        try:
            if read_metadata(path).get("model", "").split("|")[0] == model_id.split("|")[0]:
                mine.append(path)
        except Exception:  # noqa: BLE001 - an unreadable file is left alone
            continue
    mine.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    freed, total = 0, sum(p.stat().st_size for p in mine)
    for stale in reversed(mine):
        if total <= limit_bytes:
            break
        size = stale.stat().st_size
        if stale.unlink(missing_ok=True) is None:
            total -= size
            freed += size
    return freed


def best(directory: Path, model_id: str, rank: int, prompt: Sequence[int], longer_than: int) -> list[list[int]]:
    """This rank's stored strict prefixes of ``prompt`` longer than ``longer_than``, longest first."""

    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = []
    for path in directory.glob(f"*.rank{rank}.safetensors"):
        try:
            meta = read_metadata(path)
        except Exception:  # noqa: BLE001 - a bad file is skipped, never fatal
            continue
        if meta.get("format") != str(FORMAT) or meta.get("model") != model_id:
            continue
        try:
            tokens = [int(t) for t in json.loads(meta["tokens"])]
        except Exception:  # noqa: BLE001 - unreadable tokens are skipped
            continue
        if longer_than < len(tokens) < len(prompt) and list(prompt[:len(tokens)]) == tokens:
            found.append(tokens)
    return sorted(found, key=len, reverse=True)


def load(directory: Path, model_id: str, rank: int, tokens: Sequence[int], device: str | torch.device) -> object | None:
    """The snapshot of exactly ``tokens``, its tensors on ``device``; None when this rank has no such file."""

    path = rank_path(Path(directory), snapshot_key(model_id, tokens), rank)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
            data = handle.read()
    except Exception:  # noqa: BLE001 - a bad file is skipped, never fatal
        return None
    meta = header.get("__metadata__") or {}
    if meta.get("format") != str(FORMAT) or meta.get("model") != model_id:
        return None
    state = json.loads(meta["state"])

    def tensor(name: str) -> torch.Tensor | None:
        if name not in header:
            return None
        field = header[name]
        flat = torch.frombuffer(memoryview(data)[field["data_offsets"][0]:field["data_offsets"][1]],
                                dtype=NAMES[field["dtype"]])
        return flat.reshape(field["shape"]).to(device)

    from .decode import Snapshot

    snap = Snapshot([int(t) for t in json.loads(meta["tokens"])], tensor("rec"), tensor("conv"), tensor("pending"),
                    int(state["mtp_len"]), int(state["drafter_end"]))
    snap.rows = [tensor(f"rows.{i}") for i in range(int(state["rows"]))]
    snap.nbytes = sum(r.numel() * r.element_size() for r in snap.rows)
    snap.drafter_rows = [tensor(f"drafter_rows.{i}") for i in range(int(state["drafter_rows"]))]
    return snap


class SpillWriter:
    """A daemon thread that copies evicted snapshots' device tensors to disk off the request path."""

    def __init__(self, directory: Path, model_id: str, rank: int, limit_bytes: int) -> None:
        self.directory, self.model_id, self.rank, self.limit_bytes = Path(directory), model_id, rank, int(limit_bytes)
        self.jobs: queue.Queue = queue.Queue()
        self.failed: BaseException | None = None
        self.stopped = threading.Event()
        threading.Thread(target=self._work, name="glm-spill", daemon=True).start()

    def enqueue(self, snap) -> None:
        self.jobs.put(SpillJob(snap))

    def flush(self) -> None:
        """Wait until every enqueued snapshot is written; a failed write is loud here, not on the request path."""

        self.jobs.join()
        if self.failed is not None:
            raise self.failed

    def close(self) -> None:
        self.stopped.set()
        self.jobs.put(None)

    def _work(self) -> None:
        while True:
            snap = self.jobs.get()
            try:
                if snap is None or self.stopped.is_set():
                    return
                if torch.cuda.is_available():
                    torch.cuda.synchronize()      # the rows are dead to the engine; copy them whole
                write(self.directory, self.model_id, self.rank, snap, limit_bytes=self.limit_bytes)
            except Exception as exc:  # noqa: BLE001 - a failed spill costs a later prefill, never the request
                self.failed = exc
            finally:
                self.jobs.task_done()


def startup(directory: Path) -> int:
    """Delete partial writes of processes that have ended; return the bytes freed."""

    return remove_stale_partials(Path(directory))
