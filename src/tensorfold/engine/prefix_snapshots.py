"""Store system-prefix caches as tensors and JSON keyed by model and tokens, restoring layer classes by import path."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

import mlx.core as mx
import numpy as np

FORMAT = 1
DEFAULT_DIR = Path.home() / ".cache" / "tensorfold" / "prefix-snapshots"
PARTIAL = ".partial.safetensors"
# a partial without its writer's pid (0.6.0 and earlier) may be another server's write in progress until it is this old
UNNAMED_PARTIAL_SECONDS = 3600


def remove_stale_partials(directory: Path) -> int:
    """Delete the partial writes of processes that have ended (a server stopped mid-write); return the bytes freed."""

    if not directory.is_dir():
        return 0
    freed, now = 0, time.time()
    for path in directory.glob(f"*{PARTIAL}"):
        try:
            stat = path.stat()
            if not _abandoned(path.name, stat.st_mtime, now):
                continue
            path.unlink()
        except OSError:
            continue                          # gone already, or not ours to remove
        freed += stat.st_size
    return freed


def _abandoned(name: str, mtime: float, now: float) -> bool:
    parts = name[:-len(PARTIAL)].split(".")
    if len(parts) == 2 and parts[1].isdigit():
        return not _running(int(parts[1]))
    return now - mtime > UNNAMED_PARTIAL_SECONDS


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                   # another user's process: running
        return True
    return True


def snapshot_key(model_id: str, tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(model_id.encode())
    digest.update(b"\0")
    digest.update(",".join(str(int(t)) for t in tokens).encode())
    return digest.hexdigest()[:32]


def save_snapshot(directory: Path, model_id: str, tokens: Sequence[int], cache: list[Any],
                  *, keep: int = 8) -> Path | None:
    """Write one snapshot unless it is already there; keep the ``keep`` newest."""

    directory.mkdir(parents=True, exist_ok=True)
    key = snapshot_key(model_id, tokens)
    target = directory / f"{key}.safetensors"
    if target.exists():
        os.utime(target)
        return None
    arrays: dict[str, mx.array] = {}
    layers: list[dict[str, Any]] = []
    # an entry with ``stored = False`` (a stream's drafter state) is left out: the family adds a fresh one on resume
    for index, item in enumerate(item for item in cache if getattr(item, "stored", True)):
        materialize = getattr(item, "materialize", None)
        if materialize is not None:
            materialize()
        cls = type(item)
        entry: dict[str, Any] = {"class": f"{cls.__module__}:{cls.__qualname__}", "plain": {},
                                 "arrays": [], "lists": {}, "numpy": []}
        transient = set(getattr(cls, "transient", ()))    # runtime-only state the class rebuilds itself
        for name, value in vars(item).items():
            if name in transient:
                continue
            if isinstance(value, mx.array):
                arrays[f"{index}.{name}"] = value
                entry["arrays"].append(name)
            elif isinstance(value, np.ndarray):          # host-side state (token history of an n-gram layer)
                arrays[f"{index}.{name}"] = mx.array(value)
                entry["numpy"].append(name)
            elif isinstance(value, list) and any(isinstance(v, mx.array) for v in value):
                slots = []
                for slot, element in enumerate(value):
                    if isinstance(element, mx.array):
                        arrays[f"{index}.{name}.{slot}"] = element
                        slots.append(slot)
                    elif element is not None:
                        raise TypeError(f"cannot store {name}[{slot}] of {cls.__name__}")
                entry["lists"][name] = {"length": len(value), "slots": slots}
            elif value is None or isinstance(value, (bool, int, float, str)):
                entry["plain"][name] = value
            else:
                raise TypeError(f"cannot store attribute {name} of {cls.__name__}")
        layers.append(entry)
    meta = {"format": str(FORMAT), "model": model_id, "tokens": json.dumps([int(t) for t in tokens]),
            "layers": json.dumps(layers), "saved": str(time.time())}
    partial = target.with_name(f"{key}.{os.getpid()}{PARTIAL}")      # this process's own: startup knows if it ended
    try:
        mx.save_safetensors(str(partial), arrays, metadata=meta)
        partial.rename(target)
    except BaseException:
        partial.unlink(missing_ok=True)   # a full disk would otherwise keep these bytes, outside every byte budget
        raise
    # keep the newest ``keep`` of this model only: another model's blocks are not this one's to evict
    ours = []
    for path in sorted(directory.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name.endswith(".partial.safetensors"):
            continue
        try:
            same = str(read_metadata(path).get("model", "")).split("|")[0] == model_id.split("|")[0]
        except Exception:  # noqa: BLE001 - an unreadable file is left alone
            continue
        if same:
            ours.append(path)
    for stale in ours[keep:]:
        stale.unlink(missing_ok=True)
    return target


def load_snapshots(directory: Path, model_id: str, *, limit: int | None = None, allow: Any = None):
    """Yield newest model snapshots one at a time so unread blocks do not occupy memory."""

    if not directory.is_dir():
        return
    count = 0
    files = sorted(directory.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        if limit is not None and count >= limit:
            return
        if path.name.endswith(".partial.safetensors"):
            continue
        try:
            if read_metadata(path).get("model") != model_id:
                continue                          # another configuration's block: not read at all
        except Exception:  # noqa: BLE001 - a bad file is skipped, never fatal
            continue
        if allow is not None and not allow(path):
            continue
        loaded = load_snapshot(path, model_id)
        if loaded is None:
            continue
        count += 1
        yield loaded


def load_snapshot(path: Path, model_id: str) -> tuple[list[int], list[Any]] | None:
    """One stored block as (tokens, cache), evaluated in the calling thread; None if unusable."""

    try:
        arrays, meta = mx.load(str(path), return_metadata=True)
        # Evaluate lazy loads here because the scheduler thread has no CPU stream for their arrays.
        mx.eval(list(arrays.values()))
    except Exception:  # noqa: BLE001 - a bad file is skipped, never fatal
        return None
    if meta.get("format") != str(FORMAT) or meta.get("model") != model_id:
        return None
    cache: list[Any] = []
    for index, entry in enumerate(json.loads(meta["layers"])):
        module_name, qualname = entry["class"].split(":")
        cls: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            cls = getattr(cls, part)
        item = cls.__new__(cls)
        for name, value in entry["plain"].items():
            setattr(item, name, value)
        for name in entry["arrays"]:
            setattr(item, name, arrays[f"{index}.{name}"])
        for name in entry.get("numpy", []):
            setattr(item, name, np.array(arrays[f"{index}.{name}"]))
        for name, spec in entry["lists"].items():
            values: list[Any] = [None] * int(spec["length"])
            for slot in spec["slots"]:
                values[slot] = arrays[f"{index}.{name}.{slot}"]
            setattr(item, name, values)
        cache.append(item)
    return [int(t) for t in json.loads(meta["tokens"])], cache


class DiskBlocks:
    """Index stored prefixes for on-demand loading, refreshing metadata only when files change and touching used blocks for startup priority."""

    def __init__(self, directory: Path, model_id: str) -> None:
        self.directory = Path(directory)
        self.model_id = model_id
        self._known: dict[Path, tuple[float, list[int] | None]] = {}

    def blocks(self) -> list[tuple[Path, list[int]]]:
        if not self.directory.is_dir():
            return []
        known: dict[Path, tuple[float, list[int] | None]] = {}
        for path in self.directory.glob("*.safetensors"):
            if path.name.endswith(".partial.safetensors"):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            entry = self._known.get(path)
            if entry is None or entry[0] != mtime:
                tokens: list[int] | None = None
                try:
                    meta = read_metadata(path)
                    if meta.get("model") == self.model_id:
                        tokens = [int(t) for t in json.loads(meta["tokens"])]
                except Exception:  # noqa: BLE001 - an unreadable file is skipped
                    tokens = None
                entry = (mtime, tokens)
            known[path] = entry
        self._known = known
        return [(path, tokens) for path, (_, tokens) in known.items() if tokens]

    def best(self, prompt: Sequence[int], longer_than: int, usable: Any = None) -> tuple[Path, list[int]] | None:
        """The longest stored strict prefix of ``prompt`` longer than ``longer_than``, of a length ``usable`` takes."""

        best: tuple[Path, list[int]] | None = None
        for path, tokens in self.blocks():
            if usable is not None and not usable(len(tokens)):
                continue
            if longer_than < len(tokens) < len(prompt) and list(prompt[:len(tokens)]) == tokens:
                if best is None or len(tokens) > len(best[1]):
                    best = (path, tokens)
        return best

    def touch(self, tokens: Sequence[int]) -> None:
        """Mark the stored block with exactly these tokens as just used (newest first at startup)."""

        wanted = [int(t) for t in tokens]
        for path, (mtime, known) in list(self._known.items()):
            if known is not None and len(known) == len(wanted) and known == wanted:
                try:
                    os.utime(path)
                    self._known[path] = (path.stat().st_mtime, known)
                except OSError:
                    pass


def read_metadata(path: Path) -> dict[str, str]:
    """A safetensors file's metadata from its header, without loading any tensor."""

    import struct

    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return dict(header.get("__metadata__") or {})


def blocks_to_warm(directory: Path, model_id: str) -> list[list[int]]:
    """Return longest uncovered token prefixes from other kernel configurations of the same model, newest first; other models' token ids are incompatible."""

    if not directory.is_dir():
        return []
    have: list[list[int]] = []
    other: list[tuple[float, list[int]]] = []
    for path in directory.glob("*.safetensors"):
        if path.name.endswith(".partial.safetensors"):
            continue
        try:
            meta = read_metadata(path)
            tokens = [int(t) for t in json.loads(meta["tokens"])]
        except Exception:  # noqa: BLE001 - an unreadable file is skipped
            continue
        if meta.get("model") == model_id:
            have.append(tokens)
        elif str(meta.get("model", "")).split("|")[0] == model_id.split("|")[0]:
            other.append((path.stat().st_mtime, tokens))
    other.sort(key=lambda item: item[0], reverse=True)
    wanted: list[list[int]] = []
    for _, tokens in other:
        if any(tokens == h for h in have) or any(tokens == w or w[:len(tokens)] == tokens for w in wanted):
            continue
        wanted = [w for w in wanted if tokens[:len(w)] != w]  # a longer block covers its prefixes
        wanted.append(tokens)
    return wanted


__all__ = ["DEFAULT_DIR", "DiskBlocks", "blocks_to_warm", "load_snapshot", "load_snapshots", "read_metadata",
           "remove_stale_partials", "save_snapshot", "snapshot_key"]
