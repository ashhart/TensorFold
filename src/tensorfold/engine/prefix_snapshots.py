"""Computed system blocks on disk, so a new agent session never waits for them.

An agent harness sends the same system-and-tools block (~16k tokens for one coding agent)
with every new session. The lane scheduler keeps a snapshot of that block's
cache after the first session, but only in memory: every server restart threw
it away and the first window after it paid the whole prefill again. Here the
snapshot is written once to disk, keyed by the model and a hash of its
tokens, and loaded into the checkpoint store when the server starts.

A cache is a list of per-layer objects (``KVCache``, ``ArraysCache``); each is
stored generically: its ``mx.array`` attributes (and lists of arrays) as
tensors, its plain attributes as JSON, its class by import path.
"""

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
    for index, item in enumerate(cache):
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
    partial = target.with_suffix(".partial.safetensors")
    mx.save_safetensors(str(partial), arrays, metadata=meta)
    partial.rename(target)
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


def load_snapshots(directory: Path, model_id: str, *, limit: int | None = None):
    """Stored snapshots for ``model_id``, newest first, at most ``limit``, one at a time.

    A generator: each snapshot is read only when the caller asks for it. Reading all of them
    into a list first held every block in memory at once (24 files, 33 GB, 2026-09-23).
    """

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
        loaded = load_snapshot(path, model_id)
        if loaded is None:
            continue
        count += 1
        yield loaded


def load_snapshot(path: Path, model_id: str) -> tuple[list[int], list[Any]] | None:
    """One stored block as (tokens, cache), evaluated in the calling thread; None if unusable."""

    try:
        arrays, meta = mx.load(str(path), return_metadata=True)
        # mx.load is lazy and bound to this thread's CPU stream; the scheduler
        # thread that uses the cache has none ("There is no Stream(cpu, 0)").
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
    """The stored blocks' tokens, so a request can read the one its prompt starts with.

    The server loads only the newest few blocks when it starts. On 2026-09-23 three short
    blocks from one side request (3,042 / 4,578 / 5,090 tokens) were newer than the
    main 21,415-token block, so a restart loaded those and left agent sessions to prefill
    the main block again. A block the checkpoint store lacks is now read when a prompt
    that starts with it arrives (~1.5 GB: well under a second, against ~40 s of prefill),
    and a used block's file is touched, so the newest files are the blocks in use.
    Metadata is read once per file (and again only if the file changes).
    """

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

    def best(self, prompt: Sequence[int], longer_than: int) -> tuple[Path, list[int]] | None:
        """The longest stored block that is a strict prefix of ``prompt`` and longer than ``longer_than``."""

        best: tuple[Path, list[int]] | None = None
        for path, tokens in self.blocks():
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
    """System blocks saved for the same model under another configuration but not yet under ``model_id``.

    Snapshots are keyed by the kernels that computed them (MLX version, attention
    mode), so a server started with different kernels has none of the blocks the
    user's agent sessions need. Only blocks of the same model (the part of the id
    before the first ``|``) count: another model's token ids mean nothing to this
    tokenizer. Returns the longest such token lists (a block that is a prefix of
    another is covered by warming the longer one), newest first.
    """

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
           "save_snapshot", "snapshot_key"]
