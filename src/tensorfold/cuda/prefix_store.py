"""Transactional sleep snapshots; only a matching prefix returns to device memory."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from .streams import PrefixCache


def prefix_room(size: int, prompt: list[int]) -> bool:
    """Keep the normal single-request reserve beside a lazy snapshot allocation."""

    import torch

    from .capacity import GIB, available_bytes
    from .memory_gate import torch_live

    return torch_live(torch, available_bytes)() >= size + 2 * GIB


class PrefixStore:
    """One process owns its private files. Checkpoint weights are never stored here."""

    def __init__(self, directory: Path, identity: str, codec) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="sleep-", dir=directory)
        self.root = Path(self._temporary.name)
        self.identity, self.codec = identity, codec
        self.directory: Path | None = None
        self.records: list[dict] = []
        self.unusable: set[tuple[int, ...]] = set()
        self.unavailable_before = 0
        self.omitted = self.loads = self.failures = self.memory_misses = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._temporary.cleanup()

    def _manifest(self, records: list[dict]) -> dict:
        return {"format": 1, "identity": self.identity, "prefixes": records}

    def save(self, cache: PrefixCache) -> None:
        """Commit every retained live prefix before replacing the previous generation."""

        live = {tuple(ids): (ids, state, snap) for ids, state, snap in cache.entries}
        old = [r for r in self.records if tuple(r["tokens"]) not in live
               and tuple(r["tokens"]) not in self.unusable]
        selected = [(r, None) for r in old] + [(None, entry) for entry in live.values()]
        omitted = max(0, len(selected) - cache.keep)
        selected = selected[-cache.keep:] if cache.keep > 0 else []
        unavailable = self.unusable.difference(live)
        partial = Path(tempfile.mkdtemp(prefix=".partial-", dir=self.root))
        records = []
        try:
            for index, (previous, entry) in enumerate(selected):
                name = f"{index}.safetensors"
                path = partial / name
                if previous is None:
                    receipt = self.codec.save_prefix(path, *entry, identity=self.identity)
                else:
                    receipt = {k: v for k, v in previous.items() if k != "file"}
                    source = self.directory / previous["file"]
                    try:
                        self.codec.verify_prefix(source, receipt, identity=self.identity)
                    except (OSError, ValueError):
                        unavailable.add(tuple(receipt["tokens"]))
                        continue                 # its sole copy was already unavailable before this save
                    shutil.copyfile(source, path)
                os.chmod(path, 0o600)
                self.codec.verify_prefix(path, receipt, identity=self.identity)
                records.append({**receipt, "file": name})
            with (partial / "manifest.json").open("w") as stream:
                json.dump(self._manifest(records), stream)
                stream.flush()
                os.fsync(stream.fileno())
            published = self.root / ("generation-" + uuid.uuid4().hex)
            partial.rename(published)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        previous, self.directory = self.directory, published
        self.records, self.omitted = records, omitted
        self.unavailable_before = len(unavailable)
        self.unusable.clear()
        self.loads = self.failures = self.memory_misses = 0
        if previous is not None:
            shutil.rmtree(previous, ignore_errors=True)

    def verify(self) -> None:
        """A wake publishes no runtime until the whole saved generation validates."""

        if self.directory is None:
            return
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if manifest != self._manifest(self.records):
            raise ValueError("sleep cache manifest changed")
        for receipt in self.records:
            self.codec.verify_prefix(self.directory / receipt["file"], receipt, identity=self.identity)

    def snapshot(self) -> dict:
        return {"mode": "disk", "restore": "lazy", "saved_prefixes": len(self.records),
                "snapshot_bytes": sum(r["file_bytes"] for r in self.records),
                "omitted_prefixes": self.omitted,
                "unavailable_prefixes": self.unavailable_before + len(self.unusable),
                "loaded_prefixes": self.loads, "load_failures": self.failures,
                "memory_misses": self.memory_misses}

    def attach(self, cache: PrefixCache, *, device, room=None, admit=None) -> PrefixCache:
        return SavedPrefixCache(self, cache, device=device, room=room, admit=admit)

    def restore(self, prompt, *, length=0, device, room=None, admit=None):
        """Load only a strict prefix longer than the caller's resident match."""

        saved = max((r for r in self.records if length < len(r["tokens"]) < len(prompt)
                     and tuple(r["tokens"]) not in self.unusable
                     and list(prompt[:len(r["tokens"])]) == r["tokens"]),
                    key=lambda r: len(r["tokens"]), default=None)
        if saved is None:
            return None
        if admit is not None and not admit(saved["tensor_bytes"], prompt):
            self.memory_misses += 1
            return None
        try:
            entry = self.codec.load_prefix(self.directory / saved["file"], saved,
                                          identity=self.identity, device=device, room=room)
        except (OSError, ValueError, RuntimeError, MemoryError):
            # A removed/corrupted file or allocation failure leaves the resident cache intact.
            self.failures += 1
            self.unusable.add(tuple(saved["tokens"]))
            return None
        self.loads += 1
        return entry


class SavedPrefixCache(PrefixCache):
    """Resident entries retain the normal eviction policy; disk copies are immutable."""

    def __init__(self, store: PrefixStore, cache: PrefixCache, *, device, room, admit) -> None:
        super().__init__(cache.keep)
        self.entries, self.hit = cache.entries, cache.hit
        self.store, self.device, self.room, self.admit = store, device, room, admit

    def longest(self, prompt):
        best = max((e for e in self.entries if len(e[0]) < len(prompt)
                    and list(prompt[:len(e[0])]) == e[0]), key=lambda e: len(e[0]), default=None)
        length = len(best[0]) if best else 0
        loaded = self.store.restore(prompt, length=length, device=self.device, room=self.room, admit=self.admit)
        if loaded is not None:
            self.add(*loaded)
            best = self.entries[-1]
        return self._touch(best)


__all__ = ["PrefixStore"]
