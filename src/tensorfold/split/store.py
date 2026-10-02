"""The stage's weight cache: each unit one safetensors file named by its key, and a stage directory of links to them."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

ALIGN = 4096             # the data starts on a page, so direct reads of the file need no realignment


def _header(tensors: list[dict]) -> bytes:
    entries, at = {}, 0
    for t in tensors:
        entries[t["name"]] = {"dtype": t["dtype"], "shape": list(t["shape"]), "data_offsets": [at, at + int(t["nbytes"])]}
        at += int(t["nbytes"])
    raw = json.dumps(entries, separators=(",", ":")).encode()
    n = -(-(8 + len(raw)) // ALIGN) * ALIGN - 8         # the JSON padded with spaces, as safetensors allows
    return n.to_bytes(8, "little") + raw + b" " * (n - len(raw))


class Store:
    """Units under ``root/units``; stages under ``root/stages`` link the units a layer range reads."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.units = self.root / "units"
        self.stages = self.root / "stages"
        self.units.mkdir(parents=True, exist_ok=True)
        self.stages.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.units / f"{key}.safetensors"

    def has(self, unit: dict) -> bool:
        p = self.path(unit["key"])
        return p.is_file() and p.stat().st_size == len(_header(unit["tensors"])) + int(unit["nbytes"])

    def missing(self, units: list[dict]) -> list[str]:
        return [u["key"] for u in units if not self.has(u)]

    def receive(self, unit: dict, read) -> None:
        """Write a unit from ``read(n) -> bytes`` (its tensors' bytes in listed order), atomically."""

        w = self.writer(unit)
        while w.left:
            chunk = read(min(w.left, 16 << 20))
            if not chunk:
                w.abort()
                raise ConnectionError(f"the stream ended {w.left} bytes before unit {unit['label']} did")
            w.write(chunk)
        w.finish()

    def writer(self, unit: dict) -> "UnitWriter":
        return UnitWriter(self.path(unit["key"]), unit)

    def stage(self, config: dict, units: list[dict]) -> Path:
        """A directory the family loader reads: config.json plus links to the units' files."""

        h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
        for u in units:
            h.update(u["key"].encode())
        where = self.stages / h.hexdigest()[:24]
        if (where / "config.json").is_file():
            return where
        tmp = where.with_name(where.name + f".{os.getpid()}.part")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        (tmp / "config.json").write_text(json.dumps(config))
        weight_map = {}
        for u in units:
            name = f"{u['label'].replace('.', '-')}-{u['key'][:12]}.safetensors"
            (tmp / name).symlink_to(self.path(u["key"]))
            weight_map.update({t["name"]: name for t in u["tensors"]})
        (tmp / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
        shutil.rmtree(where, ignore_errors=True)
        tmp.replace(where)
        return where


class UnitWriter:
    """One unit's file, written in pieces as they arrive and renamed into place once whole."""

    def __init__(self, final: Path, unit: dict) -> None:
        self.final, self.label = final, unit["label"]
        self.tmp = final.with_name(final.name + f".{os.getpid()}.part")
        self.left = int(unit["nbytes"])
        self.f = open(self.tmp, "wb")
        self.f.write(_header(unit["tensors"]))

    def write(self, chunk) -> None:
        n = len(chunk)
        if n > self.left:
            self.abort()
            raise ValueError(f"unit {self.label}: {n - self.left} bytes more than it holds")
        self.f.write(chunk)
        self.left -= n

    def finish(self) -> None:
        if self.left:
            self.abort()
            raise ValueError(f"unit {self.label}: {self.left} bytes missing")
        self.f.close()
        self.tmp.replace(self.final)

    def abort(self) -> None:
        self.f.close()
        self.tmp.unlink(missing_ok=True)
