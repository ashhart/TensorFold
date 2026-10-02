"""A checkpoint's tensors grouped into the units a stage caches: one per layer, one for the final norm."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
HEX64 = re.compile(r"[0-9a-f]{64}")
SIDECAR = Path(os.environ.get("TF_SPLIT_DIGESTS", Path.home() / ".cache/tensorfold/split/digests.json"))


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    offset: int              # absolute file offset of the first byte
    nbytes: int
    digest: str              # the shard's sha256

    def describe(self) -> dict:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape), "nbytes": self.nbytes}


@dataclass
class Unit:
    """Tensors a stage caches together, keyed by the content they come from."""

    label: str               # "layer.12" or "final"
    tensors: list[Tensor] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def key(self) -> str:
        h = hashlib.sha256()
        for t in sorted(self.tensors, key=lambda t: t.name):
            h.update(json.dumps([t.name, t.dtype, list(t.shape), t.digest, t.offset, t.nbytes]).encode())
        return h.hexdigest()

    def describe(self) -> dict:
        return {"label": self.label, "key": self.key, "nbytes": self.nbytes,
                "tensors": [t.describe() for t in self.tensors]}


def read_header(path: Path) -> tuple[int, dict]:
    """(offset of the data, header) of a safetensors file."""

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def _load_sidecar() -> dict:
    try:
        return json.loads(SIDECAR.read_text())
    except (OSError, ValueError):
        return {}


def shard_digest(path: Path, known: dict | None = None) -> str:
    """The shard's sha256: a Hugging Face blob's name, else computed once and kept by path, size and mtime."""

    real = Path(os.path.realpath(path))
    if HEX64.fullmatch(real.name):
        return real.name
    st = real.stat()
    sidecar = _load_sidecar() if known is None else known
    tag = f"{real}|{st.st_size}|{st.st_mtime_ns}"
    if tag in sidecar:
        return sidecar[tag]
    with open(real, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    sidecar[tag] = digest
    if known is None:
        SIDECAR.parent.mkdir(parents=True, exist_ok=True)
        tmp = SIDECAR.with_suffix(".tmp")
        tmp.write_text(json.dumps(sidecar, indent=0))
        tmp.replace(SIDECAR)
    return digest


def tensors(model_dir: Path, wanted=None) -> list[Tensor]:
    """Every tensor in the checkpoint's shards that ``wanted(name)`` keeps, with its shard's digest."""

    model_dir = Path(model_dir)
    out: list[Tensor] = []
    sidecar = _load_sidecar()
    dirty = False
    for path in sorted(model_dir.glob("*.safetensors")):
        base, header = read_header(path)
        names = [n for n in header if n != "__metadata__" and (wanted is None or wanted(n))]
        if not names:
            continue
        before = len(sidecar)
        digest = shard_digest(path, sidecar)
        dirty |= len(sidecar) != before
        for name in names:
            e = header[name]
            begin, end = e["data_offsets"]
            out.append(Tensor(name, e["dtype"], tuple(e["shape"]), path, base + begin, end - begin, digest))
    if dirty:
        SIDECAR.parent.mkdir(parents=True, exist_ok=True)
        tmp = SIDECAR.with_suffix(".tmp")
        tmp.write_text(json.dumps(sidecar, indent=0))
        tmp.replace(SIDECAR)
    return out


def skipped(name: str) -> bool:
    """Tensors no stage reads: the vision tower and the MTP head (the drafter stays on the Mac)."""

    return name.startswith(("vision_tower", "visual.", "model.visual.", "mtp.")) or ".mtp." in name


def layer_of(name: str) -> int | None:
    m = LAYER.search(name)
    return int(m.group(1)) if m else None


def is_final_norm(name: str) -> bool:
    return name.endswith("model.norm.weight")


def layer_bytes(model_dir: Path, first: int, last: int | None = None) -> int:
    """Bytes of layers [first, last) in the checkpoint (headers only, nothing hashed)."""

    total = 0
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        _, header = read_header(path)
        for name, e in header.items():
            if name == "__metadata__" or skipped(name):
                continue
            i = layer_of(name)
            if i is not None and i >= first and (last is None or i < last):
                begin, end = e["data_offsets"]
                total += end - begin
    return total


def stage_units(model_dir: Path, first: int, last: int, layers: int) -> list[Unit]:
    """Units for a stage running layers [first, last) of ``layers``, plus the final norm when it runs the last layer."""

    if not 0 < first < last <= layers:
        raise ValueError(f"a stage runs layers [first, last) with 0 < first < last <= {layers}, not [{first}, {last})")

    def wanted(name: str) -> bool:
        if skipped(name):
            return False
        i = layer_of(name)
        return (first <= i < last) if i is not None else (last == layers and is_final_norm(name))

    by_label: dict[str, Unit] = {}
    for t in tensors(model_dir, wanted):
        i = layer_of(t.name)
        label = f"layer.{i}" if i is not None else "final"
        by_label.setdefault(label, Unit(label)).tensors.append(t)
    missing = [i for i in range(first, last) if f"layer.{i}" not in by_label]
    if missing:
        raise ValueError(f"the checkpoint has no tensors for layers {missing[:8]}")
    if last == layers and "final" not in by_label:
        raise ValueError("the checkpoint has no final norm (model.norm.weight)")
    order = [f"layer.{i}" for i in range(first, last)] + (["final"] if "final" in by_label else [])
    return [by_label[label] for label in order]


def read_unit(unit: Unit, sink, piece: int = 16 << 20) -> None:
    """Stream a unit's tensor bytes, in its listed order, to ``sink(bytes)``."""

    handles: dict[Path, object] = {}
    try:
        for t in unit.tensors:
            f = handles.get(t.path)
            if f is None:
                f = handles[t.path] = open(t.path, "rb")
            f.seek(t.offset)
            left = t.nbytes
            while left:
                chunk = f.read(min(piece, left))
                if not chunk:
                    raise OSError(f"{t.path} ended inside {t.name}")
                sink(chunk)
                left -= len(chunk)
    finally:
        for f in handles.values():
            f.close()
