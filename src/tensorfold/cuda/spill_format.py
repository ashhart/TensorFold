"""The spill tier's files: a safetensors header, the stored object's JSON layer, the keys and ids that name them."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
import zlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

FORMAT = "tensorfold-cuda-spill-2"
ALIGN = 4096                           # O_DIRECT offsets, lengths and buffers
BLOCK = 8 << 20                        # bytes one CRC-32 covers (spill_io.CHUNK is a multiple of it)
_SOURCES = (".py", ".cu", ".cuh", ".cpp", ".h")

_DT = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.float64: "F64", torch.int8: "I8",
       torch.uint8: "U8", torch.int16: "I16", torch.int32: "I32", torch.int64: "I64", torch.bool: "BOOL"}
for _name, _code in (("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2")):
    if hasattr(torch, _name):
        _DT[getattr(torch, _name)] = _code
_TD = {v: k for k, v in _DT.items()}


def _up(x: int) -> int:
    return -(-x // ALIGN) * ALIGN


def build_id(model_dir, drafter_dir=None) -> str:
    """What stored bits depend on beyond the layout: weights and drafter paths, version, sources (MLX key), runtime."""

    from pathlib import Path

    from tensorfold import __version__, families

    pkg = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    family = pkg.joinpath(*families.detect(model_dir).module.split(".")[1:])
    for base in (family, pkg / "cuda"):
        families.hash_sources(digest, pkg, sorted(p for p in base.rglob("*") if p.suffix in _SOURCES))
    drafter = f"|drafter={Path(drafter_dir).resolve()}" if drafter_dir is not None else ""
    tail = f"|tensorfold={__version__}|sources={digest.hexdigest()[:16]}|{_runtime()}"
    return f"{Path(model_dir).resolve()}{drafter}{tail}"


def _runtime() -> str:
    """What builds the kernels that make the bits: torch, CUDA, the GPU and its driver."""

    run = f"torch={torch.__version__}|cuda={torch.version.cuda}"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        run += f"|gpu={p.name}:{p.major}.{p.minor}"
    try:
        with open("/proc/driver/nvidia/version") as f:
            run += f"|driver={f.readline().strip()}"
    except OSError:
        pass
    return run


def weights_id(*dirs) -> str:
    """This rank's weights and drafter: each file's name, size, mtime and inode (rewritten in place: a new folder)."""

    from pathlib import Path

    digest = hashlib.sha256()
    for d in (d for d in dirs if d is not None):
        for path in sorted(Path(d).iterdir()):
            if path.suffix in (".safetensors", ".json"):
                st = path.stat()                              # through the hub's links, to the files themselves
                digest.update(f"{path.name}:{st.st_size}:{st.st_mtime_ns}:{st.st_ino}".encode())
    return digest.hexdigest()[:16]


def ids_key(ids: Sequence[int]) -> tuple[int, int]:
    h = hashlib.sha1(np.asarray(ids, dtype=np.int32).tobytes()).digest()
    return int.from_bytes(h[0:4], "big", signed=True), int.from_bytes(h[4:8], "big", signed=True)


def key_name(key: tuple[int, int], n: int) -> str:
    return f"{key[0] & 0xffffffff:08x}{key[1] & 0xffffffff:08x}_{n}"


# -- objects to files and back ---------------------------------------------------------------------------------------
def _enc(value: Any, name: str, items: list) -> dict:
    if isinstance(value, torch.Tensor):
        items.append((name, value))
        return {"t": name}
    if isinstance(value, np.ndarray):
        items.append((name, torch.from_numpy(np.ascontiguousarray(value))))
        return {"t": name, "np": 1}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"v": value}
    if isinstance(value, (list, tuple)):
        out = {"l": [_enc(v, f"{name}.{i}", items) for i, v in enumerate(value)]}
        if isinstance(value, tuple):
            out["tuple"] = 1
        return out
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"d": {k: _enc(v, f"{name}.{k}", items) for k, v in value.items()}}
    raise TypeError(f"the spill tier cannot store {type(value).__name__} at {name}")


def _dec(spec: dict, tensors: dict) -> Any:
    if "t" in spec:
        t = tensors[spec["t"]]
        return t.cpu().numpy() if spec.get("np") else t
    if "v" in spec:
        return spec["v"]
    if "l" in spec:
        out = [_dec(s, tensors) for s in spec["l"]]
        return tuple(out) if spec.get("tuple") else out
    if "d" in spec:
        return {k: _dec(s, tensors) for k, s in spec["d"].items()}
    raise ValueError(f"an unknown field spec {spec}")


def encode(obj: Any, *, skip: Sequence[str] = (),
           materialize: Callable[[Any], None] | None = None) -> tuple[list, dict]:
    """``obj``'s fields as (named tensors, JSON layer), class by import path; ``transient`` and ``skip`` left out."""

    if materialize is not None:
        materialize(obj)
    cls = type(obj)
    leave = set(getattr(cls, "transient", ())) | set(skip)
    items: list = []
    fields = {k: _enc(v, f"0.{k}", items) for k, v in vars(obj).items() if k not in leave and not k.startswith("_")}
    return items, {"class": f"{cls.__module__}:{cls.__qualname__}", "fields": fields}


def decode(layer: dict, tensors: dict, classes: Sequence[type]) -> Any:
    """``encode``'s object as one of ``classes`` (another: ValueError, never imported); missing fields take defaults."""

    cls: Any = {f"{c.__module__}:{c.__qualname__}": c for c in classes}.get(layer.get("class"))
    if cls is None:
        raise ValueError(f"a stored {layer.get('class')!r}: not a class this engine reads back")
    obj = cls.__new__(cls)
    for k, spec in layer["fields"].items():
        setattr(obj, k, _dec(spec, tensors))
    if dataclasses.is_dataclass(cls):
        for f in dataclasses.fields(cls):
            if not hasattr(obj, f.name):
                if f.default is not dataclasses.MISSING:
                    setattr(obj, f.name, f.default)
                elif f.default_factory is not dataclasses.MISSING:          # type: ignore[misc]
                    setattr(obj, f.name, f.default_factory())             # type: ignore[misc]
    return obj


def _dtype_code(t: torch.Tensor) -> str:
    try:
        return _DT[t.dtype]
    except KeyError:
        raise TypeError(f"the spill tier cannot store {t.dtype}") from None


def _crc_placeholder(nbytes: int) -> str:
    """The header's CRC field before the data is hashed: one 8-digit slot a BLOCK, so the header keeps its length."""

    return ",".join(["00000000"] * (-(-nbytes // BLOCK)))


def _blocks(n: int) -> list[tuple[int, int]]:
    """The CRC blocks of ``n`` bytes, as (start, end)."""

    return [(i, min(i + BLOCK, n)) for i in range(0, n, BLOCK)]


class _Crcs:
    """CRC-32 of each BLOCK of a byte stream fed in order."""

    def __init__(self) -> None:
        self.done: list[int] = []
        self.cur = 0
        self.fill = 0
        self.nbytes = 0

    def feed(self, mv) -> None:
        self.nbytes += len(mv)
        while len(mv):
            k = min(BLOCK - self.fill, len(mv))
            self.cur = zlib.crc32(mv[:k], self.cur)
            self.fill += k
            mv = mv[k:]
            if self.fill == BLOCK:
                self.done.append(self.cur)
                self.cur, self.fill = 0, 0

    def hex(self) -> str:
        return ",".join(f"{c:08x}" for c in self.done + ([self.cur] if self.fill else []))


def _header(items: list, meta: dict) -> tuple[bytes, int]:
    """The safetensors header for ``items``, padded so the data starts on a 4 KiB boundary, and the data's bytes."""

    table: dict = {}
    at = 0
    for name, t in items:
        nbytes = t.numel() * t.element_size()
        table[name] = {"dtype": _dtype_code(t), "shape": list(t.shape), "data_offsets": [at, at + nbytes]}
        at += nbytes
    body = json.dumps({"__metadata__": {k: str(v) for k, v in meta.items()}, **table}, separators=(",", ":")).encode()
    total = _up(8 + len(body))
    body += b" " * (total - 8 - len(body))
    return struct.pack("<Q", len(body)) + body, at


def _read_header_file(f) -> tuple[dict, int]:
    (size,) = struct.unpack("<Q", f.read(8))
    if size > 1 << 30:
        raise ValueError("not a spill file")
    return json.loads(f.read(size)), 8 + size
