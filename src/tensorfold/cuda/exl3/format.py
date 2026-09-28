"""The EXL3 format (ExLlamaV3's trellis quantization, MIT, Copyright (c) 2025 Turboderp), header-only metadata and a numpy reference decoder; docs/recipes/exl3.md has the bit arithmetic."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CODEBOOKS = ("3inst", "mcg", "mul1")
MARKERS = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}       # the marker tensors' values (int32 bit patterns)
INST3_MUL, INST3_ADD = 89226354, 64248484
MCG_MUL, MUL1_MUL = 0xCBAC1FED, 0x83DCD12D
MASK, FLIP = 0x8FFF8FFF, 0x3B603B60
MUL1_SCALE, MUL1_BIAS = 0x1EEE, 0xC931                   # fp16 bit patterns
HAD = 128
BITS = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)          # every width ExLlamaV3 writes (x.5 with mul1 only)
PARTS = ("trellis", "suh", "su", "svh", "sv", "mcg", "mul1", "bias")


# -- the codebooks and the tile ---------------------------------------------------------------------------------

def _fp16(bits: np.ndarray) -> np.ndarray:
    return bits.astype(np.uint16).view(np.float16).astype(np.float64)


@lru_cache(maxsize=3)
def codebook(name: str) -> np.ndarray:
    """The fp16 value of every 16-bit state, [65536], for codebook ``name`` ("3inst", "mcg" or "mul1")."""

    s = np.arange(65536, dtype=np.uint64)
    if name == "mul1":
        x = (s * MUL1_MUL) & 0xFFFFFFFF
        h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
        scale, bias = _fp16(np.array(MUL1_SCALE)), _fp16(np.array(MUL1_BIAS))
        return (h.astype(np.float64) * scale + bias).astype(np.float16)  # exact in float64, so one rounding
    if name == "mcg":
        x = (s * MCG_MUL) & 0xFFFFFFFF
    elif name == "3inst":
        x = (s * INST3_MUL + INST3_ADD) & 0xFFFFFFFF
    else:
        raise ValueError(f"unknown EXL3 codebook {name!r} (known: {', '.join(CODEBOOKS)})")
    x = (x & MASK) ^ FLIP
    return (_fp16(x & 0xFFFF) + _fp16(x >> 16)).astype(np.float16)  # the float64 sum is exact: one fp16 rounding


def check_bits(bits: float) -> float:
    b = float(bits)
    if b not in BITS:
        raise ValueError(f"EXL3 bits must be one of {', '.join(str(v) for v in BITS)}, got {bits}")
    return b


def stream_ends(bits: float) -> np.ndarray:
    """E(p) for p = 0..255: where value p's 16-bit window ends in the tile's bitstream (exclusive)."""

    b = check_bits(bits)
    p1 = np.arange(1, 257, dtype=np.int64)
    if b.is_integer():
        return p1 * int(b)
    k2 = int(2 * b)
    return (p1 * k2 - (p1 % 2)) // 2


@lru_cache(maxsize=1)
def tile_positions() -> tuple[np.ndarray, np.ndarray]:
    """(row, column) in its 16x16 tile of each of a tile's 256 values, in stream order."""

    p = np.arange(256)
    lane, j = p // 8, p % 8
    rows = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
    cols = lane // 4 + 8 * (j >> 2)
    return rows, cols


def tile_words(bits: float) -> int:
    """int16 words a tile: 16 * bits."""

    return int(16 * check_bits(bits))


def bits_of(trellis_shape: Iterable[int]) -> float:
    """A trellis' bits per weight from its shape (last dim / 16): 2 for [.., .., 32], 2.5 for [.., .., 40]."""

    last = int(tuple(trellis_shape)[-1])
    b = last / 16
    b = int(b) if float(b).is_integer() else b
    return check_bits(b) if last % 8 == 0 else check_bits(-1)


def _as_numpy(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):                            # a torch tensor
        t = t.detach().cpu().numpy()
    return np.ascontiguousarray(t)


def states(trellis: Any, bits: float) -> np.ndarray:
    """The 16-bit state of every value: [..., 256] uint32 in stream order, from trellis int16 [..., 16 * bits]."""

    t = _as_numpy(trellis)
    nw16 = tile_words(bits)
    if t.dtype != np.int16 or t.shape[-1] != nw16:
        raise ValueError(f"trellis must be int16 [..., {nw16}] for {bits} bits, got {t.dtype} {t.shape}")
    w = t.view(np.uint16).astype(np.uint64)
    words = w[..., 0::2] | (w[..., 1::2] << 16)          # little-endian pairs: 8 * bits 32-bit words a tile
    nw = nw16 // 2
    ring = 32 * nw
    first = stream_ends(bits) - 16 + ring                # the window's first bit, made non-negative
    i0, off = (first // 32) % nw, first % 32
    i1 = (i0 + 1) % nw
    pair = (words[..., i0] << 32) | words[..., i1]       # [..., 256]: the 64 bits from the window's first word
    return ((pair >> (48 - off).astype(np.uint64)) & 0xFFFF).astype(np.uint32)


def unpack(trellis: Any, bits: float, codebook_name: str, chunk: int = 16) -> np.ndarray:
    """W_q [K, N] fp16 in the rotated domain (ExLlamaV3's ``reconstruct``) from trellis int16 [K/16, N/16, 16 * bits], ``chunk`` k tiles at a time."""

    t = _as_numpy(trellis)
    if t.ndim != 3:
        raise ValueError(f"trellis must be [K/16, N/16, 16 * bits], got shape {t.shape}")
    kt, nt = t.shape[0], t.shape[1]
    table = codebook(codebook_name)
    rows, cols = tile_positions()
    w = np.empty((kt, 16, nt, 16), dtype=np.float16)
    for k0 in range(0, kt, chunk):
        vals = table[states(t[k0:k0 + chunk], bits).astype(np.int64)]       # [c, nt, 256]
        w[k0:k0 + chunk][:, rows, :, cols] = vals.transpose(2, 0, 1)
    return w.reshape(kt * 16, nt * 16)


def unpack_signs(packed: Any) -> np.ndarray:
    """su / sv int16 [n/16] -> fp16 [n] of +1 and -1: bit b of word w (as uint16) set means element 16w + b is -1."""

    p = _as_numpy(packed).view(np.uint16).astype(np.uint32).reshape(-1)
    bits = (p[:, None] >> np.arange(16, dtype=np.uint32)) & 1
    return (1.0 - 2.0 * bits.reshape(-1)).astype(np.float16)


# -- the layer --------------------------------------------------------------------------------------------------

@lru_cache(maxsize=4)
def hadamard(n: int = HAD) -> np.ndarray:
    """The n x n Sylvester Hadamard matrix of +1 and -1 (n a power of two): H[i, j] = (-1)^popcount(i & j)."""

    i = np.arange(n)
    parity = np.array([bin(v).count("1") & 1 for v in range(n)])
    return np.where(parity[(i[:, None] & i[None, :])] == 1, -1.0, 1.0)


def rotate(x: np.ndarray, axis: int) -> np.ndarray:
    """H / sqrt(128) applied to every block of 128 along ``axis``, in float64."""

    h = hadamard() / np.sqrt(HAD)
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    shape = x.shape
    if shape[-1] % HAD:
        raise ValueError(f"the rotated dimension must be a multiple of {HAD}, got {shape[-1]}")
    x = (x.reshape(*shape[:-1], shape[-1] // HAD, HAD) @ h).reshape(shape)
    return np.moveaxis(x, -1, axis)


def dequantize(trellis: Any, suh: Any, svh: Any, bits: float, codebook_name: str) -> np.ndarray:
    """The layer's weight W [K, N] in float64: diag(suh) @ H_K @ W_q @ H_N @ diag(svh)."""

    wq = unpack(trellis, bits, codebook_name).astype(np.float64)
    w = rotate(wq, 0) * _as_numpy(suh).astype(np.float64)[:, None]
    return rotate(w, 1) * _as_numpy(svh).astype(np.float64)[None, :]


def forward(x: Any, trellis: Any, suh: Any, svh: Any, bits: float, codebook_name: str,
            bias: Any = None) -> np.ndarray:
    """y = x @ W + bias in float64, the way the kernels order it: rotate the input, W_q, rotate the output."""

    xh = rotate(_as_numpy(x).astype(np.float64) * _as_numpy(suh).astype(np.float64), -1)
    y = rotate(xh @ unpack(trellis, bits, codebook_name).astype(np.float64), -1) * _as_numpy(svh).astype(np.float64)
    return y if bias is None else y + _as_numpy(bias).astype(np.float64)


# -- tensor groups in a checkpoint ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Exl3Tensor:
    """One EXL3 linear layer's metadata, read from the safetensors headers (no data)."""

    prefix: str               # e.g. "model.layers.0.self_attn.q_proj"
    bits: float               # 1..8, or 1.5 / 2.5 / 3.5 (mul1)
    codebook: str             # "3inst", "mcg" or "mul1"
    k: int                    # inputs (K)
    n: int                    # outputs (N)
    in_scales: str            # "suh" (fp16 scales) or "su" (packed signs)
    out_scales: str           # "svh" or "sv"
    bias: bool
    files: tuple[str, ...]    # the safetensors files holding the group

    @property
    def trellis_bytes(self) -> int:
        return self.k * self.n * int(2 * self.bits) // 16

    def describe(self) -> str:
        return f"{self.prefix}: EXL3 {self.bits:g}-bit {self.codebook}, K={self.k} N={self.n}"


def read_header(path: str | Path) -> dict[str, dict]:
    """A safetensors file's header: {name: {"dtype", "shape", "data_offsets"}} (no ``__metadata__``)."""

    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header


def read_scalar(path: str | Path, entry: dict) -> int:
    """The value of a scalar int32 tensor (a codebook marker) from its header entry."""

    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + size + int(entry["data_offsets"][0]))
        return struct.unpack("<I", f.read(4))[0]


def parse_group(prefix: str, parts: dict[str, dict], files: Iterable[str] = ()) -> Exl3Tensor:
    """An EXL3 group from its parts' header entries; ValueError, saying why, for a group ExLlamaV3's LinearEXL3 would not load."""

    def fail(why: str) -> ValueError:
        return ValueError(f"{prefix}: {why}")

    if "trellis" not in parts:
        raise fail("no .trellis")
    tr = parts["trellis"]
    shape = tuple(int(v) for v in tr["shape"])
    if tr["dtype"] != "I16" or len(shape) != 3:
        raise fail(f"trellis must be int16 [K/16, N/16, 16 * bits], got {tr['dtype']} {list(shape)}")
    try:
        bits = bits_of(shape)
    except ValueError as e:
        raise fail(str(e)) from None
    has = [m for m in ("mcg", "mul1") if m in parts]
    if len(has) > 1:
        raise fail("both mcg and mul1 markers")
    cb = has[0] if has else "3inst"
    if not float(bits).is_integer() and cb != "mul1":
        raise fail(f"{bits}-bit tiles need the mul1 codebook, found {cb}")
    k, n = 16 * shape[0], 16 * shape[1]
    if "suh" in parts:
        ins, want = "suh", ("F16", [k])
    elif "su" in parts:
        ins, want = "su", ("I16", [k // 16])
    else:
        raise fail("no input scales (.suh or .su)")
    if (parts[ins]["dtype"], list(parts[ins]["shape"])) != want:
        raise fail(f".{ins} must be {want[0]} {want[1]}, got {parts[ins]['dtype']} {parts[ins]['shape']}")
    if "svh" in parts:
        outs, want = "svh", ("F16", [n])
    elif "sv" in parts:
        outs, want = "sv", ("I16", [n // 16])
    else:
        raise fail("no output scales (.svh or .sv)")
    if (parts[outs]["dtype"], list(parts[outs]["shape"])) != want:
        raise fail(f".{outs} must be {want[0]} {want[1]}, got {parts[outs]['dtype']} {parts[outs]['shape']}")
    if k % HAD or n % HAD:
        raise fail(f"K={k} and N={n} must be multiples of {HAD} (the Hadamard blocks)")
    return Exl3Tensor(prefix, bits, cb, k, n, ins, outs, "bias" in parts, tuple(sorted(set(files))))


@dataclass
class Checkpoint:
    """Every tensor of a checkpoint folder, sorted into EXL3 groups, plain weights and groups that fail to parse."""

    groups: dict[str, Exl3Tensor]
    plain: dict[str, tuple[str, tuple[int, ...]]]       # name -> (dtype, shape) of every non-EXL3 tensor
    bad: dict[str, str]                                  # prefix -> why it is not a readable EXL3 group
    config: dict[str, Any]
    markers: dict[str, int]                              # marker tensor name -> its value, where read


def config_fields(config: dict[str, Any]) -> dict[str, Any]:
    """The EXL3 fields config.json states (version, average bits, head_bits, codebook, out_scales), top level or text config, either key."""

    blocks = [(config.get(k) or {}) for k in ("quantization_config", "quantization")]
    text = config.get("text_config") or {}
    blocks += [(text.get(k) or {}) for k in ("quantization_config", "quantization")]
    for block in blocks:
        if isinstance(block, dict) and str(block.get("quant_method", "")).lower() == "exl3":
            return dict(block)
    return {}


def require_config(config: dict[str, Any], *, where: str = "", tested: str = "", help: str = "") -> dict[str, Any]:
    """Refuse from config.json alone, before downloading, a codebook or width this module does not read; returns the fields read."""

    fields = config_fields(config)
    tail = f" Tested checkpoints: {tested}. {help}" if tested else f" {help}" if help else ""

    def refuse(why: str) -> None:
        raise ValueError(f"the EXL3 module of {where or 'this engine'} does not read this checkpoint's weights "
                         f"({why}; {describe_config(fields)}). It reads {', '.join(CODEBOOKS)} codebooks at "
                         f"{', '.join(str(v) for v in BITS)} bits per weight (the mul1 codebook is the one with "
                         f"half bits).{tail}")

    codebook = fields.get("codebook")
    if codebook is not None and str(codebook).lower() not in CODEBOOKS:
        refuse(f"codebook {codebook!r}")
    for key in ("head_bits", "mtp_bits"):
        value = fields.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            refuse(f"{key} {fields[key]!r}")
        if value not in BITS or not value.is_integer():
            refuse(f"{key} {value}")
    bits = fields.get("bits")
    if bits is not None:
        try:
            bits = float(bits)
        except (TypeError, ValueError):
            refuse(f"bits {fields['bits']!r}")
        if not 1 <= bits <= 8:                      # the average over the model, so any fraction is fine
            refuse(f"average bits {bits}")
    return fields


def describe_config(fields: dict[str, Any]) -> str:
    """``"exl3 1.4.2, mul1, average 2.50 bits, head 6"`` for a refusal message; only the fields that are there."""

    parts = ["exl3"]
    if fields.get("version"):
        parts.append(str(fields["version"]))
    if fields.get("codebook"):
        parts.append(str(fields["codebook"]))
    for key, label in (("bits", "average {} bits"), ("head_bits", "head {}"), ("mtp_bits", "MTP head {}")):
        if fields.get(key) is not None:
            parts.append(label.format(fields[key]))
    return ", ".join(parts)


def scan(model_dir: str | Path, read_markers: bool = True) -> Checkpoint:
    """Read every ``*.safetensors`` header under ``model_dir`` (no weight data) into a ``Checkpoint``."""

    root = Path(model_dir)
    entries: dict[str, tuple[str, dict]] = {}
    for path in sorted(root.glob("*.safetensors")):
        for name, entry in read_header(path).items():
            entries[name] = (path.name, entry)
    by_prefix: dict[str, dict[str, dict]] = {}
    files: dict[str, list[str]] = {}
    for name, (file, entry) in entries.items():
        prefix, _, part = name.rpartition(".")
        if part in PARTS:
            by_prefix.setdefault(prefix, {})[part] = entry
            files.setdefault(prefix, []).append(file)
    groups, bad, markers, taken = {}, {}, {}, set()
    for prefix, parts in by_prefix.items():
        if "trellis" not in parts:
            continue                                     # e.g. a plain layer's .bias
        taken.update(f"{prefix}.{p}" for p in parts)
        try:
            groups[prefix] = parse_group(prefix, parts, files[prefix])
        except ValueError as e:
            bad[prefix] = str(e).split(": ", 1)[1]
            continue
        cb = groups[prefix].codebook
        if read_markers and cb in MARKERS:
            file = entries[f"{prefix}.{cb}"][0]
            markers[f"{prefix}.{cb}"] = read_scalar(root / file, parts[cb])
    plain = {name: (entry["dtype"], tuple(entry["shape"])) for name, (_, entry) in entries.items()
             if name not in taken}
    config = json.loads((root / "config.json").read_text()) if (root / "config.json").exists() else {}
    return Checkpoint(groups, plain, bad, config, markers)


def is_exl3(model_dir: str | Path) -> bool:
    """Whether config.json (top level or text config, either key) or a sidecar file names the EXL3 format."""

    root = Path(model_dir)
    config = root / "config.json"
    if config.is_file() and config_fields(json.loads(config.read_text())):
        return True
    side = root / "quantization_config.json"
    return side.is_file() and str(json.loads(side.read_text()).get("quant_method", "")).lower() == "exl3"
