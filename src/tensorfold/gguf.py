"""Bounded GGUF container header, typed metadata, and tensor-span reader.

This module reads the GGUF magic/version header and the typed key-value
metadata section (the ``n_kv`` pairs that follow the header). :func:`parse_gguf`
stops at the end of the metadata section and never touches tensor blocks, so a
caller interested only in architecture/parameter metadata can open a model
without loading its weights.

:func:`parse_gguf_tensors` additionally reads the ``n_tensors`` tensor
descriptors (name, dims, type, offset) that follow the metadata section and
returns an exact span/type/shape inventory. Weight data itself is still never
read: only the descriptor table is parsed, and every span is validated against
the actual buffer length.

Safety contract:
  * every length/count/value read is bounds-checked against the remaining
    buffer BEFORE the read happens; a malformed or truncated file raises a
    descriptive :class:`GGUFError` instead of overreading past the buffer.
  * duplicate metadata keys are rejected (GGUF requires unique keys).
  * unknown value types and nested arrays are rejected.
  * tensor validation covers duplicate names, offset alignment, overlapping
    spans, file-bounds, shape-product overflow, and quantized block
    divisibility.

Value types are little-endian uint32 GGUF enum values. Arrays carry a separate
uint32 element type followed by a uint64 count. Strings are uint64 length plus
exactly that many bytes, without a terminator.
Tensor types follow the ggml ``enum ggml_type`` values; block sizes and bytes
per block are pinned from the ggml quant tables.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

GGUF_MAGIC = 0x46554747  # "GGUF" as a little-endian uint32
GGUF_VERSION = 3         # current spec version
SUPPORTED_VERSIONS = (2, 3)  # v1 uses a different count/string layout

# GGUF value type enum (gguf.h)
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# scalar (non-array) value types the reader understands
_SCALAR_TYPES = frozenset({
    GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_UINT16, GGUF_TYPE_INT16,
    GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32, GGUF_TYPE_BOOL,
    GGUF_TYPE_STRING, GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64,
})

_HEADER_SIZE = 24  # u32 magic + u32 version + u64 n_tensors + u64 n_kv

_TYPE_NAMES = {
    GGUF_TYPE_UINT8: "uint8", GGUF_TYPE_INT8: "int8", GGUF_TYPE_UINT16: "uint16",
    GGUF_TYPE_INT16: "int16", GGUF_TYPE_UINT32: "uint32", GGUF_TYPE_INT32: "int32",
    GGUF_TYPE_FLOAT32: "float32", GGUF_TYPE_BOOL: "bool", GGUF_TYPE_STRING: "string",
    GGUF_TYPE_UINT64: "uint64", GGUF_TYPE_INT64: "int64", GGUF_TYPE_FLOAT64: "float64",
}


class GGUFError(ValueError):
    """Malformed, truncated, or unsupported GGUF container data."""


@dataclass(frozen=True)
class GGUFHeader:
    magic: int
    version: int
    n_tensors: int
    n_kv: int


@dataclass(frozen=True)
class GGUFInfo:
    header: GGUFHeader
    metadata: dict[str, Any]


# GGUF tensor data is required to be 32-byte aligned (gguf.h GGUF_ALIGNMENT).
GGUF_ALIGNMENT = 32

# ggml tensor type -> (type name, elements per block, bytes per block).
# Block geometry pinned from the ggml quant tables (ggml-quants.h / gguf.h).
_GGML_TYPE_INFO: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 110),
    20: ("IQ4_NL", 256, 50), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4),
    27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    type_id: int
    type_name: str
    shape: tuple[int, ...]
    nelements: int
    offset: int          # offset within the tensor data section
    size: int            # byte span within the tensor data section
    start: int           # absolute file offset of the tensor data
    end: int             # absolute file offset just past the tensor data


@dataclass(frozen=True)
class GGUFTensorInventory:
    info: GGUFInfo
    data_offset: int     # absolute file offset where tensor data begins
    tensors: tuple[TensorInfo, ...]


def _align_up(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


class _Cursor:
    """Bounds-checked little-endian cursor over a bytes buffer."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.pos = 0

    def remaining(self) -> int:
        return len(self._data) - self.pos

    def need(self, n: int, what: str) -> None:
        if n > self.remaining():
            raise GGUFError(
                f"truncated {what}: need {n} bytes but only {self.remaining()} remain "
                f"(file length {len(self._data)}, pos {self.pos})")

    def unpack(self, fmt: str, size: int, what: str):
        self.need(size, what)
        value = struct.unpack_from(fmt, self._data, self.pos)[0]
        self.pos += size
        return value

    def take(self, n: int, what: str) -> bytes:
        self.need(n, what)
        out = self._data[self.pos:self.pos + n]
        self.pos += n
        return out


def _read_u8(c: _Cursor) -> int:
    return c.unpack("<B", 1, "uint8 value")


def _read_u64(c: _Cursor, what: str) -> int:
    return c.unpack("<Q", 8, what)


def _read_string(c: _Cursor) -> str:
    n = _read_u64(c, "string length")
    if n > c.remaining():
        raise GGUFError(
            f"string length {n} exceeds remaining {c.remaining()} bytes (pos {c.pos})")
    raw = c.take(n, "string data")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFError(f"key/value string is not valid UTF-8: {exc}") from exc


def _validate_type(type_: int) -> int:
    if type_ != GGUF_TYPE_ARRAY and type_ not in _SCALAR_TYPES:
        raise GGUFError(f"unsupported GGUF value type {type_}")
    return type_


def _read_scalar(c: _Cursor, type_: int):
    if type_ == GGUF_TYPE_UINT8:
        return c.unpack("<B", 1, "uint8 value")
    if type_ == GGUF_TYPE_INT8:
        return c.unpack("<b", 1, "int8 value")
    if type_ == GGUF_TYPE_UINT16:
        return c.unpack("<H", 2, "uint16 value")
    if type_ == GGUF_TYPE_INT16:
        return c.unpack("<h", 2, "int16 value")
    if type_ == GGUF_TYPE_UINT32:
        return c.unpack("<I", 4, "uint32 value")
    if type_ == GGUF_TYPE_INT32:
        return c.unpack("<i", 4, "int32 value")
    if type_ == GGUF_TYPE_FLOAT32:
        return c.unpack("<f", 4, "float32 value")
    if type_ == GGUF_TYPE_BOOL:
        return c.unpack("<B", 1, "bool value") != 0
    if type_ == GGUF_TYPE_STRING:
        return _read_string(c)
    if type_ == GGUF_TYPE_UINT64:
        return c.unpack("<Q", 8, "uint64 value")
    if type_ == GGUF_TYPE_INT64:
        return c.unpack("<q", 8, "int64 value")
    if type_ == GGUF_TYPE_FLOAT64:
        return c.unpack("<d", 8, "float64 value")
    raise GGUFError(f"unsupported GGUF value type 0x{type_:02x}")


def _read_value(c: _Cursor, type_: int) -> Any:
    if type_ == GGUF_TYPE_ARRAY:
        elem = c.unpack("<I", 4, "array element type")
        if elem not in _SCALAR_TYPES:
            raise GGUFError(f"unsupported or nested GGUF array element type {elem}")
        n = _read_u64(c, "array length")
        if n > c.remaining():
            raise GGUFError(
                f"array length {n} exceeds remaining {c.remaining()} bytes (pos {c.pos})")
        values = [_read_scalar(c, elem) for _ in range(n)]
        return values
    return _read_scalar(c, type_)


def _parse_metadata(data: bytes) -> tuple[GGUFInfo, int]:
    """Parse the GGUF header and typed metadata section.

    Returns ``(info, metadata_end)`` where ``metadata_end`` is the absolute
    file offset just past the last key-value pair (where tensor descriptors
    begin, if any).
    """
    if len(data) < _HEADER_SIZE:
        raise GGUFError(
            f"GGUF header truncated: need {_HEADER_SIZE} bytes but file has {len(data)}")

    magic, version, n_tensors, n_kv = struct.unpack_from("<IIQQ", data, 0)
    if magic != GGUF_MAGIC:
        raise GGUFError(f"bad GGUF magic 0x{magic:08x} (expected 0x{GGUF_MAGIC:08x})")
    if version not in SUPPORTED_VERSIONS:
        raise GGUFError(f"unsupported GGUF version {version} (supported: {SUPPORTED_VERSIONS})")

    cursor = _Cursor(data)
    cursor.pos = _HEADER_SIZE
    metadata: dict[str, Any] = {}
    for _ in range(n_kv):
        key = _read_string(cursor)
        if not key:
            raise GGUFError("empty GGUF metadata key")
        if key in metadata:
            raise GGUFError(f"duplicate GGUF metadata key {key!r}")
        type_ = cursor.unpack("<I", 4, "value type")
        _validate_type(type_)
        metadata[key] = _read_value(cursor, type_)

    return GGUFInfo(GGUFHeader(magic, version, n_tensors, n_kv), metadata), cursor.pos


def parse_gguf(data: bytes) -> GGUFInfo:
    """Parse a GGUF header and typed metadata section from ``data``.

    Raises :class:`GGUFError` for bad magic, unsupported versions, truncated
    headers, malformed lengths, unknown value types, nested arrays, and
    duplicate metadata keys. Tensor descriptors and weight data are never read.
    """
    info, _ = _parse_metadata(data)
    return info


def parse_gguf_tensors(data: bytes) -> GGUFTensorInventory:
    """Parse the GGUF header, metadata, and tensor descriptor inventory.

    Reads the ``n_tensors`` descriptors (name, dims, type id, offset) that
    follow the metadata section and returns each tensor's exact byte span and
    shape. Validation performed:

      * unique tensor names
      * offset aligned to :data:`GGUF_ALIGNMENT` (32)
      * shape product does not overflow 2^64
      * quantized element counts divisible by their block size
      * each span fits within the actual file bounds
      * no two spans overlap

    Weight data itself is never read.
    """
    info, metadata_end = _parse_metadata(data)
    cursor = _Cursor(data)
    cursor.pos = metadata_end

    seen: set[str] = set()
    tensors: list[TensorInfo] = []
    for _ in range(info.header.n_tensors):
        name = _read_string(cursor)
        if not name:
            raise GGUFError("empty GGUF tensor name")
        if name in seen:
            raise GGUFError(f"duplicate GGUF tensor name {name!r}")
        seen.add(name)

        n_dims = cursor.unpack("<I", 4, "tensor n_dims")
        if not 1 <= n_dims <= 4:
            raise GGUFError(f"unsupported GGUF tensor dims count {n_dims}")
        dims = [cursor.unpack("<Q", 8, "tensor dim") for _ in range(n_dims)]

        type_id = cursor.unpack("<I", 4, "tensor type id")
        try:
            type_name, block_qk, block_bytes = _GGML_TYPE_INFO[type_id]
        except KeyError:
            raise GGUFError(f"unsupported GGUF tensor type id {type_id}")

        offset = cursor.unpack("<Q", 8, "tensor offset")
        if offset % GGUF_ALIGNMENT != 0:
            raise GGUFError(
                f"tensor {name!r} offset {offset} not {GGUF_ALIGNMENT}-byte aligned")

        nelements = 1
        for d in dims:
            nelements *= d
            if nelements > (1 << 64) - 1:
                raise GGUFError(
                    f"tensor {name!r} shape overflow (element product exceeds 2^64)")

        if block_qk > 1:
            if nelements % block_qk != 0:
                raise GGUFError(
                    f"tensor {name!r} element count {nelements} not divisible by "
                    f"{type_name} block size {block_qk}")
            size = (nelements // block_qk) * block_bytes
        else:
            size = nelements * block_bytes

        if size > (1 << 64) - 1:
            raise GGUFError(f"tensor {name!r} byte span overflow")

        tensors.append(TensorInfo(
            name=name, type_id=type_id, type_name=type_name,
            shape=tuple(dims), nelements=nelements, offset=offset,
            size=size, start=0, end=0))

    # tensor weight data begins after the descriptor table, 32-byte aligned
    data_offset = _align_up(cursor.pos, GGUF_ALIGNMENT)
    if data_offset > len(data):
        raise GGUFError(
            f"GGUF tensor data offset {data_offset} exceeds file size {len(data)}")

    final: list[TensorInfo] = []
    for t in tensors:
        start = data_offset + t.offset
        end = start + t.size
        if end > len(data):
            raise GGUFError(
                f"tensor {t.name!r} span [{start}, {end}) out of bounds "
                f"(file size {len(data)})")
        final.append(TensorInfo(
            t.name, t.type_id, t.type_name, t.shape, t.nelements,
            t.offset, t.size, start, end))

    ordered = sorted(final, key=lambda t: t.start)
    for i in range(len(ordered) - 1):
        if ordered[i + 1].start < ordered[i].end:
            raise GGUFError(
                f"GGUF tensor spans overlap: {ordered[i].name!r} ends at "
                f"{ordered[i].end} before {ordered[i + 1].name!r} starts at "
                f"{ordered[i + 1].start}")

    return GGUFTensorInventory(info, data_offset, tuple(final))
