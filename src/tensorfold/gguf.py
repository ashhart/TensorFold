"""Bounded GGUF container header and typed metadata reader (G1 scope).

This module reads only the GGUF magic/version header and the typed key-value
metadata section (the ``n_kv`` pairs that follow the header). Tensor payload
descriptors and weight data are deliberately out of scope: parsing stops at
the end of the metadata section and never touches tensor blocks, so a caller
interested only in architecture/parameter metadata can open a model without
loading its weights.

Safety contract (G1 acceptance):
  * every length/count/value read is bounds-checked against the remaining
    buffer BEFORE the read happens; a malformed or truncated file raises a
    descriptive :class:`GGUFError` instead of overreading past the buffer.
  * duplicate metadata keys are rejected (GGUF requires unique keys).
  * unknown value types and nested arrays are rejected.

Value types follow the GGUF spec enum (see ggml ``gguf.h``): scalar types 0-8
and 10-13, and arrays encoded as ``9 | (element_type << 4)``. Strings are
declared with a little-endian uint64 length followed by that many bytes; a
single trailing null byte (the GGUF convention) is stripped before decode.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

GGUF_MAGIC = 0x46554747  # "GGUF" as a little-endian uint32
GGUF_VERSION = 3         # current spec version
SUPPORTED_VERSIONS = (1, 2, 3)  # header + typed metadata layout is identical

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
GGUF_TYPE_ARRAY = 9      # arrays are 9 | (element_type << 4)
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12
GGUF_TYPE_FLOAT16 = 13

# scalar (non-array) value types the reader understands
_SCALAR_TYPES = frozenset({
    GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_UINT16, GGUF_TYPE_INT16,
    GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32, GGUF_TYPE_BOOL,
    GGUF_TYPE_STRING, GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64,
    GGUF_TYPE_FLOAT16,
})

_HEADER_SIZE = 24  # u32 magic + u32 version + u64 n_tensors + u64 n_kv

_TYPE_NAMES = {
    GGUF_TYPE_UINT8: "uint8", GGUF_TYPE_INT8: "int8", GGUF_TYPE_UINT16: "uint16",
    GGUF_TYPE_INT16: "int16", GGUF_TYPE_UINT32: "uint32", GGUF_TYPE_INT32: "int32",
    GGUF_TYPE_FLOAT32: "float32", GGUF_TYPE_BOOL: "bool", GGUF_TYPE_STRING: "string",
    GGUF_TYPE_UINT64: "uint64", GGUF_TYPE_INT64: "int64", GGUF_TYPE_FLOAT64: "float64",
    GGUF_TYPE_FLOAT16: "float16",
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
    # GGUF strings are null-terminated with the null included in the length;
    # strip a single trailing null so older files that omit it still work.
    if raw.endswith(b"\x00"):
        raw = raw[:-1]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFError(f"key/value string is not valid UTF-8: {exc}") from exc


def _validate_type(type_: int) -> int:
    """Return the element type for arrays, else the scalar type itself."""
    if (type_ & 0x0F) == GGUF_TYPE_ARRAY:
        elem = (type_ >> 4) & 0x0F
        if elem == GGUF_TYPE_ARRAY or elem not in _SCALAR_TYPES:
            raise GGUFError(f"unsupported or nested GGUF array element type 0x{type_:02x}")
        return elem
    if type_ not in _SCALAR_TYPES:
        raise GGUFError(f"unsupported GGUF value type 0x{type_:02x}")
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
    if type_ == GGUF_TYPE_FLOAT16:
        return c.unpack("<e", 2, "float16 value")
    raise GGUFError(f"unsupported GGUF value type 0x{type_:02x}")


def _read_value(c: _Cursor, type_: int) -> Any:
    if (type_ & 0x0F) == GGUF_TYPE_ARRAY:
        elem = (type_ >> 4) & 0x0F
        n = _read_u64(c, "array length")
        if n > c.remaining():
            raise GGUFError(
                f"array length {n} exceeds remaining {c.remaining()} bytes (pos {c.pos})")
        values = [_read_scalar(c, elem) for _ in range(n)]
        return values
    return _read_scalar(c, type_)


def parse_gguf(data: bytes) -> GGUFInfo:
    """Parse a GGUF header and typed metadata section from ``data``.

    Raises :class:`GGUFError` for bad magic, unsupported versions, truncated
    headers, malformed lengths, unknown value types, nested arrays, and
    duplicate metadata keys. Tensor payloads are never read.
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
        type_ = cursor.unpack("<B", 1, "value type byte")
        elem = _validate_type(type_)
        # validate array element type up front so a malformed type fails before
        # reading any of its items
        if (type_ & 0x0F) == GGUF_TYPE_ARRAY and elem not in _SCALAR_TYPES:
            raise GGUFError(f"unsupported GGUF array element type 0x{type_:02x}")
        metadata[key] = _read_value(cursor, type_)

    return GGUFInfo(GGUFHeader(magic, version, n_tensors, n_kv), metadata)
