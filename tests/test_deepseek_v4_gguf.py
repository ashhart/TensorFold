"""GGUF header and typed metadata reader tests (G1 scope).

Covers the bounded GGUF reader in tensorfold.gguf:
  * magic/version validation and truncated-header rejection
  * scalar and array metadata value types
  * malformed-length rejection before any read
  * duplicate metadata key rejection
  * tensor payload is intentionally not parsed

Only the header + typed metadata section is in scope; tensor descriptors and
weights are out of scope by design (see docs/evidence/deepseek-v4-g0-testfirst.md).
"""
from __future__ import annotations

import struct

import pytest

from tensorfold.gguf import (GGUFError, GGUF_TYPE_BOOL, GGUF_TYPE_FLOAT16, GGUF_TYPE_FLOAT32,
                             GGUF_TYPE_FLOAT64, GGUF_TYPE_INT8, GGUF_TYPE_INT16, GGUF_TYPE_INT32,
                             GGUF_TYPE_INT64, GGUF_TYPE_STRING, GGUF_TYPE_UINT8, GGUF_TYPE_UINT16,
                             GGUF_TYPE_UINT32, GGUF_TYPE_UINT64, GGUF_MAGIC, GGUF_VERSION,
                             parse_gguf)

GGUF_MAGIC_LE = 0x46554747  # "GGUF"


# ---------------------------------------------------------------------------
# builders: turn values into raw GGUF header + metadata bytes
# ---------------------------------------------------------------------------

def _string(value: bytes | str) -> bytes:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return struct.pack("<Q", len(value)) + value


def _header(magic: int = GGUF_MAGIC_LE, version: int = GGUF_VERSION, n_tensors: int = 0,
            n_kv: int = 0) -> bytes:
    return struct.pack("<IIQQ", magic, version, n_tensors, n_kv)


def _kv(key: str, type_: int, payload: bytes) -> bytes:
    return _string(key.encode("utf-8")) + bytes([type_]) + payload


def _scalar(type_: int, value) -> bytes:
    if type_ == GGUF_TYPE_UINT8:
        return bytes([value])
    if type_ == GGUF_TYPE_INT8:
        return struct.pack("<b", value)
    if type_ == GGUF_TYPE_UINT16:
        return struct.pack("<H", value)
    if type_ == GGUF_TYPE_INT16:
        return struct.pack("<h", value)
    if type_ == GGUF_TYPE_UINT32:
        return struct.pack("<I", value)
    if type_ == GGUF_TYPE_INT32:
        return struct.pack("<i", value)
    if type_ == GGUF_TYPE_FLOAT32:
        return struct.pack("<f", value)
    if type_ == GGUF_TYPE_BOOL:
        return bytes([1 if value else 0])
    if type_ == GGUF_TYPE_STRING:
        return _string(value)
    if type_ == GGUF_TYPE_UINT64:
        return struct.pack("<Q", value)
    if type_ == GGUF_TYPE_INT64:
        return struct.pack("<q", value)
    if type_ == GGUF_TYPE_FLOAT64:
        return struct.pack("<d", value)
    if type_ == GGUF_TYPE_FLOAT16:
        return struct.pack("<e", value)
    raise AssertionError(f"unknown scalar type {type_}")


def _array(elem_type: int, items) -> bytes:
    out = struct.pack("<Q", len(items))
    for item in items:
        out += _scalar(elem_type, item)
    return out


def _array_type(elem_type: int) -> int:
    return 9 | (elem_type << 4)


def _file(*, header: bytes = None, n_kv: int = 0, pairs: list[tuple[str, int, bytes]] = (),
          header_kwargs: dict | None = None) -> bytes:
    if header is None:
        header = _header(n_kv=n_kv, **(header_kwargs or {}))
    out = bytearray(header)
    for key, type_, payload in pairs:
        out += _kv(key, type_, payload)
    return bytes(out)


# ---------------------------------------------------------------------------
# header: magic / version / truncated
# ---------------------------------------------------------------------------

def test_valid_header_parses():
    info = parse_gguf(_file(n_kv=0, header_kwargs={"n_tensors": 7}))
    assert info.header.magic == GGUF_MAGIC
    assert info.header.version == 3
    assert info.header.n_tensors == 7
    assert info.header.n_kv == 0
    assert info.metadata == {}


def test_bad_magic_is_rejected():
    bad = _header(magic=0x12345678, n_kv=0)  # wrong magic, same length
    with pytest.raises(GGUFError, match="magic"):
        parse_gguf(bad)


def test_unsupported_version_is_rejected():
    bad = _header(version=999, n_kv=0)
    with pytest.raises(GGUFError, match="version"):
        parse_gguf(bad)


@pytest.mark.parametrize("nbytes", [0, 1, 4, 8, 15])
def test_truncated_header_is_rejected_before_reads(nbytes):
    data = _header(n_kv=0)[:nbytes]
    with pytest.raises(GGUFError, match="header"):
        parse_gguf(data)


def test_zero_kv_with_only_header_is_valid():
    info = parse_gguf(_header(n_kv=0))
    assert info.header.n_kv == 0
    assert info.metadata == {}


# ---------------------------------------------------------------------------
# metadata scalar value types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("type_,value,expected", [
    (GGUF_TYPE_UINT8, 0, 0),
    (GGUF_TYPE_UINT8, 200, 200),
    (GGUF_TYPE_INT8, -100, -100),
    (GGUF_TYPE_UINT16, 0, 0),
    (GGUF_TYPE_UINT16, 65535, 65535),
    (GGUF_TYPE_INT16, -30000, -30000),
    (GGUF_TYPE_UINT32, 2 ** 32 - 1, 2 ** 32 - 1),
    (GGUF_TYPE_INT32, -2_000_000_000, -2_000_000_000),
    (GGUF_TYPE_UINT64, 2 ** 64 - 1, 2 ** 64 - 1),
    (GGUF_TYPE_INT64, -9_000_000_000_000_000_000, -9_000_000_000_000_000_000),
    (GGUF_TYPE_FLOAT32, 1.25, 1.25),
    (GGUF_TYPE_FLOAT64, 1.0 / 3.0, 1.0 / 3.0),
    (GGUF_TYPE_FLOAT16, 2.0, 2.0),
    (GGUF_TYPE_BOOL, True, True),
    (GGUF_TYPE_BOOL, False, False),
])
def test_scalar_types_roundtrip(type_, value, expected):
    info = parse_gguf(_file(n_kv=1, pairs=[("test.key", type_, _scalar(type_, value))]))
    assert info.metadata["test.key"] == expected


def test_string_scalar_roundtrip():
    info = parse_gguf(_file(n_kv=1, pairs=[("architecture", GGUF_TYPE_STRING, _string("deepseek-v4"))]))
    assert info.metadata["architecture"] == "deepseek-v4"


def test_multiple_kv_pairs_preserve_order_and_values():
    pairs = [
        ("general.name", GGUF_TYPE_STRING, _string("tiny")),
        ("n", GGUF_TYPE_UINT32, _scalar(GGUF_TYPE_UINT32, 42)),
        ("flag", GGUF_TYPE_BOOL, _scalar(GGUF_TYPE_BOOL, True)),
    ]
    info = parse_gguf(_file(n_kv=3, pairs=pairs))
    assert list(info.metadata) == ["general.name", "n", "flag"]
    assert info.metadata["general.name"] == "tiny"
    assert info.metadata["n"] == 42
    assert info.metadata["flag"] is True


# ---------------------------------------------------------------------------
# metadata array value types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("elem_type,items", [
    (GGUF_TYPE_UINT8, [1, 2, 3, 255]),
    (GGUF_TYPE_INT8, [-5, 0, 5]),
    (GGUF_TYPE_UINT16, [0, 123, 65535]),
    (GGUF_TYPE_INT16, [-32000, 0, 32000]),
    (GGUF_TYPE_UINT32, [0, 4_000_000_000]),
    (GGUF_TYPE_INT32, [-2_000_000_000, 2_000_000_000]),
    (GGUF_TYPE_UINT64, [0, 2 ** 64 - 1]),
    (GGUF_TYPE_INT64, [-9_000_000_000_000_000_000, 9_000_000_000_000_000_000]),
    (GGUF_TYPE_FLOAT32, [0.5, 1.25]),
    (GGUF_TYPE_FLOAT64, [1.0 / 3.0]),
    (GGUF_TYPE_FLOAT16, [2.0, 0.5]),
    (GGUF_TYPE_BOOL, [True, False, True]),
])
def test_array_types_roundtrip(elem_type, items):
    type_ = _array_type(elem_type)
    info = parse_gguf(_file(n_kv=1, pairs=[("arr", type_, _array(elem_type, items))]))
    assert info.metadata["arr"] == items


def test_string_array_roundtrip():
    items = ["a", "bb", "ccc"]
    payload = struct.pack("<Q", len(items)) + b"".join(_string(i) for i in items)
    type_ = _array_type(GGUF_TYPE_STRING)
    info = parse_gguf(_file(n_kv=1, pairs=[("strs", type_, payload)]))
    assert info.metadata["strs"] == items


def test_empty_array_roundtrip():
    type_ = _array_type(GGUF_TYPE_UINT32)
    info = parse_gguf(_file(n_kv=1, pairs=[("empty", type_, struct.pack("<Q", 0))]))
    assert info.metadata["empty"] == []


def test_unknown_value_type_is_rejected():
    data = _file(n_kv=1, pairs=[("k", 0x0F, b"\x00\x00\x00\x00")])  # type 15 is unknown
    with pytest.raises(GGUFError, match="type"):
        parse_gguf(data)


def test_nested_array_type_is_rejected():
    data = _file(n_kv=1, pairs=[("k", 0x99, b"\x00\x00\x00\x00")])  # array of array
    with pytest.raises(GGUFError, match="array"):
        parse_gguf(data)


# ---------------------------------------------------------------------------
# duplicate metadata keys
# ---------------------------------------------------------------------------

def test_duplicate_metadata_key_is_rejected():
    pairs = [
        ("dup", GGUF_TYPE_UINT32, _scalar(GGUF_TYPE_UINT32, 1)),
        ("dup", GGUF_TYPE_UINT32, _scalar(GGUF_TYPE_UINT32, 2)),
    ]
    data = _file(n_kv=2, pairs=pairs)
    with pytest.raises(GGUFError, match="duplicate"):
        parse_gguf(data)


def test_empty_metadata_key_is_rejected():
    data = _file(n_kv=1, pairs=[("", GGUF_TYPE_UINT32, _scalar(GGUF_TYPE_UINT32, 1))])
    with pytest.raises(GGUFError, match="key"):
        parse_gguf(data)


# ---------------------------------------------------------------------------
# malformed lengths are rejected before reads
# ---------------------------------------------------------------------------

def test_string_key_length_overruns_buffer():
    # key length claims 100 bytes but only 3 remain
    bad = _header(n_kv=1) + struct.pack("<Q", 100) + b"abc"
    with pytest.raises(GGUFError, match="string length|length"):
        parse_gguf(bad)


def test_string_value_length_overruns_buffer():
    # a string value whose declared length exceeds the remaining file
    key = _string("k")
    payload = struct.pack("<Q", 1000)  # string length 1000, then nothing
    data = _header(n_kv=1) + key + bytes([GGUF_TYPE_STRING]) + payload
    with pytest.raises(GGUFError, match="string length|length"):
        parse_gguf(data)


def test_array_count_overruns_buffer():
    # array declares 1000 items but the file ends immediately
    key = _string("k")
    type_ = _array_type(GGUF_TYPE_UINT8)
    payload = struct.pack("<Q", 1000)
    data = _header(n_kv=1) + key + bytes([type_]) + payload
    with pytest.raises(GGUFError, match="array length|length"):
        parse_gguf(data)


def test_truncated_value_is_rejected():
    # a uint64 value needs 8 bytes but only 3 remain
    key = _string("k")
    data = _header(n_kv=1) + key + bytes([GGUF_TYPE_UINT64]) + b"\x01\x02\x03"
    with pytest.raises(GGUFError, match="uint64|8 bytes|truncated"):
        parse_gguf(data)


def test_missing_value_is_rejected():
    # key present but the value type byte / value bytes are cut off
    data = _header(n_kv=1) + _string("k")
    with pytest.raises(GGUFError):
        parse_gguf(data)


# ---------------------------------------------------------------------------
# tensor payload is out of scope: parse stops at end of metadata
# ---------------------------------------------------------------------------

def test_tensor_payload_is_not_touched():
    # header declares 2 tensors; after the metadata section there is garbage.
    # the reader must stop at metadata and not try to read tensor data.
    info = parse_gguf(_file(header_kwargs={"n_tensors": 2}))
    assert info.header.n_tensors == 2
    assert info.metadata == {}


def test_metadata_section_ends_exactly_at_n_kv():
    # even though n_tensors is large, only the declared n_kv pairs are read.
    pairs = [("a", GGUF_TYPE_UINT32, _scalar(GGUF_TYPE_UINT32, 1))]
    info = parse_gguf(_file(n_kv=1, pairs=pairs, header_kwargs={"n_tensors": 99}))
    assert list(info.metadata) == ["a"]
    assert info.header.n_tensors == 99
