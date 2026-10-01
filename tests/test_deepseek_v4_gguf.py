"""GGUF header and typed metadata reader tests (G1 scope).

Covers the bounded GGUF reader in tensorfold.gguf:
  * magic/version validation and truncated-header rejection
  * scalar and array metadata value types
  * malformed-length rejection before any read
  * duplicate metadata key rejection
  * tensor payload is intentionally not parsed

Only the header + typed metadata section is in scope; tensor descriptors and
weights are intentionally not loaded by header-parser tests.
"""

from __future__ import annotations

import struct

import pytest

from tensorfold.gguf import (
    GGUF_MAGIC,
    GGUF_TYPE_BOOL,
    GGUF_TYPE_FLOAT32,
    GGUF_TYPE_FLOAT64,
    GGUF_TYPE_INT8,
    GGUF_TYPE_INT16,
    GGUF_TYPE_INT32,
    GGUF_TYPE_INT64,
    GGUF_TYPE_STRING,
    GGUF_TYPE_UINT8,
    GGUF_TYPE_UINT16,
    GGUF_TYPE_UINT32,
    GGUF_TYPE_UINT64,
    GGUF_VERSION,
    GGUFError,
    parse_gguf,
    parse_gguf_tensors,
)

GGUF_MAGIC_LE = 0x46554747  # "GGUF"


# ---------------------------------------------------------------------------
# builders: turn values into raw GGUF header + metadata bytes
# ---------------------------------------------------------------------------


def _string(value: bytes | str) -> bytes:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return struct.pack("<Q", len(value)) + value


def _header(magic: int = GGUF_MAGIC_LE, version: int = GGUF_VERSION, n_tensors: int = 0, n_kv: int = 0) -> bytes:
    return struct.pack("<IIQQ", magic, version, n_tensors, n_kv)


def _kv(key: str, type_: int, payload: bytes) -> bytes:
    return _string(key.encode("utf-8")) + struct.pack("<I", type_) + payload


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
    raise AssertionError(f"unknown scalar type {type_}")


def _array(elem_type: int, items) -> bytes:
    out = struct.pack("<IQ", elem_type, len(items))
    for item in items:
        out += _scalar(elem_type, item)
    return out


def _array_type(elem_type: int) -> int:
    return 9


def _file(
    *,
    header: bytes | None = None,
    n_kv: int = 0,
    pairs: list[tuple[str, int, bytes]] = (),
    header_kwargs: dict | None = None,
) -> bytes:
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


@pytest.mark.parametrize(
    "type_,value,expected",
    [
        (GGUF_TYPE_UINT8, 0, 0),
        (GGUF_TYPE_UINT8, 200, 200),
        (GGUF_TYPE_INT8, -100, -100),
        (GGUF_TYPE_UINT16, 0, 0),
        (GGUF_TYPE_UINT16, 65535, 65535),
        (GGUF_TYPE_INT16, -30000, -30000),
        (GGUF_TYPE_UINT32, 2**32 - 1, 2**32 - 1),
        (GGUF_TYPE_INT32, -2_000_000_000, -2_000_000_000),
        (GGUF_TYPE_UINT64, 2**64 - 1, 2**64 - 1),
        (GGUF_TYPE_INT64, -9_000_000_000_000_000_000, -9_000_000_000_000_000_000),
        (GGUF_TYPE_FLOAT32, 1.25, 1.25),
        (GGUF_TYPE_FLOAT64, 1.0 / 3.0, 1.0 / 3.0),
        (GGUF_TYPE_BOOL, True, True),
        (GGUF_TYPE_BOOL, False, False),
    ],
)
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


@pytest.mark.parametrize(
    "elem_type,items",
    [
        (GGUF_TYPE_UINT8, [1, 2, 3, 255]),
        (GGUF_TYPE_INT8, [-5, 0, 5]),
        (GGUF_TYPE_UINT16, [0, 123, 65535]),
        (GGUF_TYPE_INT16, [-32000, 0, 32000]),
        (GGUF_TYPE_UINT32, [0, 4_000_000_000]),
        (GGUF_TYPE_INT32, [-2_000_000_000, 2_000_000_000]),
        (GGUF_TYPE_UINT64, [0, 2**64 - 1]),
        (GGUF_TYPE_INT64, [-9_000_000_000_000_000_000, 9_000_000_000_000_000_000]),
        (GGUF_TYPE_FLOAT32, [0.5, 1.25]),
        (GGUF_TYPE_FLOAT64, [1.0 / 3.0]),
        (GGUF_TYPE_BOOL, [True, False, True]),
    ],
)
def test_array_types_roundtrip(elem_type, items):
    type_ = _array_type(elem_type)
    info = parse_gguf(_file(n_kv=1, pairs=[("arr", type_, _array(elem_type, items))]))
    assert info.metadata["arr"] == items


def test_string_array_roundtrip():
    items = ["a", "bb", "ccc"]
    payload = struct.pack("<IQ", GGUF_TYPE_STRING, len(items)) + b"".join(_string(i) for i in items)
    type_ = _array_type(GGUF_TYPE_STRING)
    info = parse_gguf(_file(n_kv=1, pairs=[("strs", type_, payload)]))
    assert info.metadata["strs"] == items


def test_empty_array_roundtrip():
    type_ = _array_type(GGUF_TYPE_UINT32)
    info = parse_gguf(_file(n_kv=1, pairs=[("empty", type_, struct.pack("<IQ", GGUF_TYPE_UINT32, 0))]))
    assert info.metadata["empty"] == []


def test_unknown_value_type_is_rejected():
    data = _file(n_kv=1, pairs=[("k", 0x0F, b"\x00\x00\x00\x00")])  # type 15 is unknown
    with pytest.raises(GGUFError, match="type"):
        parse_gguf(data)


def test_nested_array_type_is_rejected():
    data = _file(n_kv=1, pairs=[("k", 9, struct.pack("<IQ", 9, 0))])  # array of array
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
    data = _header(n_kv=1) + key + struct.pack("<I", GGUF_TYPE_STRING) + payload
    with pytest.raises(GGUFError, match="string length|length"):
        parse_gguf(data)


def test_array_count_overruns_buffer():
    # array declares 1000 items but the file ends immediately
    key = _string("k")
    type_ = _array_type(GGUF_TYPE_UINT8)
    payload = struct.pack("<IQ", GGUF_TYPE_UINT8, 1000)
    data = _header(n_kv=1) + key + struct.pack("<I", type_) + payload
    with pytest.raises(GGUFError, match="array length|length"):
        parse_gguf(data)


def test_truncated_value_is_rejected():
    # a uint64 value needs 8 bytes but only 3 remain
    key = _string("k")
    data = _header(n_kv=1) + key + struct.pack("<I", GGUF_TYPE_UINT64) + b"\x01\x02\x03"
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


# ---------------------------------------------------------------------------
# T02.2: tensor descriptor inventory (spans/type/shape) and validation
# ---------------------------------------------------------------------------
# GGUF tensor descriptor layout: name (string), n_dims (uint32),
# dims (uint64 x n_dims), type (uint32), offset (uint64).

GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_Q2_K = 10
GGML_IQ2_XXS = 16


def _align(n: int) -> int:
    return (n + 31) // 32 * 32


def _tensor_desc(name: str, dims: list, type_id: int, offset: int) -> bytes:
    return (
        _string(name.encode("utf-8"))
        + struct.pack("<I", len(dims))
        + b"".join(struct.pack("<Q", d) for d in dims)
        + struct.pack("<I", type_id)
        + struct.pack("<Q", offset)
    )


def _tensor_file(*, n_tensors, descriptors=(), data_len=0, pairs=(), n_kv=0, header_kwargs=None):
    """Header + metadata + tensor descriptors + 32-byte-aligned data section."""
    header = _header(n_tensors=n_tensors, n_kv=n_kv, **(header_kwargs or {}))
    out = bytearray(header)
    for key, type_, payload in pairs:
        out += _kv(key, type_, payload)
    out += b"".join(descriptors)
    out += b"\x00" * ((-len(out)) % 32)
    out += b"\x00" * data_len
    return bytes(out)


def test_tensor_inventory_exact_spans_and_shapes():
    # Three tensors with offsets chosen 32-byte aligned and non-overlapping:
    #   embd     F32     (8, 4)     nelem 32      size 128
    #   wq       Q8_0    (8, 32)    nelem 256     size 256//32*34 = 272
    #   experts  IQ2_XXS (256, 256) nelem 65536   size 65536//256*66 = 16896
    descs = [
        _tensor_desc("embd", [8, 4], GGML_F32, 0),
        _tensor_desc("wq", [8, 32], GGML_Q8_0, 128),
        _tensor_desc("experts", [256, 256], GGML_IQ2_XXS, 416),
    ]
    header = _header(n_tensors=3, n_kv=0)
    desc_bytes = b"".join(descs)
    # independent oracle for the data section start
    data_offset = _align(len(header) + len(desc_bytes))

    inv = parse_gguf_tensors(_tensor_file(n_tensors=3, descriptors=descs, data_len=data_offset + 416 + 16896))

    assert inv.info.header.n_tensors == 3
    assert inv.data_offset == data_offset
    assert [t.name for t in inv.tensors] == ["embd", "wq", "experts"]

    embd, wq, experts = inv.tensors
    assert embd.type_name == "F32"
    assert embd.shape == (8, 4)
    assert embd.nelements == 32
    assert embd.offset == 0 and embd.size == 128
    assert embd.start == data_offset and embd.end == data_offset + 128

    assert wq.type_name == "Q8_0"
    assert wq.shape == (8, 32)
    assert wq.nelements == 256
    assert wq.size == 272
    assert wq.offset == 128 and wq.end == data_offset + 400

    assert experts.type_name == "IQ2_XXS"
    assert experts.shape == (256, 256)
    assert experts.nelements == 65536
    assert experts.size == 16896
    assert experts.offset == 416 and experts.end == data_offset + 17312


def test_tensor_offset_not_aligned_is_rejected():
    desc = _tensor_desc("bad", [4], GGML_F32, 100)  # 100 % 32 != 0
    with pytest.raises(GGUFError, match="aligned"):
        parse_gguf_tensors(_tensor_file(n_tensors=1, descriptors=[desc], data_len=200))


def test_duplicate_tensor_name_is_rejected():
    descs = [_tensor_desc("dup", [4], GGML_F32, 0), _tensor_desc("dup", [4], GGML_F32, 32)]
    with pytest.raises(GGUFError, match="duplicate"):
        parse_gguf_tensors(_tensor_file(n_tensors=2, descriptors=descs, data_len=200))


def test_overlapping_tensor_spans_are_rejected():
    # A spans [0, 64); B spans [32, 96) -> overlap
    descs = [_tensor_desc("a", [16], GGML_F32, 0), _tensor_desc("b", [16], GGML_F32, 32)]
    with pytest.raises(GGUFError, match="overlap"):
        parse_gguf_tensors(_tensor_file(n_tensors=2, descriptors=descs, data_len=200))


def test_tensor_span_out_of_bounds_is_rejected():
    # declared size 128 but no data section is present
    desc = _tensor_desc("x", [32], GGML_F32, 0)
    with pytest.raises(GGUFError, match="out of bounds|bounds"):
        parse_gguf_tensors(_tensor_file(n_tensors=1, descriptors=[desc]))


def test_tensor_shape_overflow_is_rejected():
    # product of dims exceeds 2^64
    desc = _tensor_desc("big", [2**33, 2**33], GGML_F32, 0)
    with pytest.raises(GGUFError, match="overflow"):
        parse_gguf_tensors(_tensor_file(n_tensors=1, descriptors=[desc]))


def test_quantized_block_divisibility_is_enforced():
    # IQ2_XXS blocks hold 256 elements; 10000 is not divisible by 256
    desc = _tensor_desc("exp", [100, 100], GGML_IQ2_XXS, 0)
    with pytest.raises(GGUFError, match="divisible|block"):
        parse_gguf_tensors(_tensor_file(n_tensors=1, descriptors=[desc], data_len=40960))


def test_quantized_block_divisible_shape_is_accepted():
    # Q2_K blocks hold 256 elements; 256 is divisible -> size 84 bytes
    desc = _tensor_desc("q2k", [256], GGML_Q2_K, 0)
    inv = parse_gguf_tensors(_tensor_file(n_tensors=1, descriptors=[desc], data_len=128))
    assert inv.tensors[0].type_name == "Q2_K"
    assert inv.tensors[0].size == 84


def test_f16_tensor_type_inventory():
    # F16 is 2 bytes per element: shape (8, 4) -> 32 elements -> 64 bytes
    desc = _tensor_desc("h", [8, 4], GGML_F16, 0)
    inv = parse_gguf_tensors(_tensor_file(n_tensors=1, descriptors=[desc], data_len=128))
    assert inv.tensors[0].type_name == "F16"
    assert inv.tensors[0].shape == (8, 4)
    assert inv.tensors[0].size == 64


def test_unknown_tensor_type_is_rejected():
    desc = _tensor_desc("weird", [4], 999, 0)
    with pytest.raises(GGUFError, match="type"):
        parse_gguf_tensors(_tensor_file(n_tensors=1, descriptors=[desc], data_len=16))


def test_truncated_tensor_descriptor_is_rejected_before_read():
    # header claims one tensor but the descriptor table is missing
    with pytest.raises(GGUFError):
        parse_gguf_tensors(_tensor_file(n_tensors=1))
