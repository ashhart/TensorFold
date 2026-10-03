"""Container bounds and exact block geometry, independent of CUDA or a real checkpoint."""

import struct

import pytest

from tensorfold.gguf import GGUFError, parse_gguf, parse_gguf_tensors


def string(value):
    value = value.encode()
    return struct.pack("<Q", len(value)) + value


def container(kind, k, n, size):
    header = struct.pack("<IIQQ", 0x46554747, 3, 1, 0)
    desc = string("weight") + struct.pack("<IQQIQ", 2, k, n, kind, 0)
    data = header + desc
    return data + b"\0" * (-len(data) % 32) + b"\0" * size


@pytest.mark.parametrize(
    "kind,k,bytes_per_block", [(8, 32, 34), (9, 32, 36), (10, 256, 84), (16, 256, 66), (20, 32, 18)]
)
def test_block_spans_and_truncation(kind, k, bytes_per_block):
    data = container(kind, k, 2, bytes_per_block * 2)
    tensor = parse_gguf_tensors(data).tensors[0]
    assert tensor.size == bytes_per_block * 2 and tensor.shape == (k, 2)
    with pytest.raises(GGUFError):
        parse_gguf_tensors(data[:-1])


def test_header_and_metadata_lengths_are_checked_before_reads():
    with pytest.raises(GGUFError):
        parse_gguf(b"GGUF")
    data = struct.pack("<IIQQQ", 0x46554747, 3, 0, 1, 2**64 - 1)
    with pytest.raises(GGUFError):
        parse_gguf(data)


@pytest.mark.parametrize("k,n", [(16, 2), (0, 1)])
def test_quantized_rows_must_contain_complete_blocks(k, n):
    with pytest.raises(GGUFError):
        parse_gguf_tensors(container(8, k, n, 34))


def test_tensor_alignment_comes_from_metadata():
    header = struct.pack("<IIQQ", 0x46554747, 3, 1, 1)
    metadata = string("general.alignment") + struct.pack("<II", 4, 64)
    desc = string("weight") + struct.pack("<IQQIQ", 2, 32, 1, 8, 0)
    data = header + metadata + desc
    data += b"\0" * (-len(data) % 64) + b"\0" * 34
    tensor = parse_gguf_tensors(data).tensors[0]
    assert tensor.start % 64 == 0 and tensor.size == 34
