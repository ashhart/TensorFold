"""Literal donor-compatible GGUF wire layout, independent of fixture builders."""
import struct
from tensorfold.gguf import parse_gguf


def test_uint32_metadata_types_and_separate_array_element_type():
    # Key 'a', type ARRAY=9, element STRING=8, two length-prefixed strings.
    # GGUF strings are not NUL terminated: preserve an actual trailing NUL.
    raw = (struct.pack('<IIQQ', 0x46554747, 3, 0, 1) +
           struct.pack('<Q', 1) + b'a' + struct.pack('<IIQ', 9, 8, 2) +
           struct.pack('<Q', 2) + b'x\0' + struct.pack('<Q', 1) + b'y')
    assert parse_gguf(raw).metadata == {'a': ['x\0', 'y']}
