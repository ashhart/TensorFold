"""Packed CUDA blocks track a scalar decoder; narrow rows keep their serial bits."""

import struct

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.gguf import Packed
from tensorfold.cuda.gguf.codebook import IQ2_XXS_GRID
from tensorfold.cuda.gguf.linear import FORMATS


def reference(data, fmt):
    result = []
    _, size = FORMATS[fmt]
    for start in range(0, len(data), size):
        b = data[start : start + size]
        if fmt == "Q8_0":
            d = struct.unpack_from("<e", b)[0]
            result.extend(d * q for q in struct.unpack_from("<32b", b, 2))
        elif fmt == "Q2_K":
            d, dm = struct.unpack_from("<ee", b, 80)
            for half in range(2):
                for shift in range(4):
                    for group in range(2):
                        scale = b[half * 8 + shift * 2 + group]
                        for j in range(16):
                            q = (b[16 + half * 32 + group * 16 + j] >> (2 * shift)) & 3
                            result.append(np.float32(np.float32(d * (scale & 15)) * q) - np.float32(dm * (scale >> 4)))
        else:
            d = struct.unpack_from("<e", b)[0]
            for i in range(8):
                grids = b[2 + i * 8 : 6 + i * 8]
                bits = struct.unpack_from("<I", b, 6 + i * 8)[0]
                scale = np.float32(d * (0.5 + (bits >> 28)) * 0.25)
                for j, grid in enumerate(grids):
                    signs = (bits >> (7 * j)) & 127
                    signs |= (signs.bit_count() & 1) << 7
                    for v in range(8):
                        mag = (IQ2_XXS_GRID[grid] >> (8 * v)) & 255
                        result.append(np.float32(scale * mag) * (-1 if (signs >> v) & 1 else 1))
    return np.array(result, dtype=np.float32)


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_decode_reference_and_row_independence(fmt):
    block, size = FORMATS[fmt]
    raw = np.random.default_rng(7).integers(0, 256, (7 * 256 // block, size), dtype=np.uint8)
    if fmt == "Q2_K":
        raw[:, 80:84] = np.frombuffer(struct.pack("<ee", 0.03125, 0.015625), dtype=np.uint8)
    else:
        raw[:, :2] = np.frombuffer(struct.pack("<e", 0.03125), dtype=np.uint8)
    expected = reference(raw.tobytes(), fmt).reshape(7, 256)
    packed = Packed(torch.from_numpy(raw.copy()).cuda().flatten(), (256, 7), fmt)
    np.testing.assert_array_equal(packed.unpack().cpu().numpy(), expected)
    x = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16) * 0.01
    y = packed.linear(x)
    assert torch.equal(y, torch.cat([packed.linear(row[None]) for row in x]))
    torch.testing.assert_close(y.float(), x.float() @ torch.tensor(expected, device="cuda").T, rtol=0.01, atol=0.002)


def test_invalid_expert_ids_and_shapes_are_refused_before_routing():
    raw = torch.zeros(2 * 7 * 8 * 34, device="cuda", dtype=torch.uint8)
    packed = Packed(raw, (256, 7, 2), "Q8_0")
    x = torch.ones(17, 256, device="cuda", dtype=torch.bfloat16)
    for bad in [-1, 2]:
        with pytest.raises(ValueError, match="expert pick"):
            packed.linear(x, torch.full((17, 1), bad, device="cuda", dtype=torch.int32))
    with pytest.raises(ValueError, match="expert picks"):
        packed.linear(x, torch.ones(17, device="cuda", dtype=torch.int32))
