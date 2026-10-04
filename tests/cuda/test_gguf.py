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
    torch.manual_seed(7)
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
    y = packed.linear(x, dtype=torch.float32)
    assert torch.equal(y, torch.cat([packed.linear(row[None], dtype=torch.float32) for row in x]))
    torch.testing.assert_close(y.float(), x.float() @ torch.tensor(expected, device="cuda").T, rtol=0.01, atol=0.002)
    # Exercise grouped prefill, including a partial row tile and output tile.
    data = torch.cat((packed.data, packed.data.reshape(7, -1).flip(0).flatten()))
    experts = Packed(data, (256, 7, 2), fmt)
    x = torch.randn(65, 256, device="cuda", dtype=torch.bfloat16) * 0.01
    ref = x.float() @ torch.tensor(expected, device="cuda").bfloat16().float().T
    torch.testing.assert_close(packed.linear(x).float(), ref, rtol=0.01, atol=0.002)
    picks = (torch.arange(65, device="cuda") % 2).to(torch.int32)[:, None]
    y = experts.linear(x, picks)
    matrices = torch.tensor(np.stack((expected, expected[::-1].copy())), device="cuda").bfloat16()
    ref = torch.bmm(matrices[picks[:, 0]].float(), x.float()[:, :, None]).squeeze(-1)
    torch.testing.assert_close(y[:, 0].float(), ref, rtol=0.01, atol=0.002)
    torch.testing.assert_close(experts.linear(x[:, None], picks)[:, 0].float(), ref, rtol=0.01, atol=0.002)
    narrow = experts.linear(x[:5], picks[:5], dtype=torch.float32)
    solo = torch.cat([experts.linear(x[i : i + 1], picks[i : i + 1], dtype=torch.float32) for i in range(5)])
    assert torch.equal(narrow, solo)
    if fmt == "Q8_0":
        from tensorfold.cuda.gguf.prefill import Workspace

        out = torch.empty((65, 7), device=x.device, dtype=torch.float32)
        Workspace().matmul(packed, x, out)
        expected_dense = x.float() @ packed.unpack().bfloat16().float().T
        torch.testing.assert_close(out, expected_dense, rtol=1e-5, atol=1e-6)
        assert torch.equal(out[:1], Workspace().grouped(packed, x[:1, None], dtype=torch.float32))
        # Uneven output tiles must not write into a neighboring group's output.
        grouped = Packed(data, (256, 14), fmt)
        for rows in (1, 5, 21, 65):
            inputs = torch.randn(rows, 2, 256, device=x.device, dtype=torch.bfloat16) * 0.01
            parts = [
                Packed(data[j * packed.data.numel() : (j + 1) * packed.data.numel()], packed.shape, fmt)
                for j in range(2)
            ]
            expected = torch.cat(
                [part.linear(inputs[:, j].contiguous(), dtype=torch.float32) for j, part in enumerate(parts)], 1
            )
            assert torch.equal(grouped.linear_grouped(inputs, dtype=torch.float32), expected)
        grouped.workspace = Workspace()
        separate = []
        for j, part in enumerate(parts):
            output = torch.empty((rows, 7), device=x.device, dtype=torch.float32)
            Workspace().matmul(part, inputs[:, j].contiguous(), output)
            separate.append(output)
        assert torch.equal(grouped.workspace.grouped(grouped, inputs, dtype=torch.float32), torch.cat(separate, 1))


def test_invalid_expert_ids_and_shapes_are_refused_before_routing():
    raw = torch.zeros(2 * 7 * 8 * 34, device="cuda", dtype=torch.uint8)
    packed = Packed(raw, (256, 7, 2), "Q8_0")
    x = torch.ones(17, 256, device="cuda", dtype=torch.bfloat16)
    for bad in [-1, 2]:
        with pytest.raises(ValueError, match="expert pick"):
            packed.linear(x, torch.full((17, 1), bad, device="cuda", dtype=torch.int32))
    with pytest.raises(ValueError, match="expert picks"):
        packed.linear(x, torch.ones(17, device="cuda", dtype=torch.int32))


def test_prepare_soa_and_decode_row_sharing_match_raw():
    """SoA preparation is lossless; narrow grouped rows keep serial reductions."""
    torch.manual_seed(7)

    from tensorfold.cuda.gguf.prepare import prepare_packed

    rng = np.random.default_rng(11)
    for fmt, n, e in (("IQ2_XXS", 64, 3), ("Q2_K", 64, 3)):
        block, size = FORMATS[fmt]
        k = 256
        raw = rng.integers(0, 256, (e * n * (k // block), size), dtype=np.uint8)
        if fmt == "Q2_K":
            raw[:, 80:84] = np.frombuffer(struct.pack("<ee", 0.03125, 0.015625), dtype=np.uint8)
        else:
            raw[:, :2] = np.frombuffer(struct.pack("<e", 0.03125), dtype=np.uint8)
        data = torch.from_numpy(raw.copy()).cuda().flatten()
        packed = Packed(data, (k, n, e), fmt)
        prepared = prepare_packed(packed)
        assert prepared.layout == "soa"
        if fmt == "IQ2_XXS":
            from tensorfold.cuda.gguf.prepare import iq2_soa_bytes

            nblk = e * n * (k // 256)
            dq_bytes, total = iq2_soa_bytes(nblk)
            assert prepared.bn == dq_bytes and prepared.data.numel() == total
            dq = prepared.data[:dq_bytes].view(torch.float16)
            qs = prepared.data[dq_bytes:].view(torch.uint64)
            for ei in range(e):
                for ni in range(n):
                    for g in range(k // 256):
                        blk = (ei * n + ni) * (k // 256) + g
                        src = packed.data[blk * size : (blk + 1) * size]
                        assert src[:2].view(torch.float16).item() == dq[blk].item()
                        for w in range(8):
                            word = int.from_bytes(src[2 + w * 8 : 10 + w * 8].cpu().numpy().tobytes(), "little")
                            assert int(qs[blk * 8 + w].item()) == word
        else:
            from tensorfold.cuda.gguf.prepare import q2_soa_bytes

            nblk = e * n * (k // 256)
            dm_bytes, sc_bytes, total = q2_soa_bytes(nblk)
            assert (prepared.bn & 0xFFFFFFFF) == dm_bytes and (prepared.bn >> 32) == sc_bytes
            assert prepared.data.numel() == total
        x = torch.randn(4, k, device="cuda", dtype=torch.bfloat16) * 0.01
        picks = torch.tensor([[0], [1], [1], [2]], device="cuda", dtype=torch.int32)
        y = prepared.linear(x, picks, dtype=torch.float32)
        ref = torch.cat([prepared.linear(x[i : i + 1], picks[i : i + 1], dtype=torch.float32) for i in range(4)])
        assert torch.equal(y, ref)
        # Prefill-sized grouping (rows>16) must match the decode-width path within bf16 ULP.
        x64 = torch.randn(65, k, device="cuda", dtype=torch.bfloat16) * 0.01
        picks64 = (torch.arange(65, device="cuda") % e).to(torch.int32)[:, None]
        import os

        os.environ["TENSORFOLD_GGUF_CUDA_PREFILL"] = "1"
        cuda_out = prepared.linear(x64, picks64)
        os.environ["TENSORFOLD_GGUF_CUDA_PREFILL"] = "0"
        tri_out = prepared.linear(x64, picks64)
        os.environ.pop("TENSORFOLD_GGUF_CUDA_PREFILL", None)
        # CUDA D2R vs Triton SoA: same bf16 MMA contract (≤1–2 ULP).
        torch.testing.assert_close(cuda_out.float(), tri_out.float(), rtol=0, atol=0.001)
        if fmt == "Q2_K":
            assert torch.equal(cuda_out, tri_out)
        per_row = torch.cat([prepared.linear(x64[i : i + 1], picks64[i : i + 1]) for i in range(65)])
        torch.testing.assert_close(cuda_out[:, 0].float(), per_row.float(), rtol=0.01, atol=0.01)
