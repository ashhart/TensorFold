"""Packed FP8 key/value rows on the GPU: ``kv8.pack`` gives ``reference_pack``'s bytes and ``unpack``'s values from one
launch, for strided rows too, and a row packs alike in any window."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels import kv8  # noqa: E402

dev = "cuda"


def _bits(x):
    return x.contiguous().view(torch.int16)


def _random(rows, heads, seed):
    """Rows with max |x| from about 2^-100 to 2^100: their values times 2^e stay normal fp32 on any GPU."""

    g = torch.Generator().manual_seed(seed)
    shift = torch.randint(-100, 100, (rows, heads, 1), generator=g).double()
    return (torch.randn(rows, heads, 256, generator=g, dtype=torch.float64) * torch.exp2(shift)).bfloat16().to(dev)


def _edges():
    """(rows, 1, 256): ties between e4m3 neighbours and subnormals at three exponents, amax on either side of the
    1.75 mantissa boundary, one that rounds down to 448 * 2^(e - 1), and rows of +0.0 and -0.0."""

    g = torch.Generator().manual_seed(7)
    ties = [448.0, 432.0, 400.0, 2.0 ** -10, 3 * 2.0 ** -10, -0.0, -2.0 ** -11, 1.0, -448.0, 17.0]
    rows = []
    for scale in (1.0, 2.0 ** 20, 2.0 ** -40):
        rows.append(torch.tensor(ties + [0.0] * 246, dtype=torch.float64) * scale)
    for amax in (448.0, 450.0, 446.0, 454.0):
        for k in (-60, 0, 60):
            row = torch.randn(256, generator=g, dtype=torch.float64).clamp(-3, 3) * 2.0 ** k
            row[17] = -amax * 2.0 ** k
            rows.append(row)
    rows += [torch.zeros(256, dtype=torch.float64), -torch.zeros(256, dtype=torch.float64)]
    return torch.stack(rows).bfloat16().view(-1, 1, 256).to(dev)


@pytest.mark.parametrize("make", [lambda: _random(37, 4, 0), lambda: _random(1, 1, 1), _edges])
def test_pack_matches_the_reference_byte_for_byte(make):
    x = make()
    rounded, packed = kv8.pack(x)
    assert packed.shape == (*x.shape[:2], kv8.ROW8) and packed.dtype == torch.uint8 and rounded.dtype == torch.bfloat16
    assert packed.cpu().equal(kv8.reference_pack(x.cpu()))
    assert _bits(rounded).equal(_bits(kv8.unpack(packed)))
    assert _bits(rounded.cpu()).equal(_bits(kv8.unpack(kv8.reference_pack(x.cpu()))))


def test_strided_rows_pack_as_their_copies():
    both = _random(24, 8, 2)
    kv = torch.randn(24, 2 * 4 * 256, device=dev).bfloat16()
    views = [both[:, 4:],                                      # rows and heads stride
             kv[:, 4 * 256:].reshape(24, 4, 256),              # the values of a [k | v] projection
             torch.randn(24, 256, 4, device=dev).bfloat16().transpose(1, 2)]       # values that stride: a copy
    for x in views:
        rounded, packed = kv8.pack(x)
        want_r, want_p = kv8.pack(x.contiguous())
        assert packed.equal(want_p) and _bits(rounded).equal(_bits(want_r))
        assert packed.cpu().equal(kv8.reference_pack(x.cpu()))


def test_a_row_packs_alike_in_any_window():
    x = _random(19, 4, 3)
    rounded, packed = kv8.pack(x)
    for r in (0, 7, 18):
        one_r, one_p = kv8.pack(x[r:r + 1])
        assert one_p.equal(packed[r:r + 1]) and _bits(one_r).equal(_bits(rounded[r:r + 1]))
    empty_r, empty_p = kv8.pack(x[:0])
    assert empty_r.shape == (0, 4, 256) and empty_p.shape == (0, 4, kv8.ROW8)


def test_the_simulation_rounds_with_the_same_quantizer():
    from tensorfold.families.qwen3_5.cuda import glue

    x = torch.cat([_random(8, 1, 4), _edges()])
    assert _bits(glue.fp8_round(x).cpu()).equal(_bits(kv8.unpack(kv8.reference_pack(x.cpu()))))
