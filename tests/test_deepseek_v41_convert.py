"""DeepSeek-V4.1's load-time weight rewrites change no value: FP8 32x32 blocks, MXFP4 experts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dsv41_layouts import mx8_from_block

from tensorfold.cuda.nvfp4 import experts as nvfp4
from tensorfold.families.deepseek_v41.cuda import convert

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
MODEL = os.environ.get("TF_DSV41_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set TF_DSV41_MODEL to the checkpoint")


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view({torch.float32: torch.int32, torch.float64: torch.int64}[x.dtype])


def _e4m3(gen: torch.Generator, n: int, k: int) -> torch.Tensor:
    b = torch.randint(0, 256, (n, k), generator=gen, dtype=torch.int32)
    return torch.where((b & 0x7F) == 0x7F, b - 1, b).to(torch.uint8).view(torch.float8_e4m3fn)   # no NaN codes


def _block_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """W[o, i] = e4m3(q[o, i]) * 2^(S[o // 32, i // 32] - 127), one 32x32 block at a time."""

    w = weight.to(torch.float64)
    s = scale.contiguous().view(torch.uint8).to(torch.float64)
    out = torch.empty_like(w)
    for b in range(s.shape[0]):
        for c in range(s.shape[1]):
            rows, cols = slice(32 * b, 32 * b + 32), slice(32 * c, 32 * c + 32)
            out[rows, cols] = w[rows, cols] * torch.pow(2.0, s[b, c] - 127)
    return out


def _mx8_dequant(lin) -> torch.Tensor:
    """What ``Mx8Linear`` stores, back to values: its e4m3 rows times each row's 1x32 powers of two."""

    s = lin.scale_rows().to(torch.float64).repeat_interleave(32, dim=1)
    return lin.w8_rows().to(torch.float64) * torch.pow(2.0, s - 127)


def _mxfp4_dequant(words: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """DeepSeek's MXFP4: low nibble = even element, high = odd, times 2^(S[o, i // 32] - 127)."""

    b = words.contiguous().view(torch.uint8).to(torch.int64)
    codes = torch.stack([b & 0xF, b >> 4], dim=-1).reshape(b.shape[0], -1)
    s = scale.contiguous().view(torch.uint8).to(torch.float64).repeat_interleave(32, dim=1)
    return torch.tensor(E2M1, dtype=torch.float64)[codes] * torch.pow(2.0, s - 127)


def _mxfp4(gen: torch.Generator, n: int, k: int, low: int, top: int) -> tuple[torch.Tensor, torch.Tensor]:
    words = torch.randint(-128, 128, (n, k // 2), generator=gen, dtype=torch.int8)
    e = torch.randint(low, top + 1, (n, k // 32), generator=gen, dtype=torch.uint8)
    e[0, 0], e[-1, -1] = low, top                      # the span's ends both present
    return words, e.view(torch.float8_e8m0fnu)


def _assert_experts_exact(ex, gate, up, down) -> None:
    for i in range(len(gate)):
        for which, (w, s) in (("gate", gate[i]), ("up", up[i]), ("down", down[i])):
            got, want = nvfp4.dense(ex, i, which), _mxfp4_dequant(w, s)
            assert torch.equal(_bits(got), _bits(want.to(torch.float32))), (i, which)


@pytest.mark.parametrize("n", [64, 100, 33])
def test_row_repeat_equals_block_dequant(n):
    gen = torch.Generator().manual_seed(n)
    k = 128
    weight = _e4m3(gen, n, k)
    scale = torch.randint(100, 141, (-(-n // 32), k // 32), generator=gen, dtype=torch.uint8)
    lin = mx8_from_block(weight, scale.view(torch.float8_e8m0fnu))
    assert (lin.n, lin.k) == (n, k)
    assert torch.equal(_bits(_mx8_dequant(lin)), _bits(_block_dequant(weight, scale)))


def test_row_repeat_refuses_scales_of_another_height():
    with pytest.raises(ValueError, match="do not cover"):
        convert.fp8_block_rows(torch.zeros((3, 4), dtype=torch.uint8), 100)


def test_mxfp4_experts_dense_equal_the_float_dequant():
    gen = torch.Generator().manual_seed(7)
    spans = [(115, 128), (111, 128), (0, 17), (237, 254)]          # the pack's range, span 17 at both ends
    gate = [_mxfp4(gen, 64, 128, *spans[i]) for i in range(4)]
    up = [_mxfp4(gen, 64, 128, *spans[3 - i]) for i in range(4)]
    down = [_mxfp4(gen, 128, 64, *spans[(i + 1) % 4]) for i in range(4)]
    ex = convert.make_experts4(gate, up, down)
    assert (ex.count, ex.width, ex.dims, ex.limit) == (4, 64, 128, 10.0)
    _assert_experts_exact(ex, gate, up, down)


def test_mxfp4_conversion_keeps_words_and_scales_each_matrix_alone():
    gen = torch.Generator().manual_seed(3)
    words, e = _mxfp4(gen, 32, 64, 120, 125)
    got, sixteen, whole = convert.mxfp4_to_nvfp4(words, e)
    assert got.dtype == torch.uint8 and torch.equal(got, words.view(torch.uint8))
    assert sixteen.shape == (32, 4) and whole.dtype == torch.float32 and float(whole) == 2.0 ** (125 - 135)
    want = torch.pow(2.0, e.view(torch.uint8).to(torch.float64) - 125 + 8).repeat_interleave(2, dim=1)
    assert torch.equal(sixteen.view(torch.float8_e4m3fn).to(torch.float64), want)


def test_low_nibble_is_the_even_element():
    words = torch.full((32, 16), 0x21, dtype=torch.uint8).view(torch.int8)
    e = torch.full((32, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    ex = convert.make_experts4([(words, e)], [(words, e)], [(torch.zeros((32, 16), dtype=torch.int8), e)])
    row = nvfp4.dense(ex, 0, "gate")[0]
    assert row[0::2].eq(0.5).all() and row[1::2].eq(1.0).all()


def test_mxfp4_span_18_is_refused():
    gen = torch.Generator().manual_seed(5)
    words, e = _mxfp4(gen, 32, 64, 110, 128)
    with pytest.raises(ValueError, match="span 18"):
        convert.mxfp4_to_nvfp4(words, e)
    nan = torch.full((32, 2), 0xFF, dtype=torch.uint8)
    with pytest.raises(ValueError, match="NaN"):
        convert.mxfp4_to_nvfp4(words, nan)


def _pack(name: str) -> torch.Tensor:
    safetensors = pytest.importorskip("safetensors")
    root = Path(MODEL)
    shard = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"][name]
    with safetensors.safe_open(str(root / shard), framework="pt") as f:
        return f.get_tensor(name)


@needs_model
def test_real_wq_b_row_repeat_is_exact():
    weight, scale = _pack("layers.0.attn.wq_b.weight"), _pack("layers.0.attn.wq_b.scale")
    lin = mx8_from_block(weight, scale)
    assert torch.equal(_bits(_mx8_dequant(lin)), _bits(_block_dequant(weight, scale)))


@needs_model
@pytest.mark.parametrize("rank", [0, 1])
def test_real_dspark_expert_rank_half_is_exact(rank):
    name = "mtp.0.ffn.experts.0.{}.{}"
    w1, w3, w2 = ((_pack(name.format(m, "weight")), _pack(name.format(m, "scale"))) for m in ("w1", "w3", "w2"))
    rows = slice(1152 * rank, 1152 * rank + 1152)
    gate, up = [(w1[0][rows], w1[1][rows])], [(w3[0][rows], w3[1][rows])]
    down = [(w2[0][:, 576 * rank:576 * rank + 576], w2[1][:, 36 * rank:36 * rank + 36])]
    ex = convert.make_experts4(gate, up, down)
    assert (ex.width, ex.dims) == (1152, 5120)
    _assert_experts_exact(ex, gate, up, down)
