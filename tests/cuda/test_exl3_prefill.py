"""The EXL3 prompt matmul (``cuda/exl3/prefill.py``) on a GPU, on synthetic layers: every path a row can take gives
the same bits whatever the row count, and the paths that replace one another (tile changes, kernels that rebuild the
same decoded weights) give the bits of the path they replace."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3 import linear, prefill

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

ROWS = (1, 3, 16, 17, 33, 48, 49, 64, 128, 129, 300)
FOLD_COMBOS = [("mul1", 4), ("mul1", 3), ("mul1", 2.5), ("mul1", 6), ("3inst", 2), ("3inst", 8), ("mcg", 4)]


def _layer(codebook: str, bits: float, k: int, n: int, seed: int = 0, bias: bool = False) -> linear.Exl3Linear:
    """A synthetic group: random trellis words, small scales; K and N multiples of 128."""

    rng = np.random.default_rng(seed)
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(k // 16, n // 16, fmt.tile_words(bits)))
                               .astype(np.int16))
    suh = torch.from_numpy((rng.standard_normal(k) * 0.05).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(n) * 0.05).astype(np.float16))
    b = torch.from_numpy((rng.standard_normal(n) * 0.05).astype(np.float16)) if bias else None
    return linear.Exl3Linear.from_tensors(trellis, suh, svh, codebook, b)


def _rows(m: int, k: int, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((m, k), generator=gen, device="cuda").to(torch.bfloat16)


def _run(layer: linear.Exl3Linear, x: torch.Tensor, ws: prefill.Workspace) -> torch.Tensor:
    out = torch.empty((x.shape[0], layer.n), dtype=torch.bfloat16, device=x.device)
    return prefill.matmul(layer, x, out, ws)


@pytest.mark.parametrize("codebook,bits,k,n", [("mul1", 4, 5120, 6144), ("mul1", 3, 1024, 2048),
                                                ("3inst", 2, 512, 1024), ("mcg", 6, 1024, 512)])
def test_the_k_step_leaves_the_bits_unchanged(monkeypatch, codebook, bits, k, n):
    """K steps of 64 sum the same products in the same order as steps of 32: the same output bits."""

    layer = _layer(codebook, bits, k, n, bias=True)
    x = _rows(300, k)
    now = _run(layer, x, prefill.Workspace())
    monkeypatch.setattr(prefill, "tiles", lambda k, n: (128, 32, 8, 4, 8))
    assert torch.equal(_run(layer, x, prefill.Workspace()), now)



@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("codebook,bits", FOLD_COMBOS)
def test_unpack_fold2_equals_unpack_fold(codebook, bits, dtype):
    """The fast fold kernel gives the straightforward one's W'' bit for bit, in bf16 and fp16."""

    layer = _layer(codebook, bits, 512, 1024, seed=3)
    ext = linear._ext()
    a = torch.empty((layer.k, layer.n), dtype=dtype, device="cuda")
    b = torch.empty_like(a)
    ext.unpack_fold(layer.words, layer.suh, a, *layer.strides, layer.k2, linear.CODEBOOK_IDS[codebook])
    ext.unpack_fold2(layer.words, layer.suh, b, *layer.strides, layer.k2, linear.CODEBOOK_IDS[codebook])
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("codebook,bits", [("mul1", 4), ("3inst", 3), ("mcg", 6)])
def test_folded_weights_match_the_float64_reference(codebook, bits):
    rng = np.random.default_rng(5)
    k, n = 256, 384
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(k // 16, n // 16, fmt.tile_words(bits)))
                               .astype(np.int16))
    suh = torch.from_numpy((rng.standard_normal(k) * 0.05).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(n) * 0.05).astype(np.float16))
    layer = linear.Exl3Linear.from_tensors(trellis, suh, svh, codebook)
    want = suh.double().numpy()[:, None] * fmt.rotate(fmt.rotate(fmt.unpack(trellis, bits, codebook)
                                                                 .astype(np.float64), 1), 0)
    got = torch.empty((k, n), dtype=torch.bfloat16, device="cuda")
    linear._ext().unpack_fold2(layer.words, layer.suh, got, *layer.strides, layer.k2, linear.CODEBOOK_IDS[codebook])
    err = np.linalg.norm(got.double().cpu().numpy() - want) / np.linalg.norm(want)
    assert err < 4e-3                                          # one bf16 rounding (2^-9 relative at most)


def test_the_folded_path_matches_the_float64_reference():
    rng = np.random.default_rng(6)
    k, n, bits = 512, 256, 4
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(k // 16, n // 16, fmt.tile_words(bits)))
                               .astype(np.int16))
    suh = torch.from_numpy((rng.standard_normal(k) * 0.05).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(n) * 0.05).astype(np.float16))
    bias = torch.from_numpy((rng.standard_normal(n) * 0.05).astype(np.float16))
    layer = linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1", bias)
    x = _rows(37, k)
    got = _run(layer, x, prefill.Workspace(fold=True))
    w64 = suh.double().numpy()[:, None] * fmt.rotate(fmt.rotate(fmt.unpack(trellis, bits, "mul1")
                                                                .astype(np.float64), 1), 0)
    want = (x.double().cpu().numpy() @ w64) * svh.double().numpy() + bias.double().numpy()
    err = np.linalg.norm(got.double().cpu().numpy() - want) / np.linalg.norm(want)
    assert err < 6e-3                                          # bf16 weights and output


@pytest.mark.parametrize("codebook,bits,k,n", [("mul1", 4, 5120, 6144), ("mul1", 3, 1024, 2048),
                                                ("3inst", 2, 512, 1024), ("mcg", 6, 1024, 512)])
def test_folded_rows_never_depend_on_the_row_count(codebook, bits, k, n):
    """A row's bits on the folded path are the same alone, in any window and at any offset."""

    layer = _layer(codebook, bits, k, n, bias=True)
    ws = prefill.Workspace(fold=True)
    x = _rows(300, k)
    whole = _run(layer, x, ws)
    for m in ROWS:
        assert torch.equal(_run(layer, x[:m], ws), whole[:m]), f"rows 0..{m}"
        r0 = min(5, 300 - m)
        assert torch.equal(_run(layer, x[r0:r0 + m], ws), whole[r0:r0 + m]), f"rows {r0}..{r0 + m}"
    for r in (0, 1, 47, 128, 299):
        assert torch.equal(_run(layer, x[r:r + 1], ws), whole[r:r + 1]), f"row {r} alone"


def test_the_folded_path_is_opt_in(monkeypatch):
    """A plain Workspace keeps the W_q path's bits; TENSORFOLD_EXL3_FOLD=0 puts an opted-in family back on it."""

    layer = _layer("mul1", 4, 1024, 512)
    x = _rows(40, 1024)
    plain = _run(layer, x, prefill.Workspace())
    assert not torch.equal(_run(layer, x, prefill.Workspace(fold=True)), plain)
    monkeypatch.setattr(prefill, "FOLD", False)
    assert torch.equal(_run(layer, x, prefill.Workspace(fold=True)), plain)


@pytest.mark.parametrize("codebook,k,n,bias", [("mul1", 5120, 6144, False), ("mul1", 17408, 5120, False),
                                                ("mul1", 6144, 5120, True), ("3inst", 1024, 512, True),
                                                ("mcg", 512, 1024, False)])
def test_fdirect_gives_unpack_fold2_and_the_gemms_bits(monkeypatch, codebook, k, n, bias):
    """4-bit calls of up to FDIRECT_ROWS rows rebuild W'' inside one kernel: the bits of W'' decoded and then
    multiplied, at every row count it may take (one kernel per 128-row slice above 128)."""

    if bias:
        pytest.xfail("fdirect with a bias does not yet match _gemm_fold's epilogue rounding; fdirect is off by default")
    layer = _layer(codebook, 4, k, n, seed=k, bias=bias)
    ws = prefill.Workspace(fold=True)
    x = _rows(200, k, seed=2)
    monkeypatch.setattr(prefill, "FDIRECT_ROWS", 0)
    want = _run(layer, x, ws)
    monkeypatch.setattr(prefill, "FDIRECT_ROWS", 200)
    for m in (1, 3, 16, 17, 33, 48, 64, 100, 128, 200):
        assert torch.equal(_run(layer, x[:m], ws), want[:m]), f"{m} rows"
