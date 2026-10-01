"""simd_qmm_bits, simd_qmm's row-exact matmul for 5-, 6- and 8-bit codes in groups of 64: every row identical whatever
the row count, the scalar twin's 1-4-row calls equal to the matrix kernel's rows, and as accurate as MLX."""

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.dense.v1 import simd_qmm_bits  # noqa: E402

SHAPES = [(17408, 5120), (5120, 17408), (5120, 6144), (1024, 5120), (48, 5120)]


def _same(a, b):
    return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def _weights(n, k, bits, seed=7):
    mx.random.seed(seed)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    return mx.quantize(w, group_size=64, bits=bits)


@pytest.mark.parametrize("bits", simd_qmm_bits.BITS)
@pytest.mark.parametrize("n,k", SHAPES)
def test_rows_do_not_depend_on_row_count(n, k, bits):
    q, s, b = _weights(n, k, bits)
    x = (mx.random.normal((64, k)) * 0.5).astype(mx.bfloat16)
    full = simd_qmm_bits.qmm(x, q, s, b, bits)
    mx.eval(full)
    for m in (1, 2, 3, 5, 8, 9, 16, 17, 33, 64):
        assert _same(simd_qmm_bits.qmm(x[:m], q, s, b, bits), full[:m]), f"rows 0..{m - 1} changed with the row count"
    for r in (0, 7, 20, 63):
        assert _same(simd_qmm_bits.qmm(x[r:r + 1], q, s, b, bits), full[r:r + 1]), f"row {r} alone differs"


@pytest.mark.parametrize("bits", simd_qmm_bits.BITS)
@pytest.mark.parametrize("n,k", SHAPES)
def test_scalar_twin_matches_matrix_kernel(n, k, bits):
    q, s, b = _weights(n, k, bits, seed=3)
    assert simd_qmm_bits.check(q, s, b, bits)


@pytest.mark.parametrize("bits", simd_qmm_bits.BITS)
def test_as_accurate_as_mlx(bits):
    n, k = 1024, 5120
    q, s, b = _weights(n, k, bits, seed=5)
    x = (mx.random.normal((8, k)) * 0.5).astype(mx.bfloat16)
    ref = x.astype(mx.float32) @ mx.dequantize(q, s, b, group_size=64, bits=bits).astype(mx.float32).T
    scale = float(mx.abs(ref).max().item())
    ours = float(mx.abs(simd_qmm_bits.qmm(x, q, s, b, bits).astype(mx.float32) - ref).max().item()) / scale
    theirs = float(mx.abs(mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=bits)
                          .astype(mx.float32) - ref).max().item()) / scale
    assert ours <= max(theirs, 0.005)


def test_fits_takes_5_6_8_bits_in_groups_of_64():
    q, s, b = _weights(64, 256, 5)
    assert simd_qmm_bits.fits(q, s, b, 64, 5)
    assert not simd_qmm_bits.fits(q, s, b, 64, 4)
    q3, s3, b3 = mx.quantize(mx.zeros((64, 256), dtype=mx.bfloat16), group_size=32, bits=5)
    assert not simd_qmm_bits.fits(q3, s3.astype(mx.bfloat16), b3.astype(mx.bfloat16), 32, 5)


G128_SHAPES = [(17408, 5120), (5120, 17408), (5120, 6144), (1024, 5120)]


def _quant(n, k, bits, group, seed):
    mx.random.seed(seed)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    return mx.quantize(w, group_size=group, bits=bits)


@pytest.mark.parametrize("bits", simd_qmm_bits.BITS)
@pytest.mark.parametrize("n,k", G128_SHAPES)
def test_group_128_matches_two_groups_of_64(n, k, bits):
    q, s, b = _quant(n, k, bits, 128, seed=11)
    x = (mx.random.normal((16, k)) * 0.5).astype(mx.bfloat16)
    s2, b2 = mx.repeat(s, 2, axis=1), mx.repeat(b, 2, axis=1)
    native = simd_qmm_bits.qmm(x, q, s, b, bits, 128)
    wide = simd_qmm_bits.qmm(x, q, s2, b2, bits, 64)
    mx.eval(native, wide)
    assert _same(native, wide)
    assert _same(simd_qmm_bits.qmm(x[:1], q, s, b, bits, 128), native[:1])
    assert simd_qmm_bits.check(q, s, b, bits, 128)


def test_group_128_is_as_accurate_as_mlx():
    n, k, bits = 1024, 5120, 5
    q, s, b = _quant(n, k, bits, 128, seed=5)
    x = (mx.random.normal((8, k)) * 0.5).astype(mx.bfloat16)
    ref = x.astype(mx.float32) @ mx.dequantize(q, s, b, group_size=128, bits=bits).astype(mx.float32).T
    scale = float(mx.abs(ref).max().item())
    ours = float(mx.abs(simd_qmm_bits.qmm(x, q, s, b, bits, 128).astype(mx.float32) - ref).max().item()) / scale
    theirs = float(mx.abs(mx.quantized_matmul(
        x, q, s, b, transpose=True, group_size=128, bits=bits).astype(mx.float32) - ref).max().item()) / scale
    assert ours <= max(theirs, 0.005)


def test_fits_takes_groups_of_128():
    q, s, b = _quant(64, 256, 8, 128, seed=1)
    assert simd_qmm_bits.fits(q, s, b, 128, 8)
    assert not simd_qmm_bits.fits(q, s, b, 128, 4)
