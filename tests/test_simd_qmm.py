"""simd_qmm, the row-exact 4-bit matmul for GPUs without the M5's tensor units: every row identical whatever the
row count, one-row calls (scalar kernel) equal to window rows (MMA kernel), and fp32-accurate. Runs on any Apple
GPU."""

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.dense.v1 import simd_qmm  # noqa: E402

SHAPES = [(17408, 5120), (5120, 17408), (1024, 5120), (48, 5120), (2688, 1856)]


def _same(a, b):
    return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def _weights(n, k, seed=7):
    mx.random.seed(seed)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    return mx.quantize(w, group_size=64, bits=4)


@pytest.mark.parametrize("n,k", SHAPES)
def test_rows_do_not_depend_on_row_count(n, k):
    q, s, b = _weights(n, k)
    x = (mx.random.normal((128, k)) * 0.5).astype(mx.bfloat16)
    full = simd_qmm.qmm(x, q, s, b)
    mx.eval(full)
    for m in (1, 2, 5, 8, 9, 16, 17, 33, 100, 128):
        assert _same(simd_qmm.qmm(x[:m], q, s, b), full[:m]), f"rows 0..{m - 1} changed with the row count"
    # a row computed alone (the scalar kernel) equals the same row anywhere inside a window (the MMA kernel)
    for r in (0, 7, 20, 127):
        assert _same(simd_qmm.qmm(x[r:r + 1], q, s, b), full[r:r + 1]), f"row {r} alone differs"


@pytest.mark.parametrize("n,k", SHAPES)
def test_scalar_kernel_matches_mma_kernel(n, k):
    q, s, b = _weights(n, k, seed=3)
    assert simd_qmm.check(q, s, b)


@pytest.mark.parametrize("n,k", [(5120, 17408), (1024, 5120)])
def test_as_accurate_as_mlx(n, k):
    q, s, b = _weights(n, k, seed=5)
    x = (mx.random.normal((8, k)) * 0.5).astype(mx.bfloat16)
    ref = x.astype(mx.float32) @ mx.dequantize(q, s, b, group_size=64, bits=4).astype(mx.float32).T
    scale = float(mx.abs(ref).max().item())
    ours = float(mx.abs(simd_qmm.qmm(x, q, s, b).astype(mx.float32) - ref).max().item()) / scale
    theirs = float(mx.abs(mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=4)
                          .astype(mx.float32) - ref).max().item()) / scale
    assert ours <= max(theirs, 0.005)


def test_prologue_gives_the_unfused_bits():
    header = r"""
inline uint4 scale8(const device bfloat* X, const device bfloat* E, int r, int j, int K) {
  uint4 out;
  for (int h = 0; h < 4; h++) {
    const int k = 8 * j + 2 * h;
    const bfloat a = bfloat(float(X[size_t(r) * K + k]) * float(E[k]));
    const bfloat b = bfloat(float(X[size_t(r) * K + k + 1]) * float(E[k + 1]));
    out[h] = uint(as_type<ushort>(a)) | (uint(as_type<ushort>(b)) << 16);
  }
  return out;
}
"""
    pro = simd_qmm.Prologue("scale", "scale8(X, E, (r), (j), K)", ("E",), header)
    n, k = 1024, 5120
    q, s, b = _weights(n, k, seed=11)
    e = mx.random.uniform(0.5, 1.5, (k,)).astype(mx.bfloat16)
    for rows in (1, 5, 40):
        x = (mx.random.normal((rows, k)) * 0.5).astype(mx.bfloat16)
        ref = simd_qmm.qmm((x.astype(mx.float32) * e.astype(mx.float32)).astype(mx.bfloat16), q, s, b)
        assert _same(simd_qmm.qmm(x, q, s, b, prologue=pro, extra=[e]), ref)


def test_fits_needs_groups_of_64_and_outputs_in_eights():
    import mlx.nn as nn

    ok = nn.QuantizedLinear(1856, 2688, bias=False, group_size=64, bits=4)
    ok.scales = ok.scales.astype(mx.bfloat16)
    assert simd_qmm.fits(ok)
    g32 = nn.QuantizedLinear(1856, 2688, bias=False, group_size=32, bits=4)
    g32.scales = g32.scales.astype(mx.bfloat16)
    assert not simd_qmm.fits(g32)
