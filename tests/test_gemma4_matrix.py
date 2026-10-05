"""Gemma 4's matrix decode projections: every width and group row-exact, stable and near the fp32 product."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

if not mx.metal.is_available():
    pytest.skip("the Gemma decode kernels are Metal kernels", allow_module_level=True)

from tensorfold.kernels.gemma.v1.matmul import Projection, matrix_kind, prefers_matrix  # noqa: E402

FORMATS = [(4, 32), (4, 64), (4, 128), (5, 64), (6, 64), (8, 64), (5, 128), (6, 128), (8, 128)]


def _linear(n_in, n_out, bits, group, seed):
    mx.random.seed(seed)
    lin = nn.QuantizedLinear(n_in, n_out, bias=False, group_size=group, bits=bits)
    w = (mx.random.normal((n_out, n_in)) * 0.05).astype(mx.bfloat16)
    lin.weight, s, b = mx.quantize(w, group_size=group, bits=bits)
    lin.scales, lin.biases = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    return lin


def _fp32(lin, x):
    w = mx.dequantize(lin.weight, lin.scales, lin.biases, group_size=lin.group_size, bits=lin.bits)
    return np.array(x.astype(mx.float32)) @ np.array(w.astype(mx.float32)).T


def _err(y, ref):
    return float(np.abs(np.array(y.astype(mx.float32)) - ref).max() / max(np.abs(ref).max(), 1e-6))


@pytest.mark.parametrize("bits,group", FORMATS)
@pytest.mark.parametrize("n_in,n_out", [(512, 192), (1024, 2304)])
def test_matrix_reads_every_width_with_one_rows_bits_at_any_row_count(bits, group, n_in, n_out):
    lin = _linear(n_in, n_out, bits, group, seed=bits * 1000 + group + n_out)
    proj = Projection([lin], "matrix")
    assert proj.kinds[0] in ("simd", "simd_bits"), proj.kinds
    mx.random.seed(9)
    x = (mx.random.normal((16, n_in)) * 0.5).astype(mx.bfloat16)
    y = proj(x)
    mx.eval(y)
    assert bool(mx.array_equal(proj(x), y).item())                  # the same bits run to run
    for rows in range(1, 17):                                       # every row count: each row's one-row bits
        assert bool(mx.array_equal(proj(x[:rows]), y[:rows]).item()), rows
    for at in (0, 7, 15):
        assert bool(mx.array_equal(proj(x[at:at + 1])[0], y[at]).item()), at
    ref = _fp32(lin, x)
    rows_err = _err(Projection([lin], "rows")(x), ref)
    matrix_err = _err(y, ref)
    assert matrix_err <= 0.01 and matrix_err <= rows_err + 0.004, (matrix_err, rows_err)


def test_matrix_rows_past_a_tile_keep_their_bits():
    lin = _linear(1024, 2304, 6, 64, seed=3)
    proj = Projection([lin], "matrix")
    x = (mx.random.normal((64, 1024)) * 0.5).astype(mx.bfloat16)
    y = proj(x)
    for rows in (17, 24, 33, 63):
        assert bool(mx.array_equal(proj(x[:rows]), y[:rows]).item()), rows
    assert bool(mx.array_equal(proj(x[40:41])[0], y[40]).item())


def test_a_width_no_matrix_kernel_reads_stays_on_the_row_kernels():
    for bits, group in ((2, 64), (3, 64), (3, 128)):
        lin = _linear(512, 192, bits, group, seed=bits)
        assert not matrix_kind(lin.weight, lin.scales, lin.biases, bits, group)
        assert Projection([lin], "matrix").kinds == ("affine",)


@pytest.mark.parametrize("bits,group,n_in,n_out,want", [
    (4, 64, 512, 192, "q4"), (4, 64, 2048, 4096, "simd"), (4, 32, 2048, 4096, "simd"), (4, 128, 512, 192, "simd_bits"),
    (5, 64, 512, 192, "simd_bits"), (8, 64, 2048, 4096, "simd_bits"), (3, 64, 2048, 4096, "affine")])
def test_auto_fixes_each_linear_s_kernel_by_its_format_and_shape_alone(bits, group, n_in, n_out, want):
    lin = _linear(n_in, n_out, bits, group, seed=n_out + bits)
    proj = Projection([lin], "auto")
    assert proj.kinds == (want,)
    x = (mx.random.normal((16, n_in)) * 0.5).astype(mx.bfloat16)
    y = proj(x)
    for rows in (1, 2, 3, 5, 8, 16):
        assert proj.kinds == (want,)
        assert bool(mx.array_equal(proj(x[:rows]), y[:rows]).item()), rows


def test_a_stack_of_mixed_widths_on_matrix_is_each_linear():
    specs = [(5, 64), (5, 64), (4, 128), (8, 64), (3, 64), (4, 64), (4, 32)]
    linears = [_linear(512, 64 * (i % 3 + 1), bits, group, seed=i) for i, (bits, group) in enumerate(specs)]
    proj = Projection(linears, "matrix")
    assert proj.kinds == ("simd_bits", "simd_bits", "simd_bits", "affine", "simd", "simd")
    x = (mx.random.normal((13, 512)) * 0.5).astype(mx.bfloat16)
    y = proj(x)
    ref = np.concatenate([_fp32(l, x) for l in linears], axis=-1)
    assert _err(y, ref) <= 0.01
    for at in (0, 5, 12):
        assert bool(mx.array_equal(proj(x[at:at + 1])[0], y[at]).item()), at
