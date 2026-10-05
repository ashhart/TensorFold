"""GLM-5.3-Flash's decode linears: one kernel per format and shape (``linear.choose``), the same at every row count."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.glm5_next import linear  # noqa: E402
from tensorfold.families.glm5_next.kda import KDA  # noqa: E402


@pytest.fixture
def gpu():
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    linear._MATRIX.clear()
    yield
    linear._MATRIX.clear()
    mx.set_default_device(previous)


def _q(n: int, k: int, bits: int, group: int, seed: int) -> linear.Q:
    w = (0.02 * mx.random.normal((n, k), key=mx.random.key(seed))).astype(mx.bfloat16)
    return linear.Q(*mx.quantize(w, group_size=group, bits=bits), bits=bits, group=group)


def _x(rows: int, k: int, seed: int = 3) -> mx.array:
    return (0.5 * mx.random.normal((rows, k), key=mx.random.key(seed))).astype(mx.bfloat16)


def _lane() -> bool:
    return linear._backend() == "lane"


FORMATS = [(8, 64), (6, 64), (5, 64), (4, 64), (8, 128), (5, 128), (4, 128)]


def test_choice_reads_format_and_shape_only():
    for bits, group in FORMATS:
        for n, k in [(4096, 4096), (512, 4096), (8192, 128), (20480, 1536)]:
            for lane in (False, True):
                once = linear.choose(bits, group, n, k, lane=lane)
                assert all(linear.choose(bits, group, n, k, lane=lane) == once for _ in range(3))
    # the measured table: windows of 2 to 16 rows win on the matrix kernel, except where a row kernel shares reads
    assert linear.choose(5, 64, 24896, 4096, lane=False) and linear.choose(6, 64, 16384, 1536, lane=False)
    assert linear.choose(8, 64, 4096, 4096, lane=False) and linear.choose(8, 128, 4096, 4096, lane=False)
    assert linear.choose(4, 128, 4096, 4096, lane=False)
    assert not linear.choose(4, 64, 4096, 4096, lane=False)                  # qmv_rows: faster to four rows
    assert not linear.choose(8, 64, 8192, 128, lane=False)                   # qmv_quad_rows (KDA's f_b / g_b)
    assert linear.choose(5, 64, 8192, 128, lane=False)
    assert not linear.choose(8, 128, 4096, 4096, lane=True)                  # tensor units: groups of 64 measured


def test_same_shape_same_kernel(gpu):
    for bits, group in FORMATS:
        a, b = _q(1024, 2048, bits, group, 1), _q(1024, 2048, bits, group, 2)
        assert linear.on_matrix(a) == linear.on_matrix(b) == linear.choose(bits, group, 1024, 2048, lane=_lane())
        linear.project(_x(5, 2048), a, rows_exact=True)
        assert linear.on_matrix(a) == linear.on_matrix(b)                    # unchanged by the rows it ran


@pytest.mark.parametrize("forced", [True, False])
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("shape", [(1024, 2048), (2208, 4096)])
def test_every_row_count_takes_one_kernel(gpu, monkeypatch, forced, fmt, shape):
    monkeypatch.setattr(linear, "choose", lambda *a, **k: forced)
    bits, group = fmt
    q = _q(*shape, bits, group, 5)
    x = _x(16, shape[1])
    one = mx.concatenate([linear.project(x[r:r + 1], q, rows_exact=True) for r in range(16)])
    for rows in range(2, 17):
        assert bool(mx.array_equal(linear.project(x[:rows], q, rows_exact=True), one[:rows]).item()), rows


def test_split_linear_takes_each_parts_kernel_at_one_row(gpu):
    parts = [_q(512, 2048, 8, 64, 6), _q(256, 2048, 4, 64, 7), _q(1024, 2048, 5, 64, 8)]
    split = linear.Q.stack(parts)
    assert isinstance(split, linear.QSplit) and linear.on_matrix(split)
    x = _x(16, 2048)
    one = mx.concatenate([linear.project(x[r:r + 1], split, rows_exact=True) for r in range(16)])
    each = mx.concatenate([linear.project(x[:1], p, rows_exact=True) for p in parts], axis=-1)
    assert bool(mx.array_equal(one[:1], each).item())
    for rows in (2, 4, 8, 16):
        assert bool(mx.array_equal(linear.project(x[:rows], split, rows_exact=True), one[:rows]).item()), rows


@pytest.mark.parametrize("bits", [8, 5])
def test_kda_small_projections_one_kernel(gpu, bits):
    q = _q(8192, 128, bits, 64, 9)
    x = _x(16, 128)
    one = mx.concatenate([KDA._small(q, x[r:r + 1], True) for r in range(16)])
    for rows in (2, 3, 4, 8, 16):
        assert bool(mx.array_equal(KDA._small(q, x[:rows], True), one[:rows]).item()), rows


@pytest.mark.parametrize("fmt", FORMATS)
def test_matrix_kernel_against_fp32(gpu, monkeypatch, fmt):
    bits, group = fmt
    q = _q(2048, 4096, bits, group, 10)
    x = _x(16, 4096, 11).at[:, 3].multiply(20.0)                             # an outlier channel
    ref = x.astype(mx.float32) @ mx.dequantize(q.weight, q.scales, q.biases, group_size=group,
                                               bits=bits).astype(mx.float32).T
    rms = mx.sqrt(mx.mean(ref * ref))
    err = {}
    for forced in (True, False):                                             # the matrix kernel, MLX's row kernels
        monkeypatch.setattr(linear, "choose", lambda *a, **k: forced)
        linear._MATRIX.clear()
        y = linear.project(x, q, rows_exact=True).astype(mx.float32)
        err[forced] = (float((mx.mean(mx.abs(y - ref)) / rms).item()), float((mx.max(mx.abs(y - ref)) / rms).item()))
    assert err[True][0] <= 2.5e-3 and err[True][1] <= 4e-2
    assert err[True][0] <= 1.05 * err[False][0]                              # no less accurate than the row kernels


def test_native_g128_reads_the_same_values_as_two_groups_of_64(gpu):
    if _lane():
        pytest.skip("tensor units read a group of 128 as two of 64")
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm_bits

    q = _q(2048, 4096, 4, 128, 12)
    assert simd_qmm_bits.reads(q.weight, q.scales, q.biases, 128, 4) and simd_qmm_bits.check(q.weight, q.scales,
                                                                                               q.biases, 4, 128)
    x = _x(8, 4096)
    native = simd_qmm_bits.qmm(x, q.weight, q.scales, q.biases, 4, 128)
    s2, b2 = mx.repeat(q.scales, 2, axis=1), mx.repeat(q.biases, 2, axis=1)
    halves = simd_qmm_bits.qmm(x, q.weight, s2, b2, 4, 64)
    assert bool(mx.array_equal(native, halves).item())                        # one scale load, the same arithmetic
