"""DeepSeek-V4.1-Flash's decode windows: every row of a window keeps its one-row call's bits on the row kernels."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.deepseek_v41.model import top_ids  # noqa: E402
from tensorfold.families.deepseek_v41.quant import Linear  # noqa: E402
from tensorfold.kernels.deepseek.v41 import rows as KV  # noqa: E402

FORMATS = [("mxfp8", 8), ("mxfp4", 4)]


@pytest.fixture
def gpu():
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def quantized(mode: str, bits: int, n: int = 1024, k: int = 4096):
    w = mx.random.normal((n, k), key=mx.random.key(1)) * 0.05
    return mx.quantize(w, group_size=32, bits=bits, mode=mode)


def window(rows: int, k: int, dtype=mx.bfloat16) -> mx.array:
    x = mx.random.normal((rows, k), key=mx.random.key(2))
    return (x * mx.random.uniform(0.1, 10, (rows, 1), key=mx.random.key(3))).astype(dtype)


@pytest.mark.parametrize("mode,bits", FORMATS)
@pytest.mark.parametrize("rows", [2, 5, 16])
def test_fpqmv_rows_give_one_row_bits(gpu, mode, bits, rows):
    """Each row of the shared-read kernel equals MLX's one-row quantized matmul."""

    w, s = quantized(mode, bits)
    x = window(rows, 4096)
    assert KV.fpqmv_rows_fits(x, bits, 4096, 1024)
    joint = KV.fpqmv_rows(x, w, s, bits, 4096, 1024)
    for r in range(rows):
        one = mx.quantized_matmul(x[r:r + 1], w, s, None, transpose=True, group_size=32, bits=bits, mode=mode)
        assert mx.array_equal(joint[r:r + 1], one).item(), f"row {r}"


@pytest.mark.parametrize("mode,bits", FORMATS)
def test_quantized_linear_rows_inside_a_decode_window(gpu, mode, bits):
    """A quantized projection inside ``decode_rows`` gives every row its one-row bits."""

    w, s = quantized(mode, bits)
    linear = Linear(w, s, bits=bits, mode=mode, fp8_input=False)
    x = window(16, 4096)
    with KV.decode_rows():
        joint = linear(x)
    for r in range(16):
        assert mx.array_equal(joint[r:r + 1], linear(x[r:r + 1])).item(), f"row {r}"


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("groups", [1, 4])
def test_unquantized_matmul_rows_inside_a_decode_window(gpu, dtype, groups):
    """An unquantized (grouped) matmul inside ``decode_rows`` gives every row its one-row bits."""

    w = (mx.random.normal((1024, 1024), key=mx.random.key(4)) * 0.05).astype(dtype)
    x = window(6, 1024 * groups, dtype)
    with KV.decode_rows():
        joint = KV.matmul(x, w, dtype, groups=groups)
    for r in range(6):
        one = KV.matmul(x[r:r + 1], w, dtype, groups=groups)
        assert mx.array_equal(joint[r:r + 1], one).item(), f"row {r}"


def test_decode_rows_nests_and_restores():
    """``decode_rows`` turns row mode on for its block and restores the previous mode after."""

    assert not KV.rows_mode()
    with KV.decode_rows():
        assert KV.rows_mode()
        with KV.decode_rows():
            assert KV.rows_mode()
        assert KV.rows_mode()
    assert not KV.rows_mode()


def test_top_ids_fixed_keeps_count_columns():
    """``fixed`` pads a narrow score row with -1 to ``count`` columns; unfixed keeps the narrow width."""

    scores = mx.array([[0.5, float("-inf"), 2.0]])
    assert top_ids(scores, 5, True).tolist() == [[-1, -1, -1, 0, 2]]
    assert top_ids(scores, 5).tolist() == [[-1, 0, 2]]
    assert top_ids(mx.zeros((2, 0)), 3, True).tolist() == [[-1, -1, -1], [-1, -1, -1]]
