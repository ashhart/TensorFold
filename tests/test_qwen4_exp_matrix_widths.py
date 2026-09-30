"""TF_FLASH_DENSE=matrix: a window matches its one-row steps, and stays near MLX's matmul."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.families.qwen4_exp import decode  # noqa: E402


def _linear(n, k, bits, group, seed):
    mx.random.seed(seed)
    holder = nn.Sequential(nn.Linear(k, n, bias=False))
    holder.set_dtype(mx.bfloat16)
    nn.quantize(holder, group_size=group, bits=bits)
    mx.eval(holder.parameters())
    return holder.layers[0]


@pytest.mark.parametrize("bits, group", [(5, 64), (6, 64), (8, 64), (5, 128), (8, 128), (4, 64), (4, 32)])
def test_matrix_rows_equal_one_row_steps(bits, group):
    linear = _linear(512, 1024, bits, group, seed=bits * 1000 + group)
    x = (mx.random.normal((16, 1024)) * 0.5).astype(mx.bfloat16)
    try:
        window = decode._matrix_project(x, linear)
        steps = mx.concatenate([decode._matrix_project(x[r:r + 1], linear) for r in range(16)])
        mx.eval(window, steps)
    except RuntimeError as exc:                      # no Metal matrix kernels here
        pytest.skip(str(exc).splitlines()[0][:80])
    assert bool(mx.array_equal(window, steps).item())
    ref = linear(x).astype(mx.float32)
    err = float(mx.max(mx.abs(window.astype(mx.float32) - ref)).item())
    assert err <= 0.02 * float(mx.max(mx.abs(ref)).item())
