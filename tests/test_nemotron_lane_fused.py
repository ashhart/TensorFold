"""Nemotron's M5 decode launches that also write the next lane matmul's input sums keep the unfused bits exactly."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.kernels.nemotron.lightning.v1 import kernels as K  # noqa: E402

pytestmark = pytest.mark.skipif(not K.tensor_units(), reason="lane matmuls run on M5 tensor units")


def _xsum(x):
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm

    rows, k = int(x.shape[0]), int(x.shape[1])
    mp = 16 * ((rows + 15) // 16)
    return lane_qmm._kernel("xsum")(inputs=[x, lane_qmm._mdims(rows, mp)], template=[("K", k), ("GS", 64)],
                                    grid=(k // 64, mp, 1), threadgroup=(min(k // 64, 256), 1, 1),
                                    output_shapes=[(k // 64, mp)], output_dtypes=[mx.float32])[0]


def _same(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("rows", [1, 3, 17, 40])
def test_up_relu2_writes_the_unfused_activation_and_its_sums(rows):
    from tensorfold.kernels.nemotron.lightning.v1 import lane_fused
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm

    holder = nn.Module()
    holder.up = nn.QuantizedLinear(256, 192, bias=False, group_size=64, bits=4)
    holder.up.scales = holder.up.scales.astype(mx.bfloat16)          # checkpoints store bf16 scales
    holder.up.biases = holder.up.biases.astype(mx.bfloat16)
    was = lane_qmm.enabled
    lane_qmm.install(holder, tile=True, wide=True)
    try:
        assert lane_fused.takes_relu2(holder.up)
        x = (mx.random.normal((rows, 256), key=mx.random.key(rows)) * 0.7).astype(mx.bfloat16)
        act, sums = lane_fused.up_relu2(x, _xsum(x), holder.up)
        want = mx.square(mx.maximum(holder.up(x), 0))
        assert _same(act, want) and _same(sums[:, :rows], _xsum(want)[:, :rows])
    finally:
        if not was:
            lane_qmm.uninstall()


@pytest.mark.parametrize("rows", [1, 3, 17, 40])
def test_group_norm_sums_match_group_norm_then_xsum(rows):
    from tensorfold.kernels.nemotron.lightning.v1 import lane_fused

    y = (mx.random.normal((rows, 1024), key=mx.random.key(50 + rows)) * 0.7).astype(mx.bfloat16)
    weight = (mx.random.normal((1024,), key=mx.random.key(7)) * 0.1 + 1).astype(mx.bfloat16)
    eps = mx.array([1e-5], dtype=mx.float32)
    want = K.group_norm(y, weight, eps, 512)
    got, sums = lane_fused.group_norm_sums(y, weight, eps, 512)
    assert _same(got, want) and _same(sums[:, :rows], _xsum(want)[:, :rows])
    assert bool(mx.all(sums[:, rows:] == 0).item())
