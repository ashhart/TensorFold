"""The lane matmul: every row identical whatever the row count, and as accurate as MLX's kernels."""

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.dense.v1 import lane_qmm  # noqa: E402


def _needs_tensor_units():
    try:
        w = mx.zeros((32, 8), dtype=mx.uint32)
        s = mx.ones((32, 1), dtype=mx.bfloat16)
        y = lane_qmm.lane_matmul(mx.ones((1, 64), dtype=mx.bfloat16), w, lane_qmm.pack_scales(s, s))
        mx.eval(y)
    except Exception as exc:  # noqa: BLE001 - no Metal 4 tensor ops on this machine
        pytest.skip(f"tensor-unit kernels unavailable: {str(exc).splitlines()[0][:80]}")


def _same(a, b):
    return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


@pytest.mark.parametrize("n,k", [(17408, 5120), (5120, 17408), (1024, 5120), (48, 5120), (5120, 6144)])
def test_rows_do_not_depend_on_row_count(n, k):
    _needs_tensor_units()
    mx.random.seed(7)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    sbt = lane_qmm.pack_scales(s, b)
    x = (mx.random.normal((128, k)) * 0.5).astype(mx.bfloat16)
    full = lane_qmm.lane_matmul(x, q, sbt)
    mx.eval(full)
    for m in (1, 2, 5, 9, 16, 17, 31, 33, 47, 64, 65, 100, 128):
        part = lane_qmm.lane_matmul(x[:m], q, sbt)
        assert _same(part, full[:m]), f"rows 0..{m - 1} changed with the row count"
    # a row computed alone equals the same row inside a window starting elsewhere
    alone = lane_qmm.lane_matmul(x[20:21], q, sbt)
    assert _same(alone, full[20:21])


def test_accuracy_matches_mlx():
    _needs_tensor_units()
    mx.random.seed(3)
    n, k = 4096, 5120
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    x = (mx.random.normal((8, k)) * 0.5).astype(mx.bfloat16)
    ref = x.astype(mx.float32) @ mx.dequantize(q, s, b, group_size=64, bits=4).astype(mx.float32).T
    ours = lane_qmm.lane_matmul(x, q, lane_qmm.pack_scales(s, b)).astype(mx.float32)
    theirs = mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=4).astype(mx.float32)
    err_ours = mx.max(mx.abs(ours - ref)).item()
    err_theirs = mx.max(mx.abs(theirs - ref)).item()
    assert err_ours <= 2.5 * err_theirs + 1e-6


def test_split_depends_only_on_shape():
    assert lane_qmm.split_k(17408, 5120) == lane_qmm.split_k(17408, 5120)
    for n, k in [(17408, 5120), (5120, 17408), (48, 5120), (248320, 5120)]:
        sk = lane_qmm.split_k(n, k)
        assert 1 <= sk <= 8 and (k // 64) // sk >= 8


@pytest.mark.parametrize("n,k", [(17408, 5120), (5120, 17408), (1024, 5120), (5120, 6144), (4096, 5120)])
def test_tiled_weights_give_the_same_bits(n, k):
    _needs_tensor_units()
    mx.random.seed(11)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    sbt = lane_qmm.pack_scales(s, b)
    qt = lane_qmm.tile_weight(q)
    assert bool(mx.all(lane_qmm.untile_weight(qt) == q).item())
    x = (mx.random.normal((128, k)) * 0.5).astype(mx.bfloat16)
    full = lane_qmm.lane_matmul(x, qt, sbt, tiled=True)
    for m in (1, 7, 11, 12, 13, 15, 16, 17, 32, 33, 64, 128):
        plain = lane_qmm.lane_matmul(x[:m], q, sbt)
        tiled = lane_qmm.lane_matmul(x[:m], qt, sbt, tiled=True)
        assert _same(plain, tiled), f"{m} rows: tiled weights changed the bits"
        assert _same(tiled, full[:m]), f"{m} rows: the row count changed the bits"


def test_install_tiles_in_place_and_uninstall_restores():
    _needs_tensor_units()
    import mlx.nn as nn

    mx.random.seed(5)
    model = nn.Sequential(nn.Linear(512, 256, bias=False), nn.Linear(256, 48, bias=False))
    model.set_dtype(mx.bfloat16)                                # bf16 scales, as the real checkpoint has
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    wide, big = model.layers[0], model.layers[1]
    q = wide.weight
    x = (mx.random.normal((200, 512)) * 0.5).astype(mx.bfloat16)
    mlx_wide = mx.quantized_matmul(x, q, wide.scales, wide.biases, transpose=True, group_size=64, bits=4)
    plain = lane_qmm.lane_matmul(x[:9], q, lane_qmm.pack_scales(wide.scales, wide.biases))
    mx.eval(mlx_wide, plain)
    try:
        lane_qmm.install(model, rows=lane_qmm.MAX_ROWS)
        assert getattr(wide, "_lane_tiled", False) and not getattr(big, "_lane_tiled", False)   # 48 rows stay as MLX packs them
        assert wide.weight.shape == q.shape and not bool(mx.all(wide.weight == q).item())
        assert _same(wide(x[:9]), plain)                        # lane kernel on the tiled layout
        assert _same(wide(x), mlx_wide)                         # 200 rows: MLX's kernel on the layout rebuilt
    finally:
        lane_qmm.uninstall()
    assert bool(mx.all(wide.weight == q).item()) and not getattr(wide, "_lane_tiled", True)
