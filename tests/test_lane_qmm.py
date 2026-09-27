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
    """The same bits, and finite (a lost dispatch can leave the same NaN on both sides)."""

    return bool(mx.all(mx.isfinite(a)).item()) and bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


@pytest.mark.parametrize("bits", [4, 3, 2])
@pytest.mark.parametrize("n,k", [(17408, 5120), (5120, 17408), (1024, 5120), (48, 5120), (5120, 6144)])
def test_rows_do_not_depend_on_row_count(n, k, bits):
    _needs_tensor_units()
    mx.random.seed(7)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
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


@pytest.mark.parametrize("bits", [4, 3, 2])
@pytest.mark.parametrize("n,k,m,tiled", [(4096, 5120, 8, False), (17408, 5120, 48, True), (5120, 17408, 48, True),
                                         (6144, 5120, 128, False)])
def test_accuracy_matches_mlx(bits, n, k, m, tiled):
    _needs_tensor_units()
    mx.random.seed(3)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
    ref = x.astype(mx.float32) @ mx.dequantize(q, s, b, group_size=64, bits=bits).astype(mx.float32).T
    ours = lane_qmm.lane_matmul(x, lane_qmm.tile_weight(q, bits=bits) if tiled else q, lane_qmm.pack_scales(s, b),
                                tiled=tiled).astype(mx.float32)
    theirs = mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=bits).astype(mx.float32)
    err_ours = mx.max(mx.abs(ours - ref)).item()
    err_theirs = mx.max(mx.abs(theirs - ref)).item()
    assert err_ours <= 2.5 * err_theirs + 1e-6


@pytest.mark.parametrize("bits", [3, 2])
@pytest.mark.parametrize("n,k,tiled", [(64, 128, False), (64, 128, True), (64, 1024, False), (64, 1024, True),
                                       (64, 5120, False), (64, 5120, True), (48, 5120, False)])
def test_lowbit_values_are_mlx_packing(n, k, tiled, bits):
    """One-hot rows read every 3- or 2-bit value back where MLX's dequantize puts it, over one K slice or several."""

    _needs_tensor_units()
    mx.random.seed(13)
    words = (n, k * bits // 32)                                               # every bit of every word, MLX's packing
    hi, lo = (mx.random.randint(0, 2**16, words).astype(mx.uint32) for _ in range(2))
    q = (hi << 16) | lo
    s, b = mx.ones((n, k // 64), dtype=mx.bfloat16), mx.zeros((n, k // 64), dtype=mx.bfloat16)
    expect = mx.dequantize(q, s, b, group_size=64, bits=bits)                 # (n, k): integers 0 .. 2**bits - 1
    w = lane_qmm.tile_weight(q, bits=bits) if tiled else q
    sbt = lane_qmm.pack_scales(s, b)
    eye = mx.eye(k, dtype=mx.bfloat16)
    y = mx.concatenate([lane_qmm.lane_matmul(eye[i:i + 128], w, sbt, tiled=tiled) for i in range(0, k, 128)])
    assert bool(mx.all(y.T == expect).item())


@pytest.mark.parametrize("bits", [4, 3, 2])
def test_repeated_calls_give_the_same_bits(bits):
    """The same call again and again gives the same bits (determinism across calls, tiled and not)."""

    _needs_tensor_units()
    mx.random.seed(17)
    n, k = 2048, 5120
    q, s, b = mx.quantize((mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16), group_size=64, bits=bits)
    qt, sbt = lane_qmm.tile_weight(q, bits=bits), lane_qmm.pack_scales(s, b)
    x = (mx.random.normal((128, k)) * 0.5).astype(mx.bfloat16)
    for m in (1, 16, 128):
        for w, tiled in ((q, False), (qt, True)):
            first = lane_qmm.lane_matmul(x[:m], w, sbt, tiled=tiled)
            again = [lane_qmm.lane_matmul(x[:m], w, sbt, tiled=tiled) for _ in range(20)]
            mx.eval(first, *again)
            assert all(_same(a, first) for a in again), f"{m} rows, tiled={tiled}: the bits changed between calls"


def test_other_widths_are_refused():
    x = mx.zeros((1, 128), dtype=mx.bfloat16)
    s = mx.ones((32, 2), dtype=mx.bfloat16)
    for bits in (5, 6, 8):
        w = mx.zeros((32, 128 * bits // 32), dtype=mx.uint32)
        assert not lane_qmm.supports(w, s, x, bits, 64, "affine")
        with pytest.raises(ValueError):
            lane_qmm.lane_matmul(x, w, lane_qmm.pack_scales(s, s))
    w3 = mx.zeros((64, 12), dtype=mx.uint32)
    assert lane_qmm.supports(w3, mx.ones((64, 2), dtype=mx.bfloat16), x, 3, 64, "affine")
    assert not lane_qmm.supports(w3, mx.ones((64, 2), dtype=mx.bfloat16), x, 3, 32, "affine")
    with pytest.raises(ValueError):                                # 3-bit tiles are 32 columns wide
        lane_qmm.lane_matmul(x, w3, lane_qmm.pack_scales(s, s), tiled=True, nt=64)
    w2, s2 = mx.zeros((64, 8), dtype=mx.uint32), mx.ones((64, 2), dtype=mx.bfloat16)
    assert lane_qmm.supports(w2, s2, x, 2, 64, "affine")


def test_split_depends_only_on_shape():
    assert lane_qmm.split_k(17408, 5120) == lane_qmm.split_k(17408, 5120)
    for n, k in [(17408, 5120), (5120, 17408), (48, 5120), (248320, 5120)]:
        sk = lane_qmm.split_k(n, k)
        assert 1 <= sk <= 8 and (k // 64) // sk >= 8


@pytest.mark.parametrize("bits", [4, 3, 2])
@pytest.mark.parametrize("n,k", [(17408, 5120), (5120, 17408), (1024, 5120), (5120, 6144), (4096, 5120)])
def test_tiled_weights_give_the_same_bits(n, k, bits):
    _needs_tensor_units()
    mx.random.seed(11)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    sbt = lane_qmm.pack_scales(s, b)
    qt = lane_qmm.tile_weight(q, bits=bits)
    assert bool(mx.all(lane_qmm.untile_weight(qt, bits=bits) == q).item())
    x = (mx.random.normal((128, k)) * 0.5).astype(mx.bfloat16)
    full = lane_qmm.lane_matmul(x, qt, sbt, tiled=True)
    for m in (1, 7, 11, 12, 13, 15, 16, 17, 32, 33, 64, 128):
        plain = lane_qmm.lane_matmul(x[:m], q, sbt)
        tiled = lane_qmm.lane_matmul(x[:m], qt, sbt, tiled=True)
        assert _same(plain, tiled), f"{m} rows: tiled weights changed the bits"
        assert _same(tiled, full[:m]), f"{m} rows: the row count changed the bits"


@pytest.mark.parametrize("bits", [4, 3, 2])
def test_install_tiles_in_place_and_uninstall_restores(bits):
    _needs_tensor_units()
    import mlx.nn as nn

    mx.random.seed(5)
    model = nn.Sequential(nn.Linear(512, 256, bias=False), nn.Linear(256, 48, bias=False))
    model.set_dtype(mx.bfloat16)                                # bf16 scales, as the real checkpoint has
    nn.quantize(model, group_size=64, bits=bits)
    mx.eval(model.parameters())
    wide, big = model.layers[0], model.layers[1]
    q = wide.weight
    x = (mx.random.normal((200, 512)) * 0.5).astype(mx.bfloat16)
    mlx_wide = mx.quantized_matmul(x, q, wide.scales, wide.biases, transpose=True, group_size=64, bits=bits)
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


@pytest.mark.parametrize("bits", [4, 3, 2])
def test_install_wide_is_the_plain_kernels_bits(bits):
    """wide=True tiles 4-bit weights 64 wide and other widths 32 wide, with the plain kernel's bits."""

    _needs_tensor_units()
    import mlx.nn as nn

    mx.random.seed(19)
    model = nn.Sequential(nn.Linear(1024, 512, bias=False), nn.Linear(512, 96, bias=False))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=bits)
    mx.eval(model.parameters())
    plain = {id(m): (m.weight, lane_qmm.pack_scales(m.scales, m.biases)) for m in model.layers}
    xs = {id(m): (mx.random.normal((48, int(m.weight.shape[1]) * 32 // bits)) * 0.5).astype(mx.bfloat16)
          for m in model.layers}
    want = {i: lane_qmm.lane_matmul(xs[i], *plain[i]) for i in plain}
    mx.eval(want)
    try:
        lane_qmm.install(model, rows=lane_qmm.MAX_ROWS, wide=True)
        assert model.layers[0]._lane_nt == (64 if bits == 4 else 32)
        assert model.layers[1]._lane_nt == 32                    # 96 rows: 32-wide tiles either way
        assert lane_qmm.warm(model) == 2
        for m in model.layers:
            for rows in (1, 16, 48):
                assert _same(m(xs[id(m)][:rows]), want[id(m)][:rows]), f"{rows} rows changed with the wide layout"
    finally:
        lane_qmm.uninstall()


def test_install_leaves_other_widths_to_mlx_and_reports_them():
    """A mixed checkpoint: 6-bit and unquantized layers keep MLX's layout and kernels; uncovered() names them."""

    _needs_tensor_units()
    import mlx.nn as nn

    mx.random.seed(23)
    model = nn.Sequential(*(nn.Linear(256, 128, bias=False) for _ in range(4)))
    model.set_dtype(mx.bfloat16)
    for i, bits in enumerate((4, 3, 6)):
        nn.quantize(model, group_size=64, bits=bits, class_predicate=lambda path, _m, i=i: path == f"layers.{i}")
    mx.eval(model.parameters())
    six = model.layers[2]
    q6 = six.weight
    x = (mx.random.normal((8, 256)) * 0.5).astype(mx.bfloat16)
    want6 = mx.quantized_matmul(x, q6, six.scales, six.biases, transpose=True, group_size=64, bits=6)
    assert lane_qmm.uncovered(model) == {"6-bit g64": 1, "unquantized": 1}
    try:
        lane_qmm.install(model, rows=lane_qmm.MAX_ROWS)
        assert getattr(model.layers[0], "_lane_tiled", False) and getattr(model.layers[1], "_lane_tiled", False)
        assert getattr(six, "_lane_sbt", None) is None and not getattr(six, "_lane_tiled", False)
        assert six.weight is q6
        assert _same(six(x), want6)
    finally:
        lane_qmm.uninstall()


@pytest.mark.parametrize("n,k,nt", [(2560, 6144, 64), (16480, 2560, 32), (13952, 2560, 64), (79592, 2560, 0)])
def test_groups_of_32_rows_do_not_depend_on_row_count(n, k, nt):
    _needs_tensor_units()
    mx.random.seed(17)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=32, bits=4)
    sbt = lane_qmm.pack_scales(s, b)
    qt = lane_qmm.tile_weight(q, nt, group=32, bits=4) if nt else q
    if nt:
        assert bool(mx.all(lane_qmm.untile_weight(qt, nt, group=32, bits=4) == q).item())
    x = (mx.random.normal((128, k)) * 0.5).astype(mx.bfloat16)
    full = lane_qmm.lane_matmul(x, qt, sbt, tiled=bool(nt), nt=nt or lane_qmm.NT, group=32)
    for m in (1, 2, 5, 16, 17, 24, 32, 33, 64, 100):
        part = lane_qmm.lane_matmul(x[:m], qt, sbt, tiled=bool(nt), nt=nt or lane_qmm.NT, group=32)
        assert _same(part, full[:m]), f"{m} rows: the row count changed the bits"
    for r in (0, 9, 63, 127):
        alone = lane_qmm.lane_matmul(x[r:r + 1], qt, sbt, tiled=bool(nt), nt=nt or lane_qmm.NT, group=32)
        assert _same(alone, full[r:r + 1]), f"row {r} alone differs"


def test_groups_of_32_as_accurate_as_mlx():
    _needs_tensor_units()
    mx.random.seed(19)
    n, k = 4096, 2560
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=32, bits=4)
    x = (mx.random.normal((8, k)) * 0.5).astype(mx.bfloat16)
    ref = x.astype(mx.float32) @ mx.dequantize(q, s, b, group_size=32, bits=4).astype(mx.float32).T
    tiled = lane_qmm.tile_weight(q, 64, group=32, bits=4)
    ours = lane_qmm.lane_matmul(x, tiled, lane_qmm.pack_scales(s, b), tiled=True, nt=64, group=32).astype(mx.float32)
    theirs = mx.quantized_matmul(x, q, s, b, transpose=True, group_size=32, bits=4).astype(mx.float32)
    assert mx.max(mx.abs(ours - ref)).item() <= 2.5 * mx.max(mx.abs(theirs - ref)).item() + 1e-6
