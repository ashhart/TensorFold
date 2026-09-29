"""DeepSeek-V4-Flash's Metal row kernels give every window row MLX's one-row bits (fp32 qmv, mxfp4 experts)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():
    pytest.skip("Metal kernels", allow_module_level=True)

from tensorfold.families.deepseek_v4.moe import FP4, swiglu  # noqa: E402
from tensorfold.families.glm5_next.linear import Q  # noqa: E402
from tensorfold.kernels.deepseek.v4 import rows as RK  # noqa: E402
from tensorfold.kernels.glm.flash.v1 import kernels as GK  # noqa: E402


def affine(outs: int, ins: int) -> Q:
    w = (0.05 * mx.random.normal((outs, ins))).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    return Q(q, s, b, bits=4, group=64)


def fp4(experts: int, outs: int, ins: int) -> FP4:
    w = (0.05 * mx.random.normal((experts, outs, ins))).astype(mx.bfloat16)
    return FP4(*mx.quantize(w, group_size=32, bits=4, mode="mxfp4"))


@pytest.mark.parametrize("rows", [2, 7, 16])
def test_fp32_rows_equal_one_row_calls(rows):
    mx.random.seed(rows)
    q = affine(2048, 4096)
    x = mx.random.normal((rows, 4096)).astype(mx.bfloat16).astype(mx.float32)
    joint = RK.qmv_rows_f32(x, q)
    for r in range(rows):
        one = q(x[r:r + 1])
        assert one.dtype == mx.float32
        assert mx.array_equal(joint[r:r + 1], one).item(), f"row {r}"


@pytest.mark.parametrize("rows", [2, 9, 16])
def test_fp4_expert_rows_equal_one_row_gathers(rows):
    mx.random.seed(100 + rows)
    experts, top, dims, inter = 8, 6, 4096, 512
    gate, down = fp4(experts, inter, dims), fp4(experts, dims, inter)
    x = mx.random.normal((rows, dims)).astype(mx.bfloat16)
    idx = mx.sort(mx.stack([mx.random.permutation(experts)[:top] for _ in range(rows)]).astype(mx.int32), axis=-1)
    group = GK.expert_group(idx, experts)
    g = RK.expert_rows_fp4(x, idx, group, gate, per_pick=False)
    act = swiglu(g, g, 10.0)
    y = RK.expert_rows_fp4(act, idx, group, down, per_pick=True)
    for r in range(rows):
        h = mx.expand_dims(x[r:r + 1], (-2, -3))
        one_g = gate(h, idx[r:r + 1], False).squeeze(-2)
        assert mx.array_equal(g[r:r + 1], one_g).item(), f"gate row {r}"
        one_y = down(swiglu(one_g, one_g, 10.0)[:, :, None, :], idx[r:r + 1], False).squeeze(-2)
        assert mx.array_equal(y[r:r + 1], one_y).item(), f"down row {r}"


@pytest.mark.parametrize("ratio,dims,first,count", [(4, 512, 0, 3), (4, 512, 5, 4), (4, 128, 0, 1), (4, 128, 9, 2),
                                                     (128, 512, 0, 1), (128, 512, 3, 2)])
def test_pool_kernel_equals_the_ops_path(ratio, dims, first, count):
    from tensorfold.families.deepseek_v4 import compressor as C
    from tensorfold.families.deepseek_v4.caches import LayerCache

    mx.random.seed(ratio + dims + first)
    width = (2 if ratio == 4 else 1) * dims
    inv = 1.0 / (10000 ** (mx.arange(0, 64, 2, dtype=mx.float32) / 64))
    comp = C.Compressor(affine(width, 256), affine(width, 256), mx.random.normal((ratio, width)),
                        (1 + 0.1 * mx.random.normal((dims,))).astype(mx.bfloat16), ratio, 1e-6, inv)
    comp.col = 64                                                   # past another compressor's columns
    total = 64 + 2 * width
    lo = max(0, (first - int(ratio == 4)) * ratio)
    span = mx.random.normal((((first + count) * ratio) - lo, total))
    cache = LayerCache(ratio)
    cache.write_proj(span, lo)
    for source in (None, (span, lo)):
        if source is None and (first + count) * ratio - lo > cache.proj_ring():
            continue
        ops = comp._pool_ops(cache, first, first + count, source)
        data, base = source if source is not None else (cache.proj, 0)
        kern = C.PK.pool_rows(data, comp.ape, comp.norm, inv, first=first, count=count, ratio=ratio,
                              overlap=ratio == 4, col=comp.col, base=base,
                              ring=0 if source is not None else int(cache.proj.shape[0]))
        assert mx.array_equal(kern, ops).item(), (source is None, mx.abs(kern.astype(mx.float32) - ops).max())


@pytest.mark.parametrize("rows", [1, 5])
def test_bf16_inputs_read_as_fp32(rows):
    from types import SimpleNamespace

    from tensorfold.kernels.glm.flash.v1 import moe as MK

    mx.random.seed(10 + rows)
    q = affine(1024, 4096)
    x = mx.random.normal((rows, 4096)).astype(mx.bfloat16)
    wide = RK.qmv_rows_f32(x.astype(mx.float32), q)
    assert mx.array_equal(RK.qmv_rows_f32(x, q), wide).item()
    assert mx.array_equal(wide[:1], q(x[:1].astype(mx.float32))).item()
    router = (0.05 * mx.random.normal((256, 4096))).astype(mx.bfloat16)
    moe = SimpleNamespace(router=mx.contiguous(router.astype(mx.float32).T), router_packed=MK.pack_router(router))
    assert mx.array_equal(MK.router_rows(x, moe), MK.router_rows(x.astype(mx.float32), moe)).item()


@pytest.mark.parametrize("rows,base", [(1, 0), (4, 0), (6, 120)])
def test_attention_rotates_back_as_norm_rope(rows, base):
    from tensorfold.kernels.deepseek.v4 import attention as ATT
    from tensorfold.kernels.deepseek.v4 import rope as NR

    mx.random.seed(20 + rows)
    q = mx.random.normal((rows, 64, 512)).astype(mx.bfloat16)
    pool = mx.random.normal((40, 512)).astype(mx.bfloat16)
    ring = mx.random.normal((144, 512)).astype(mx.bfloat16)
    ends = [130 + i for i in range(rows)]
    windows = [(e - 127, e) for e in ends]
    sink = mx.random.normal((64,))
    inv = 1.0 / (160000 ** (mx.arange(0, 64, 2, dtype=mx.float32) / 64))
    counts = [30 + i for i in range(rows)]
    plain = ATT.attend_rows(q, pool, None, counts, ring, windows, sink, 0.04)
    want = NR.norm_rope(plain, mx.array(ends) + base, inv, norm=False, inverse=True)
    got = ATT.attend_rows(q, pool, None, counts, ring, windows, sink, 0.04, inv, base)
    assert mx.array_equal(got, want).item()


@pytest.mark.parametrize("splits", [0, 4])
@pytest.mark.parametrize("rows,chosen", [(1, False), (5, False), (5, True)])
def test_attention_rows_are_one_row_calls_and_near_fp32(splits, rows, chosen):
    from tensorfold.kernels.deepseek.v4 import attention as ATT

    mx.random.seed(30 + rows)
    ring = mx.random.normal((144, 512)).astype(mx.bfloat16)
    pool = mx.random.normal((300, 512)).astype(mx.bfloat16)
    sink = mx.random.normal((64,))
    q = mx.random.normal((rows, 64, 512)).astype(mx.bfloat16)
    windows = [(200 + i - 127, 200 + i) for i in range(rows)]
    counts = [64 if chosen else 40 + 3 * i for i in range(rows)]
    picks = [mx.sort(mx.random.permutation(300)[:64]).astype(mx.int32) for _ in range(rows)]
    pidx = mx.stack(picks) if chosen else None
    previous = ATT.SPLITS
    ATT.SPLITS = splits
    try:
        joint = ATT.attend_rows(q, pool, pidx, counts, ring, windows, sink, 512 ** -0.5)
        for i in range(rows):
            one = ATT.attend_rows(q[i:i + 1], pool, None if pidx is None else pidx[i:i + 1], counts[i:i + 1], ring,
                                  windows[i:i + 1], sink, 512 ** -0.5)
            assert mx.array_equal(joint[i:i + 1], one).item(), i
            keys = mx.concatenate([pool[pidx[i]] if chosen else pool[:counts[i]],
                                   ring[mx.arange(windows[i][0], windows[i][1] + 1) % 144]]).astype(mx.float32)
            s = (q[i].astype(mx.float32) @ keys.T) * 512 ** -0.5
            m = s.max(axis=-1, keepdims=True)
            e = mx.exp(s - m)
            ref = (e @ keys) / (e.sum(-1, keepdims=True) + mx.exp(sink[:, None] - m))
            assert mx.abs(joint[i].astype(mx.float32) - ref).max().item() < 2e-2 * mx.abs(ref).max().item()
    finally:
        ATT.SPLITS = previous
