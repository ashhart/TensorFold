"""GLM-5.3-Flash's decode-row kernels: each row of a window keeps its one-row call's bits (fallbacks: the calls)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.glm.flash.v1 import kernels as K  # noqa: E402
from tensorfold.kernels.glm.flash.v1 import widths as W  # noqa: E402
from tensorfold.families.glm5_next import config, linear, mla, mlp, weights  # noqa: E402


@pytest.fixture
def gpu():
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _same(a: mx.array, b: mx.array) -> bool:
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def _experts(e: int, n: int, k: int, seed: int) -> linear.Q:
    mx.random.seed(seed)
    w = (0.02 * mx.random.normal((e, n, k))).astype(mx.bfloat16)
    return linear.Q(*mx.quantize(w, group_size=64, bits=4), bits=4, group=64)


def _picks(rows: int, top: int, experts: int, seed: int) -> mx.array:
    """Distinct experts per row, with some experts shared between rows (as consecutive tokens do)."""

    mx.random.seed(seed)
    idx = mx.stack([mx.random.permutation(experts)[:top] for _ in range(rows)]).astype(mx.uint32)
    for r in range(1, rows):
        if int(idx[0, 0].item()) not in [int(v) for v in idx[r].tolist()]:
            idx[r, r % top] = idx[0, 0]
    return idx


def _one_row(x: mx.array, idx: mx.array, q: linear.Q) -> mx.array:
    """What model.MoE.experts runs for one row: mx.gather_qmm on [1, k, 1, K] -> [1, k, N]."""

    return mx.gather_qmm(x[None], q.weight, q.scales, q.biases, rhs_indices=idx, transpose=True, group_size=64,
                         bits=4).squeeze(-2)


def test_row_kernel_rule(monkeypatch):
    assert config.ENABLED == frozenset(config.ROW_KERNELS)
    monkeypatch.setattr(config, "ENABLED", frozenset({"router"}))
    assert config.row_kernel("router", 2, True)
    assert not config.row_kernel("router", 1, True) and not config.row_kernel("router", 2, False)
    assert not config.row_kernel("experts", 2, True)


def test_fallbacks_are_the_one_row_calls(cpu):
    """Without Metal the row functions are the one-row MLX calls, row after row."""

    assert K.expert_group(mx.zeros((2, 4), dtype=mx.uint32), 16) is None
    gate = _experts(16, 64, 512, seed=1)
    idx = _picks(3, 4, 16, seed=2)
    x = mx.random.normal((3, 512)).astype(mx.bfloat16)
    got = K.expert_qmv(x, idx, None, gate, per_pick=False)
    want = mx.concatenate([_one_row(x[r:r + 1][:, None, :], idx[r:r + 1], gate) for r in range(3)])
    assert _same(got, want)
    m = mx.random.normal((96, 40))
    xs = mx.random.normal((5, 96))
    assert _same(K.matmul_rows(xs, m, transposed=True), mx.concatenate([xs[r:r + 1] @ m for r in range(5)]))
    assert _same(K.matmul_rows(xs, m.T, transposed=False), mx.concatenate([xs[r:r + 1] @ m for r in range(5)]))


def test_expert_group_lists_each_experts_picks(gpu):
    idx = mx.array([[5, 1, 9], [9, 2, 5]], dtype=mx.uint32)
    uids, umem, count = K.expert_group(idx, 40)
    n = int(count[0].item())                                     # the count is padded (kernels.inputs)
    assert n == 4
    assert uids[:n].tolist() == [1, 2, 5, 9]
    members = [[m for m in row if m >= 0] for row in umem[:n].tolist()]
    assert members == [[1], [4], [0, 5], [2, 3]]                  # row * 3 + slot, rows in order


@pytest.mark.parametrize("dims", [(512, 1024), (2048, 512)])
def test_expert_qmv_gives_each_pick_its_one_row_bits(gpu, dims):
    n, k = dims
    experts, top = 24, 4
    w = _experts(experts, n, k, seed=3)
    for rows in (2, 3, 5, 8, 16):
        idx = _picks(rows, top, experts, seed=rows)
        group = K.expert_group(idx, experts)
        x = mx.random.normal((rows, k)).astype(mx.bfloat16)
        shared = K.expert_qmv(x, idx, group, w, per_pick=False)
        want = mx.concatenate([_one_row(x[r:r + 1][:, None, :], idx[r:r + 1], w) for r in range(rows)])
        assert _same(shared, want), rows
        act = mx.random.normal((rows, top, k)).astype(mx.bfloat16)
        own = K.expert_qmv(act, idx, group, w, per_pick=True)
        want = mx.concatenate([_one_row(act[r][:, None, :], idx[r:r + 1], w) for r in range(rows)])
        assert _same(own, want), rows


@pytest.mark.parametrize("case", [
    ("router", True, "float32", 4096, 288),        # MoE router logits, x @ W [D, E]
    ("hc mix", False, "float32", 16384, 24),       # hyper-connection mix, z @ fn.T
    ("indexer gate", True, "bfloat16", 4096, 128),  # x @ igate [D, 128]
])
def test_matmul_rows_gives_mlx_one_row_bits(gpu, case):
    _, transposed, dtype, k, n = case
    dt = getattr(mx, dtype)
    mx.random.seed(4)
    m = (0.05 * mx.random.normal((k, n) if transposed else (n, k))).astype(dt)
    x = mx.random.normal((16, k)).astype(dt)
    mat = m if transposed else m.T
    one = mx.concatenate([x[r:r + 1] @ mat for r in range(16)])
    for rows in range(1, 17):
        assert _same(K.matmul_rows(x[:rows], m, transposed=transposed), one[:rows]), rows


def _moe(dims: int = 512, width: int = 512, experts: int = 24, top: int = 4) -> mlp.MoE:
    cfg = config.Config.from_dict({
        "hidden_size": dims, "num_hidden_layers": 1, "layer_types": ["linear_attention"], "mlp_layer_types": ["sparse"],
        "vocab_size": 16, "rms_norm_eps": 1e-5, "num_attention_heads": 1, "q_lora_rank": 64, "kv_lora_rank": 64,
        "qk_nope_head_dim": 64, "v_head_dim": 64, "index_n_heads": 1, "index_head_dim": 64, "index_topk": 16,
        "n_routed_experts": experts, "num_experts_per_tok": top, "moe_intermediate_size": width,
        "intermediate_size": width, "n_shared_experts": 1, "routed_scaling_factor": 2.5, "norm_topk_prob": True,
        "swiglu_limit": 10.0, "eos_token_id": [0]})
    mx.random.seed(5)
    gate, up, down = _experts(experts, width, dims, 6), _experts(experts, width, dims, 7), _experts(experts, dims,
                                                                                                    width, 8)

    def lin(n: int, k: int) -> linear.Q:
        w = (0.05 * mx.random.normal((n, k))).astype(mx.bfloat16)
        return linear.Q(*mx.quantize(w, group_size=64, bits=4), bits=4, group=64)

    shared = mlp.DenseMLP(lin(width, dims), lin(width, dims), lin(dims, width), 10.0)
    router = (0.3 * mx.random.normal((experts, dims))).astype(mx.float32)
    bias = (0.1 * mx.random.normal((experts,))).astype(mx.float32)
    return mlp.MoE(router, bias, gate, up, down, shared, cfg)


@pytest.mark.parametrize("enabled", [("experts", "router"), ("router",), ()])
def test_moe_window_rows_are_one_row_steps(gpu, monkeypatch, enabled):
    """The MoE block on a window (row kernels, router only, or row by row) gives every row its one-row bits."""

    monkeypatch.setattr(config, "ENABLED", frozenset(enabled))
    moe = _moe()
    x = (0.5 * mx.random.normal((16, 512))).astype(mx.bfloat16)
    one = mx.concatenate([moe(x[r:r + 1], True) for r in range(16)])
    for rows in (2, 3, 4, 8, 16):
        assert _same(moe(x[:rows], True), one[:rows]), (enabled, rows)


@pytest.mark.parametrize("shape", [(8192, 128), (200, 128), (256, 64)])  # KDA's f_b / g_b: [H d, d]
def test_qmv_quad_rows_gives_mlx_one_row_bits(gpu, shape):
    n, k = shape
    mx.random.seed(9)
    w = (0.05 * mx.random.normal((n, k))).astype(mx.bfloat16)
    q = linear.Q(*mx.quantize(w, group_size=64, bits=4), bits=4, group=64)
    x = mx.random.normal((16, k)).astype(mx.bfloat16)
    one = mx.concatenate([q(x[r:r + 1]) for r in range(16)])
    for rows in range(2, 17):
        assert _same(K.qmv_quad_rows(x[:rows], q), one[:rows]), rows


def test_every_row_kernel_switch_keeps_windows_exact(gpu, monkeypatch, tmp_path):
    """Each row kernel alone and all of them keep 2/3/4/8-row windows exact on the tiny checkpoint."""

    from glm5_fakes import write_checkpoint
    from tensorfold.families.glm5_next.runtime import GLMFlash

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        path = write_checkpoint(tmp_path / "glm5")
    finally:
        mx.set_default_device(previous)
    model = weights.load_backbone(path)
    for enabled in [(name,) for name in config.ROW_KERNELS] + [config.ROW_KERNELS]:
        monkeypatch.setattr(config, "ENABLED", frozenset(enabled))
        runtime = GLMFlash(model, check=True)
        assert runtime.multi_row_exact, (enabled, runtime.check_report)


@pytest.mark.parametrize("start", [2045, 4093])                  # windows crossing index_topk / block boundaries
def test_indexer_choices_are_each_rows_own(gpu, start):
    """A window's sparse key choice equals each row's own at GLM's indexer shape, with bf16 ties."""

    import types

    cfg = types.SimpleNamespace(index_topk=2048, index_kpool=4, index_tail=True)
    stub = types.SimpleNamespace(cfg=cfg)
    stub.index_scores = lambda iq, iw, pool: mla.MLA.index_scores(stub, iq, iw, pool)
    mx.random.seed(12)
    rows = 16
    pool = mx.round(4 * mx.random.normal(((start + rows) // 4 + 1, 128))).astype(mx.bfloat16) / 4
    cache = types.SimpleNamespace(pool=pool)
    iq = mx.round(2 * mx.random.normal((rows, 32, 128))).astype(mx.bfloat16) / 2
    iw = mx.random.normal((rows, 32)).astype(mx.bfloat16)
    got = mla.MLA._choices(stub, iq, iw, cache, start)
    for r in range(rows):
        position = start + r
        if position + 1 <= cfg.index_topk:
            assert got[r] is None
            continue
        blocks = (position + 1) // 4
        want = mla.MLA.selected(stub, stub.index_scores(iq[r][None], iw[r][None], pool[:blocks]), position)
        assert _same(got[r], want), r


@pytest.mark.skipif(not __import__("os").environ.get("TF_GLM5_MODEL"), reason="set TF_GLM5_MODEL to the checkpoint")
def test_real_weights_long_context_windows_are_exact(gpu):
    """The real first 8 layers past 4,096 keys: 2-16-row windows keep every row's one-row bits."""

    import os

    from tensorfold.engine.lane_engine import LaneEngine

    model = weights.load_backbone(os.environ["TF_GLM5_MODEL"], layers=8)
    assert config.ENABLED == frozenset(config.ROW_KERNELS)
    base = model.make_cache()
    prompt = [1000 + (37 * i) % 50_000 for i in range(4093)]
    for c0 in range(0, len(prompt), 2048):
        mx.eval(model.hidden(mx.array([prompt[c0:c0 + 2048]]), base))
    tokens = [3001 + 17 * r for r in range(16)]
    one = LaneEngine.copy_single_cache(base)
    serial = mx.concatenate([model.head(model.hidden(mx.array([[t]]), one)) for t in tokens], axis=1)
    for width in (2, 3, 4, 8, 16):
        many = LaneEngine.copy_single_cache(base)
        window = model.head(model.hidden(mx.array([tokens[:width]]), many))
        assert _same(window, serial[:, :width]), width


# -- a mixed-bit checkpoint's widths: 5-, 6- and 8-bit tensors through the row kernels ---------------------------
def _qbits(n: int, k: int, bits: int, seed: int, scale: float = 0.05) -> linear.Q:
    mx.random.seed(seed)
    return linear.Q(*mx.quantize((scale * mx.random.normal((n, k))).astype(mx.bfloat16), group_size=64, bits=bits),
                    bits=bits, group=64)


@pytest.mark.parametrize("bits", [8, 6, 5])
@pytest.mark.parametrize("shape", [(1024, 4096), (4096, 1536), (512, 2048), (32, 4096), (8, 512)])
def test_qmv_rows_other_bits_give_mlx_one_row_bits(gpu, bits, shape):
    """8-, 6- and 5-bit group-64 weights: every row of a 2-32-row window gets MLX's one-row qmv_fast bits."""

    n, k = shape
    q = _qbits(n, k, bits, seed=bits + n)
    assert W.fast_shape(q, n) and K.qmv_rows_fits(q, 2)
    x = mx.random.normal((32, k)).astype(mx.bfloat16)
    one = mx.concatenate([q(x[r:r + 1]) for r in range(32)])
    for rows in (2, 3, 4, 5, 8, 16, 32):
        assert _same(K.qmv_rows(x[:rows], q), one[:rows]), (bits, shape, rows)


def test_fast_shape_is_mlx_s_qmv_fast_rule():
    assert not W.fast_shape(_qbits(64, 4096, 4, 1), 64)          # 4-bit has its own kernel
    assert W.fast_shape(_qbits(64, 256, 8, 1), 64) and not W.fast_shape(_qbits(64, 384, 8, 1), 64)   # K % 256
    assert W.fast_shape(_qbits(64, 512, 5, 1), 64) and not W.fast_shape(_qbits(64, 256, 5, 1), 64)   # K % 512
    assert not W.fast_shape(_qbits(60, 4096, 8, 1), 60)          # N % 8
    assert not W.fast_shape(_qbits(64, 128, 8, 1), 64)           # qmv_quad territory
    assert not K.qmv_rows_fits(_qbits(60, 4096, 8, 1), 2)
    assert K.qmv_quad_rows_fits(_qbits(64, 128, 8, 1), 2) and not K.qmv_quad_rows_fits(_qbits(64, 128, 5, 1), 2)


@pytest.mark.parametrize("shape", [(8192, 128), (200, 128), (256, 64)])
def test_qmv_quad_rows_8bit_gives_mlx_one_row_bits(gpu, shape):
    n, k = shape
    q = _qbits(n, k, 8, seed=10)
    x = mx.random.normal((16, k)).astype(mx.bfloat16)
    one = mx.concatenate([q(x[r:r + 1]) for r in range(16)])
    for rows in range(2, 17):
        assert _same(K.qmv_quad_rows(x[:rows], q), one[:rows]), rows


def _experts_bits(e: int, n: int, k: int, bits: int, seed: int) -> linear.Q:
    mx.random.seed(seed)
    w = (0.02 * mx.random.normal((e, n, k))).astype(mx.bfloat16)
    return linear.Q(*mx.quantize(w, group_size=64, bits=bits), bits=bits, group=64)


def _one_row_bits(x: mx.array, idx: mx.array, q: linear.Q) -> mx.array:
    return mx.gather_qmm(x[None], q.weight, q.scales, q.biases, rhs_indices=idx, transpose=True, group_size=64,
                         bits=q.bits).squeeze(-2)


@pytest.mark.parametrize("bits", [8, 6, 5])
def test_expert_qmv_other_bits_gives_each_pick_its_one_row_bits(gpu, bits):
    n, k, experts, top = 512, 1024, 24, 4
    w = _experts_bits(experts, n, k, bits, seed=bits)
    assert K.expert_qmv_fits(w, 2)
    for rows in (2, 3, 5, 8, 16):
        idx = _picks(rows, top, experts, seed=rows)
        group = K.expert_group(idx, experts)
        x = mx.random.normal((rows, k)).astype(mx.bfloat16)
        shared = K.expert_qmv(x, idx, group, w, per_pick=False)
        want = mx.concatenate([_one_row_bits(x[r:r + 1][:, None, :], idx[r:r + 1], w) for r in range(rows)])
        assert _same(shared, want), (bits, rows)
        act = mx.random.normal((rows, top, k)).astype(mx.bfloat16)
        own = K.expert_qmv(act, idx, group, w, per_pick=True)
        want = mx.concatenate([_one_row_bits(act[r][:, None, :], idx[r:r + 1], w) for r in range(rows)])
        assert _same(own, want), (bits, rows)


def _requant(q: linear.Q, bits: int) -> linear.Q:
    w = mx.dequantize(q.weight, q.scales, q.biases, group_size=64, bits=q.bits)
    return linear.Q(*mx.quantize(w, group_size=64, bits=bits), bits=bits, group=64)


def test_moe_window_with_8bit_shared_expert_is_row_by_row(gpu, monkeypatch):
    """The fused MoE with an 8-bit shared expert gives every row of a window the row-by-row block's bits."""

    from tensorfold.kernels.glm.flash.v1 import moe as F

    moe = _moe()
    moe.shared = mlp.DenseMLP(_requant(_rows_part(moe.shared.gate_up, 0), 8), _requant(_rows_part(moe.shared.gate_up, 1), 8),
                              _requant(moe.shared.down, 8), 10.0)
    assert moe.shared.gate_up.bits == 8 and moe.shared.down.bits == 8
    moe.fused_ok = F.moe_fits(moe)
    assert moe.fused_ok and F.SPLIT_SHARED
    monkeypatch.setattr(config, "ENABLED", frozenset())
    monkeypatch.setattr(config, "FUSED", frozenset())
    x = (0.5 * mx.random.normal((16, 512))).astype(mx.bfloat16)
    one = mx.concatenate([moe(x[r:r + 1], True) for r in range(16)])
    for rows in (2, 3, 4, 8, 16):
        assert _same(F.moe_rows(moe, x[:rows]), one[:rows]), rows


def _rows_part(gate_up: linear.Q, half: int) -> linear.Q:
    n = gate_up.outs // 2
    return linear._rows(gate_up, half * n, (half + 1) * n)


def test_kda_rows_with_8bit_f_b_g_b_is_row_by_row(gpu, tmp_path, monkeypatch):
    """The fused KDA step with 8-bit f_b / g_b: a window equals its rows one at a time, and y equals the ops path's."""

    from glm5_fakes import write_checkpoint
    from tensorfold.kernels.glm.flash.v1 import kda as KDA_K

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        path = write_checkpoint(tmp_path / "glm5")
    finally:
        mx.set_default_device(previous)
    model = weights.load_backbone(path)
    kda = model.layers[0].attn
    kda.f_b, kda.g_b = _requant(kda.f_b, 8), _requant(kda.g_b, 8)
    assert KDA_K.fits(kda)                           # the fused step takes 8-bit f_b / g_b too
    for rows in (1, 2, 3, 5, 8, 16):
        mx.random.seed(rows)
        proj = (0.5 * mx.random.normal((rows, kda.in_proj.outs))).astype(mx.bfloat16)
        conv = (0.5 * mx.random.normal((kda.taps - 1, 3 * kda.width))).astype(mx.bfloat16)
        state = (0.1 * mx.random.normal((1, kda.heads, kda.dim, kda.dim))).astype(mx.float32)
        y, st, cs = KDA_K.kda_rows(kda, proj, conv, state)
        # the window equals the same kernel one row at a time (row invariance: what drafting needs)
        ys, s1, c1 = [], state, conv
        for r in range(rows):
            yr, s1, c1 = KDA_K.kda_rows(kda, mx.contiguous(proj[r:r + 1]), c1, s1)
            ys.append(yr)
        assert _same(y, mx.concatenate(ys)) and _same(st, s1) and _same(cs, c1), rows
        # y equals the ops path's; the fp32 state may differ in its last bits (printed: the kernel is the decode path)
        y2, st2, cs2 = KDA_K.kda_rows_ops(kda, proj, conv, state)
        assert _same(y, y2) and _same(cs, cs2), rows
        if not _same(st, st2):
            d = mx.abs(st - st2).max().item()
            print(f"rows {rows}: 8-bit kda_rows state vs ops max |diff| {d:.3g}")
