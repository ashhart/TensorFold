"""GLM-5.3-Flash decode-row kernels: each row of a window keeps the bits its one-row call gives it.

The Metal tests run at GLM-5.3-Flash's own shapes where they matter (the router, the hyper-connection mix, the
indexer gate) and at smaller multiples of the kernels' blocks for the experts; on Linux the fallbacks are checked
to be the one-row calls themselves."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.glm.flash.v1 import kernels as K  # noqa: E402
from tensorfold.families.glm5_next import model as glm  # noqa: E402


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


def _experts(e: int, n: int, k: int, seed: int) -> glm.Q:
    mx.random.seed(seed)
    w = (0.02 * mx.random.normal((e, n, k))).astype(mx.bfloat16)
    return glm.Q(*mx.quantize(w, group_size=64, bits=4))


def _picks(rows: int, top: int, experts: int, seed: int) -> mx.array:
    """Distinct experts per row, with some experts shared between rows (as consecutive tokens do)."""

    mx.random.seed(seed)
    idx = mx.stack([mx.random.permutation(experts)[:top] for _ in range(rows)]).astype(mx.uint32)
    for r in range(1, rows):
        if int(idx[0, 0].item()) not in [int(v) for v in idx[r].tolist()]:
            idx[r, r % top] = idx[0, 0]
    return idx


def _one_row(x: mx.array, idx: mx.array, q: glm.Q) -> mx.array:
    """What model.MoE.experts runs for one row: mx.gather_qmm on [1, k, 1, K] -> [1, k, N]."""

    return mx.gather_qmm(x[None], q.weight, q.scales, q.biases, rhs_indices=idx, transpose=True, group_size=64,
                         bits=4).squeeze(-2)


def test_row_kernel_switch(monkeypatch):
    for value, expected in (("all", set(glm.ROW_KERNELS)), ("none", set()), ("router,hc", {"router", "hc"})):
        monkeypatch.setenv("TF_GLM5_ROW_KERNELS", value)
        assert glm._row_kernels() == expected
    monkeypatch.setenv("TF_GLM5_ROW_KERNELS", "router,nope")
    with pytest.raises(ValueError):
        glm._row_kernels()
    monkeypatch.setattr(glm, "ENABLED", frozenset({"router"}))
    assert glm.row_kernel("router", 2, True)
    assert not glm.row_kernel("router", 1, True) and not glm.row_kernel("router", 2, False)
    assert not glm.row_kernel("experts", 2, True)


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
    n = int(count.item())
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


def _moe(dims: int = 512, width: int = 512, experts: int = 24, top: int = 4) -> glm.MoE:
    cfg = glm.Config.from_dict({
        "hidden_size": dims, "num_hidden_layers": 1, "layer_types": ["linear_attention"], "mlp_layer_types": ["sparse"],
        "vocab_size": 16, "rms_norm_eps": 1e-5, "num_attention_heads": 1, "q_lora_rank": 64, "kv_lora_rank": 64,
        "qk_nope_head_dim": 64, "v_head_dim": 64, "index_n_heads": 1, "index_head_dim": 64, "index_topk": 16,
        "n_routed_experts": experts, "num_experts_per_tok": top, "moe_intermediate_size": width,
        "intermediate_size": width, "n_shared_experts": 1, "routed_scaling_factor": 2.5, "norm_topk_prob": True,
        "swiglu_limit": 10.0, "eos_token_id": [0]})
    mx.random.seed(5)
    gate, up, down = _experts(experts, width, dims, 6), _experts(experts, width, dims, 7), _experts(experts, dims,
                                                                                                    width, 8)

    def lin(n: int, k: int) -> glm.Q:
        return glm.Q(*mx.quantize((0.05 * mx.random.normal((n, k))).astype(mx.bfloat16), group_size=64, bits=4))

    shared = glm.DenseMLP(lin(width, dims), lin(width, dims), lin(dims, width), 10.0)
    router = (0.3 * mx.random.normal((experts, dims))).astype(mx.float32)
    bias = (0.1 * mx.random.normal((experts,))).astype(mx.float32)
    return glm.MoE(router, bias, gate, up, down, shared, cfg)


@pytest.mark.parametrize("enabled", [("experts", "router"), ("router",), ()])
def test_moe_window_rows_are_one_row_steps(gpu, monkeypatch, enabled):
    """The MoE block on a window (router + expert kernels, router only, or row by row) gives every row its
    one-row bits."""

    monkeypatch.setattr(glm, "ENABLED", frozenset(enabled))
    moe = _moe()
    x = (0.5 * mx.random.normal((16, 512))).astype(mx.bfloat16)
    one = mx.concatenate([moe(x[r:r + 1], True) for r in range(16)])
    for rows in (2, 3, 4, 8, 16):
        assert _same(moe(x[:rows], True), one[:rows]), (enabled, rows)


@pytest.mark.parametrize("shape", [(8192, 128), (200, 128), (256, 64)])  # KDA's f_b / g_b: [H d, d]
def test_qmv_quad_rows_gives_mlx_one_row_bits(gpu, shape):
    n, k = shape
    mx.random.seed(9)
    q = glm.Q(*mx.quantize((0.05 * mx.random.normal((n, k))).astype(mx.bfloat16), group_size=64, bits=4))
    x = mx.random.normal((16, k)).astype(mx.bfloat16)
    one = mx.concatenate([q(x[r:r + 1]) for r in range(16)])
    for rows in range(2, 17):
        assert _same(K.qmv_quad_rows(x[:rows], q), one[:rows]), rows


def test_every_row_kernel_switch_keeps_windows_exact(gpu, monkeypatch, tmp_path):
    """On the tiny checkpoint: with each row kernel alone and with all of them, 2/3/4/8-row windows give every row
    its one-row bits (the load-time check)."""

    from glm5_fakes import write_checkpoint
    from tensorfold.families.glm5_next.runtime import GLMFlash

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        path = write_checkpoint(tmp_path / "glm5")
    finally:
        mx.set_default_device(previous)
    model = glm.load_backbone(path)
    for enabled in [(name,) for name in glm.ROW_KERNELS] + [glm.ROW_KERNELS]:
        monkeypatch.setattr(glm, "ENABLED", frozenset(enabled))
        runtime = GLMFlash(model, check=True)
        assert runtime.multi_row_exact, (enabled, runtime.check_report)


@pytest.mark.parametrize("start", [2045, 4093])                  # windows crossing index_topk / block boundaries
def test_indexer_choices_are_each_rows_own(gpu, start):
    """A window's sparse key choice (rows scored and ranked in groups sharing a block count) equals each row's
    one-row choice, at GLM-5.3-Flash's indexer shape (32 heads of 128, top 512 blocks of 4) with bf16 ties."""

    import types

    cfg = types.SimpleNamespace(index_topk=2048, index_kpool=4, index_tail=True)
    stub = types.SimpleNamespace(cfg=cfg)
    stub.index_scores = lambda iq, iw, pool: glm.MLA.index_scores(stub, iq, iw, pool)
    mx.random.seed(12)
    rows = 16
    pool = mx.round(4 * mx.random.normal(((start + rows) // 4 + 1, 128))).astype(mx.bfloat16) / 4
    cache = types.SimpleNamespace(pool=pool)
    iq = mx.round(2 * mx.random.normal((rows, 32, 128))).astype(mx.bfloat16) / 2
    iw = mx.random.normal((rows, 32)).astype(mx.bfloat16)
    got = glm.MLA._choices(stub, iq, iw, cache, start)
    for r in range(rows):
        position = start + r
        if position + 1 <= cfg.index_topk:
            assert got[r] is None
            continue
        blocks = (position + 1) // 4
        want = glm.MLA.selected(stub, stub.index_scores(iq[r][None], iw[r][None], pool[:blocks]), position)
        assert _same(got[r], want), r


@pytest.mark.skipif(not __import__("os").environ.get("TF_GLM5_MODEL"), reason="set TF_GLM5_MODEL to the checkpoint")
def test_real_weights_long_context_windows_are_exact(gpu):
    """The real checkpoint's first 8 layers (two sparse-attention layers, five MoE) and head past 4,096 keys, where
    each query reads its chosen 512 blocks: 2-16-row windows give every row its one-row bits, row kernels on."""

    import os

    from tensorfold.engine.lane_engine import LaneEngine

    model = glm.load_backbone(os.environ["TF_GLM5_MODEL"], layers=8)
    assert glm.ENABLED == frozenset(glm.ROW_KERNELS)
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
