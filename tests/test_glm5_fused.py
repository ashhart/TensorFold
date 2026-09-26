"""GLM-5.3-Flash fused decode kernels (kernels/glm/flash/v1/fused.py):
a window's rows keep their one-row bits, and the fused MoE / hyper-connection / router kernels give the row-by-row
path's bits."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from glm5_fakes import write_checkpoint  # noqa: E402
from tensorfold.families.glm5_next import model as glm  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return write_checkpoint(tmp_path_factory.mktemp("glm5f"))
    finally:
        mx.set_default_device(previous)


@pytest.fixture(params=["cpu", "gpu"])
def device(request):
    if request.param == "gpu" and not mx.metal.is_available():
        pytest.skip("needs Metal")
    previous = mx.default_device()
    mx.set_default_device(getattr(mx, request.param))
    yield request.param
    mx.set_default_device(previous)


def _same(a, b) -> bool:
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def test_fused_switch(monkeypatch):
    for value, expected in (("all", set(glm.FUSED_KERNELS)), ("none", set()), ("moe", {"moe"}),
                            ("moe,hc", {"moe", "hc"})):
        monkeypatch.setenv("TF_GLM5_FUSED", value)
        assert glm._fused() == expected
    monkeypatch.setenv("TF_GLM5_FUSED", "kda")   # the KDA step is kda.py (TF_GLM5_FUSED_KDA)
    with pytest.raises(ValueError):
        glm._fused()


def test_fused_model_drafts_exact(checkpoint, monkeypatch, device):
    from test_glm5_next_family import _run_engine, tokens
    from tensorfold.families.glm5_next import mtp as glm_mtp
    from tensorfold.families.glm5_next.runtime import GLMFlash

    monkeypatch.setattr(glm, "FUSED", frozenset(glm.FUSED_KERNELS))
    model = glm.load_backbone(checkpoint)
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=3)
    assert runtime.multi_row_exact, runtime.check_report
    prompt = tokens(30, seed=8)
    engine_a, a = _run_engine(runtime, prompt, 20)
    _, b = _run_engine(GLMFlash(model, None, drafts=0), prompt, 20)
    assert engine_a.drafted > 0 and a.emitted == b.emitted


@pytest.fixture(params=[1, 0], ids=["shared-split", "shared-in-slot"])
def split_shared(request, monkeypatch):
    from tensorfold.kernels.glm.flash.v1 import fused as F

    monkeypatch.setattr(F, "SPLIT_SHARED", request.param)
    return request.param


def test_fused_moe_is_the_row_by_row_block(monkeypatch, split_shared):
    """On Metal the five-kernel MoE gives every row of a 1-16-row window the row-by-row block's bits (router,
    top-k order with ties, normalised weights, experts, SwiGLU, combine, shared expert)."""

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    from test_glm5_row_kernels import _moe

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        moe = _moe()
        assert moe.fused_ok
        x = (0.5 * mx.random.normal((16, 512))).astype(mx.bfloat16)
        monkeypatch.setattr(glm, "FUSED", frozenset())
        monkeypatch.setattr(glm, "ENABLED", frozenset())
        ref = mx.concatenate([moe(x[r:r + 1], True) for r in range(16)])
        monkeypatch.setattr(glm, "FUSED", frozenset({"moe"}))
        for rows in (1, 2, 3, 4, 8, 16):
            assert _same(moe(x[:rows], True), ref[:rows]), rows
        # ties in the router: equal logits pick the lower expert id, as mx.argpartition does
        moe.router = mx.zeros_like(moe.router)
        moe.bias = mx.zeros_like(moe.bias)
        monkeypatch.setattr(glm, "FUSED", frozenset())
        ref = mx.concatenate([moe(x[r:r + 1], True) for r in range(4)])
        monkeypatch.setattr(glm, "FUSED", frozenset({"moe"}))
        assert _same(moe(x[:4], True), ref)
    finally:
        mx.set_default_device(previous)


def test_fused_hc_boundary_is_the_row_by_row_path(monkeypatch):
    """At GLM-5.3-Flash's width (4 streams of 4,096): write-back + split + RMSNorm in three kernels give the
    row-by-row path's streams, normed input, post and comb for 1-16 rows, the first block (no write-back) and the
    last write-back (no split)."""

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    from tensorfold.kernels.glm.flash.v1 import fused as F

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        cfg = glm.Config.from_dict({
            "hidden_size": 4096, "num_hidden_layers": 1, "layer_types": ["linear_attention"],
            "mlp_layer_types": ["dense"], "vocab_size": 16, "rms_norm_eps": 1e-5, "num_attention_heads": 1,
            "q_lora_rank": 64, "kv_lora_rank": 64, "qk_nope_head_dim": 64, "v_head_dim": 64, "index_n_heads": 1,
            "index_head_dim": 64, "index_topk": 16, "n_routed_experts": 4, "num_experts_per_tok": 2,
            "moe_intermediate_size": 64, "intermediate_size": 64, "routed_scaling_factor": 1.0,
            "eos_token_id": [0]})
        mx.random.seed(21)
        hc = glm.HC(0.05 * mx.random.normal((24, 16384)), 0.3 * mx.random.normal((24,)),
                    mx.array([0.5, 0.5, 0.5]), cfg)
        w = (1 + 0.1 * mx.random.normal((4096,))).astype(mx.bfloat16)
        x = mx.random.normal((16, 4, 4096)).astype(mx.bfloat16)
        branch = mx.random.normal((16, 4096)).astype(mx.bfloat16)
        post = mx.random.uniform(0, 2, (16, 4))
        comb = mx.random.uniform(shape=(16, 4, 4))
        assert F.hc_fits(hc, 4096)
        monkeypatch.setattr(glm, "ENABLED", frozenset())

        def reference(xs, b, p, c, first):
            new = xs if first else glm.hc_expand(b, xs, p, c, True)
            xc, po, co = hc.split(new, True)
            return new, mx.fast.rms_norm(xc, w, 1e-5), po, co

        for first in (True, False):
            ref = [mx.concatenate(parts) for parts in zip(*[
                reference(x[r:r + 1], branch[r:r + 1], post[r:r + 1], comb[r:r + 1], first) for r in range(16)])]
            for rows in (1, 2, 3, 8, 16):
                pending = None if first else (branch[:rows], post[:rows], comb[:rows])
                got = F.hc_step(x[:rows], pending, hc, w, 1e-5)
                for a, b in zip(got, ref):
                    assert _same(a, b[:rows]), (first, rows)
        last = F.hc_step(x, (branch, post, comb), None, None, 1e-5)[0]
        assert _same(last, glm.hc_expand(branch, x, post, comb, False))
    finally:
        mx.set_default_device(previous)


def test_router_kernel_gives_mlx_one_row_bits():
    """The repacked router (288 experts x 4,096, stored bf16) gives MLX's one-row fp32 matmul bits, 1-16 rows,
    with the fetching threadgroup and without."""

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    import types

    from tensorfold.kernels.glm.flash.v1 import fused as F

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        mx.random.seed(31)
        gate_w = (0.05 * mx.random.normal((288, 4096))).astype(mx.bfloat16)
        moe = types.SimpleNamespace(router=mx.contiguous(gate_w.astype(mx.float32).T),
                                    router_packed=F.pack_router(gate_w))
        x = mx.random.normal((16, 4096))
        one = mx.concatenate([x[r:r + 1] @ moe.router for r in range(16)])
        for tg in (1024, 0):
            F.ROUTER_TG, keep = tg, F.ROUTER_TG
            try:
                for rows in (1, 2, 3, 5, 8, 16):
                    assert _same(F.router_rows(x[:rows], moe), one[:rows]), (tg, rows)
            finally:
                F.ROUTER_TG = keep
    finally:
        mx.set_default_device(previous)


def test_draft_depth_follows_acceptance(monkeypatch):
    """GLM picks the depth from the conditional acceptance and the verify cost of each depth: shallow at low
    acceptance, deeper as it rises, never past the cap; TF_GLM_DEPTH fixes it."""

    from tensorfold.families.glm5_next.runtime import GLMFlash

    rt = GLMFlash.__new__(GLMFlash)
    rt.drafts = 4
    monkeypatch.delenv("TF_GLM_DEPTH", raising=False)
    depths = [rt.depth_for(a) for a in (0.5, 0.7, 0.8, 0.85, 0.9, 0.99)]
    assert depths == sorted(depths) and depths[0] == 1 and depths[-1] == 4
    rt.drafts = 2
    assert rt.depth_for(0.99) == 2
    rt.drafts = 4
    # a chained draft that lands less often than the first one keeps the chain short (M3 Ultra: w1 think
    # 0.81 then 0.69, w2 code 0.87 then 0.60)
    assert rt.depth_for([0.81, 0.69]) == 1 and rt.depth_for([0.87, 0.60]) == 1
    assert rt.depth_for([0.95, 0.95, 0.95]) >= 3
    assert rt.depth_for([0.8]) == 1          # fresh stream: later positions assumed to land less often
    monkeypatch.setenv("TF_GLM_DEPTH", "3")
    rt.drafts = 4
    assert rt.depth_for(0.1) == 3


def test_engine_uses_the_models_depth(monkeypatch):
    """The serial engine asks a model with depth_for for the depth, fed with the conditional acceptance (drafts
    past the first rejection are not counted)."""

    import types

    from tensorfold.engine.family_engine import SerialEngine

    seen = []
    model = types.SimpleNamespace(drafts=4, depth_for=lambda rates: seen.append(rates) or 4)
    engine = SerialEngine.__new__(SerialEngine)
    engine.model, engine._accept_rate, engine._pos_rate = model, {}, {}
    stream = types.SimpleNamespace(stream_id="s")
    assert engine._depth(stream, [1, 2, 3, 4], 2) == 4       # draft 0 landed, draft 1 was rejected, 2-3 unread
    hit = 0.875 * 0.8 + 0.125
    miss = 0.875 * (0.8 * hit)                               # a new position starts below the one before it
    assert len(seen[-1]) == 2 and abs(seen[-1][0] - hit) < 1e-12 and abs(seen[-1][1] - miss) < 1e-12
    engine._depth(stream, [1], 1)                            # only position 0 checked (rejected)
    assert abs(seen[-1][0] - 0.875 * hit) < 1e-12 and abs(seen[-1][1] - miss) < 1e-12
