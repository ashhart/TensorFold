"""GLM-5.3-Flash's fused decode kernels give a window's rows the row-by-row path's one-row bits."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from glm5_fakes import write_checkpoint  # noqa: E402
from tensorfold.families.glm5_next import config, model, weights  # noqa: E402


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


def test_fused_model_drafts_exact(checkpoint, monkeypatch, device):
    from test_glm5_next_family import _run_engine, tokens
    from tensorfold.families.glm5_next import mtp as glm_mtp
    from tensorfold.families.glm5_next.runtime import GLMFlash

    monkeypatch.setattr(config, "FUSED", frozenset(config.FUSED_KERNELS))
    model = weights.load_backbone(checkpoint)
    runtime = GLMFlash(model, glm_mtp.load(model), drafts=3)
    assert runtime.multi_row_exact, runtime.check_report
    prompt = tokens(30, seed=8)
    engine_a, a = _run_engine(runtime, prompt, 20)
    _, b = _run_engine(GLMFlash(model, None, drafts=0), prompt, 20)
    assert engine_a.drafted > 0 and a.emitted == b.emitted


@pytest.fixture(params=[1, 0], ids=["shared-split", "shared-in-slot"])
def split_shared(request, monkeypatch):
    from tensorfold.kernels.glm.flash.v1 import moe as F

    monkeypatch.setattr(F, "SPLIT_SHARED", request.param)
    return request.param


def test_fused_moe_is_the_row_by_row_block(monkeypatch, split_shared):
    """On Metal the five-kernel MoE gives every row of a 1-16-row window the row-by-row block's bits."""

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    from test_glm5_row_kernels import _moe

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        moe = _moe()
        assert moe.fused_ok
        x = (0.5 * mx.random.normal((16, 512))).astype(mx.bfloat16)
        monkeypatch.setattr(config, "FUSED", frozenset())
        monkeypatch.setattr(config, "ENABLED", frozenset())
        ref = mx.concatenate([moe(x[r:r + 1], True) for r in range(16)])
        monkeypatch.setattr(config, "FUSED", frozenset({"moe"}))
        for rows in (1, 2, 3, 4, 8, 16):
            assert _same(moe(x[:rows], True), ref[:rows]), rows
        # ties in the router: equal logits pick the lower expert id, as mx.argpartition does
        moe.router = mx.zeros_like(moe.router)
        moe.bias = mx.zeros_like(moe.bias)
        monkeypatch.setattr(config, "FUSED", frozenset())
        ref = mx.concatenate([moe(x[r:r + 1], True) for r in range(4)])
        monkeypatch.setattr(config, "FUSED", frozenset({"moe"}))
        assert _same(moe(x[:4], True), ref)
    finally:
        mx.set_default_device(previous)


def test_fused_hc_boundary_is_the_row_by_row_path(monkeypatch):
    """At GLM's width the three hyper-connection kernels give the row-by-row bits for 1-16 rows and at both ends."""

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    from tensorfold.kernels.glm.flash.v1 import hc as F

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        cfg = config.Config.from_dict({
            "hidden_size": 4096, "num_hidden_layers": 1, "layer_types": ["linear_attention"],
            "mlp_layer_types": ["dense"], "vocab_size": 16, "rms_norm_eps": 1e-5, "num_attention_heads": 1,
            "q_lora_rank": 64, "kv_lora_rank": 64, "qk_nope_head_dim": 64, "v_head_dim": 64, "index_n_heads": 1,
            "index_head_dim": 64, "index_topk": 16, "n_routed_experts": 4, "num_experts_per_tok": 2,
            "moe_intermediate_size": 64, "intermediate_size": 64, "routed_scaling_factor": 1.0,
            "eos_token_id": [0]})
        mx.random.seed(21)
        hc = model.HC(0.05 * mx.random.normal((24, 16384)), 0.3 * mx.random.normal((24,)),
                    mx.array([0.5, 0.5, 0.5]), cfg)
        w = (1 + 0.1 * mx.random.normal((4096,))).astype(mx.bfloat16)
        x = mx.random.normal((16, 4, 4096)).astype(mx.bfloat16)
        branch = mx.random.normal((16, 4096)).astype(mx.bfloat16)
        post = mx.random.uniform(0, 2, (16, 4))
        comb = mx.random.uniform(shape=(16, 4, 4))
        assert F.hc_fits(hc, 4096)
        monkeypatch.setattr(config, "ENABLED", frozenset())

        def reference(xs, b, p, c, first):
            new = xs if first else model.hc_expand(b, xs, p, c, True)
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
        assert _same(last, model.hc_expand(branch, x, post, comb, False))
    finally:
        mx.set_default_device(previous)


def test_router_kernel_gives_mlx_one_row_bits():
    """The repacked router gives MLX's one-row fp32 matmul bits for 1-16 rows, with or without the fetching group."""

    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    import types

    from tensorfold.kernels.glm.flash.v1 import moe as F

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
