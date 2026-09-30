"""GLM-5.3-Flash's prompt kernels give a prompt chunk the bits its MLX ops give it (KDA glue, SwiGLU, combine, HC)."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.glm5_next import config, linear, mlp, model  # noqa: E402
from tensorfold.families.glm5_next.caches import KDACache  # noqa: E402
from tensorfold.families.glm5_next.kda import KDA  # noqa: E402


@pytest.fixture
def gpu():
    if not mx.metal.is_available():
        pytest.skip("needs Metal")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


def _same(a: mx.array, b: mx.array) -> bool:
    return a.shape == b.shape and bool(mx.array_equal(a, b).item())


def _q(n: int, k: int, seed: int, scale: float = 0.05, lead: tuple = ()) -> linear.Q:
    mx.random.seed(seed)
    w = (scale * mx.random.normal((*lead, n, k))).astype(mx.bfloat16)
    return linear.Q(*mx.quantize(w, group_size=64, bits=4), bits=4, group=64)


def _cfg(dims: int, **extra) -> config.Config:
    base = {"hidden_size": dims, "num_hidden_layers": 1, "layer_types": ["linear_attention"],
            "mlp_layer_types": ["sparse"], "vocab_size": 16, "rms_norm_eps": 1e-5, "num_attention_heads": 1,
            "q_lora_rank": 64, "kv_lora_rank": 64, "qk_nope_head_dim": 64, "v_head_dim": 64, "index_n_heads": 1,
            "index_head_dim": 64, "index_topk": 16, "n_routed_experts": 32, "num_experts_per_tok": 8,
            "moe_intermediate_size": 256, "intermediate_size": 256, "n_shared_experts": 1,
            "routed_scaling_factor": 2.5, "norm_topk_prob": True, "swiglu_limit": 10.0, "eos_token_id": [0]}
    return config.Config.from_dict({**base, **extra})


def _kda(dims: int, heads: int, dim: int) -> KDA:
    cfg = _cfg(dims, linear_attn_config={"num_heads": heads, "head_dim": dim, "short_conv_kernel_size": 4,
                                         "gate_lower_bound": -5.0})
    width = heads * dim
    w = {name: _q(width, dims, 3 + i) for i, name in enumerate(("q_proj", "k_proj", "v_proj"))}
    w.update(f_a_proj=_q(dim, dims, 7), g_a_proj=_q(dim, dims, 8), b_proj=_q(heads, dims, 9, 0.3),
             f_b_proj=_q(width, dim, 10, 0.3), g_b_proj=_q(width, dim, 11, 0.3), o_proj=_q(dims, width, 12))
    mx.random.seed(13)
    for c in "qkv":
        w[f"{c}_conv1d"] = (0.4 * mx.random.normal((width, 1, 4))).astype(mx.bfloat16)
    w["o_norm"] = (1 + 0.1 * mx.random.normal((dim,))).astype(mx.bfloat16)
    w["A_log"] = 0.5 * mx.random.normal((heads,))
    w["dt_bias"] = 0.5 * mx.random.normal((width,))
    return KDA(w, cfg)


@pytest.mark.parametrize("shape", [(512, 64, 128), (256, 2, 64)])
def test_prompt_kda_glue_is_the_ops_path(gpu, monkeypatch, shape):
    """Two prompt chunks through a KDA layer: output rows, conv window and fp32 state equal the MLX ops' bits."""

    from tensorfold.kernels.glm.flash.v1 import prompt as PK

    kda = _kda(*shape)
    assert PK.kda_fits(kda)
    mx.random.seed(14)
    x = (0.5 * mx.random.normal((2048 + 257, shape[0]))).astype(mx.bfloat16)
    for rows in (17, 300, 2048):
        got = {}
        for fused in (False, True):
            monkeypatch.setattr(config, "FUSED", frozenset({"kda"} if fused else ()))
            cache = KDACache()
            first = kda(x[:rows], [cache], (rows,), False)
            second = kda(x[rows:rows + 257], [cache], (257,), False)
            mx.eval(first, second, cache.conv, cache.ssm)
            got[fused] = (first, second, cache.conv, cache.ssm)
        assert all(_same(a, b) for a, b in zip(got[False], got[True])), (shape, rows)


def _moe(limit: float) -> mlp.MoE:
    dims, width, experts = 512, 256, 32
    cfg = _cfg(dims, swiglu_limit=limit)
    gate, up = _q(width, dims, 20, 0.02, (experts,)), _q(width, dims, 21, 0.02, (experts,))
    down = _q(dims, width, 22, 0.02, (experts,))
    shared = mlp.DenseMLP(_q(width, dims, 23), _q(width, dims, 24), _q(dims, width, 25), limit)
    mx.random.seed(26)
    router = 0.3 * mx.random.normal((experts, dims))
    bias = 0.1 * mx.random.normal((experts,))
    return mlp.MoE(router, bias, gate, up, down, shared, cfg)


@pytest.mark.parametrize("limit", [10.0, 0.0])
def test_prompt_moe_glue_is_the_ops_path(gpu, monkeypatch, limit):
    """A prompt chunk's MoE (fused SwiGLU, the combine through the unsort) and dense MLP equal the MLX ops' bits."""

    moe = _moe(limit)
    for rows in (17, 300, 2048):
        mx.random.seed(rows)
        x = (2.0 * mx.random.normal((rows, 512))).astype(mx.bfloat16)
        monkeypatch.setattr(config, "FUSED", frozenset())
        ref = (moe(x, False), moe.shared(x, False))
        monkeypatch.setattr(config, "FUSED", frozenset({"moe"}))
        got = (moe(x, False), moe.shared(x, False))
        assert _same(got[0], ref[0]) and _same(got[1], ref[1]), (limit, rows)


def test_prompt_hc_boundary_is_the_prompt_path(gpu):
    """At GLM's width the boundary kernels with MLX's GEMM for the mixes give prompt rows the prompt path's bits."""

    from tensorfold.kernels.glm.flash.v1 import hc as F

    cfg = _cfg(4096)
    mx.random.seed(21)
    hc = model.HC(0.05 * mx.random.normal((24, 16384)), 0.3 * mx.random.normal((24,)), mx.array([0.5, 0.5, 0.5]), cfg)
    w = (1 + 0.1 * mx.random.normal((4096,))).astype(mx.bfloat16)
    assert F.hc_fits(hc, 4096)
    for rows in (17, 300, 2048):
        x = mx.random.normal((rows, 4, 4096)).astype(mx.bfloat16)
        branch = mx.random.normal((rows, 4096)).astype(mx.bfloat16)
        post = mx.random.uniform(0, 2, (rows, 4))
        comb = mx.random.uniform(shape=(rows, 4, 4))
        for first in (True, False):
            new = x if first else model.hc_expand(branch, x, post, comb, False)
            xc, po, co = hc.split(new, False)
            ref = (new, mx.fast.rms_norm(xc, w, 1e-5), po, co)
            got = F.hc_step(x, None if first else (branch, post, comb), hc, w, 1e-5, prompt=True)
            assert all(_same(a, b) for a, b in zip(got, ref)), (rows, first)
        last = F.hc_step(x, (branch, post, comb), None, None, 1e-5, prompt=True)[0]
        assert _same(last, model.hc_expand(branch, x, post, comb, False)), rows


def test_prompt_forward_is_the_ops_forward(gpu, tmp_path, monkeypatch):
    """A tiny checkpoint's two prompt chunks: every hidden row and cache array equal with the kernels on and off."""

    from glm5_fakes import write_checkpoint
    from tensorfold.families.glm5_next import weights

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        path = write_checkpoint(tmp_path / "glm5")
    finally:
        mx.set_default_device(previous)
    backbone = weights.load_backbone(path)
    tokens = mx.array([[(37 * i + 11) % 250 + 3 for i in range(300)]], dtype=mx.uint32)
    got = {}
    for fused in (False, True):
        monkeypatch.setattr(config, "FUSED", frozenset(config.FUSED_KERNELS) if fused else frozenset())
        cache = backbone.make_cache()
        first = backbone.hidden(tokens[:, :200], cache)
        second = backbone.hidden(tokens[:, 200:], cache)
        mx.eval(first, second)
        got[fused] = [first, second] + [a for c in cache for a in c.state]
    assert all(_same(a, b) for a, b in zip(got[False], got[True]))


def test_tensor_unit_chips_take_the_ops_prompt_path(gpu, tmp_path, monkeypatch):
    """With tensor units (M5) a prompt takes next-0.6.0's selection: no prompt kernel runs, the rows and cache arrays
    are the ops path's, and decode keeps its fused boundary; the same checkpoint on M1-M4 takes the kernels."""

    from glm5_fakes import write_checkpoint
    from tensorfold.families.glm5_next import weights
    from tensorfold.kernels.glm.flash.v1 import hc as HCK
    from tensorfold.kernels.glm.flash.v1 import prompt as PK
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm as PM

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        path = write_checkpoint(tmp_path / "glm5")
    finally:
        mx.set_default_device(previous)
    backbone = weights.load_backbone(path)
    tokens = mx.array([[(37 * i + 11) % 250 + 3 for i in range(300)]], dtype=mx.uint32)
    calls = []
    for name in ("kda_pre", "kda_post", "scan", "swiglu", "combine", "index_scores"):
        real = getattr(PK, name)
        monkeypatch.setattr(PK, name, lambda *a, _n=name, _f=real, **k: (calls.append(_n), _f(*a, **k))[1])

    def run(fused):
        monkeypatch.setattr(config, "FUSED", frozenset(config.FUSED_KERNELS) if fused else frozenset())
        cache = backbone.make_cache()
        rows = [backbone.hidden(tokens[:, :200], cache), backbone.hidden(tokens[:, 200:], cache)]
        mx.eval(*rows)
        return rows + [a for c in cache for a in c.state]

    def boundary(decode):
        """Which way a boundary goes, the fused HC kernels taken to fit (they take 4,096-wide streams only)."""

        with monkeypatch.context() as m:
            m.setattr(HCK, "hc_step", lambda *a: ("fused", a[-1]))
            m.setattr(backbone, "hc_fused_ok", lambda: True)
            layer = backbone.layers[1]
            x = mx.ones((3, 4, backbone.args.hidden_size), dtype=mx.bfloat16)
            out = backbone.boundary(x, None, layer.attn_hc, layer.in_norm, decode)
            return out if isinstance(out[0], str) else "ops"

    monkeypatch.setattr(PM, "_tensor_units", lambda: False)             # M1-M4: the prompt kernels serve
    run(True)
    assert {"kda_pre", "swiglu"} <= set(calls)
    assert boundary(False) == ("fused", True) and boundary(True) == ("fused", False)
    monkeypatch.setattr(PM, "_tensor_units", lambda: True)              # M5: MLX's matmuls and gathers as well
    monkeypatch.setattr(PM, "_tiles", [False])
    calls.clear()
    ops, got = run(False), run(True)
    assert calls == []
    assert all(_same(a, b) for a, b in zip(ops, got))
    assert boundary(False) == "ops" and boundary(True) == ("fused", False)


@pytest.mark.parametrize("dims", [(64, 128), (2, 64)])
def test_staged_scan_is_mlx_lm_s_kernel(gpu, dims):
    """The staged scan gives mlx-lm's gated delta kernel's outputs and fp32 state bit for bit."""

    from mlx_lm.models import gated_delta as gd

    from tensorfold.kernels.glm.flash.v1 import prompt as PK

    heads, d = dims
    for steps in (17, 300, 2048):
        mx.random.seed(steps)
        q = (0.1 * mx.random.normal((1, steps, heads, d))).astype(mx.bfloat16)
        k = (0.1 * mx.random.normal((1, steps, heads, d))).astype(mx.bfloat16)
        v = mx.random.normal((1, steps, heads, d)).astype(mx.bfloat16)
        g = mx.exp(-mx.random.uniform(0, 0.5, (1, steps, heads, d)))
        beta = mx.random.uniform(shape=(1, steps, heads)).astype(mx.bfloat16)
        state = 0.1 * mx.random.normal((1, heads, d, d))
        assert PK.scan_fits(q, g)
        y1, s1 = gd.gated_delta_kernel(q, k, v, g, beta, state)
        y2, s2 = PK.scan(q, k, v, g, beta, state)
        assert _same(y1, y2) and _same(s1, s2), (dims, steps)


@pytest.mark.parametrize("fmt", [(4, 64), (8, 64)])
def test_prompt_projections_keep_mlx_s_bits(gpu, fmt):
    """GLM's prompt projections through the 64-row-tile qmm (split-K shapes stay MLX's) give MLX's bits."""

    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm as PM

    bits, group = fmt
    shapes = [(24896, 4096), (4096, 8192), (1536, 4096), (128, 4096), (16384, 1536), (4096, 16384), (4096, 2048),
              (8192, 128)]
    for n, k in shapes:
        mx.random.seed(n + k)
        w, s, b = mx.quantize((0.05 * mx.random.normal((n, k))).astype(mx.bfloat16), group_size=group, bits=bits)
        for rows in (64, 300, 2048):
            x = mx.random.normal((rows, k)).astype(mx.bfloat16)
            ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=group, bits=bits)
            assert _same(PM.matmul(x, w, s, b, group=group, bits=bits), ref), (fmt, n, k, rows)


def test_fused_index_scores_are_the_three_ops(gpu):
    """The DSA index scores in one kernel give iq @ pool.T, relu, the weight product and the head sum's bits."""

    from tensorfold.families.glm5_next.mla import MLA
    from tensorfold.kernels.glm.flash.v1 import prompt as PK

    for rows, blocks in ((512, 4096), (512, 513), (17, 700), (100, 16384), (1, 512), (512, 32768)):
        mx.random.seed(rows + blocks)
        iq = (0.3 * mx.random.normal((rows, 32, 128))).astype(mx.bfloat16)
        iw = (mx.random.normal((rows, 32)) * 0.2).astype(mx.bfloat16)
        pool = mx.random.normal((blocks, 128)).astype(mx.bfloat16)
        assert PK.index_fits(iq, pool)
        ref = MLA.index_scores(None, iq, iw, pool)
        assert _same(PK.index_scores(iq, iw, pool), ref), (rows, blocks)
