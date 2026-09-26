"""Gemma 4 fused decode kernels (GPU): each kernel is its MLX op sequence to bf16 rounding, and a tiny random 4-bit
Gemma 4 decodes through FusedDecode as mlx_lm's forward does (same greedy tokens, logits to bf16 noise)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
gemma4_text = pytest.importorskip("mlx_lm.models.gemma4_text")

if not mx.metal.is_available():
    pytest.skip("the fused kernels are Metal kernels", allow_module_level=True)

from tensorfold.kernels.gemma.v1 import kernels as K  # noqa: E402

EPS = mx.array([1e-6], dtype=mx.float32)


def bf(shape, seed, scale=1.0, shift=0.0):
    mx.random.seed(seed)
    return (mx.random.normal(shape) * scale + shift).astype(mx.bfloat16)


def close(a, b, rel=0.02):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    return np.abs(a - b).max() <= rel * max(np.abs(b).max(), 1e-3)


def rms(x, w=None):
    return mx.fast.rms_norm(x, w, 1e-6)


@pytest.mark.parametrize("values_are_keys", [False, True])
def test_qkv_norm_is_the_three_head_norms(values_are_keys):
    heads, kv, dh = 4, 2, 64
    width = (heads + kv + (0 if values_are_keys else kv)) * dh
    qkv, qw, kw = bf((3, width), 1), bf((dh,), 2, 0.1, 1.0), bf((dh,), 3, 0.1, 1.0)
    q, k, v = K.qkv_norm(qkv, qw, kw, EPS, heads=heads, kv_heads=kv, head_dim=dh, values_are_keys=values_are_keys)
    rq = qkv[:, :heads * dh].reshape(3, heads, dh)
    rk = qkv[:, heads * dh:(heads + kv) * dh].reshape(3, kv, dh)
    rv = rk if values_are_keys else qkv[:, (heads + kv) * dh:].reshape(3, kv, dh)
    assert close(q, rms(rq, qw)) and close(k, rms(rk, kw)) and close(v, rms(rv))


def test_attn_tail_is_norm_add_and_three_norms():
    d = 512
    h, o = bf((2, d), 4), bf((2, d), 5, 3.0)
    wa, w1, w2, w3 = (bf((d,), s, 0.1, 1.0) for s in (6, 7, 8, 9))
    hn, n1, n2, n3 = K.attn_tail(h, o, wa, w1, w2, w3, EPS)
    ref = h + rms(o, wa)
    assert close(hn, ref) and close(n1, rms(ref, w1)) and close(n2, rms(ref, w2)) and close(n3, rms(ref, w3))


def test_route_picks_the_top_k_with_softmax_weights():
    scores, scale = bf((3, 128), 10, 2.0), bf((128,), 11, 0.1, 1.0)
    ids, weights = K.route(scores, scale, 8)
    for r in range(3):
        s = np.array(scores[r].astype(mx.float32))
        top = np.argsort(-s, kind="stable")[:8]
        assert list(np.array(ids[r])) == list(top)
        p = np.exp(s[top] - s[top].max())
        p /= p.sum()
        want = p * np.array(scale.astype(mx.float32))[top]
        assert np.allclose(np.array(weights[r].astype(mx.float32)), want, rtol=0.02, atol=1e-3)


def test_moe_tail_is_the_three_post_norms_residual_scalar_and_next_norm():
    d = 512
    h, y1, y2 = bf((2, d), 12), bf((2, d), 13, 2.0), bf((2, d), 14, 0.5)
    w1, w2, wp, wn = (bf((d,), s, 0.1, 1.0) for s in (15, 16, 17, 18))
    scalar = mx.array([0.75], dtype=mx.bfloat16)
    hn, nxt = K.moe_tail(h, y1, y2, w1, w2, wp, scalar, wn, EPS)
    ref = (h + rms(rms(y1, w1) + rms(y2, w2), wp)) * scalar
    assert close(hn, ref) and close(nxt, rms(ref, wn))


def _switch(experts, n_in, n_out, seed):
    lin = nn.QuantizedLinear(n_in, n_out, bias=False, group_size=64, bits=4)
    mx.random.seed(seed)
    w = mx.random.normal((experts, n_out, n_in)) * 0.05
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    lin.weight, lin.scales, lin.biases = q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    return lin


def test_expert_kernels_are_the_gated_experts_and_their_weighted_sum():
    experts, d, width, top = 16, 2816, 704, 8           # Gemma 4 26B-A4B's shapes: 2,816 is not a multiple of 512
    gate, up, down = _switch(experts, d, width, 20), _switch(experts, d, width, 21), _switch(experts, width, d, 22)
    x = bf((2, d), 23)
    ids = mx.array([[3, 0, 15, 7, 9, 1, 12, 4], [5, 5, 2, 8, 11, 14, 6, 10]], dtype=mx.uint32)   # a repeat too
    weights = mx.softmax(bf((2, top), 24), axis=-1).astype(mx.bfloat16)
    act = K.expert_gateup(x, ids, gate, up)
    out = K.expert_down(act, ids, weights, down)

    def deq(lin, e):
        return mx.dequantize(lin.weight[e], lin.scales[e], lin.biases[e], group_size=64, bits=4)

    xf = x.astype(mx.float32)
    for r in range(2):
        total = mx.zeros((d,))
        for k in range(top):
            e = int(ids[r, k].item())
            a = nn.gelu_approx(xf[r] @ deq(gate, e).T) * (xf[r] @ deq(up, e).T)
            assert close(act[r * top + k], a, rel=0.03)
            total = total + weights[r, k].astype(mx.float32) * (a @ deq(down, e).T)
        assert close(out[r], total, rel=0.03)


TINY = {
    "model_type": "gemma4_text", "hidden_size": 128, "num_hidden_layers": 4, "intermediate_size": 128,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64, "global_head_dim": 128,
    "num_global_key_value_heads": 1, "attention_k_eq_v": True, "vocab_size": 97, "vocab_size_per_layer_input": 97,
    "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0, "sliding_window": 8,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "enable_moe_block": True, "num_experts": 32, "top_k_experts": 4, "moe_intermediate_size": 512,
    "use_double_wide_mlp": False, "final_logit_softcapping": 30.0, "tie_word_embeddings": True,
}


def tiny_text():
    mx.random.seed(0)
    text = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(TINY))
    nn.quantize(text, group_size=64, bits=4)
    for layer in text.model.layers:                     # a non-trivial router scale and layer scalar
        layer.router.per_expert_scale = (mx.random.uniform(shape=(32,)) + 0.5).astype(mx.bfloat16)
        layer.layer_scalar = mx.array([0.8], dtype=mx.bfloat16)
    text.set_dtype(mx.bfloat16)
    mx.eval(text.parameters())
    return text


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_each_fused_layer_is_mlx_lm_s_layer(seed):
    """From the same attention output and residual, a fused layer's output is mlx_lm's to bf16 rounding with the
    same experts. (End to end, a random tiny model amplifies bf16 rounding ~3x a layer, so the whole-model check
    lives in tools/gemma4_truth_eval.py on the real weights.)"""

    text = tiny_text()
    fused = K.FusedDecode(text)
    mx.random.seed(100 + seed)
    for i, layer in enumerate(text.model.layers):
        attn = layer.self_attn
        h = (mx.random.normal((1, TINY["hidden_size"])) * 2).astype(mx.bfloat16)
        out = mx.random.normal((1, attn.n_heads * attn.head_dim)).astype(mx.bfloat16)
        got, _ = fused._back(i)(out, h)
        ha = h + layer.post_attention_layernorm(attn.o_proj(out))
        dense = layer.post_feedforward_layernorm_1(layer.mlp(layer.pre_feedforward_layernorm(ha)))
        ids, weights = layer.router(ha)
        routed = layer.post_feedforward_layernorm_2(layer.experts(layer.pre_feedforward_layernorm_2(ha), ids, weights))
        want = (ha + layer.post_feedforward_layernorm(dense + routed)) * layer.layer_scalar
        assert close(got, want, rel=0.02), i
        mine, _ = K.route(layer.router.proj(mx.fast.rms_norm(ha, fused.router_norm[i], 1e-6)),
                          layer.router.per_expert_scale, TINY["top_k_experts"])
        assert sorted(np.array(mine[0]).tolist()) == sorted(np.array(ids[0]).tolist()), i


def test_fused_decode_steps_advance_the_caches_like_mlx_lm(monkeypatch):
    from tensorfold.families.gemma4.model import Gemma4

    text = tiny_text()
    monkeypatch.setenv("TF_GEMMA4_FUSED", "0")
    plain = Gemma4(text)
    monkeypatch.setenv("TF_GEMMA4_FUSED", "1")
    fused = Gemma4(text)
    assert fused.fused is not None and plain.fused is None
    prompt = [int(t) for t in np.random.default_rng(3).integers(6, 97, size=6)]
    caches = plain.make_cache(), fused.make_cache()
    for model, cache in zip((plain, fused), caches):
        mx.eval(model(mx.array([prompt]), cache))
        for t in range(10, 24):                          # past the 8-token sliding window: the caches rotate
            logits = model(mx.array([[t]]), cache)
            assert logits.shape == (1, 1, TINY["vocab_size"])
            mx.eval(logits)
    assert [c.offset for c in caches[0]] == [c.offset for c in caches[1]]
