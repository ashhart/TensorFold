"""Gemma 4 decode kernels (Metal): each is its MLX ops to bf16 rounding, row-exact, with one Metal signature."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
gemma4_text = pytest.importorskip("mlx_lm.models.gemma4_text")

if not mx.metal.is_available():
    pytest.skip("the Gemma decode kernels are Metal kernels", allow_module_level=True)

from gemma4_tiny import TINY, tiny_text  # noqa: E402
from kernel_signatures import changed, recording  # noqa: E402
from tensorfold.kernels.gemma.v1 import attention, glue, moe  # noqa: E402
from tensorfold.kernels.gemma.v1.decode import RowDecode, inverse_frequencies  # noqa: E402
from tensorfold.kernels.inputs import ints  # noqa: E402

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
@pytest.mark.parametrize("head_dim", [256, 512])
def test_qkv_prep_is_the_head_norms_and_rope_at_each_rows_position(values_are_keys, head_dim):
    from mlx_lm.models.rope_utils import initialize_rope

    heads, kv = 4, 2
    width = (heads + kv + (0 if values_are_keys else kv)) * head_dim
    positions = [5, 1023, 70001]
    qkv, qw, kw = bf((3, width), 1), bf((head_dim,), 2, 0.1, 1.0), bf((head_dim,), 3, 0.1, 1.0)
    scaling = ({"rope_type": "proportional", "partial_rotary_factor": 0.25} if head_dim == 512
               else {"rope_type": "default"})
    rope = initialize_rope(head_dim, 10_000.0 if head_dim == 256 else 1_000_000.0, False, scaling, 262_144)
    q, k, v = glue.qkv_prep(qkv, qw, kw, inverse_frequencies(rope, head_dim), ints(positions), EPS, heads=heads,
                            kv_heads=kv, head_dim=head_dim, values_are_keys=values_are_keys)
    rq = qkv[:, :heads * head_dim].reshape(3, heads, head_dim)
    rk = qkv[:, heads * head_dim:(heads + kv) * head_dim].reshape(3, kv, head_dim)
    rv = rk if values_are_keys else qkv[:, (heads + kv) * head_dim:].reshape(3, kv, head_dim)
    for r, p in enumerate(positions):
        want_q = rope(rms(rq[r], qw)[None, :, None], offset=p)[0, :, 0]      # mlx_lm's [B, H, L, D]
        want_k = rope(rms(rk[r], kw)[None, :, None], offset=p)[0, :, 0]
        assert close(q[r], want_q, 0.01) and close(k[:, r], want_k, 0.01) and close(v[:, r], rms(rv[r]))


@pytest.mark.parametrize("values_are_keys", [False, True])
@pytest.mark.parametrize("head_dim", [256, 512])
def test_qkv_rows_is_the_row_matvec_then_qkv_prep(values_are_keys, head_dim):
    from tensorfold.kernels.nemotron.lightning.v1 import rows as row_kernels

    heads, kv, dims = 4, 2, 2816
    width = (heads + kv + (0 if values_are_keys else kv)) * head_dim
    mx.random.seed(40)
    weight, scales, biases = mx.quantize(mx.random.normal((width, dims)) * 0.05, group_size=64, bits=4)
    scales, biases = scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)
    x, qw, kw = bf((3, dims), 41), bf((head_dim,), 42, 0.1, 1.0), bf((head_dim,), 43, 0.1, 1.0)
    inv = mx.array(np.linspace(1.0, 1e-4, head_dim // 2, dtype=np.float32))
    shape = dict(heads=heads, kv_heads=kv, head_dim=head_dim, values_are_keys=values_are_keys)
    at = ints([7, 900, 40000])
    fused = glue.qkv_rows(x, weight, scales, biases, 64, qw, kw, inv, at, EPS, **shape)
    apart = glue.qkv_prep(row_kernels.qmv(x, weight, scales, biases, 64), qw, kw, inv, at, EPS, **shape)
    for a, b in zip(fused, apart):
        assert bool(mx.array_equal(a, b).item())


def test_attn_tail_is_norm_add_and_three_norms():
    d = 512
    h, o = bf((2, d), 4), bf((2, d), 5, 3.0)
    wa, w1, w2, w3 = (bf((d,), s, 0.1, 1.0) for s in (6, 7, 8, 9))
    hn, n1, n2, n3 = glue.attn_tail(h, o, wa, w1, w2, w3, EPS)
    ref = h + rms(o, wa)
    assert close(hn, ref) and close(n1, rms(ref, w1)) and close(n2, rms(ref, w2)) and close(n3, rms(ref, w3))


def test_moe_tail_is_the_three_post_norms_residual_scalar_and_next_norm():
    d = 512
    h, y1, y2 = bf((2, d), 12), bf((2, d), 13, 2.0), bf((2, d), 14, 0.5)
    w1, w2, wp, wn = (bf((d,), s, 0.1, 1.0) for s in (15, 16, 17, 18))
    scalar = mx.array([0.75], dtype=mx.bfloat16)
    hn, nxt = glue.moe_tail(h, y1, y2, w1, w2, wp, scalar, wn, EPS)
    ref = (h + rms(rms(y1, w1) + rms(y2, w2), wp)) * scalar
    assert close(hn, ref) and close(nxt, rms(ref, wn))


def test_route_picks_the_top_k_with_softmax_weights():
    scores, scale = bf((3, 128), 10, 2.0), bf((128,), 11, 0.1, 1.0)
    ids, weights = moe.route(scores, scale, 8)
    for r in range(3):
        s = np.array(scores[r].astype(mx.float32))
        top = np.argsort(-s, kind="stable")[:8]
        assert list(np.array(ids[8 * r:8 * r + 8])) == list(top)
        p = np.exp(s[top] - s[top].max())
        p /= p.sum()
        want = p * np.array(scale.astype(mx.float32))[top]
        assert np.allclose(np.array(weights[8 * r:8 * r + 8].astype(mx.float32)), want, rtol=0.02, atol=1e-3)


def test_router_logits_is_the_8_bit_matvec():
    proj = nn.QuantizedLinear(2816, 128, bias=False, group_size=64, bits=8)
    w = mx.random.normal((128, 2816), key=mx.random.key(3)) * 0.05
    proj.weight, scales, biases = mx.quantize(w, group_size=64, bits=8)
    proj.scales, proj.biases = scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)
    x = bf((5, 2816), 4)
    assert close(moe.router_logits(x, proj), proj(x), 0.01)


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
    act = moe.expert_gateup(x, ids.reshape(-1), top, gate, up)
    out = moe.expert_down(act, ids.reshape(-1), weights.reshape(-1), top, down)

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


def _ring(heads, slots, dims, seed):
    return bf((1, heads, slots, dims), seed), bf((1, heads, slots, dims), seed + 1)


@pytest.mark.parametrize("dims, heads, kv_heads, window, ring", [(256, 16, 8, 1024, 1152), (512, 16, 2, 0, 0),
                                                                 (64, 2, 1, 8, 136)])
def test_attention_is_softmax_over_each_rows_keys_and_rows_are_independent(dims, heads, kv_heads, window, ring):
    top = 1500 if window else 700
    slots = ring or 768
    keys, values = _ring(kv_heads, slots, dims, 30)
    positions = [top - 6 + r for r in range(6)]
    q = bf((6, heads, dims), 32)

    def own(rows):                                     # the rows' own keys and values, as the q|k|v kernel gives them
        at = mx.array([pos % (ring or slots) for pos in rows])
        return keys[0][:, at], values[0][:, at]

    out = attention.attend(q, keys, values, attention.Rows(positions, window, ring, dims), *own(positions))
    group = heads // kv_heads
    for r, p in enumerate(positions):
        lo = max(0, p - window + 1) if window else 0
        at = [pos % (ring or slots) for pos in range(lo, p + 1)]
        for h in range(heads):
            k = keys[0, h // group][mx.array(at)].astype(mx.float32)
            v = values[0, h // group][mx.array(at)].astype(mx.float32)
            want = mx.softmax(k @ q[r, h].astype(mx.float32), axis=-1) @ v
            assert close(out[r, h], want, 0.02), (r, h)
        alone = attention.attend(q[r:r + 1], keys, values, attention.Rows([p], window, ring, dims), *own([p]))
        assert bool(mx.array_equal(alone[0], out[r]).item()), r


def test_each_kernel_keeps_one_metal_signature_at_every_row_count(monkeypatch):
    """MLX 0.31 recompiles a kernel whose input crosses 8 elements, which can drop a queued dispatch (mlx#3662)."""

    from tensorfold.families.gemma4.cache import make_cache
    from tensorfold.kernels.gemma.v1.base import Kernel
    from tensorfold.kernels.nemotron.lightning.v1 import rows

    for module in (attention, glue, moe):             # kernels built earlier would not pass the recorder
        for value in vars(module).values():
            if isinstance(value, Kernel):
                monkeypatch.setattr(value, "compiled", {})
    monkeypatch.setattr(rows, "_kernels", {})
    text = tiny_text()
    decode = RowDecode(text, "rows")
    with recording() as seen:
        for rows in (1, 2, 3, 5, 8, 9, 13):
            cache = make_cache(text)
            mx.eval(decode.logits(decode(mx.array(list(range(10, 10 + rows)), dtype=mx.uint32), [(cache, rows, 0)])))
            assert cache[0].offset == rows
    assert seen and not changed(seen), changed(seen)


@pytest.mark.parametrize("seed", [0, 1])
def test_each_fused_layer_is_mlx_lm_s_layer(seed):
    """A fused layer is mlx_lm's to bf16 rounding on the experts it picked (its top k breaks ties to the lower id)."""

    text = tiny_text()
    decode = RowDecode(text, "rows")
    top = TINY["top_k_experts"]
    mx.random.seed(100 + seed)
    for i, layer in enumerate(text.model.layers):
        attn = layer.self_attn
        h = (mx.random.normal((1, TINY["hidden_size"])) * 2).astype(mx.bfloat16)
        out = mx.random.normal((1, attn.n_heads * attn.head_dim)).astype(mx.bfloat16)
        got, _ = decode._back(i)(out, h)
        ha = h + layer.post_attention_layernorm(attn.o_proj(out))
        normed = mx.fast.rms_norm(ha, decode.router_norm[i], 1e-6)
        logits = moe.router_logits(normed, layer.router.proj)
        assert bool(mx.array_equal(logits, layer.router.proj(normed)).item()), i
        ids, weights = moe.route(logits, layer.router.per_expert_scale, top)
        theirs, _ = layer.router(ha)
        scores = np.array(logits[0].astype(mx.float32))
        assert np.sort(scores[np.array(ids[:top])])[0] == np.sort(scores[np.array(theirs[0])])[0], i
        dense = layer.post_feedforward_layernorm_1(layer.mlp(layer.pre_feedforward_layernorm(ha)))
        routed = layer.post_feedforward_layernorm_2(layer.experts(layer.pre_feedforward_layernorm_2(ha),
                                                                  ids[:top][None], weights[:top][None]))
        want = (ha + layer.post_feedforward_layernorm(dense + routed)) * layer.layer_scalar
        assert close(got, want, rel=0.02), i
