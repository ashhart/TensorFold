"""Kimi K3's text forward in NumPy, written from the official semantics: fp64 sums, bf16 where the model rounds."""
from dataclasses import dataclass, field

import numpy as np

from k3_synth import bf16, mxfp4_decode


@dataclass
class Config:
    """The text model's shape; defaults are Kimi-K3's config.json."""
    hidden: int = 7168
    layers: int = 93
    vocab: int = 163840
    eps: float = 1e-5
    dense_inter: int = 33792
    experts: int = 896
    topk: int = 16
    shared: int = 2
    moe_inter: int = 3072
    latent: int = 3584
    kda_heads: int = 96
    kda_dim: int = 128
    conv: int = 4
    lower_bound: float = -5.0
    mla_heads: int = 96
    q_lora: int = 1536
    kv_lora: int = 512
    nope: int = 128
    rope: int = 64
    v_dim: int = 128
    block: int = 12
    situ_beta: float = 4.0
    situ_linear: float = 25.0
    full_attn: tuple = tuple(range(4, 93, 4)) + (93,)

    def is_kda(self, i: int) -> bool:
        """Layer i (0-based) is KDA unless i + 1 is in the 1-based full-attention list."""
        return (i + 1) not in self.full_attn


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def linear(x, w):
    """bf16 x @ w.T with an exact-as-fp64 sum, rounded to bf16 once (F.linear in bf16)."""
    return bf16(np.asarray(x, np.float64) @ np.asarray(w, np.float64).T)


def rms_norm(x, w, eps):
    """KimiRMSNorm: normalise in fp32, round to bf16, then the bf16 weight times it, rounded again."""
    xf = np.asarray(x, np.float64)
    n = bf16(xf / np.sqrt((xf * xf).mean(-1, keepdims=True) + eps))
    return bf16(np.asarray(w, np.float64) * n)


def situ(gate, up, beta, lin):
    """SituAndMul in fp32: beta tanh(g/beta) sigmoid(g) * lin tanh(u/lin), rounded to bf16."""
    g, u = np.asarray(gate, np.float64), np.asarray(up, np.float64)
    return bf16(beta * np.tanh(g / beta) * sigmoid(g) * (lin * np.tanh(u / lin)))


def mlp(x, wg, wu, wd, cfg):
    return linear(situ(linear(x, wg), linear(x, wu), cfg.situ_beta, cfg.situ_linear), wd)


def route(h, wr, bias, cfg):
    """Sigmoid router: top-k on score + bias (ties to the lower id), weights = scores renormalised (fp32 path)."""
    s = sigmoid(np.asarray(h, np.float64) @ np.asarray(wr, np.float64).T)
    choice = s + np.asarray(bias, np.float64)
    idx = np.argsort(-choice, axis=1, kind="stable")[:, :cfg.topk]
    w = np.take_along_axis(s, idx, 1)
    return idx, w / (w.sum(1, keepdims=True) + 1e-20)


def expert_weights(W, p, e):
    """Routed expert e's dense w1, w3, w2 decoded from MXFP4."""
    return tuple(mxfp4_decode(W[f"{p}experts.{e}.{n}.weight_packed"], W[f"{p}experts.{e}.{n}.weight_scale"])
                 for n in ("w1", "w3", "w2"))


def moe(h, W, p, cfg, decoded=expert_weights):
    """Latent MoE: router on h, experts on the 3584-wide latent, norm and up-projection, plus the shared MLP."""
    idx, wts = route(h, W[p + "gate.weight"], W[p + "gate.e_score_correction_bias"], cfg)
    lat = linear(h, W[p + "routed_expert_down_proj.weight"])
    y = np.zeros((h.shape[0], cfg.latent))
    for r in range(h.shape[0]):
        for j, e in enumerate(idx[r]):
            w1, w3, w2 = decoded(W, p, int(e))
            y[r] += wts[r, j] * mlp(lat[r:r + 1], w1, w3, w2, cfg)[0]
    y = rms_norm(bf16(y), W[p + "routed_expert_norm.weight"], cfg.eps)
    up = linear(y, W[p + "routed_expert_up_proj.weight"])
    sh = mlp(h, *(W[f"{p}shared_experts.{n}_proj.weight"] for n in ("gate", "up", "down")), cfg)
    return bf16(up + sh), idx, wts


@dataclass
class KdaState:
    """A stream's KDA layer: fp32-exact recurrent state [H, K, V] and the last conv-1 projected rows."""
    S: np.ndarray
    conv: np.ndarray


def kda_state(cfg) -> KdaState:
    w = cfg.kda_heads * cfg.kda_dim
    return KdaState(np.zeros((cfg.kda_heads, cfg.kda_dim, cfg.kda_dim)), np.zeros((cfg.conv - 1, 3 * w), np.float32))


def kda(x, W, p, cfg, st: KdaState, trace=None):
    """KDA over one stream's rows in order: short conv + silu, l2-normed q/k, gated delta rule, gated norm."""
    H, D = cfg.kda_heads, cfg.kda_dim
    R = x.shape[0]
    qkv = np.concatenate([linear(x, W[p + n + "_proj.weight"]) for n in "qkv"], 1)
    hist = np.concatenate([st.conv, qkv], 0).astype(np.float64)
    cw = np.concatenate([W[p + n + "_conv1d.weight"].reshape(H * D, cfg.conv) for n in "qkv"], 0)
    acc = sum(hist[j:j + R] * cw[:, j] for j in range(cfg.conv))
    act = bf16(acc * sigmoid(acc)).astype(np.float64)
    st.conv = hist[R:].astype(np.float32)
    f = linear(linear(x, W[p + "f_a_proj.weight"]), W[p + "f_b_proj.weight"]).reshape(R, H, D)
    beta = sigmoid(linear(x, W[p + "b_proj.weight"]).astype(np.float64))
    a = np.exp(np.asarray(W[p + "A_log"], np.float64)[:H])
    gk = cfg.lower_bound * sigmoid(a[None, :, None] * (f + np.asarray(W[p + "dt_bias"], np.float64).reshape(H, D)))
    q, k, v = (act[:, i * H * D:(i + 1) * H * D].reshape(R, H, D) for i in range(3))
    o = np.zeros((R, H, D))
    for t in range(R):
        qn = q[t] / np.sqrt((q[t] ** 2).sum(-1, keepdims=True) + 1e-6) * D ** -0.5
        kn = k[t] / np.sqrt((k[t] ** 2).sum(-1, keepdims=True) + 1e-6)
        st.S *= np.exp(gk[t])[:, :, None]
        u = (v[t] - np.einsum("hkv,hk->hv", st.S, kn)) * beta[t][:, None]
        st.S += kn[:, :, None] * u[:, None, :]
        o[t] = np.einsum("hkv,hk->hv", st.S, qn)
    o = bf16(o).astype(np.float64)
    g = linear(x, W[p + "g_proj.weight"]).reshape(R, H, D).astype(np.float64)
    y = o / np.sqrt((o * o).mean(-1, keepdims=True) + cfg.eps) * np.asarray(W[p + "o_norm.weight"], np.float64)
    y = bf16(y * sigmoid(g)).reshape(R, H * D)
    if trace is not None:
        trace.update(qkv=qkv, act=act, f=f, beta=beta, gk=gk, o=o, g=g, y=y)
    return linear(y, W[p + "o_proj.weight"])


def mla_project(x, W, p, cfg):
    """MLA's per-row projections: q heads [R, H, nope+rope] and the cached latent [R, kv_lora + rope]."""
    qa = rms_norm(linear(x, W[p + "q_a_proj.weight"]), W[p + "q_a_layernorm.weight"], cfg.eps)
    q = linear(qa, W[p + "q_b_proj.weight"]).reshape(x.shape[0], cfg.mla_heads, cfg.nope + cfg.rope)
    ckv = linear(x, W[p + "kv_a_proj_with_mqa.weight"])
    c = rms_norm(ckv[:, :cfg.kv_lora], W[p + "kv_a_layernorm.weight"], cfg.eps)
    return q, np.concatenate([c, ckv[:, cfg.kv_lora:]], 1)


def mla(x, W, p, cfg, cache: list, trace=None):
    """MLA (NoPE, output gate) in absorbed form: q into the latent space, attend over cached latents, then W_uv."""
    H, L, N = cfg.mla_heads, cfg.kv_lora, cfg.nope
    q, lat = mla_project(x, W, p, cfg)
    wkv = np.asarray(W[p + "kv_b_proj.weight"], np.float64).reshape(H, N + cfg.v_dim, L)
    out = np.zeros((x.shape[0], H, cfg.v_dim))
    scale = (N + cfg.rope) ** -0.5
    for t in range(x.shape[0]):
        cache.append(lat[t])
        keys = np.asarray(cache, np.float64)
        qlat = np.einsum("hn,hnl->hl", q[t, :, :N].astype(np.float64), wkv[:, :N])
        s = (qlat @ keys[:, :L].T + q[t, :, N:].astype(np.float64) @ keys[:, L:].T) * scale
        pr = np.exp(s - s.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        out[t] = np.einsum("hl,hvl->hv", pr @ keys[:, :L], wkv[:, N:])
    attn = bf16(out).reshape(x.shape[0], H * cfg.v_dim)
    gate = bf16(sigmoid(linear(x, W[p + "g_proj.weight"]).astype(np.float64)))
    if trace is not None:
        trace.update(q=q, lat=lat, attn=attn, gate=gate)
    return linear(bf16(attn * gate), W[p + "o_proj.weight"])


def mla_plain(x, W, p, cfg, cache: list):
    """MLA as Moonshot's file computes it (bf16 K and V per head, fp32 scores): the check for the absorbed form."""
    H, L, N, V = cfg.mla_heads, cfg.kv_lora, cfg.nope, cfg.v_dim
    q, lat = mla_project(x, W, p, cfg)
    out = np.zeros((x.shape[0], H, V))
    for t in range(x.shape[0]):
        cache.append(lat[t])
        keys = np.asarray(cache, np.float64)
        kv = linear(keys[:, :L], W[p + "kv_b_proj.weight"]).reshape(-1, H, N + V).astype(np.float64)
        for h in range(H):
            s = (q[t, h, :N] @ kv[:, h, :N].T + q[t, h, N:] @ keys[:, L:].T) * (N + cfg.rope) ** -0.5
            pr = np.exp(s - s.max())
            out[t, h] = (pr / pr.sum()) @ kv[:, h, N:]
    gate = bf16(sigmoid(linear(x, W[p + "g_proj.weight"]).astype(np.float64)))
    return linear(bf16(bf16(out).reshape(x.shape[0], H * V) * gate), W[p + "o_proj.weight"])


def attn_res(prefix, blocks, w_norm, w_proj, eps):
    """Attention residual: softmax over [blocks..., prefix] of RMS-normed scores against norm.w * proj.w."""
    v = np.stack([np.asarray(b, np.float64) for b in blocks] + [np.asarray(prefix, np.float64)], 1)
    k = v / np.sqrt((v * v).mean(-1, keepdims=True) + eps)
    sw = np.asarray(w_norm, np.float64) * np.asarray(w_proj, np.float64).reshape(-1)
    s = (k * sw).sum(-1)
    pr = np.exp(s - s.max(-1, keepdims=True))
    pr /= pr.sum(-1, keepdims=True)
    return bf16((pr[:, :, None] * v).sum(1))


@dataclass
class Stream:
    """One stream's caches across layers: KDA states and MLA latent lists."""
    kda: dict = field(default_factory=dict)
    mla: dict = field(default_factory=dict)


def layer(i, prefix, blocks, W, cfg, stream: Stream, trace=None, decoded=expert_weights):
    """One decoder layer with attention residuals; rows are one stream's tokens in order."""
    p = f"layers.{i}."
    t = {} if trace is None else trace
    hs = attn_res(prefix, blocks, W[p + "self_attention_res_norm.weight"], W[p + "self_attention_res_proj.weight"],
                  cfg.eps) if blocks else prefix
    if i % cfg.block == 0:
        blocks, prefix = blocks + [prefix], None
    x = rms_norm(hs, W[p + "input_layernorm.weight"], cfg.eps)
    if cfg.is_kda(i):
        a = kda(x, W, p + "self_attn.", cfg, stream.kda.setdefault(i, kda_state(cfg)), t)
    else:
        a = mla(x, W, p + "self_attn.", cfg, stream.mla.setdefault(i, []), t)
    prefix = a if prefix is None else bf16(prefix + a)
    hs = attn_res(prefix, blocks, W[p + "mlp_res_norm.weight"], W[p + "mlp_res_proj.weight"], cfg.eps)
    h = rms_norm(hs, W[p + "post_attention_layernorm.weight"], cfg.eps)
    if i == 0:
        m = mlp(h, *(W[f"{p}mlp.{n}_proj.weight"] for n in ("gate", "up", "down")), cfg)
    else:
        m, t["ids"], t["weights"] = moe(h, W, p + "block_sparse_moe.", cfg, decoded)
    t.update(attn_in=x, attn=a, prefix=prefix, mlp_in=h, mlp=m)
    return bf16(prefix + m), blocks


def head(prefix, blocks, W, cfg):
    """Output attention residual, final norm and the LM head: bf16 logits."""
    h = attn_res(prefix, blocks, W["output_attn_res_norm.weight"], W["output_attn_res_proj.weight"], cfg.eps)
    return linear(rms_norm(h, W["norm.weight"], cfg.eps), W["lm_head.weight"])


def forward(tokens, W, cfg, stream: Stream, layers=None):
    """Logits for one stream's next tokens given its caches (prefill and decode alike)."""
    prefix, blocks = W["embed_tokens.weight"][np.asarray(tokens)], []
    for i in range(cfg.layers if layers is None else layers):
        prefix, blocks = layer(i, prefix, blocks, W, cfg, stream)
    return head(prefix, blocks, W, cfg)
