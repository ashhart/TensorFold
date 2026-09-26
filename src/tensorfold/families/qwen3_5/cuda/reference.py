"""A plain PyTorch forward of Qwen3.8 dense, fp32 arithmetic, to check the kernels against.

It follows mlx_lm's ``qwen3_5`` math: RMSNorm (stored weights), Gated DeltaNet with a depthwise
conv of width 4, q and k RMS-normalised and scaled, g = exp(-exp(A_log) softplus(a + dt_bias)),
beta = sigmoid(b), the delta-rule recurrence in fp32, gated RMSNorm with silu(z); attention with
[q | gate] per head, q/k norms, RoPE on the first rope_dims dims (rotate-half), sigmoid gate;
SwiGLU MLP. Activations are rounded to bf16 where the model stores them. Slow: every call
dequantizes the weights it uses.
"""

from __future__ import annotations

import torch

from tensorfold.families.qwen3_5.cuda.qmm import dequantize
from tensorfold.families.qwen3_5.cuda.weights import QLinear, Weights


def _linear(x: torch.Tensor, q: QLinear) -> torch.Tensor:
    from tensorfold.families.qwen3_5.cuda.qmm_fast import untile

    q = untile(q)
    return (x.float() @ dequantize(q.weight, q.scales, q.biases).T).to(torch.bfloat16)


def _rms(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return y * w.float() if w is not None else y


def _rope(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """x (T, H, D) fp32, rotate-half over the first 2 * len(inv_freq) dims."""

    half = inv_freq.numel()
    ang = pos.float()[:, None] * inv_freq[None, :]            # (T, half)
    cos, sin = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:2 * half]
    out = x.clone()
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:2 * half] = x2 * cos + x1 * sin
    return out


class State:
    """Per-layer state for a sequence: KV for attention layers, conv and recurrent state for GDN."""

    def __init__(self, w: Weights):
        c = w.config
        self.pos = 0
        self.kv: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        self.conv: list[torch.Tensor | None] = []
        self.rec: list[torch.Tensor | None] = []
        dev = w.norm.device
        for layer in w.layers:
            if layer.linear:
                cd = 2 * c.k_heads * c.dk + c.v_heads * c.dv
                self.conv.append(torch.zeros((c.conv_kernel - 1, cd), dtype=torch.bfloat16, device=dev))
                self.rec.append(torch.zeros((c.v_heads, c.dv, c.dk), dtype=torch.float32, device=dev))
                self.kv.append(None)
            else:
                self.conv.append(None)
                self.rec.append(None)
                self.kv.append((torch.zeros((0, c.kv_heads, c.head_dim), dtype=torch.bfloat16, device=dev),
                                torch.zeros((0, c.kv_heads, c.head_dim), dtype=torch.bfloat16, device=dev)))


def _gdn(layer, x: torch.Tensor, st: State, i: int, c) -> torch.Tensor:
    g_ = layer.gdn
    T = x.shape[0]
    qkv = _linear(x, g_.qkv)
    z = _linear(x, g_.z).float().reshape(T, c.v_heads, c.dv)
    b = _linear(x, g_.b).float()
    a = _linear(x, g_.a).float()
    inp = torch.cat([st.conv[i], qkv], 0).float()              # (3 + T, C)
    st.conv[i] = torch.cat([st.conv[i], qkv], 0)[-(c.conv_kernel - 1):].clone()
    w = g_.conv.float()                                        # (C, 4)
    conv = sum(inp[j:j + T] * w[:, j] for j in range(c.conv_kernel))
    conv = torch.nn.functional.silu(conv).to(torch.bfloat16).float()
    kd = c.k_heads * c.dk
    q = conv[:, :kd].reshape(T, c.k_heads, c.dk)
    k = conv[:, kd:2 * kd].reshape(T, c.k_heads, c.dk)
    v = conv[:, 2 * kd:].reshape(T, c.v_heads, c.dv)
    q = (_rms(q, None, 1e-6) * (1.0 / c.dk)).to(torch.bfloat16).float()
    k = (_rms(k, None, 1e-6) * (c.dk ** -0.5)).to(torch.bfloat16).float()
    beta = torch.sigmoid(b)
    g = torch.exp(-torch.exp(g_.A_log) * torch.nn.functional.softplus(a + g_.dt_bias))
    rep = c.v_heads // c.k_heads
    S = st.rec[i]
    ys = []
    for t in range(T):
        qt = q[t].repeat_interleave(rep, 0)
        kt = k[t].repeat_interleave(rep, 0)                   # (Hv, Dk)
        S = S * g[t][:, None, None]
        kv_mem = (S * kt[:, None, :]).sum(-1)
        delta = (v[t] - kv_mem) * beta[t][:, None]
        S = S + kt[:, None, :] * delta[:, :, None]
        ys.append((S * qt[:, None, :]).sum(-1))
    st.rec[i] = S
    y = torch.stack(ys).to(torch.bfloat16)                      # (T, Hv, Dv)
    yn = _rms(y, g_.norm, c.eps)
    out = (torch.nn.functional.silu(z) * yn).to(torch.bfloat16).reshape(T, -1)
    return _linear(out, g_.out)


def _attention(layer, x: torch.Tensor, st: State, i: int, c, inv_freq: torch.Tensor) -> torch.Tensor:
    at = layer.attn
    T = x.shape[0]
    qg = _linear(x, at.q).reshape(T, c.heads, 2 * c.head_dim)
    q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:].reshape(T, -1)
    k = _linear(x, at.k).reshape(T, c.kv_heads, c.head_dim)
    v = _linear(x, at.v).reshape(T, c.kv_heads, c.head_dim)
    pos = torch.arange(st.pos, st.pos + T, device=x.device)
    q = _rope(_rms(q, at.q_norm, c.eps).to(torch.bfloat16).float(), pos, inv_freq).to(torch.bfloat16)
    k = _rope(_rms(k, at.k_norm, c.eps).to(torch.bfloat16).float(), pos, inv_freq).to(torch.bfloat16)
    K0, V0 = st.kv[i]
    K, V = torch.cat([K0, k]), torch.cat([V0, v])
    st.kv[i] = (K, V)
    L = K.shape[0]
    rep = c.heads // c.kv_heads
    Kf = K.float().repeat_interleave(rep, 1)                   # (L, H, D)
    Vf = V.float().repeat_interleave(rep, 1)
    s = torch.einsum("thd,lhd->htl", q.float(), Kf) * c.head_dim ** -0.5
    limit = (st.pos + torch.arange(T, device=x.device))[:, None]
    s = s.masked_fill(torch.arange(L, device=x.device)[None, :] > limit, float("-inf"))
    o = torch.einsum("htl,lhd->thd", torch.softmax(s, -1), Vf).to(torch.bfloat16).reshape(T, -1)
    o = (o.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    return _linear(o, at.o)


@torch.no_grad()
def forward(w: Weights, tokens: torch.Tensor, st: State) -> torch.Tensor:
    """Logits (T, V) fp32 for ``tokens`` continuing the sequence in ``st``."""

    c = w.config
    T = tokens.shape[0]
    e = w.embed
    ids = tokens.long()
    rows = QLinear(e.weight[ids], e.scales[ids], e.biases[ids])
    x = dequantize(rows.weight, rows.scales, rows.biases).to(torch.bfloat16)
    for i, layer in enumerate(w.layers):
        h = _rms(x, layer.input_norm, c.eps).to(torch.bfloat16)
        r = _gdn(layer, h, st, i, c) if layer.linear else _attention(layer, h, st, i, c, w.inv_freq)
        x = (x.float() + r.float()).to(torch.bfloat16)
        h = _rms(x, layer.post_norm, c.eps).to(torch.bfloat16)
        act = (torch.nn.functional.silu(_linear(h, layer.gate).float()) * _linear(h, layer.up).float()).to(torch.bfloat16)
        x = (x.float() + _linear(act, layer.down).float()).to(torch.bfloat16)
    st.pos += T
    h = _rms(x, w.norm, c.eps).to(torch.bfloat16).float()
    from tensorfold.families.qwen3_5.cuda.qmm_fast import untile

    hd = untile(w.head)
    parts = []
    for s0 in range(0, hd.n, 32768):                            # the head in slices: 5 GB as one fp32 matrix
        s1 = min(hd.n, s0 + 32768)
        parts.append(h @ dequantize(hd.weight[s0:s1], hd.scales[s0:s1], hd.biases[s0:s1]).T)
    return torch.cat(parts, dim=1)
