"""fp32 PyTorch forward from the kernels' own 4-bit tables, so its only gap to the kernels is their bf16 activations."""

from __future__ import annotations

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.families.qwen3_5.cuda.qmm import dequantize
from tensorfold.families.qwen3_5.cuda.qmm_fast import untile

from .weights import MoE, Weights


def _dense(x: torch.Tensor, q) -> torch.Tensor:
    q = untile(q)
    return x @ dequantize(q.weight, q.scales, q.biases).T


def _expert(table: torch.Tensor, e: int, gs: int) -> torch.Tensor:
    """Expert e of one matrix of a shared-kernel table ([E, N/32, K/gs, 1, words]) as fp32 (N, K)."""

    words, scales, biases = grouped.unpack(table[e:e + 1, :, :, 0], gs)
    return dequantize(words[0], scales[0], biases[0])


def _rms(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * w.float() if w is not None else y


def route(logits: torch.Tensor, bias: torch.Tensor, top_k: int, scaling: float, norm: bool):
    """Top-k by sigmoid score + bias (ties to the lower expert id); weights from the plain scores."""

    score = torch.sigmoid(logits)
    sel = score + bias
    order = torch.sort(sel, dim=-1, descending=True, stable=True).indices[:, :top_k]
    w = torch.gather(score, 1, order)
    if norm:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return order, w * scaling


def moe(m: MoE, x: torch.Tensor, top_k: int, scaling: float, norm: bool) -> torch.Tensor:
    logits = x @ m.router.float().T
    idx, wts = route(logits, m.bias, top_k, scaling, norm)
    out = torch.zeros_like(x)
    ex = m.experts
    experts = ex.count - 2                # the last two are the shared expert's halves
    for e in list(torch.unique(idx).tolist()) + [experts, experts + 1]:
        if e < experts:
            rows, slot = (idx == e).nonzero(as_tuple=True)
            weight = wts[rows, slot][:, None]
        else:
            rows = torch.arange(x.shape[0], device=x.device)
            weight = torch.ones((x.shape[0], 1), device=x.device)
        up = x[rows] @ _expert(ex.up, e, ex.gs).T
        act = torch.relu(up).pow(2)
        out.index_add_(0, rows, weight * (act @ _expert(ex.down, e, ex.gs).T))
    return out


def mamba(m, x: torch.Tensor, c) -> torch.Tensor:
    T = x.shape[0]
    proj = _dense(x, m.in_proj)
    z, xbc, dtr = proj[:, :c.xd], proj[:, c.xd:c.xd + c.conv_dim], proj[:, c.xd + c.conv_dim:]
    kc = c.conv_kernel
    padded = torch.cat([torch.zeros((kc - 1, c.conv_dim), device=x.device), xbc])
    conv = m.conv_b + sum(m.conv_w[k] * padded[k:k + T] for k in range(kc))
    conv = torch.nn.functional.silu(conv)
    xs = conv[:, :c.xd].reshape(T, c.m_heads, c.m_head_dim)
    gs = c.m_groups * c.m_state
    B = conv[:, c.xd:c.xd + gs].reshape(T, c.m_groups, c.m_state)
    C = conv[:, c.xd + gs:].reshape(T, c.m_groups, c.m_state)
    dt = torch.nn.functional.softplus(dtr + m.dt_bias).clamp(c.dt_min, c.dt_max)
    rep = c.m_heads // c.m_groups
    state = torch.zeros((c.m_heads, c.m_head_dim, c.m_state), device=x.device)
    ys = []
    for t in range(T):
        Bt = B[t].repeat_interleave(rep, 0)                           # (H, DS)
        Ct = C[t].repeat_interleave(rep, 0)
        state = state * torch.exp(m.a * dt[t])[:, None, None] + (xs[t] * dt[t][:, None])[:, :, None] * Bt[:, None, :]
        ys.append((state * Ct[:, None, :]).sum(-1) + m.d[:, None] * xs[t])
    y = torch.stack(ys).reshape(T, c.xd) * torch.nn.functional.silu(z)
    y = _rms(y.reshape(T, c.m_groups, -1), None, c.eps).reshape(T, c.xd) * m.gnorm.float()
    return _dense(y, m.out_proj)


def attention(a, x: torch.Tensor, c, cache: list | None = None) -> torch.Tensor:
    T = x.shape[0]
    qkv = _dense(x, a.qkv)
    qd, kd = c.heads * c.head_dim, c.kv_heads * c.head_dim
    q = qkv[:, :qd].reshape(T, c.heads, c.head_dim)
    k = qkv[:, qd:qd + kd].reshape(T, c.kv_heads, c.head_dim)
    v = qkv[:, qd + kd:].reshape(T, c.kv_heads, c.head_dim)
    past = 0
    if cache is not None:
        if cache:
            past = cache[0].shape[0]
            k, v = torch.cat([cache[0], k]), torch.cat([cache[1], v])
        cache[:] = [k, v]
    rep = c.heads // c.kv_heads
    kf, vf = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = torch.einsum("thd,lhd->htl", q, kf) * c.head_dim ** -0.5
    L = k.shape[0]
    mask = torch.arange(L, device=x.device)[None, :] > (past + torch.arange(T, device=x.device))[:, None]
    s = s.masked_fill(mask, float("-inf"))
    o = torch.einsum("htl,lhd->thd", torch.softmax(s, -1), vf).reshape(T, qd)
    return _dense(o, a.o)


def embed(w: Weights, tokens: torch.Tensor) -> torch.Tensor:
    e = w.embed
    ids = tokens.long()
    return dequantize(e.weight[ids], e.scales[ids], e.biases[ids])


@torch.no_grad()
def hidden(w: Weights, tokens: torch.Tensor) -> torch.Tensor:
    """The final normed hidden states (T, D) fp32 of a whole sequence from an empty state."""

    c = w.config
    x = embed(w, tokens)
    for blk in w.blocks:
        h = _rms(x, blk.norm, c.eps)
        if blk.kind == "M":
            x = x + mamba(blk.mamba, h, c)
        elif blk.kind == "*":
            x = x + attention(blk.attn, h, c)
        else:
            x = x + moe(blk.moe, h, c.top_k, c.scaling, c.norm_topk)
    return _rms(x, w.norm_f, c.eps)


@torch.no_grad()
def logits(w: Weights, h: torch.Tensor, rows_per_slice: int = 32768) -> torch.Tensor:
    head = untile(w.head)
    parts = []
    for s0 in range(0, head.n, rows_per_slice):
        s1 = min(head.n, s0 + rows_per_slice)
        parts.append(h @ dequantize(head.weight[s0:s1], head.scales[s0:s1], head.biases[s0:s1]).T)
    return torch.cat(parts, dim=1)


@torch.no_grad()
def forward(w: Weights, tokens: torch.Tensor) -> torch.Tensor:
    """Logits (T, V) fp32 for ``tokens`` from an empty state."""

    return logits(w, hidden(w, tokens))


@torch.no_grad()
def mtp_logits(w: Weights, h: torch.Tensor, next_tokens: torch.Tensor) -> torch.Tensor:
    """The MTP head over a whole sequence: h (T, D) the main model's final hidden states, next_tokens (T,)."""

    c, m = w.config, w.mtp
    e = embed(w, next_tokens)
    x = _dense(torch.cat([_rms(e, m.enorm, c.eps), _rms(h, m.hnorm, c.eps)], dim=-1), m.eh_proj)
    x = x + attention(m.attn, _rms(x, m.attn_norm, c.eps), c)
    x = x + moe(m.moe, _rms(x, m.moe_norm, c.eps), c.top_k, c.scaling, c.norm_topk)
    return logits(w, _rms(x, m.final_norm, c.eps))
