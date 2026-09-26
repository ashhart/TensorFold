"""Flash Next's small kernels on CUDA (Triton), each row on its own.

A program reads only its own row (or its own row and head), so a row's bits never depend on the other
rows of a window. Arithmetic follows the checkpoint's training framework: fp32 math, one bf16 rounding
where the model stores a bf16 tensor. Kernels that feed a 4-bit matmul also write the fp32 sums of each
32-input group of the bf16 values they store, so the matmul need not read its input twice.

Hyper-connection block (4 residual streams of D):
    hc_writeback  streams += bf16(branch * inject[s]) (the branch: a projection's output, or the MoE's slots
                  combined in slot order), partial sums of squares per stream
    hc_normed     bf16(h * rsqrt(mean(h^2) + eps) * scale) and its group sums
    hc_act        from the down projection [low | inject]: bf16(silu(bf16(v / S))) and 2 sigmoid(v / S)
    hc_mix        mean over streams of bf16(sigmoid(up) * normed)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bsig(x):
    return (1.0 / (1.0 + tl.exp(-x))).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _bsilu(x):
    return (x / (1.0 + tl.exp(-x))).to(tl.bfloat16).to(tl.float32)


# -- embedding ------------------------------------------------------------------------------------
@triton.jit
def _embed(IDS, W, S, B, OUT, D: tl.constexpr, S_COPIES: tl.constexpr):
    """Row r, group g: dequantize 32 values of token IDS[r]'s row (MLX layout), bf16, into S_COPIES streams."""

    row = tl.program_id(0)
    g = tl.program_id(1)
    tok = tl.load(IDS + row).to(tl.int64)
    words = tl.load(W + tok * (D // 8) + g * 4 + tl.arange(0, 4))
    q = tl.reshape((words[:, None] >> (tl.arange(0, 8) * 4)[None, :]) & 0xF, (32,)).to(tl.float32)
    s = tl.load(S + tok * (D // 32) + g).to(tl.float32)
    b = tl.load(B + tok * (D // 32) + g).to(tl.float32)
    v = (q * s + b).to(tl.bfloat16)
    d = g * 32 + tl.arange(0, 32)
    for c in tl.static_range(S_COPIES):
        tl.store(OUT + row * (S_COPIES * D) + c * D + d, v)


def embed(ids: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, dims: int,
          copies: int = 1, out: torch.Tensor | None = None) -> torch.Tensor:
    """ids (R,) int32 -> (R, copies * dims) bf16 rows of the 4-bit embedding (MLX layout, group 32)."""

    rows = ids.shape[0]
    if out is None:
        out = torch.empty((rows, copies * dims), dtype=torch.bfloat16, device=ids.device)
    _embed[(rows, dims // 32)](ids, weight, scales, biases, out, D=dims, S_COPIES=copies, num_warps=1)
    return out


# -- hyper-connections ------------------------------------------------------------------------------
@triton.jit
def _hc_writeback(H, HOUT, PSS, BR, INJ, Y, WTS, RS,
                  D: tl.constexpr, S: tl.constexpr, MODE: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr,
                  BLOCK: tl.constexpr, WORLD: tl.constexpr):
    """Program (r, c): dims [c BLOCK, (c + 1) BLOCK) of every stream. MODE 0: no write-back; 1: branch BR [R, D]
    bf16; 2: the MoE's slots Y [R, SLOTS, D] fp32 combined as fp32 sum_k w_k y_k (k < TOPK, in order) + w_s y_s
    (the shared expert, slot TOPK), rounded once; 3: tensor-parallel fp32 partials gathered in rank order
    (BR [WORLD] blocks RS apart, each [R, D]) summed rank 0 first, rounded once; 4: a matmul's unreduced K slices
    (BR [WORLD] slices RS apart) summed in slice order and rounded once, the bits of ``qmm._reduce``. Streams
    h_s = bf16(h_s + bf16(branch * inject_s)); PSS[r, c, s] = the chunk's fp32 sum of h_s^2."""

    r = tl.program_id(0)
    c = tl.program_id(1)
    NC: tl.constexpr = D // BLOCK
    d = c * BLOCK + tl.arange(0, BLOCK)
    if MODE == 1:
        branch = tl.load(BR + r * D + d).to(tl.float32)
    elif MODE == 3 or MODE == 4:
        acc = tl.load(BR + r * D + d)
        for k in tl.static_range(1, WORLD):
            acc = acc + tl.load(BR + k * RS + r * D + d)
        branch = acc.to(tl.bfloat16).to(tl.float32)
    elif MODE == 2:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(TOPK + 1):
            wk = tl.load(WTS + r * SLOTS + k)
            yk = tl.load(Y + (r * SLOTS + k) * D + d)
            acc = acc + yk * wk
        branch = acc.to(tl.bfloat16).to(tl.float32)
    for s in tl.static_range(S):
        hv = tl.load(H + r * (S * D) + s * D + d).to(tl.float32)
        if MODE != 0:
            inj = tl.load(INJ + r * S + s).to(tl.float32)
            hv = (hv + (branch * inj).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            tl.store(HOUT + r * (S * D) + s * D + d, hv.to(tl.bfloat16))
        tl.store(PSS + (r * NC + c) * S + s, tl.sum(hv * hv, axis=0))


@triton.jit
def _hc_normed(H, PSS, SCALE, NORMED, XS, eps,
               D: tl.constexpr, S: tl.constexpr, NC: tl.constexpr, BLOCK: tl.constexpr):
    """Program (r, j): elements [j BLOCK, (j + 1) BLOCK) of row r's S*D streams (inside one stream):
    bf16(h * rinv_s * scale), and its 32-group sums. rinv_s from the stream's NC partial sums in order."""

    r = tl.program_id(0)
    j = tl.program_id(1)
    e = j * BLOCK + tl.arange(0, BLOCK)
    s = (j * BLOCK) // D
    total = 0.0
    for c in range(NC):
        total += tl.load(PSS + (r * NC + c) * S + s)
    rinv = 1.0 / tl.sqrt(total / D + eps)
    hv = tl.load(H + r * (S * D) + e).to(tl.float32)
    w = tl.load(SCALE + e).to(tl.float32)
    y = (hv * rinv * w).to(tl.bfloat16)
    tl.store(NORMED + r * (S * D) + e, y)
    sums = tl.sum(tl.reshape(y.to(tl.float32), (BLOCK // 32, 32)), axis=1)
    tl.store(XS + r * (S * D // 32) + j * (BLOCK // 32) + tl.arange(0, BLOCK // 32), sums)


def hc_writeback(h: torch.Tensor, hout: torch.Tensor, pss: torch.Tensor, streams: int, mode: int,
                 branch: torch.Tensor | None = None, inject: torch.Tensor | None = None,
                 y: torch.Tensor | None = None, wts: torch.Tensor | None = None, block: int = 256) -> None:
    """mode 3: ``branch`` is the gathered partials [world, R, D] fp32 (contiguous); mode 4: a matmul's K slices
    [SK, R, D] fp32."""

    rows, wide = h.shape
    d = wide // streams
    top = (wts.shape[1] - 1) if wts is not None else 1
    slots = wts.shape[1] if wts is not None else 1
    world = branch.shape[0] if mode in (3, 4) else 1
    rs = rows * d
    dummy = h
    _hc_writeback[(rows, d // block)](h, hout, pss, branch if branch is not None else dummy,
                                      inject if inject is not None else dummy, y if y is not None else dummy,
                                      wts if wts is not None else dummy, rs, D=d, S=streams, MODE=mode, TOPK=top,
                                      SLOTS=slots, BLOCK=block, WORLD=world, num_warps=2)


@triton.jit
def _moe_partial(Y, WTS, OUT, D: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr, BLOCK: tl.constexpr):
    """A tensor-parallel rank's MoE share, fp32: sum_k w_k y_k (slots in order) + w_s y_s, not rounded."""

    r = tl.program_id(0)
    c = tl.program_id(1)
    d = c * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(TOPK + 1):
        acc = acc + tl.load(Y + (r * SLOTS + k) * D + d) * tl.load(WTS + r * SLOTS + k)
    tl.store(OUT + r * D + d, acc)


def moe_partial(y: torch.Tensor, wts: torch.Tensor, out: torch.Tensor, rows: int) -> None:
    d = out.shape[1]
    _moe_partial[(rows, d // 256)](y, wts, out, D=d, TOPK=wts.shape[1] - 1, SLOTS=wts.shape[1], BLOCK=256,
                                   num_warps=2)


def hc_normed(h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor, normed: torch.Tensor, xs: torch.Tensor,
              streams: int, eps: float, block: int = 512) -> None:
    rows, wide = h.shape
    d = wide // streams
    _hc_normed[(rows, wide // block)](h, pss, scale, normed, xs, eps, D=d, S=streams, NC=pss.shape[1],
                                      BLOCK=block, num_warps=4)


@triton.jit
def _hc_act(DN, ACT, XS, INJ, S: tl.constexpr, LOW: tl.constexpr, LOWP: tl.constexpr, HAS_INJ: tl.constexpr,
            NDN: tl.constexpr):
    """Row r of the down projection [low | inject]: act = bf16(silu(bf16(v / S))) and its 32-group sums; inject
    gates bf16(2 bf16(sigmoid(bf16(v / S))))."""

    r = tl.program_id(0)
    i = tl.arange(0, LOWP)
    ok = i < LOW
    v = tl.load(DN + r * NDN + i, mask=ok, other=0.0).to(tl.float32)
    v = (v / S).to(tl.bfloat16).to(tl.float32)
    a = _bsilu(v)
    a = tl.where(ok, a, 0.0)
    tl.store(ACT + r * LOW + i, a.to(tl.bfloat16), mask=ok)
    sums = tl.sum(tl.reshape(a, (LOWP // 32, 32)), axis=1)
    g = tl.arange(0, LOWP // 32)
    tl.store(XS + r * (LOW // 32) + g, sums, mask=g < LOW // 32)
    if HAS_INJ:
        s = tl.arange(0, S)
        iv = tl.load(DN + r * NDN + LOW + s).to(tl.float32)
        iv = (iv / S).to(tl.bfloat16).to(tl.float32)
        tl.store(INJ + r * S + s, (2.0 * _bsig(iv)).to(tl.bfloat16))


@triton.jit
def _hc_reduce_act(PART, ACT, XS, INJ, SK: tl.constexpr, M, S: tl.constexpr, LOW: tl.constexpr,
                   LOWP: tl.constexpr, HAS_INJ: tl.constexpr, NDN: tl.constexpr):
    """``qmm``'s split-K sum (slices in order, one bf16 rounding) fused with ``hc_act``: the same bits."""

    r = tl.program_id(0)
    i = tl.arange(0, LOWP)
    ok = i < LOW
    v = tl.load(PART + r * NDN + i, mask=ok, other=0.0)
    for s_ in tl.static_range(1, SK):
        v = v + tl.load(PART + (s_ * M + r) * NDN + i, mask=ok, other=0.0)
    v = v.to(tl.bfloat16).to(tl.float32)
    v = (v / S).to(tl.bfloat16).to(tl.float32)
    a = _bsilu(v)
    a = tl.where(ok, a, 0.0)
    tl.store(ACT + r * LOW + i, a.to(tl.bfloat16), mask=ok)
    sums = tl.sum(tl.reshape(a, (LOWP // 32, 32)), axis=1)
    g = tl.arange(0, LOWP // 32)
    tl.store(XS + r * (LOW // 32) + g, sums, mask=g < LOW // 32)
    if HAS_INJ:
        s = tl.arange(0, S)
        iv = tl.load(PART + r * NDN + LOW + s)
        for s_ in tl.static_range(1, SK):
            iv = iv + tl.load(PART + (s_ * M + r) * NDN + LOW + s)
        iv = iv.to(tl.bfloat16).to(tl.float32)
        iv = (iv / S).to(tl.bfloat16).to(tl.float32)
        tl.store(INJ + r * S + s, (2.0 * _bsig(iv)).to(tl.bfloat16))


def hc_reduce_act(part: torch.Tensor, act: torch.Tensor, xs: torch.Tensor, inj: torch.Tensor | None, streams: int,
                  low: int) -> None:
    """part: the down projection's unreduced slices [SK, R, N]."""

    sk, rows, ndn = part.shape
    _hc_reduce_act[(rows,)](part, act, xs, inj if inj is not None else act, SK=sk, M=rows, S=streams, LOW=low,
                            LOWP=triton.next_power_of_2(low), HAS_INJ=inj is not None, NDN=ndn, num_warps=4)


def hc_act(dn: torch.Tensor, act: torch.Tensor, xs: torch.Tensor, inj: torch.Tensor | None, streams: int,
           low: int) -> None:
    rows, ndn = dn.shape
    _hc_act[(rows,)](dn, act, xs, inj if inj is not None else act, S=streams, LOW=low,
                     LOWP=triton.next_power_of_2(low), HAS_INJ=inj is not None, NDN=ndn, num_warps=4)


@triton.jit
def _hc_mix(UP, NORMED, MIXED, XS, D: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    """Program (r, c): mixed[d] = bf16((sum over streams, in order, of bf16(bf16(sigmoid(up_s)) * normed_s)) / S)
    for dims [c BLOCK, (c + 1) BLOCK), and its 32-group sums."""

    r = tl.program_id(0)
    c = tl.program_id(1)
    d = c * BLOCK + tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(S):
        u = tl.load(UP + r * (S * D) + s * D + d).to(tl.float32)
        n = tl.load(NORMED + r * (S * D) + s * D + d).to(tl.float32)
        total += (_bsig(u) * n).to(tl.bfloat16).to(tl.float32)
    m = (total / S).to(tl.bfloat16)
    tl.store(MIXED + r * D + d, m)
    sums = tl.sum(tl.reshape(m.to(tl.float32), (BLOCK // 32, 32)), axis=1)
    tl.store(XS + r * (D // 32) + c * (BLOCK // 32) + tl.arange(0, BLOCK // 32), sums)


def hc_mix(up: torch.Tensor, normed: torch.Tensor, mixed: torch.Tensor, xs: torch.Tensor, streams: int,
           block: int = 256) -> None:
    rows, wide = up.shape
    d = wide // streams
    _hc_mix[(rows, d // block)](up, normed, mixed, xs, D=d, S=streams, BLOCK=block, num_warps=2)


# -- plain RMSNorm (MTP inputs, PLE norms) ------------------------------------------------------------
@triton.jit
def _rmsnorm(X, W, OUT, XS, eps, x_stride, D: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    """Program (r, grp): RMSNorm of each run of G features on its own (G = D: one norm), scale W, bf16 out,
    and 32-group sums. BLOCK >= G."""

    r = tl.program_id(0)
    grp = tl.program_id(1)
    i = tl.arange(0, BLOCK)
    ok = i < G
    x = tl.load(X + r * x_stride + grp * G + i, mask=ok, other=0.0).to(tl.float32)
    rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / G + eps)
    w = tl.load(W + grp * G + i, mask=ok, other=0.0).to(tl.float32)
    y = (x * rinv * w).to(tl.bfloat16)
    tl.store(OUT + r * D + grp * G + i, y, mask=ok)
    sums = tl.sum(tl.reshape(tl.where(ok, y.to(tl.float32), 0.0), (BLOCK // 32, 32)), axis=1)
    gi = tl.arange(0, BLOCK // 32)
    tl.store(XS + r * (D // 32) + grp * (G // 32) + gi, sums, mask=gi < G // 32)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, group: int | None = None,
            out: torch.Tensor | None = None, xs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    rows, d = x.shape
    g = group or d
    if out is None:
        out = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
    if xs is None:
        xs = torch.empty((rows, d // 32), dtype=torch.float32, device=x.device)
    _rmsnorm[(rows, d // g)](x, w, out, xs, eps, x.stride(0), D=d, G=g, BLOCK=triton.next_power_of_2(g),
                             num_warps=8 if g > 4096 else 4)
    return out, xs


# -- attention --------------------------------------------------------------------------------------
@triton.jit
def _attn_prep(P, POS0, QW, KW, IW, INV, Q, KC, VC, IQ, IKC, eps,
               PW: tl.constexpr, NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, NI: tl.constexpr,
               IHD: tl.constexpr, HALF: tl.constexpr):
    """Program (r, head) over the stacked projection [q|gate pairs (NQ x 2HD) | k (NKV HD) | v (NKV HD) |
    indexer q (NI IHD) | indexer key (IHD)]. Heads [0, NQ): queries; [NQ, NQ + NKV): keys, written to the
    cache at position POS0 + r with the values; then NI indexer queries; the last: the raw indexer key,
    written to its cache. RMSNorm with the stored scale in fp32, bf16, then RoPE on the first 2 HALF dims
    (rotate-half) at the row's position, bf16."""

    r = tl.program_id(0)
    head = tl.program_id(1)
    pos = tl.load(POS0) + r
    d = tl.arange(0, HD)
    if head < NQ + NKV + NI:
        is_q = head < NQ
        is_k = (head >= NQ) & (head < NQ + NKV)
        width = tl.where(head >= NQ + NKV, IHD, HD)
        live = d < width
        if head < NQ:
            src = r * PW + head * 2 * HD
        elif head < NQ + NKV:
            src = r * PW + NQ * 2 * HD + (head - NQ) * HD
        else:
            src = r * PW + NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD
        x = tl.load(P + src + d, mask=live, other=0.0).to(tl.float32)
        rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / width + eps)
        if head < NQ:
            w = tl.load(QW + d).to(tl.float32)
        elif head < NQ + NKV:
            w = tl.load(KW + d).to(tl.float32)
        else:
            w = tl.load(IW + d, mask=live, other=0.0).to(tl.float32)
        xn = (x * rinv * w).to(tl.bfloat16).to(tl.float32)
        partner = tl.where(d < HALF, d + HALF, tl.where(d < 2 * HALF, d - HALF, d))
        xp = tl.load(P + src + partner, mask=live, other=0.0).to(tl.float32)
        if head < NQ:
            wp = tl.load(QW + partner).to(tl.float32)
        elif head < NQ + NKV:
            wp = tl.load(KW + partner).to(tl.float32)
        else:
            wp = tl.load(IW + partner, mask=live, other=0.0).to(tl.float32)
        xpn = (xp * rinv * wp).to(tl.bfloat16).to(tl.float32)
        i = tl.where(d < HALF, d, tl.where(d < 2 * HALF, d - HALF, 0))
        ang = pos.to(tl.float32) * tl.load(INV + i)
        cos = tl.cos(ang)
        sin = tl.sin(ang)
        rot = tl.where(d < HALF, xn * cos - xpn * sin, tl.where(d < 2 * HALF, xpn * sin + xn * cos, xn))
        out = rot.to(tl.bfloat16)
        if head < NQ:
            tl.store(Q + (r * NQ + head) * HD + d, out)
        elif head < NQ + NKV:
            tl.store(KC + (pos.to(tl.int64) * NKV + head - NQ) * HD + d, out)
            v = tl.load(P + r * PW + NQ * 2 * HD + NKV * HD + (head - NQ) * HD + d)
            tl.store(VC + (pos.to(tl.int64) * NKV + head - NQ) * HD + d, v)
        else:
            tl.store(IQ + (r * NI + head - NQ - NKV) * IHD + d, out, mask=live)
    else:
        live = d < IHD
        raw = tl.load(P + r * PW + NQ * 2 * HD + 2 * NKV * HD + NI * IHD + d, mask=live, other=0.0)
        tl.store(IKC + pos.to(tl.int64) * IHD + d, raw, mask=live)


def attn_prep(p: torch.Tensor, pos0: torch.Tensor, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, iq, ikc,
              eps: float, *, q_heads: int, kv_heads: int, head_dim: int, index_heads: int, index_dim: int) -> None:
    rows, pw = p.shape
    _attn_prep[(rows, q_heads + kv_heads + index_heads + 1)](
        p, pos0, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, iq, ikc, eps, PW=pw, NQ=q_heads, NKV=kv_heads,
        HD=head_dim, NI=index_heads, IHD=index_dim, HALF=inv_freq.numel(), num_warps=2)


@triton.jit
def _attn_gate(O, P, OUT, XS, PW: tl.constexpr, NQ: tl.constexpr, HD: tl.constexpr):
    """Program (r, h): bf16(o * sigmoid(gate)) (fp32 math), gate from the [q | gate] pairs, and group sums."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, HD)
    o = tl.load(O + (r * NQ + h) * HD + d).to(tl.float32)
    g = tl.load(P + r * PW + h * 2 * HD + HD + d).to(tl.float32)
    out = (o / (1.0 + tl.exp(-g))).to(tl.bfloat16)
    tl.store(OUT + r * (NQ * HD) + h * HD + d, out)
    sums = tl.sum(tl.reshape(out.to(tl.float32), (HD // 32, 32)), axis=1)
    tl.store(XS + r * (NQ * HD // 32) + h * (HD // 32) + tl.arange(0, HD // 32), sums)


def attn_gate(o: torch.Tensor, p: torch.Tensor, out: torch.Tensor, xs: torch.Tensor, *, q_heads: int,
              head_dim: int) -> None:
    rows = o.shape[0]
    _attn_gate[(rows, q_heads)](o, p, out, xs, PW=p.shape[1], NQ=q_heads, HD=head_dim, num_warps=2)


# -- n-gram embedding (PLE) ---------------------------------------------------------------------------
@triton.jit
def _ple_embed(W, S, B, OUT, XS, HEADS: tl.constexpr, DH: tl.constexpr):
    """Program (r, h): gathered n-gram row r * HEADS + h (DH values, MLX layout, group 32) ->
    OUT[r, h DH: (h + 1) DH] bf16 and its group sums."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    G: tl.constexpr = DH // 32
    row = (r * HEADS + h).to(tl.int64)
    gi = tl.arange(0, 8)
    gok = gi < G
    wi = tl.arange(0, 4)
    words = tl.load(W + row * (DH // 8) + gi[:, None] * 4 + wi[None, :], mask=gok[:, None], other=0)
    q = tl.reshape((words[:, :, None] >> (tl.arange(0, 8) * 4)[None, None, :]) & 0xF, (8, 32)).to(tl.float32)
    s = tl.load(S + row * G + gi, mask=gok, other=0.0).to(tl.float32)
    b = tl.load(B + row * G + gi, mask=gok, other=0.0).to(tl.float32)
    v = (q * s[:, None] + b[:, None]).to(tl.bfloat16)
    k = tl.arange(0, 32)
    tl.store(OUT + r * (HEADS * DH) + h * DH + gi[:, None] * 32 + k[None, :], v, mask=gok[:, None])
    tl.store(XS + r * (HEADS * DH // 32) + h * G + gi, tl.sum(v.to(tl.float32), axis=1), mask=gok)


def ple_embed(rows: int, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, heads: int, dh: int,
              out: torch.Tensor, xs: torch.Tensor) -> None:
    """Gathered rows (``weights.HostTable.gather``, row r * heads + h) -> out [rows, heads * dh] bf16."""

    _ple_embed[(rows, heads)](weight, scales, biases, out, xs, HEADS=heads, DH=dh, num_warps=1)


@triton.jit
def _ple_gate(KEYS, VALS, H, NK, NQ, GATED, PSS, eps,
              D: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    """Program r. keys = bf16(norm_key(key_proj)) per stream, queries = bf16(norm_query(h)) per stream;
    gate_s = sum(keys_s * queries_s) / sqrt(D) (fp32), signed sqrt, sigmoid; gated_s = bf16(sigmoid * values);
    PSS[r, s] = sum of gated_s^2 (for norm_conv). Two passes over D per stream."""

    r = tl.program_id(0)
    NCH: tl.constexpr = D // BLOCK
    for s in tl.static_range(S):
        # RMS of the key and query streams
        kss = 0.0
        qss = 0.0
        for c in range(NCH):
            d = c * BLOCK + tl.arange(0, BLOCK)
            kv = tl.load(KEYS + r * (S * D) + s * D + d).to(tl.float32)
            hv = tl.load(H + r * (S * D) + s * D + d).to(tl.float32)
            kss += tl.sum(kv * kv, axis=0)
            qss += tl.sum(hv * hv, axis=0)
        krinv = 1.0 / tl.sqrt(kss / D + eps)
        qrinv = 1.0 / tl.sqrt(qss / D + eps)
        dot = 0.0
        for c in range(NCH):
            d = c * BLOCK + tl.arange(0, BLOCK)
            kv = tl.load(KEYS + r * (S * D) + s * D + d).to(tl.float32)
            hv = tl.load(H + r * (S * D) + s * D + d).to(tl.float32)
            kn = (kv * krinv * tl.load(NK + s * D + d).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            qn = (hv * qrinv * tl.load(NQ + s * D + d).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            dot += tl.sum((kn * qn).to(tl.bfloat16).to(tl.float32), axis=0)
        gate = (dot / tl.sqrt(D * 1.0)).to(tl.bfloat16).to(tl.float32)
        mag = tl.sqrt(tl.maximum(tl.abs(gate), 1e-6))
        gate = tl.where(gate > 0, mag, tl.where(gate < 0, -mag, 0.0)).to(tl.bfloat16).to(tl.float32)
        sg = _bsig(gate)
        gss = 0.0
        for c in range(NCH):
            d = c * BLOCK + tl.arange(0, BLOCK)
            v = tl.load(VALS + r * D + d).to(tl.float32)
            gv = (sg * v).to(tl.bfloat16)
            tl.store(GATED + r * (S * D) + s * D + d, gv)
            gf = gv.to(tl.float32)
            gss += tl.sum(gf * gf, axis=0)
        tl.store(PSS + r * S + s, gss)


def ple_gate(keys, vals, h, norm_key, norm_query, gated, pss, eps: float, streams: int) -> None:
    rows, wide = h.shape
    _ple_gate[(rows,)](keys, vals, h, norm_key, norm_query, gated, pss, eps, D=wide // streams, S=streams,
                       BLOCK=512, num_warps=4)


@triton.jit
def _ple_conv(GATED, PSS, NC, TAIL, CW, H, HOUT, NROW, eps, R,
              D: tl.constexpr, S: tl.constexpr, TAPS: tl.constexpr, DIL: tl.constexpr, BLOCK: tl.constexpr):
    """Program (r, c): normed = bf16(gated * rinv_s * norm_conv) for row r; conv over [TAIL rows; normed rows]
    (depthwise, TAPS taps DIL apart, fp32), SiLU (bf16); h = bf16(h + bf16(gated + silu)). NROW[r] receives
    row r's normed values (the conv window's next rows)."""

    r = tl.program_id(0)
    c = tl.program_id(1)
    W: tl.constexpr = S * D
    NT: tl.constexpr = (TAPS - 1) * DIL
    e = c * BLOCK + tl.arange(0, BLOCK)
    s = (c * BLOCK) // D
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for t in tl.static_range(TAPS):
        at = r + t * DIL                              # index into [tail (NT rows); rows]
        if at < NT:
            xv = tl.load(TAIL + at * W + e).to(tl.float32)
        else:
            rr = at - NT
            rinv = 1.0 / tl.sqrt(tl.load(PSS + rr * S + s) / D + eps)
            g = tl.load(GATED + rr * W + e).to(tl.float32)
            xv = (g * rinv * tl.load(NC + e).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        acc += tl.load(CW + e * TAPS + t).to(tl.float32) * xv
    conv = acc.to(tl.bfloat16).to(tl.float32)
    act = _bsilu(conv)
    g = tl.load(GATED + r * W + e).to(tl.float32)
    ple = (g + act).to(tl.bfloat16).to(tl.float32)
    hv = tl.load(H + r * W + e).to(tl.float32)
    tl.store(HOUT + r * W + e, (hv + ple).to(tl.bfloat16))
    rinv = 1.0 / tl.sqrt(tl.load(PSS + r * S + s) / D + eps)
    own = (g * rinv * tl.load(NC + e).to(tl.float32)).to(tl.bfloat16)
    tl.store(NROW + r * W + e, own)


def ple_conv(gated, pss, norm_conv, tail, conv_w, h, hout, nrow, eps: float, streams: int, dilation: int) -> None:
    rows, wide = h.shape
    taps = conv_w.shape[1]
    _ple_conv[(rows, wide // 512)](gated, pss, norm_conv, tail, conv_w, h, hout, nrow, eps, rows,
                                   D=wide // streams, S=streams, TAPS=taps, DIL=dilation, BLOCK=512, num_warps=4)


# -- MTP input -----------------------------------------------------------------------------------------
@triton.jit
def _add_streams(E, HS, OUT, D: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    """out[r, s, :] = bf16(e[r] + hs[r, s]) (the MTP input: the embedding branch added to each stream)."""

    r = tl.program_id(0)
    c = tl.program_id(1)
    d = c * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(E + r * D + d).to(tl.float32)
    for s in tl.static_range(S):
        hv = tl.load(HS + (r * S + s) * D + d).to(tl.float32)
        tl.store(OUT + r * (S * D) + s * D + d, (e + hv).to(tl.bfloat16))


def add_streams(e: torch.Tensor, hs: torch.Tensor, out: torch.Tensor, streams: int) -> None:
    rows, d = e.shape
    _add_streams[(rows, d // 256)](e, hs, out, D=d, S=streams, BLOCK=256, num_warps=2)
