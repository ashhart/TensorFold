"""Release Flash Next text rotary kernels retained for byte-exact image-path regression checks."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.families.qwen4_exp.cuda.kvquant import h32, quant_groups_4, quant_groups_8

@triton.jit
def _attn_prep(P, POS0, QW, KW, IW, INV, Q, KC, VC, KS, VS, IQ, IKC, eps,
               PW: tl.constexpr, NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, NI: tl.constexpr,
               IHD: tl.constexpr, HALF: tl.constexpr, BITS: tl.constexpr):
    """The release text-only arithmetic, with unchanged reductions and stored rounding."""

    r = tl.program_id(0)
    _prep_row(P, tl.load(POS0) + r, r, tl.program_id(1), QW, KW, IW, INV, Q, KC, VC, KS, VS, IQ, IKC, eps, PW, NQ,
              NKV, HD, NI, IHD, HALF, BITS)



@triton.jit
def _prep_row(P, pos, r, head, QW, KW, IW, INV, Q, KC, VC, KS, VS, IQ, IKC, eps, PW: tl.constexpr, NQ: tl.constexpr,
              NKV: tl.constexpr, HD: tl.constexpr, NI: tl.constexpr, IHD: tl.constexpr, HALF: tl.constexpr,
              BITS: tl.constexpr):
    """The release text-only arithmetic, with unchanged reductions and stored rounding."""

    d = tl.arange(0, HD)
    if head < NQ + NKV + NI:
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
            if BITS:                                 # the query rides the cache's rotation (see the docstring)
                out = tl.reshape(h32(tl.reshape(out.to(tl.float32), (HD // 32, 32)), M=HD // 32),
                                 (HD,)).to(tl.bfloat16)
            tl.store(Q + (r * NQ + head) * HD + d, out)
        elif head < NQ + NKV:
            slot = pos.to(tl.int64) * NKV + head - NQ
            if BITS:
                gg = tl.arange(0, HD // 32)
                block = tl.reshape(out.to(tl.float32), (HD // 32, 32))
                v = tl.load(P + r * PW + NQ * 2 * HD + NKV * HD + (head - NQ) * HD + d)
                vblock = tl.reshape(v.to(tl.float32), (HD // 32, 32))
                if BITS == 4:
                    kc, ks = quant_groups_4(block, M=HD // 32)
                    vc, vs = quant_groups_4(vblock, M=HD // 32)
                else:
                    kc, ks = quant_groups_8(block, M=HD // 32)
                    vc, vs = quant_groups_8(vblock, M=HD // 32)
                tl.store(KS + slot * (HD // 32) + gg, ks)
                tl.store(VS + slot * (HD // 32) + gg, vs)
                if BITS == 4:
                    gb = tl.arange(0, 16)
                    tl.store(KC + slot * (HD // 2) + gg[:, None] * 16 + gb[None, :], kc)
                    tl.store(VC + slot * (HD // 2) + gg[:, None] * 16 + gb[None, :], vc)
                else:
                    gd = tl.arange(0, 32)
                    tl.store(KC + slot * HD + gg[:, None] * 32 + gd[None, :], kc)
                    tl.store(VC + slot * HD + gg[:, None] * 32 + gd[None, :], vc)
            else:
                tl.store(KC + slot * HD + d, out)
                v = tl.load(P + r * PW + NQ * 2 * HD + NKV * HD + (head - NQ) * HD + d)
                tl.store(VC + slot * HD + d, v)
        else:
            tl.store(IQ + (r * NI + head - NQ - NKV) * IHD + d, out, mask=live)
    else:
        live = d < IHD
        raw = tl.load(P + r * PW + NQ * 2 * HD + 2 * NKV * HD + NI * IHD + d, mask=live, other=0.0)
        tl.store(IKC + pos.to(tl.int64) * IHD + d, raw, mask=live)



def attn_prep(p: torch.Tensor, pos0: torch.Tensor, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, iq, ikc,
              eps: float, *, q_heads: int, kv_heads: int, head_dim: int, index_heads: int, index_dim: int,
              ks: torch.Tensor | None = None, vs: torch.Tensor | None = None, bits: int = 0) -> None:
    """The release text-only arithmetic, with unchanged reductions and stored rounding."""

    rows, pw = p.shape
    if bits and (ks is None or vs is None):
        raise ValueError("a quantized KV cache needs its scale tensors")
    if ks is None:
        ks = vs = kc
    _attn_prep[(rows, q_heads + kv_heads + index_heads + 1)](
        p, pos0, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, ks, vs, iq, ikc, eps, PW=pw, NQ=q_heads, NKV=kv_heads,
        HD=head_dim, NI=index_heads, IHD=index_dim, HALF=inv_freq.numel(), BITS=bits, num_warps=2)



@triton.jit
def _pool(IKC, POOLED, POS0, W, INV, eps, R, DI: tl.constexpr, HALF: tl.constexpr, RATIO: tl.constexpr):
    """The release text-only arithmetic, with unchanged reductions and stored rounding."""

    _pool_block(IKC, POOLED, tl.load(POS0), tl.program_id(0), W, INV, eps, R, DI, HALF, RATIO)



@triton.jit
def _pool_block(IKC, POOLED, p0, i, W, INV, eps, R, DI: tl.constexpr, HALF: tl.constexpr, RATIO: tl.constexpr):
    """The release text-only arithmetic, with unchanged reductions and stored rounding."""

    b = p0 // RATIO + i
    if RATIO * b + RATIO <= p0 + R:
        d = tl.arange(0, DI)
        acc = tl.load(IKC + (RATIO * b).to(tl.int64) * DI + d).to(tl.float32)
        for k in tl.static_range(1, RATIO):
            acc = acc + tl.load(IKC + (RATIO * b + k).to(tl.int64) * DI + d).to(tl.float32)
        x = (acc / RATIO).to(tl.bfloat16).to(tl.float32)
        rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / DI + eps)
        xn = (x * rinv * tl.load(W + d)).to(tl.bfloat16).to(tl.float32)
        partner = tl.where(d < HALF, d + HALF, tl.where(d < 2 * HALF, d - HALF, d))
        xp = tl.load(IKC + (RATIO * b).to(tl.int64) * DI + partner).to(tl.float32)
        for k in tl.static_range(1, RATIO):
            xp = xp + tl.load(IKC + (RATIO * b + k).to(tl.int64) * DI + partner).to(tl.float32)
        xp = (xp / RATIO).to(tl.bfloat16).to(tl.float32)
        xpn = (xp * rinv * tl.load(W + partner)).to(tl.bfloat16).to(tl.float32)
        j = tl.where(d < HALF, d, tl.where(d < 2 * HALF, d - HALF, 0))
        ang = (RATIO * b).to(tl.float32) * tl.load(INV + j)
        cos, sin = tl.cos(ang), tl.sin(ang)
        rot = tl.where(d < HALF, xn * cos - xpn * sin, tl.where(d < 2 * HALF, xpn * sin + xn * cos, xn))
        tl.store(POOLED + b.to(tl.int64) * DI + d, rot.to(tl.bfloat16))



def qsa_pool(ikc: torch.Tensor, pooled: torch.Tensor, pos0: torch.Tensor, ik_scale: torch.Tensor,
             inv_freq: torch.Tensor, eps: float, scratch, rows: int) -> None:
    """The release text-only arithmetic, with unchanged reductions and stored rounding."""

    _pool[(rows // scratch.ratio + 2,)](ikc, pooled, pos0, ik_scale, inv_freq, eps, rows, DI=ikc.shape[1],
                                        HALF=inv_freq.numel(), RATIO=scratch.ratio, num_warps=1)
