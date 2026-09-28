"""Mamba-2 kernels: the state lags a window, whose kept rows replay first in the same loop body, so bits stay serial."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

SCAN_BD, SCAN_WARPS = 32, 8          # channels a scan program, warps; part of the arithmetic, fixed


@triton.jit(do_not_specialize=["R"])
def _conv(P, BASE, RAW, XC, CW, CB, META, R, PROJ: tl.constexpr, XOFF: tl.constexpr, CD: tl.constexpr,
          RMAX: tl.constexpr, BC: tl.constexpr):
    """BASE holds the last three committed inputs; RAW[1 - parity]'s kept rows replay first, then the window's."""

    ch = tl.program_id(0) * BC + tl.arange(0, BC)
    ok = ch < CD
    parity = tl.load(META + 1)
    pk = tl.load(META + 2)
    t0 = tl.load(BASE + 0 * CD + ch, mask=ok, other=0.0).to(tl.float32)
    t1 = tl.load(BASE + 1 * CD + ch, mask=ok, other=0.0).to(tl.float32)
    t2 = tl.load(BASE + 2 * CD + ch, mask=ok, other=0.0).to(tl.float32)
    w0 = tl.load(CW + 0 * CD + ch, mask=ok, other=0.0)
    w1 = tl.load(CW + 1 * CD + ch, mask=ok, other=0.0)
    w2 = tl.load(CW + 2 * CD + ch, mask=ok, other=0.0)
    w3 = tl.load(CW + 3 * CD + ch, mask=ok, other=0.0)
    bias = tl.load(CB + ch, mask=ok, other=0.0)
    for i in range(pk + R):
        prev = i < pk
        cur_prev = tl.load(RAW + ((1 - parity) * RMAX + i) * CD + ch, mask=ok & prev, other=0.0)
        cur_new = tl.load(P + (i - pk) * PROJ + XOFF + ch, mask=ok & (i >= pk), other=0.0)
        cur = tl.where(prev, cur_prev, cur_new).to(tl.float32)
        acc = bias
        acc = acc + w0 * t0
        acc = acc + w1 * t1
        acc = acc + w2 * t2
        acc = acc + w3 * cur
        cv = acc.to(tl.bfloat16).to(tl.float32)
        out = (cv * tl.sigmoid(cv)).to(tl.bfloat16)
        new_row = (parity * RMAX + (i - pk)) * CD + ch
        tl.store(XC + new_row, out, mask=ok & (i >= pk))
        tl.store(RAW + new_row, cur.to(tl.bfloat16), mask=ok & (i >= pk))
        t0 = t1
        t1 = t2
        t2 = cur
        last_kept = i == pk - 1
        tl.store(BASE + 0 * CD + ch, t0.to(tl.bfloat16), mask=ok & last_kept)
        tl.store(BASE + 1 * CD + ch, t1.to(tl.bfloat16), mask=ok & last_kept)
        tl.store(BASE + 2 * CD + ch, t2.to(tl.bfloat16), mask=ok & last_kept)


def conv(proj, base, raw, xc, conv_w, conv_b, meta, rows: int, *, xd: int) -> None:
    cd = base.shape[1]
    if conv_w.shape[0] != 4:
        raise ValueError("the conv kernel is written for kernel size 4")
    bc = 256
    _conv[(triton.cdiv(cd, bc),)](proj, base, raw, xc, conv_w, conv_b, meta, rows, PROJ=proj.shape[1], XOFF=xd, CD=cd,
                                  RMAX=raw.shape[1], BC=bc, num_warps=4)


@triton.jit
def _tap(P, BASE, r, ch, live, back: tl.constexpr, PROJ: tl.constexpr, XOFF: tl.constexpr, CD: tl.constexpr):
    """Raw inputs of rows r - back: the chunk's own, or the committed BASE (its last three) before row 0."""

    src = r - back
    inside = src >= 0
    new = tl.load(P + tl.where(inside, src, 0)[:, None] * PROJ + XOFF + ch[None, :], mask=live & inside[:, None],
                  other=0.0)
    old = tl.load(BASE + tl.where(inside, 0, src + 3)[:, None] * CD + ch[None, :], mask=live & ~inside[:, None],
                  other=0.0)
    return tl.where(inside[:, None], new, old).to(tl.float32)


@triton.jit(do_not_specialize=["R"])
def _conv_rows(P, BASE, XC, CW, CB, R, PROJ: tl.constexpr, XOFF: tl.constexpr, CD: tl.constexpr, BR: tl.constexpr,
               BC: tl.constexpr):
    """A prompt chunk's rows in parallel, with the decode loop's arithmetic; BASE is read, not written."""

    ch = tl.program_id(1) * BC + tl.arange(0, BC)
    ok = ch < CD
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    live = (r < R)[:, None] & ok[None, :]
    t0 = _tap(P, BASE, r, ch, live, 3, PROJ, XOFF, CD)
    t1 = _tap(P, BASE, r, ch, live, 2, PROJ, XOFF, CD)
    t2 = _tap(P, BASE, r, ch, live, 1, PROJ, XOFF, CD)
    cur = _tap(P, BASE, r, ch, live, 0, PROJ, XOFF, CD)
    w0 = tl.load(CW + 0 * CD + ch, mask=ok, other=0.0)[None, :]
    w1 = tl.load(CW + 1 * CD + ch, mask=ok, other=0.0)[None, :]
    w2 = tl.load(CW + 2 * CD + ch, mask=ok, other=0.0)[None, :]
    w3 = tl.load(CW + 3 * CD + ch, mask=ok, other=0.0)[None, :]
    acc = tl.broadcast_to(tl.load(CB + ch, mask=ok, other=0.0)[None, :], (BR, BC))
    acc = acc + w0 * t0
    acc = acc + w1 * t1
    acc = acc + w2 * t2
    acc = acc + w3 * cur
    cv = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(XC + r[:, None] * CD + ch[None, :], (cv * tl.sigmoid(cv)).to(tl.bfloat16), mask=live)


@triton.jit(do_not_specialize=["R"])
def _conv_commit(P, BASE, R, PROJ: tl.constexpr, XOFF: tl.constexpr, CD: tl.constexpr, BC: tl.constexpr):
    """BASE <- the last three raw inputs of [BASE; the chunk's R rows], once the chunk's conv has read BASE."""

    ch = tl.program_id(0) * BC + tl.arange(0, BC)
    ok = ch < CD
    j = tl.arange(0, 4)
    src = R - 3 + j                              # entry j of the new BASE: chunk row src, or old BASE row R + j
    inside = src >= 0
    keep = (j < 3)[:, None] & ok[None, :]
    new = tl.load(P + tl.where(inside, src, 0)[:, None] * PROJ + XOFF + ch[None, :], mask=keep & inside[:, None],
                  other=0.0)
    old = tl.load(BASE + tl.where(inside, 0, R + j)[:, None] * CD + ch[None, :], mask=keep & ~inside[:, None],
                  other=0.0)
    rows = tl.where(inside[:, None], new, old)
    tl.debug_barrier()
    tl.store(BASE + j[:, None] * CD + ch[None, :], rows, mask=keep)


def conv_rows(proj, base, xc, conv_w, conv_b, rows: int, *, xd: int) -> None:
    cd = base.shape[1]
    br, bc = 16, 128
    _conv_rows[(triton.cdiv(rows, br), triton.cdiv(cd, bc))](proj, base, xc, conv_w, conv_b, rows,
                                                             PROJ=proj.shape[1], XOFF=xd, CD=cd, BR=br, BC=bc,
                                                             num_warps=4)
    _conv_commit[(triton.cdiv(cd, bc),)](proj, base, rows, PROJ=proj.shape[1], XOFF=xd, CD=cd, BC=bc, num_warps=4)


@triton.jit
def _ssm_step(st, x, dt, bv, a):
    """The state update: one code path for replayed and new rows."""

    da = tl.exp(a * dt)
    return st * da + (x * dt)[:, None] * bv[None, :]


@triton.jit(do_not_specialize=["R"])
def _scan(P, XC, DT, S, A, DSK, DTB, META, Y, R, lo, hi, PROJ: tl.constexpr, XD: tl.constexpr, CD: tl.constexpr,
          DTOFF: tl.constexpr, H: tl.constexpr, DH: tl.constexpr, NG: tl.constexpr, DS: tl.constexpr,
          RMAX: tl.constexpr, BD: tl.constexpr):
    """Replays the last window's kept rows into S, then the window: y = bf16(C . s + D x) * bf16(silu(z)) -> Y bf16."""

    h = tl.program_id(0)
    dblk = tl.program_id(1)
    g = h // (H // NG)
    d = dblk * BD + tl.arange(0, BD)
    si = tl.arange(0, DS)
    parity = tl.load(META + 1)
    pk = tl.load(META + 2)
    a = tl.load(A + h)
    dsk = tl.load(DSK + h)
    dtb = tl.load(DTB + h)
    sptr = S + (h * DH + d[:, None]) * DS + si[None, :]
    st = tl.load(sptr)
    for i in range(pk + R):
        prev = i < pk
        buf = tl.where(prev, 1 - parity, parity)
        row = tl.where(prev, i, i - pk)
        base = (buf * RMAX + row) * CD
        x = tl.load(XC + base + h * DH + d).to(tl.float32)
        bv = tl.load(XC + base + XD + g * DS + si).to(tl.float32)
        cv = tl.load(XC + base + XD + NG * DS + g * DS + si).to(tl.float32)
        dt_old = tl.load(DT + (buf * RMAX + row) * H + h, mask=prev, other=0.0)
        v = tl.load(P + row * PROJ + DTOFF + h, mask=i >= pk, other=0.0).to(tl.float32) + dtb
        dt_new = tl.maximum(v, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(v)))
        dt_new = tl.minimum(tl.maximum(dt_new, lo), hi)
        dt = tl.where(prev, dt_old, dt_new)
        tl.store(DT + (buf * RMAX + row) * H + h, dt, mask=(i >= pk) & (dblk == 0))
        st = _ssm_step(st, x, dt, bv, a)
        tl.store(sptr, st, mask=(i == pk - 1) & (d[:, None] < DH))
        y = (tl.sum(st * cv[None, :], axis=1) + x * dsk).to(tl.bfloat16).to(tl.float32)
        z = tl.load(P + row * PROJ + h * DH + d, mask=i >= pk, other=0.0).to(tl.float32)
        gz = (z * tl.sigmoid(z)).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + row * XD + h * DH + d, (gz * y).to(tl.bfloat16), mask=(i >= pk) & (d < DH))


def scan(proj, xc, dt, state, a, d_skip, dt_bias, meta, rows: int, *, heads: int, head_dim: int, groups: int,
         state_dim: int, lo: float, hi: float) -> torch.Tensor:
    xd = heads * head_dim
    cd = xc.shape[2]
    y = torch.empty((rows, xd), dtype=torch.bfloat16, device=proj.device)
    _scan[(heads, triton.cdiv(head_dim, SCAN_BD))](
        proj, xc, dt, state, a, d_skip, dt_bias, meta, y, rows, lo, hi, PROJ=proj.shape[1], XD=xd, CD=cd,
        DTOFF=xd + cd, H=heads, DH=head_dim, NG=groups, DS=state_dim, RMAX=xc.shape[1], BD=SCAN_BD,
        num_warps=SCAN_WARPS)
    return y


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_nemotron_scan_rows", sources=[str(here / "scan_rows.cpp"), str(here / "scan_rows.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


def scan_rows(proj, xc, state, a, d_skip, dt_bias, rows: int, *, heads: int, head_dim: int, groups: int,
              state_dim: int, lo: float, hi: float) -> torch.Tensor:
    """A prompt chunk's scan in ``scan_rows.cu``; ``state`` ends holding the state after the chunk's last row."""

    if state_dim != 128 or head_dim % 32:
        raise ValueError("the prompt scan is written for 128 states and value rows in blocks of 32")
    xd = heads * head_dim
    y = torch.empty((rows, xd), dtype=torch.bfloat16, device=proj.device)
    _ext().scan_rows(proj[:rows], xc[:rows], state, a, d_skip, dt_bias, y, xd + xc.shape[1], groups, lo, hi)
    return y


@triton.jit
def _group_rmsnorm(X, W, OUT, XS, eps, XD: tl.constexpr, GS: tl.constexpr):
    row = tl.program_id(0)
    grp = tl.program_id(1)
    offs = grp * GS + tl.arange(0, GS)
    v = tl.load(X + row * XD + offs).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(v * v, axis=0) / GS + eps)
    n = (v * inv).to(tl.bfloat16).to(tl.float32)
    out = (tl.load(W + offs).to(tl.float32) * n).to(tl.bfloat16)
    tl.store(OUT + row * XD + offs, out)
    og = tl.reshape(out.to(tl.float32), (GS // 64, 64))
    tl.store(XS + row * (XD // 64) + grp * (GS // 64) + tl.arange(0, GS // 64), tl.sum(og, axis=1))


def group_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, groups: int):
    """(R, XD) -> weight * RMSNorm over groups of XD / groups (bf16, as mlx_lm), and its 64-input group sums."""

    rows, xd = x.shape
    out = torch.empty_like(x)
    xs = torch.empty((rows, xd // 64), dtype=torch.float32, device=x.device)
    _group_rmsnorm[(rows, groups)](x, w, out, xs, eps, XD=xd, GS=xd // groups, num_warps=4)
    return out, xs
