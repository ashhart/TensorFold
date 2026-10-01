"""The 27B's prefill: bits never depend on chunking but differ from decode's, so only prompt-end states resume."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from tensorfold.cuda import moe, prompt_precision
from tensorfold.cuda.kernels import gdn as deltanet
from tensorfold.cuda.kernels import qmm as shared
from tensorfold.cuda.kernels.prefill_attention import attention

from . import glue
from . import prefill_bf16, prefill_glue
from .decode import clone_state
from .forward import State, grow as _grow
from .qmm_fast import matmul, matmul_partial, tile
from .weights import Plain, QLinear, Weights

CHUNK = 4096
TAP_LAYERS = (5, 19, 33, 47, 61)


def _mm(x, w: QLinear, f32: bool = False) -> torch.Tensor:
    """``x``: bf16 rows (``prefill_bf16``), or e4m3 rows with group sums and row scales (``prefill_glue``, FP8)."""

    if isinstance(x, tuple):
        return shared.prefill_matmul8(x, tile(w), f32=f32) if isinstance(w, QLinear) else w.prefill8(x)
    if not isinstance(w, QLinear):
        return w.prefill(x)                               # an EXL3 pack's or an NVFP4 checkpoint's projection
    packed = tile(w)
    if packed.fast:                                       # each weight rounded once to bf16, one fp32 chain over K
        return shared.prefill_matmul(x, packed, f32=f32, tile=shared.prompt_tile(x.shape[0], packed.n))
    return matmul_partial(x, packed) if f32 else matmul(x, packed)


def _mm_group(x, ws: list) -> list[torch.Tensor]:
    """Projections of one input: checkpoint-math ones under one input scale quantize the rows once (same bits)."""

    shared = [i for i, w in enumerate(ws) if getattr(w, "act", None) is not None] if torch.is_tensor(x) else []
    got = {}
    if len(shared) > 1:
        from tensorfold.cuda.nvfp4 import checkpoint

        outs = checkpoint.matmul_group(x, [ws[i] for i in shared], prompt_rows=True)
        got = dict(zip(shared, outs)) if outs is not None else {}
    plain = [i for i, w in enumerate(ws) if isinstance(w, Plain) and i not in got] if torch.is_tensor(x) else []
    if len(plain) == 2:                                    # the GDN gates b and a: one launch, each its own bits
        from .b16 import prompt_pair

        got.update(zip(plain, prompt_pair(x, ws[plain[0]].weight, ws[plain[1]].weight)))
    return [got[i] if i in got else _mm(x, w) for i, w in enumerate(ws)]


FUSED_MLP = True           # checkpoint-math NVFP4 MLPs: SwiGLU inside the gate|up GEMM, its rows straight to down


def _mlp(h, layer, pg, tp: bool) -> torch.Tensor:
    """A dense layer's MLP over prompt rows; checkpoint-math NVFP4 layers take the fused gate|up epilogue."""

    if FUSED_MLP and not tp and torch.is_tensor(h) and getattr(layer.down, "act", None) is not None:
        from tensorfold.cuda.nvfp4 import checkpoint

        y = checkpoint.mlp_prompt(h, layer.gate, layer.up, layer.down)
        if y is not None:
            return y
    return _row_mm(pg.swiglu(*_mm_group(h, [layer.gate, layer.up])), layer.down, tp)


def _row_mm(x, w: QLinear, tp: bool) -> torch.Tensor:
    if not tp:
        return _mm(x, w)
    from .distributed import gather_rank_partials

    return gather_rank_partials(_mm(x, w))                 # bf16 partials: half the bytes of fp32 over the link


@torch.no_grad()
def prefill_chunk(w: Weights, tokens: torch.Tensor, st: State, *, tp: bool = False, capture_taps: bool = False,
                  last: bool = True, every: bool = False, cut: int = 0, vision=None):
    """Commit ``tokens`` at [st.pos, st.pos + W) into ``st`` without writing through its entries (``every``: all rows' final normed states; ``cut``: also the state after the first ``cut`` rows, the GDN chains run as two launches with one launch's bits)."""

    c = w.config
    pg = prefill_glue if w.fast_prefill and prompt_precision.fp8() else prefill_bf16   # e4m3 rows when prompts take FP8
    W = int(tokens.shape[0])
    if not 0 <= cut < W:
        raise ValueError(f"cut {cut} is not inside a chunk of {W} rows")
    p0 = st.pos
    keep = c.conv_kernel - 1
    dev = tokens.device
    pos = (torch.arange(p0, p0 + W, device=dev, dtype=torch.int32) if vision is None
           else vision.positions[:, p0:p0 + W].contiguous())
    windows = (torch.arange(W, device=dev, dtype=torch.int32)[:, None]
               + torch.arange(keep + 1, device=dev, dtype=torch.int32)[None, :])
    x = glue.embedding(tokens.to(torch.int32), w.embed)
    if vision is not None:
        from tensorfold.vision.qwen_cuda import replace_rows

        x = replace_rows(x, vision, p0, p0 + W)
    pending: torch.Tensor | None = None
    taps: list[torch.Tensor] = []
    part = clone_state(st) if cut else None     # its attention buffers are the chunk's, through the shared list
    for i, layer in enumerate(w.layers):
        x, h = pg.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        if layer.linear:
            gdn = layer.gdn
            if gdn.zba is not None:
                qkv, zba = _mm_group(h, [gdn.qkv, gdn.zba])
                vd = c.v_heads * c.dv
                z = zba[:, :vd].contiguous().reshape(W, c.v_heads, c.dv)
                b = zba[:, vd:vd + c.v_heads].contiguous()
                a = zba[:, vd + c.v_heads:].contiguous()
            else:
                qkv, z, b, a = _mm_group(h, [gdn.qkv, gdn.z, gdn.b, gdn.a])
                z = z.reshape(W, c.v_heads, c.dv)
            q, k, v, g, beta = glue.gdn_pre(qkv, st.conv[i], gdn.conv, windows, a, b, gdn.A_log, gdn.dt_bias,
                                            kh=c.k_heads, vh=c.v_heads, dk=c.dk)
            final = torch.empty_like(st.rec[i])
            if part is None:
                yr = deltanet.chain(q, k, v, g, beta, st.rec[i], final)
            else:
                part.rec[i] = torch.empty_like(st.rec[i])
                yr = torch.cat([deltanet.chain(q[:cut], k[:cut], v[:cut], g[:cut], beta[:cut], st.rec[i], part.rec[i]),
                                deltanet.chain(q[cut:], k[cut:], v[cut:], g[cut:], beta[cut:], part.rec[i], final)])
                part.conv[i] = torch.cat([st.conv[i], qkv[max(0, cut - keep):cut]])[-keep:].contiguous()
            r = _row_mm(pg.gated_norm(yr, z, gdn.norm, c.eps), gdn.out, tp)
            st.conv[i] = torch.cat([st.conv[i], qkv[-keep:]])[-keep:].contiguous()
            st.rec[i] = final
        else:
            attn = layer.attn
            if attn.kv is not None:
                qg, kv = _mm_group(h, [attn.q, attn.kv])
                kd = c.kv_heads * c.head_dim
                key = kv[:, :kd].contiguous()
                value = kv[:, kd:].contiguous().reshape(W, c.kv_heads, c.head_dim)
            else:
                qg, key, value = _mm_group(h, [attn.q, attn.k, attn.v])
                value = value.reshape(W, c.kv_heads, c.head_dim)
            q, key = glue.attn_prep(qg, key, attn.q_norm, attn.k_norm, pos, w.inv_freq, c.eps, heads=c.heads,
                                    kv_heads=c.kv_heads, head_dim=c.head_dim, mrope_section=c.mrope_section)
            kbuf, vbuf = _grow(st, i, p0 + W)
            kbuf[p0:p0 + W] = key.view(W, c.kv_heads, c.head_dim)
            vbuf[p0:p0 + W] = value
            out = attention(q.view(W, c.heads, c.head_dim), kbuf, vbuf, p0, scale=c.head_dim ** -0.5)
            r = _row_mm(pg.gate_mul(out, qg, heads=c.heads, head_dim=c.head_dim), attn.o, tp)
        if layer.moe is not None:                          # routed experts read bf16 rows (their prefill form)
            x, h, _ = glue.add_rmsnorm(x, r, layer.post_norm, c.eps)
            pending = moe.run(h, layer.moe, prefill=True)
        else:
            x, h = pg.add_rmsnorm(x, r, layer.post_norm, c.eps)
            pending = _mlp(h, layer, pg, tp)
        if capture_taps and i in TAP_LAYERS:
            taps.append((x.float() + pending.float()).to(torch.bfloat16))
    st.pos = p0 + W
    normed = None
    if every:
        _, normed, _ = glue.add_rmsnorm(x, pending, w.norm, c.eps)
    elif last:
        _, normed, _ = glue.add_rmsnorm(x[-1:].contiguous(), pending[-1:].contiguous(), w.norm, c.eps)
    taps_out = torch.cat(taps, dim=-1) if capture_taps else None
    if part is None:
        return normed, taps_out
    part.pos = p0 + cut                             # the chunk's buffers: their rows below part.pos stay as committed
    return normed, taps_out, part


def chunks(start: int, end: int, size: int = CHUNK) -> list[tuple[int, int]]:
    """Even chunks of at most ``size`` rows (a short one costs a whole weight pass); any bounds give the same bits."""

    n = -(-(end - start) // size)
    return [(start + (end - start) * j // n, start + (end - start) * (j + 1) // n) for j in range(n)] if n else []


@torch.no_grad()
def prefill_state(w: Weights, prompt: Sequence[int], st: State, *, tp: bool = False, draft=None,
                  size: int | None = None, keep_at: int | None = None, vision=None):
    """Commit prompt[st.pos:] into ``st``, tapping the drafter's window; ``keep_at``: ``(normed, (state, snapshot))``, the state after prompt[:keep_at] from a cut chunk."""

    dev = w.norm.device
    base, n = st.pos, len(prompt)
    if keep_at is not None and not base <= keep_at <= n:
        raise ValueError(f"keep_at {keep_at} is outside the prefilled range [{base}, {n}]")
    if vision is not None:                         # a later prefill step goes on with the rope its first step set
        if base and getattr(st, "rope_delta", None) is not vision.rope_delta:
            raise ValueError("image prompts require a fresh prefill state")
        st.rope_delta = vision.rope_delta
    ids = torch.tensor(list(prompt[base:]), dtype=torch.int32, device=dev)
    normed, kept = None, None
    end = n if keep_at is None else keep_at        # the drafter's window then also covers the kept point
    tap_from = base
    if draft is not None and end - draft.window > base:
        tap_from = end - draft.window
        draft.skip(tap_from - base)
    spans = chunks(base, n, size or getattr(w, "prompt_rows", CHUNK))       # stand-in weights take 4096
    for j, (a, b) in enumerate(spans):
        if keep_at == a:
            kept = (clone_state(st), draft.snapshot() if draft is not None else None)
        cut = keep_at - a if keep_at is not None and a < keep_at < b else 0
        want = draft is not None and b > tap_from
        normed, taps, *part = prefill_chunk(w, ids[a - base:b - base], st, tp=tp, capture_taps=want,
                                            last=j == len(spans) - 1, cut=cut, vision=vision)
        snap = None
        if want:
            rows = taps[max(0, tap_from - a):]
            if cut:                                # the drafter at the point, then the rest of the chunk
                split = keep_at - max(a, tap_from)
                if split:
                    draft.add_taps(rows[:split])
                snap, rows = draft.snapshot(), rows[split:]
            draft.add_taps(rows)
        if part:
            kept = (part[0], snap)
    if keep_at is None:
        return normed
    if keep_at == n:
        kept = (clone_state(st), draft.snapshot() if draft is not None else None)
    return normed, kept


@dataclass
class Piece:
    """A stream's step of a batched prefill: ``prompt`` through the step's stop (rows from ``st.pos`` on), the state it
    commits into, where to keep a state (or None) and the drafter's context before it (None: this stream drafts not)."""

    prompt: Sequence[int]
    st: State
    keep_at: int | None = None
    snap: Any = None


def _pinned(values: np.ndarray, device) -> torch.Tensor:
    """An int32 table on the GPU from pinned memory: no stream sync, as a pageable copy would make."""

    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.int32)).pin_memory().to(device, non_blocking=True)


@torch.no_grad()
def prefill_rows(w: Weights, items: list[tuple[Sequence[int], State, int]], *, tp: bool = False,
                 capture_taps: bool = False) -> list[tuple[torch.Tensor, torch.Tensor | None, State | None]]:
    """``prefill_chunk`` for several streams in one forward (dense text layers): ``items`` are (ids, state, cut).
    Projections, norms and MLPs run over every row at once, each row's bits its own; each stream's convolution
    windows, DeltaNet chains and attention read its own state. Per item: (last row normed, taps, the state at cut)."""

    c = w.config
    if any(layer.moe is not None for layer in w.layers):
        raise ValueError("a batched prefill takes dense layers only")
    pg = prefill_glue if w.fast_prefill and prompt_precision.fp8() else prefill_bf16
    sts = [st for _, st, _ in items]
    sizes = [len(ids) for ids, _, _ in items]
    starts = np.concatenate([[0], np.cumsum(sizes)]).tolist()
    W, keep, dev = starts[-1], c.conv_kernel - 1, w.norm.device
    p0s = [st.pos for st in sts]
    for (_, _, cut), n in zip(items, sizes):
        if not 0 <= cut < n:
            raise ValueError(f"cut {cut} is not inside a piece of {n} rows")
    local = [np.arange(n)[:, None] + np.arange(keep + 1)[None, :] for n in sizes]
    windows = _pinned(np.concatenate([np.where(t < keep, t, t + o) for t, o in zip(local, starts)]), dev)
    sids = _pinned(np.repeat(np.arange(len(items)), sizes), dev)
    pos = _pinned(np.concatenate([np.arange(p, p + n) for p, n in zip(p0s, sizes)]), dev)
    ids = _pinned(np.concatenate([np.asarray(ids, dtype=np.int64) for ids, _, _ in items]), dev)
    x = glue.embedding(ids, w.embed)
    pending: torch.Tensor | None = None
    taps: list[torch.Tensor] = []
    parts = [clone_state(st) if cut else None for _, st, cut in items]
    spans = list(zip(starts, sizes))
    for i, layer in enumerate(w.layers):
        x, h = pg.add_rmsnorm(x, pending, layer.input_norm, c.eps)
        if layer.linear:
            gdn = layer.gdn
            if gdn.zba is not None:
                qkv, zba = _mm_group(h, [gdn.qkv, gdn.zba])
                vd = c.v_heads * c.dv
                z = zba[:, :vd].contiguous().reshape(W, c.v_heads, c.dv)
                b = zba[:, vd:vd + c.v_heads].contiguous()
                a = zba[:, vd + c.v_heads:].contiguous()
            else:
                qkv, z, b, a = _mm_group(h, [gdn.qkv, gdn.z, gdn.b, gdn.a])
                z = z.reshape(W, c.v_heads, c.dv)
            q, k, v, g, beta = glue.gdn_pre(qkv, torch.cat([st.conv[i] for st in sts]), gdn.conv, windows, a, b,
                                            gdn.A_log, gdn.dt_bias, kh=c.k_heads, vh=c.v_heads, dk=c.dk,
                                            stream_ids=sids, nkeep=keep)
            ys = []
            for st, part, (_, _, cut), (o, n) in zip(sts, parts, items, spans):
                final = torch.empty_like(st.rec[i])
                rows = lambda lo, hi: (q[lo:hi], k[lo:hi], v[lo:hi], g[lo:hi], beta[lo:hi])  # noqa: E731
                if part is None:
                    ys.append(deltanet.chain(*rows(o, o + n), st.rec[i], final))
                else:                                      # two launches, one launch's bits: the state at the cut
                    part.rec[i] = torch.empty_like(st.rec[i])
                    ys.append(deltanet.chain(*rows(o, o + cut), st.rec[i], part.rec[i]))
                    ys.append(deltanet.chain(*rows(o + cut, o + n), part.rec[i], final))
                    part.conv[i] = torch.cat([st.conv[i], qkv[o + max(0, cut - keep):o + cut]])[-keep:].contiguous()
                st.conv[i] = torch.cat([st.conv[i], qkv[o + max(0, n - keep):o + n]])[-keep:].contiguous()
                st.rec[i] = final
            r = _row_mm(pg.gated_norm(torch.cat(ys), z, gdn.norm, c.eps), gdn.out, tp)
        else:
            attn = layer.attn
            if attn.kv is not None:
                qg, kv = _mm_group(h, [attn.q, attn.kv])
                kd = c.kv_heads * c.head_dim
                key = kv[:, :kd].contiguous()
                value = kv[:, kd:].contiguous().reshape(W, c.kv_heads, c.head_dim)
            else:
                qg, key, value = _mm_group(h, [attn.q, attn.k, attn.v])
                value = value.reshape(W, c.kv_heads, c.head_dim)
            q, key = glue.attn_prep(qg, key, attn.q_norm, attn.k_norm, pos, w.inv_freq, c.eps, heads=c.heads,
                                    kv_heads=c.kv_heads, head_dim=c.head_dim, mrope_section=c.mrope_section)
            q, key = q.view(W, c.heads, c.head_dim), key.view(W, c.kv_heads, c.head_dim)
            outs = []
            for st, p0, (o, n) in zip(sts, p0s, spans):
                kbuf, vbuf = _grow(st, i, p0 + n)
                kbuf[p0:p0 + n] = key[o:o + n]
                vbuf[p0:p0 + n] = value[o:o + n]
                outs.append(attention(q[o:o + n], kbuf, vbuf, p0, scale=c.head_dim ** -0.5))
            r = _row_mm(pg.gate_mul(torch.cat(outs), qg, heads=c.heads, head_dim=c.head_dim), attn.o, tp)
        x, h = pg.add_rmsnorm(x, r, layer.post_norm, c.eps)
        pending = _mlp(h, layer, pg, tp)
        if capture_taps and i in TAP_LAYERS:
            taps.append((x.float() + pending.float()).to(torch.bfloat16))
    every = torch.cat(taps, dim=-1) if capture_taps else None
    out = []
    for st, part, p0, (_, _, cut), (o, n) in zip(sts, parts, p0s, items, spans):
        st.pos = p0 + n
        _, normed, _ = glue.add_rmsnorm(x[o + n - 1:o + n].contiguous(), pending[o + n - 1:o + n].contiguous(),
                                        w.norm, c.eps)
        if part is not None:
            part.pos = p0 + cut                     # the piece's buffers: their rows below part.pos stay as committed
        out.append((normed, None if every is None else every[o:o + n], part))
    return out


@torch.no_grad()
def prefill_batch(w: Weights, pieces: list[Piece], *, tp: bool = False, draft=None) -> list[tuple]:
    """``prefill_state`` for several streams in one forward (``prefill_rows``): every result has the bits
    ``prefill_state`` gives its piece alone. Per piece: (the last row's normed state, ``(state, snapshot)`` at
    ``keep_at`` or None, the drafter's context after or None). The caller keeps the rows within one forward's."""

    plans = []
    for p in pieces:
        base, n = p.st.pos, len(p.prompt)
        if not base < n:
            raise ValueError(f"nothing to prefill: the state is at {base} of a {n}-token prompt")
        if p.keep_at is not None and not base <= p.keep_at <= n:
            raise ValueError(f"keep_at {p.keep_at} is outside the prefilled range [{base}, {n}]")
        drafting = draft is not None and p.snap is not None
        end = n if p.keep_at is None else p.keep_at        # the drafter's window then also covers the kept point
        tap_from = end - draft.window if drafting and end - draft.window > base else base
        cut = p.keep_at - base if p.keep_at is not None and base < p.keep_at < n else 0
        kept = None
        if p.keep_at == base:
            kept = (clone_state(p.st), (list(p.snap[0]), list(p.snap[1]), p.snap[2], p.snap[3]) if drafting else None)
        plans.append((base, n, drafting, tap_from, cut, kept))
    outs = prefill_rows(w, [(p.prompt[base:], p.st, cut) for p, (base, _, _, _, cut, _) in zip(pieces, plans)],
                        tp=tp, capture_taps=any(plan[2] for plan in plans))
    results = []
    for p, (base, n, drafting, tap_from, cut, kept), (normed, taps, part) in zip(pieces, plans, outs):
        snap = at_cut = None
        if drafting:                                       # each stream's drafter context in turn, as alone
            draft.restore(p.snap)
            if tap_from > base:
                draft.skip(tap_from - base)
            rows = taps[tap_from - base:]
            if cut:
                split = p.keep_at - tap_from
                if split:
                    draft.add_taps(rows[:split])
                at_cut, rows = draft.snapshot(), rows[split:]
            draft.add_taps(rows)
            snap = draft.snapshot()
        if part is not None:
            kept = (part, at_cut)
        if p.keep_at == n:
            kept = (clone_state(p.st), snap)
        results.append((normed, kept, snap))
    if draft is not None and any(plan[2] for plan in plans):
        draft.skip(0)                                      # no context of its own: rounds read the streams' snaps
    return results
