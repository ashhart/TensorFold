"""DSpark drafting on one rank: stage rings, a 5-row block through three stages, then Markov draws."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from tensorfold.cuda.comm import fast_gather
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import glue, qmm

from ..config import Config
from . import attn_kernel, moe, mx8, norms, quant, rope, sample
from .buffers import Buffers, State
from .forward import _gather
from .weights import StageW, Weights


class Work:
    """The drafter's own device scratch: block and Markov input ids, the block rows' anchors and lists, a bias row."""

    def __init__(self, cfg: Config, world: int, device: torch.device | str = "cuda") -> None:
        dev, i32 = torch.device(device), torch.int32
        B, win = cfg.dspark_block_size, cfg.sliding_window
        self.bids = torch.full((B,), cfg.dspark_noise_token_id, dtype=i32, device=dev)    # [y, noise, ...]
        self.mids = torch.zeros((B,), dtype=i32, device=dev)          # out_i: Markov step i's input token
        self.anchors = torch.zeros((B,), dtype=i32, device=dev)       # p for every block row (M:1021-1029)
        self.lists = torch.arange(B, dtype=i32, device=dev).repeat(B, 1)    # every row sees all block rows
        self.counts = torch.full((B,), B, dtype=i32, device=dev)
        self.bias = torch.empty((1, cfg.vocab_size // world), dtype=torch.float32, device=dev)
        self.rows = torch.arange(win, dtype=torch.int64, device=dev)
        self.at = torch.empty((win,), dtype=torch.int64, device=dev)   # absorbed positions
        self.sampling = 0.0                                             # host seconds of the last propose's draws


@torch.no_grad()
def absorb(e, b: Buffers, n: int, *, prompt: bool) -> None:
    """The last ``n`` committed positions' taps into every stage's ring (M:1039-1051, 1128-1130)."""

    w, st, k = e.w, e.st, e.dwork
    cfg, ds = w.cfg, w.dspark
    if not 0 < n <= min(b.taps.shape[0], st.pos):
        raise ValueError(f"absorb: {n} rows at {st.pos} committed positions, {b.taps.shape[0]} tap rows")
    eps, x = cfg.rms_norm_eps, b.mx[:n]
    taps = b.taps[:n].view(n, -1)
    if w.world == 1:
        mx8.mm(ds.main_proj, taps, x, prompt=prompt)
    else:                               # the rank's output columns, gathered and laid side by side in rank order
        part = b.mx.view(-1)[:x.numel() // w.world].view(n, -1)
        mx8.mm(ds.main_proj, taps, part, prompt=prompt)
        recv = b.mx_gat.view(-1)[:x.numel()].view(w.world, n, -1)
        fast_gather(w.comm, part, recv)
        x.view(n, w.world, -1).copy_(recv.transpose(0, 1))
    main_x = norms.rmsnorm(x, ds.main_norm, eps, b.xn[:n])
    at = torch.add(k.rows[:n], st.pos_dev, out=k.at[:n]).sub_(n)
    slots = at.remainder(st.rings.shape[1])
    for sw in ds.stages:
        qakv = mx8.mm(sw.attn.wqa_kv, main_x, b.qakv[:n], prompt=prompt)
        kv = norms.rmsnorm(qakv[:, cfg.q_lora_rank:], sw.attn.kv_norm, eps, b.kvw[sw.index, :n])
        quant.fp8_qdq_1x32(rope.apply(kv, at, w.rope[sw.role.rope]))
        st.rings[sw.index].index_copy_(0, slots, kv)


def _stage(sw: StageW, w: Weights, st: State, b: Buffers, k: Work) -> None:
    """One stage (M:968-994, attention as M:1054-1074) on the block stream ``b.xd`` in place."""

    cfg, a, L = w.cfg, sw.attn, sw.index
    B, eps, table, pos = cfg.dspark_block_size, cfg.rms_norm_eps, w.rope[sw.role.rope], st.pos_dev
    flat = b.xd.view(B, -1)
    h = sw.hc_attn
    norms.hc_mix(flat, h.fn, h.base, h.scale, b.hcpart[:B], b.pre_a[:B], b.post_a[:B], b.comb_a[:B], eps,
                 cfg.hc_eps, cfg.hc_sinkhorn_iters)
    xa = norms.collapse_norm(flat, b.pre_in[:B], sw.attn_norm, b.xn[:B], eps)
    qakv = mx8.mm(a.wqa_kv, xa, b.qakv[:B])
    qr = norms.rmsnorm(qakv[:, :cfg.q_lora_rank], a.q_norm, eps, b.qr[:B])
    kv = norms.rmsnorm(qakv[:, cfg.q_lora_rank:], a.kv_norm, eps, b.bkv)
    q = b.q[:B]
    mx8.mm(a.wq_b, qr, q.view(B, -1))
    rope.apply(q, pos, table)
    quant.fp8_qdq_1x32(rope.apply(kv, pos, table))
    o = attn_kernel.attention(q, st.rings[L], None, pos, k.anchors, kv, k.lists, k.counts, a.sink, b.o[:B],
                              prompt=False, part=b.attn_part)
    rope.apply(o, pos, table, inverse=True)
    u = mx8.grouped(a.wo_a, o.view(B, -1), b.u[:B])
    mx8.mm(a.wo_b, u, b.part[:B], f32=True)
    glue.hc_post(flat, flat, _gather(w, b, B), b.post_a[:B], b.comb_a[:B])
    h = sw.hc_ffn
    norms.hc_mix(flat, h.fn, h.base, h.scale, b.hcpart[:B], b.pre_f[:B], b.post_f[:B], b.comb_f[:B], eps,
                 cfg.hc_eps, cfg.hc_sinkhorn_iters)
    xf = norms.collapse_norm(flat, b.pre_a[:B], sw.ffn_norm, b.xn[:B], eps)    # delayed mHC: the attention's pre
    moe.dspark_moe(cfg, sw.moe, xf, b)
    glue.hc_post(flat, flat, _gather(w, b, B), b.post_f[:B], b.comb_f[:B])
    b.pre_in[:B].copy_(b.pre_f[:B])


@torch.no_grad()
def block(e) -> torch.Tensor:
    """The three stages, collapse and norm for ``dwork.bids`` (M:1128-1146), then the head into ``dbuf.dlog``."""

    w, st, b, k = e.w, e.st, e.dbuf, e.dwork
    cfg, ds = w.cfg, w.dspark
    B = cfg.dspark_block_size
    glue.embed(k.bids, w.embed, cfg.hidden_size, cfg.hc_mult, b.xd)
    b.pre_in[:B].zero_()
    b.pre_in[:B, 0].fill_(1.0)
    torch.sub(st.pos_dev.expand(B), 1, out=k.anchors)
    for sw in ds.stages:
        _stage(sw, w, st, b, k)
    x = norms.collapse_norm(b.xd.view(B, -1), b.pre_in[:B], ds.norm, b.fnormed[:B], cfg.rms_norm_eps,
                            raw=b.hidden[:B])
    return mx8.mm(w.head, x, b.dlog, f32=True)


@triton.jit
def _markov_in(IDS, EMB, X, PROJ, ME, CONF, i, D: tl.constexpr, RK: tl.constexpr, BLOCK: tl.constexpr):
    """Row i: me = markov_embed[ids[i]] (a bf16 copy); confidence = proj . f32(cat(x_i, me)) as one fp32 sum."""

    n = tl.arange(0, BLOCK)
    tok = tl.load(IDS + i).to(tl.int64)
    isx = n < D
    ism = (n >= D) & (n < D + RK)
    me = tl.load(EMB + tok * RK + (n - D), mask=ism, other=0.0)
    tl.store(ME + i * RK + (n - D), me, mask=ism)
    x = tl.load(X + i * D + n, mask=isx, other=0.0)
    v = tl.where(isx, x.to(tl.float32), me.to(tl.float32))
    tl.store(CONF + i, tl.sum(v * tl.load(PROJ + n, mask=n < D + RK, other=0.0), axis=0))


def markov_input(e, i: int) -> None:
    """Row i's Markov embedding into ``dbuf.me[i]`` and confidence logit into ``dbuf.conf[i]`` (M:1149-1156)."""

    w, b, k = e.w, e.dbuf, e.dwork
    D, RK = w.cfg.hidden_size, w.cfg.dspark_markov_rank
    _markov_in[(1,)](k.mids, w.dspark.markov_embed, b.hidden, w.dspark.conf, b.me, b.conf, i, D=D, RK=RK,
                     BLOCK=triton.next_power_of_2(D + RK), num_warps=8)


def markov_step(e, i: int) -> torch.Tensor:
    """Markov step i's device part: ``dbuf.dlog[i] += markov_head(me_i)``, fp32 on the rank's vocabulary rows."""

    w, b, k = e.w, e.dbuf, e.dwork
    qmm.matmul(b.me[i:i + 1], w.dspark.markov_head, out=k.bias, f32=True)
    return b.dlog[i:i + 1].add_(k.bias)


def _sigmoid(c: float) -> float:
    return 1.0 / (1.0 + math.exp(-c)) if c >= 0 else math.exp(c) / (1.0 + math.exp(c))


def policy(confidence: Sequence[float], d_fixed: int, conf_threshold: float | None = None) -> int:
    """Drafts a round: ``d_fixed``, or the most whose sigmoid product is >= ``conf_threshold``, at least 1."""

    if conf_threshold is None:
        return d_fixed
    n, prod = min(d_fixed, len(confidence)), 1.0
    for i, c in enumerate(confidence[:n]):
        prod *= _sigmoid(float(c))
        if prod < conf_threshold:
            return max(i, 1)
    return n


@torch.no_grad()
def propose(e, y: int, p: int, sampling: Sampling | None, d: int, conf_threshold: float | None = None, *,
            steps=None, landed=None) -> tuple[list[int], list[float]]:
    """The block for ``y`` after position ``p``, then Markov draws -> (drafts, confidence logits), alike on ranks."""

    w, st, b, k = e.w, e.st, e.dbuf, e.dwork
    B = w.cfg.dspark_block_size
    if p != st.pos - 1 or p < 0 or not 1 <= d <= B:
        raise ValueError(f"propose: p {p} with {st.pos} committed positions, {d} drafts of a {B}-row block")
    run_block, run_input, run_step = steps or (block, markov_input, markov_step)
    k.bids[:1].fill_(y)
    k.mids[:1].fill_(y)
    run_block(e)
    drafts: list[int] = []
    conf: list[float] = []
    k.sampling = 0.0
    for i in range(d):
        run_input(e, i)
        if conf_threshold is not None:
            conf.append(float(b.conf[i]))
            if policy(conf, d, conf_threshold) <= i:
                break
        run_step(e, i)
        (tok,), seconds = sample.draft_rows(w, b.dlog[i:i + 1], [p + 2 + i], sampling)
        k.sampling += seconds
        drafts.append(tok)
        if landed is not None:
            landed(i, tok)
        if i + 1 < B:
            k.mids[i + 1:i + 2].fill_(tok)
    return drafts, (conf if conf_threshold is not None else b.conf[:d].tolist())[:len(drafts)]
