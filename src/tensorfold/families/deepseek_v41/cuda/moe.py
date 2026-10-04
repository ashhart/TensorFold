"""DeepSeek's MoE on one rank: routing, EXL3 routed experts, the shared expert, DSpark's NVFP4 experts."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.exl3 import experts as exl3
from tensorfold.cuda.nvfp4 import experts as nvx
from tensorfold.families.glm5_next.cuda import glue

from . import mx8

GUARD_CALLS = 160                   # eager routed passes a scratch checks for fp16 overflow: the first forwards' layers


@triton.jit
def _route(L, BIAS, PICK, WTS, temp, scale, NE: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr,
           BLOCK: tl.constexpr, SLOTP: tl.constexpr):
    """Routing for one row: sqrt-softplus scores, top-k of score + bias (lower ids win ties), normalized."""

    r = tl.program_id(0).to(tl.int64)
    ar = tl.arange(0, BLOCK)
    ak = tl.arange(0, SLOTP)
    ok = ar < NE
    z = tl.math.div_rn(tl.load(L + r * NE + ar, mask=ok, other=0.0), temp)
    score = tl.sqrt_rn(tl.where(z > 20.0, z, libdevice.log1p(libdevice.exp(z))))
    choice = tl.where(ok, score + tl.load(BIAS + ar, mask=ok, other=0.0), float("-inf"))
    picks = tl.zeros((SLOTP,), dtype=tl.int32)
    wts = tl.zeros((SLOTP,), dtype=tl.float32)
    total = 0.0
    for k in tl.static_range(TOPK):
        top = tl.max(choice, axis=0)
        idx = tl.min(tl.where(choice == top, ar, BLOCK), axis=0)
        sk = tl.sum(tl.where(ar == idx, score, 0.0), axis=0)
        picks = tl.where(ak == k, idx, picks)
        wts = tl.where(ak == k, sk, wts)
        total += sk
        choice = tl.where(ar == idx, float("-inf"), choice)
    wts = tl.math.div_rn(wts, total + 1e-20) * scale
    picks = tl.where(ak >= TOPK, NE, picks)
    wts = tl.where(ak >= TOPK, 1.0, wts)
    tl.store(PICK + r * SLOTS + ak, picks, mask=ak < SLOTS)
    tl.store(WTS + r * SLOTS + ak, wts, mask=ak < SLOTS)


def route(xf: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, mlog: torch.Tensor, pick: torch.Tensor,
          wts: torch.Tensor, top_k: int, scale: float, temp: float = 1.0) -> None:
    """Router logits of xf [R, D] into mlog fp32, then the routing into pick/wts [R, slots], shared last."""

    rows, slots = pick.shape
    experts = gate.shape[0]
    if slots not in (top_k, top_k + 1) or tuple(wts.shape) != (rows, slots) or tuple(mlog.shape) != (rows, experts):
        raise ValueError(f"route: pick {tuple(pick.shape)}, wts {tuple(wts.shape)}, mlog {tuple(mlog.shape)} "
                         f"for top-{top_k} of {experts}")
    if not (pick.is_contiguous() and wts.is_contiguous() and mlog.is_contiguous()):
        raise ValueError("route: pick, wts and mlog must be contiguous")
    glue.router(xf, gate, mlog)
    _route[(rows,)](mlog, bias, pick, wts, float(temp), float(scale), NE=experts, TOPK=top_k, SLOTS=slots,
                    BLOCK=triton.next_power_of_2(experts), SLOTP=triton.next_power_of_2(slots), num_warps=4,
                    enable_fp_fusion=False)


@triton.jit
def _swiglu(GU, g_stride, OUT, o_stride, limit, I: tl.constexpr, BLOCK: tl.constexpr):
    """SwiGLU: bf16(silu(min(g, limit)) * clamp(u, -limit, limit)) from bf16 [g | u], the math in fp32."""

    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = c < I
    g = tl.minimum(tl.load(GU + r * g_stride + c, mask=ok, other=0.0).to(tl.float32), limit)
    u = tl.load(GU + r * g_stride + I + c, mask=ok, other=0.0).to(tl.float32)
    u = tl.minimum(tl.maximum(u, -limit), limit)
    act = tl.math.div_rn(g, 1.0 + libdevice.exp(-g)) * u
    tl.store(OUT + r * o_stride + c, act.to(tl.bfloat16), mask=ok)


def swiglu(gu: torch.Tensor, out: torch.Tensor, limit: float) -> torch.Tensor:
    """SwiGLU: gu [R, 2I] bf16 (gate columns, then up) -> out [R, I] bf16."""

    rows, width = out.shape
    if gu.shape != (rows, 2 * width) or gu.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError(f"swiglu: gu {tuple(gu.shape)} -> out {tuple(out.shape)}, unit-stride rows")
    block = 1024
    _swiglu[(rows, triton.cdiv(width, block))](gu, gu.stride(0), out, out.stride(0), float(limit), I=width,
                                                BLOCK=block, num_warps=4, enable_fp_fusion=False)
    return out


def shared(cfg, m, xf: torch.Tensor, b, *, prompt: bool = False) -> torch.Tensor:
    """The rank's half of the shared expert: Mx8 [w1 | w3] rows, SwiGLU, Mx8 w2 columns -> b.sd [R, D] fp32."""

    R = xf.shape[0]
    mx8.mm(m.shared_gu, xf, b.sgu[:R], prompt=prompt)
    swiglu(b.sgu[:R], b.sact[:R], cfg.swiglu_limit)
    return mx8.mm(m.shared_d, b.sact[:R], b.sd[:R], f32=True, prompt=prompt)


def _guard(s: exl3.Scratch, pairs: int, comm=None) -> None:
    """Raise on every rank if any rank's eager fp16 activations overflowed; none in capture or past GUARD_CALLS."""

    left = getattr(s, "guard_left", GUARD_CALLS)
    if left <= 0 or torch.cuda.is_current_stream_capturing():
        return
    s.guard_left = left - 1
    flag = (torch.isinf(s.xg[:pairs]).any() | torch.isinf(s.xu[:pairs]).any() | torch.isinf(s.xd[:pairs]).any())
    flag = flag.to(torch.int32).reshape(1)
    if comm is not None:                # xd holds this rank's intermediate columns: the verdict is every rank's
        both = torch.empty((comm.world,), dtype=torch.int32, device=flag.device)
        comm.all_gather(flag, both)
        flag = both
    if bool(flag.any()):
        raise FloatingPointError("EXL3 expert activations overflow fp16")


def backbone(cfg, m, xf: torch.Tensor, b, *, prompt: bool = False, comm=None) -> torch.Tensor:
    """A backbone layer's MoE for xf [R, D] bf16 -> the rank's fp32 share b.part [R, D]."""

    R = xf.shape[0]
    route(xf, m.gate, m.bias, b.mlog[:R], b.pick[:R], b.wts[:R], cfg.num_experts_per_tok,
          cfg.routed_scaling_factor, cfg.gate_temp)
    s = b.exl3
    for r0 in range(0, R, s.rows):
        n = min(s.rows, R - r0)
        exl3.routed(xf[r0:r0 + n], b.pick[r0:r0 + n], b.wts[r0:r0 + n], m.experts, s, b.part[r0:r0 + n], n,
                    cfg.swiglu_limit, exl3.ACT_F32)
        _guard(s, n * s.slots, comm)
    shared(cfg, m, xf, b, prompt=prompt)
    return b.part[:R].add_(b.sd[:R])


def dspark_moe(cfg, m, xf: torch.Tensor, b) -> torch.Tensor:
    """A DSpark stage's MoE for its block rows xf [B, D] bf16 -> the rank's fp32 share b.part [B, D]."""

    R = xf.shape[0]
    E = cfg.dspark_n_routed_experts
    route(xf, m.gate, m.bias, b.dmlog[:R], b.dpick[:R], b.dwts[:R], cfg.dspark_num_experts_per_tok,
          cfg.routed_scaling_factor, cfg.gate_temp)
    grouped.route(b.dpick[:R], b.dplan)
    act, ey = b.dact[:R].view(-1, m.experts.width), b.dey[:R].view(-1, m.experts.dims)
    nvx.gate_up(xf, m.experts, b.dplan, act, R, skip=E)
    nvx.down(act, m.experts, b.dplan, ey, R, skip=E)
    b.dey[:R, -1].copy_(shared(cfg, m, xf, b))
    glue.combine(b.dey[:R], b.dwts[:R], b.part[:R])
    return b.part[:R]
