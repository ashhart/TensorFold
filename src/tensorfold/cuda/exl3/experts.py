"""Routed EXL3 experts of any codebook, a width per expert, one grouped launch a projection; rows never depend on the window."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch

CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2
ACT_BF16, ACT_F32 = 0, 1          # SwiGLU with the GLM family's bf16 roundings / in fp32
# Half-bits a value: 1..8 bits (2, 4, .. 16) and every half-integer rate 1.5..7.5 (3, 5, .. 15).
K2_SUPPORTED = tuple(range(2, 17))

# (n tiles a block, warps, K splits, tiles in flight): GLM's settings, whose arithmetic order this keeps bit for bit
GLM_GATEUP = (8, 4, 4, 1)
GLM_DOWN = (8, 4, 1, 1)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    srcs = [str(here / f) for f in ("experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu")]
    return load(name="tensorfold_exl3_experts_v1", sources=srcs, extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=False)


def codebook_id(name: str) -> int:
    """'3inst' / 'mcg' / 'mul1' (the checkpoint's quantization_config.codebook, or the marker tensor's name)."""

    ids = {"3inst": CB_3INST, "mcg": CB_MCG, "mul1": CB_MUL1}
    if name not in ids:
        raise ValueError(f"unknown EXL3 codebook {name!r}")
    return ids[name]


def k2_of(trellis: torch.Tensor) -> int:
    """Half-bits a value of a trellis int16 [K/16, N/16, 16 * K] (16 * K + 8 for the half-integer rates)."""

    w = trellis.shape[-1]
    if w % 8:
        raise ValueError(f"trellis last dim {w} is not a multiple of 8")
    k2 = w // 8
    if k2 not in K2_SUPPORTED:
        raise ValueError(f"unsupported EXL3 bit width {k2 / 2}")
    return k2


@dataclass
class Exl3RoutedExperts:
    """One layer's routed experts: trellis pointers and widths per projection, stacked suh/svh, the tensors keeping the trellises alive."""

    gate_ptr: torch.Tensor    # int64 [E]
    up_ptr: torch.Tensor
    down_ptr: torch.Tensor
    gate_k2: torch.Tensor     # int32 [E], half-bits a value
    up_k2: torch.Tensor
    down_k2: torch.Tensor
    suh_g: torch.Tensor       # fp16 [E, D]
    suh_u: torch.Tensor
    svh_g: torch.Tensor       # fp16 [E, I]
    svh_u: torch.Tensor
    suh_d: torch.Tensor       # fp16 [E, I]
    svh_d: torch.Tensor       # fp16 [E, D]
    count: int                # E
    dims: int                 # D (model width)
    width: int                # I (expert width on this rank)
    cb: int
    k2_gu: tuple[int, int]    # (min, max) K2 over gate and up
    k2_d: tuple[int, int]
    trellis_bytes: torch.Tensor   # int64 [E], gate + up + down trellis bytes of each expert (for GB/s)
    keep: list = field(default_factory=list, repr=False)

    def nbytes_read(self, ids: Sequence[int]) -> int:
        return int(self.trellis_bytes[list(ids)].sum())


def prepare(gate: Sequence[tuple], up: Sequence[tuple], down: Sequence[tuple], codebook: int | str,
            device="cuda") -> Exl3RoutedExperts:
    """A layer from per-expert (trellis, suh, svh) triples, trellises referenced in place, each at its own width."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    E = len(gate)
    if not (len(up) == len(down) == E) or E == 0:
        raise ValueError("gate, up and down need the same, non-zero number of experts")
    D, I = gate[0][0].shape[0] * 16, gate[0][0].shape[1] * 16
    keep = []

    def table(mats, k, n):
        ptrs, k2s = [], []
        for t, _, _ in mats:
            if t.dtype != torch.int16 or not t.is_contiguous() or t.device.type != "cuda":
                raise ValueError("trellis must be a contiguous CUDA int16 tensor")
            if t.shape[0] * 16 != k or t.shape[1] * 16 != n:
                raise ValueError(f"expert shape {tuple(t.shape)} does not match [{k // 16}, {n // 16}, *]")
            k2s.append(k2_of(t))
            ptrs.append(t.data_ptr())
            keep.append(t)
        return (torch.tensor(ptrs, dtype=torch.int64, device=device),
                torch.tensor(k2s, dtype=torch.int32, device=device), k2s)

    gp, gk, gks = table(gate, D, I)
    upp, uk, uks = table(up, D, I)
    dp, dk, dks = table(down, I, D)

    def stack(mats, j, n):
        out = torch.empty((E, n), dtype=torch.float16, device=device)
        for e, m in enumerate(mats):
            out[e].copy_(m[j].reshape(-1))
        return out

    tb = torch.tensor([(D * I // 256) * (gks[e] + uks[e] + dks[e]) * 16 for e in range(E)], dtype=torch.int64)
    return Exl3RoutedExperts(gp, upp, dp, gk, uk, dk, stack(gate, 1, D), stack(up, 1, D), stack(gate, 2, I),
                             stack(up, 2, I), stack(down, 1, I), stack(down, 2, D), E, D, I, cb,
                             (min(gks + uks), max(gks + uks)), (min(dks), max(dks)), tb, keep)


def prepare_stacked(gt: torch.Tensor, ut: torch.Tensor, dt: torch.Tensor, suh_g, suh_u, svh_g, svh_u, suh_d, svh_d,
                    codebook: int | str) -> Exl3RoutedExperts:
    """A uniform-width layer stacked per projection: trellis [E, K/16, N/16, 16K] (int16, or GLM's int32 words), suh/svh [E, n]."""

    def as16(t):
        return t.view(torch.int16) if t.dtype == torch.int32 else t

    gt, ut, dt = as16(gt), as16(ut), as16(dt)
    E = gt.shape[0]
    return prepare([(gt[e], suh_g[e], svh_g[e]) for e in range(E)], [(ut[e], suh_u[e], svh_u[e]) for e in range(E)],
                   [(dt[e], suh_d[e], svh_d[e]) for e in range(E)], codebook, device=gt.device)


@dataclass
class NemotronExl3Experts:
    """Gateless ReLU² experts with a padded EXL3 intermediate; ``keep`` owns the trellises behind the pointers."""

    up_ptr: torch.Tensor
    down_ptr: torch.Tensor
    up_k2: torch.Tensor
    down_k2: torch.Tensor
    suh_u: torch.Tensor
    svh_u: torch.Tensor
    suh_d: torch.Tensor
    svh_d: torch.Tensor
    count: int
    dims: int
    width: int                  # stored, padded intermediate width
    intermediate_size: int      # logical width (padded activations must be zero before down)
    cb: int
    k2_u: tuple[int, int]
    k2_d: tuple[int, int]
    keep: list = field(default_factory=list, repr=False)


def prepare_nemotron(up: Sequence[tuple], down: Sequence[tuple], codebook: int | str, *,
                     intermediate_size: int) -> NemotronExl3Experts:
    """Prepare ordered per-expert ``(trellis, suh, svh)`` up/down triples, already on one CUDA device.

    ``intermediate_size`` is the unpadded width; trellises include the padded columns and rows.
    ``suh``/``svh`` are fp16 scale vectors (not packed sign words).
    """
    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    if cb not in (CB_3INST, CB_MCG, CB_MUL1):
        raise ValueError(f"unknown EXL3 codebook id {cb}")
    if not up or len(up) != len(down):
        raise ValueError("up and down need the same, non-zero number of experts")
    first = up[0][0]
    if first.ndim != 3:
        raise ValueError("up trellis must be [K/16, N/16, 8 * k2]")
    D, I = first.shape[0] * 16, first.shape[1] * 16
    if D % 128 or I % 128 or not 0 < intermediate_size <= I:
        raise ValueError("D and padded width must be multiples of 128; logical intermediate_size must be in (0, width]")
    device = first.device
    if device.type != "cuda":
        raise ValueError("expert trellises must be on CUDA")
    keep = []

    def table(mats, k, n):
        ptrs, k2s, ins, outs = [], [], [], []
        for trellis, suh, svh in mats:
            if (trellis.ndim != 3 or trellis.shape[:2] != (k // 16, n // 16) or
                    trellis.dtype != torch.int16 or not trellis.is_contiguous() or trellis.device != device):
                raise ValueError(f"trellis must be contiguous CUDA int16 [{k // 16}, {n // 16}, 8 * k2]")
            if suh.dtype != torch.float16 or suh.numel() != k or svh.dtype != torch.float16 or svh.numel() != n:
                raise ValueError(f"suh/svh must be fp16 vectors of lengths {k}/{n}")
            ptrs.append(trellis.data_ptr())
            k2s.append(k2_of(trellis))
            ins.append(suh.reshape(-1).to(device))
            outs.append(svh.reshape(-1).to(device))
            keep.append(trellis)
        return (torch.tensor(ptrs, dtype=torch.int64, device=device),
                torch.tensor(k2s, dtype=torch.int32, device=device),
                torch.stack(ins), torch.stack(outs), (min(k2s), max(k2s)))

    upp, uk, su, sv, urange = table(up, D, I)
    dp, dk, sd, vd, drange = table(down, I, D)
    return NemotronExl3Experts(upp, dp, uk, dk, su, sv, sd, vd, len(up), D, I, intermediate_size, cb,
                               urange, drange, keep)


def default_config(K: int, N: int, gateup: bool) -> tuple[int, int, int, int]:
    """The tile setting for a K -> N projection (GLM's where it divides): the shape's alone, so rows stay independent."""

    cands = [GLM_GATEUP if gateup else GLM_DOWN, (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2)]
    for nt, w, sk, pf in cands:
        if K % (16 * sk * w) == 0 and N % (16 * nt) == 0:
            return nt, w, sk, pf
    raise ValueError(f"no tile setting divides K={K}, N={N}")


class Scratch:
    """Buffers for up to ``rows`` rows of ``slots`` slots; slots whose pick is not a routed expert are left to the caller."""

    def __init__(self, ex: Exl3RoutedExperts, rows: int, slots: int, cfg_gu=None, cfg_d=None, device="cuda") -> None:
        D, I = ex.dims, ex.width
        self.cfg_gu = cfg_gu or default_config(D, I, True)
        self.cfg_d = cfg_d or default_config(I, D, False)
        P = rows * slots
        self.xg = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xu = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xd = torch.zeros((P, I), dtype=torch.float16, device=device)
        # gate and up write 2 * splits * P * I partials, down splits * P * D
        self.z = torch.zeros((max(2 * self.cfg_gu[2] * I, self.cfg_d[2] * D) * P,), dtype=torch.float32, device=device)
        self.y = torch.zeros((P, D), dtype=torch.float32, device=device)
        maxu = min(P, ex.count)
        self.ids = torch.zeros((maxu,), dtype=torch.int32, device=device)
        self.count = torch.zeros((1,), dtype=torch.int32, device=device)
        self.members_buf = torch.full((maxu * rows,), -1, dtype=torch.int32, device=device)
        self.rows, self.slots, self.count_experts = rows, slots, ex.count

    def window(self, R: int):
        """(ids, members) sized for R rows: the grids only span what R rows can use."""

        maxu = min(R * self.slots, self.count_experts)
        return self.ids[:maxu], self.members_buf[:maxu * R].view(maxu, R)


def routed(x: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor | None, ex: Exl3RoutedExperts, s: Scratch,
           out: torch.Tensor | None, R: int, limit: float = math.inf, act_mode: int = ACT_F32,
           group: bool = True) -> torch.Tensor:
    """Routed experts of R rows (picks >= E skipped): Y per slot, or ``out`` = the wts-weighted sum when ``wts``; no host sync."""

    ext = _ext()
    D, I, E = ex.dims, ex.width, ex.count
    slots = s.slots
    P = R * slots
    if R > s.rows:
        raise ValueError(f"{R} rows but the scratch holds {s.rows}")
    ids, members = s.window(R)
    if group:
        ext.group(pick, ids, s.count, members, R, slots, E)
    ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E)
    nt, w, sk, pf = s.cfg_gu
    ext.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2, D, I,
                P, sk, slots, ex.cb, nt, w, pf, ex.k2_gu[0], ex.k2_gu[1])
    ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, sk, slots, E, float(limit), act_mode)
    nt, w, sk, pf = s.cfg_d
    ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1, I,
                D, P, sk, slots, ex.cb, nt, w, pf, ex.k2_d[0], ex.k2_d[1])
    if wts is None:
        ext.down_epilogue(s.z, pick, ex.svh_d, s.y, R, P, D, sk, slots, E)
        return s.y[:P]
    if out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    # the down epilogue and the combine in one launch (the same arithmetic in the same order as the two)
    ext.down_combine(s.z, pick, ex.svh_d, s.y, wts, out, R, P, D, sk, slots, E)
    return out


class NemotronScratch:
    """Reusable one-GPU scratch for up to ``rows`` rows and ``slots`` picks per row."""

    def __init__(self, ex: NemotronExl3Experts, rows: int, slots: int, cfg_up=None, cfg_down=None) -> None:
        if rows < 1 or not 1 <= slots <= 32:
            raise ValueError("rows must be positive and slots must be in 1..32")
        D, I, dev = ex.dims, ex.width, ex.up_ptr.device
        self.cfg_up = cfg_up or default_config(D, I, True)
        self.cfg_down = cfg_down or default_config(I, D, False)
        P = rows * slots
        self.xu = torch.empty((P, D), dtype=torch.float16, device=dev)
        self.xd = torch.empty((P, I), dtype=torch.float16, device=dev)
        self.z = torch.empty((max(self.cfg_up[2] * I, self.cfg_down[2] * D) * P,),
                             dtype=torch.float32, device=dev)
        self.y = torch.empty((P, D), dtype=torch.float32, device=dev)
        maxu = min(P, ex.count)
        self.ids = torch.empty((maxu,), dtype=torch.int32, device=dev)
        self.count = torch.empty((1,), dtype=torch.int32, device=dev)
        self.members_buf = torch.empty((maxu * rows,), dtype=torch.int32, device=dev)
        self.rows, self.slots, self.count_experts = rows, slots, ex.count

    def window(self, R: int):
        maxu = min(R * self.slots, self.count_experts)
        return self.ids[:maxu], self.members_buf[:maxu * R].view(maxu, R)


def routed_nemotron(x: torch.Tensor, pick: torch.Tensor, ex: NemotronExl3Experts, s: NemotronScratch,
                    R: int) -> torch.Tensor:
    """Evaluate precomputed int32 ``pick[R, slots]`` (< E routed, >= E skipped), per-slot fp32 [R, slots, D].

    Skipped slots are zero, so a separate shared expert can be added by the caller. The returned view
    aliases scratch and is overwritten on the next call. No host synchronization or route/weight computation.
    """
    D, I, E, slots = ex.dims, ex.width, ex.count, s.slots
    if not 1 <= R <= s.rows or x.shape != (R, D) or pick.shape != (R, slots):
        raise ValueError(f"expected x [{R}, {D}], pick [{R}, {slots}] and R in 1..{s.rows}")
    if x.device != ex.up_ptr.device or pick.device != x.device or pick.dtype != torch.int32:
        raise ValueError("x, experts and int32 pick must be on the same CUDA device")
    if x.dtype not in (torch.float16, torch.bfloat16) or x.stride(1) != 1 or not pick.is_contiguous():
        raise ValueError("x must be row-strided fp16/bf16; pick must be contiguous int32")
    ext = _ext()
    P = R * slots
    ids, members = s.window(R)
    ext.group(pick, ids, s.count, members, R, slots, E)
    ext.rot_in_single(x, x.stride(0), pick, ex.suh_u, s.xu, R, D, slots, E)
    nt, w, sk, pf = s.cfg_up
    ext.grouped(s.xu, s.xu, ex.up_ptr, ex.up_ptr, ex.up_k2, ex.up_k2, ids, s.count, members, s.z,
                1, D, I, P, sk, slots, ex.cb, nt, w, pf, ex.k2_u[0], ex.k2_u[1])
    ext.relu2_epilogue(s.z, pick, ex.svh_u, ex.suh_d, s.xd, R, P, I, ex.intermediate_size, sk, slots, E)
    nt, w, sk, pf = s.cfg_down
    ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z,
                1, I, D, P, sk, slots, ex.cb, nt, w, pf, ex.k2_d[0], ex.k2_d[1])
    s.y[:P].zero_()
    ext.down_epilogue(s.z, pick, ex.svh_d, s.y, R, P, D, sk, slots, E)
    return s.y[:P].view(R, slots, D)


def dequant(trellis: torch.Tensor, codebook: int | str) -> torch.Tensor:
    """W_q [K, N] fp16 of one matrix through the kernels' own lane decode (ExLlamaV3's ``reconstruct``); for tests."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    k2 = k2_of(trellis)
    K, N = trellis.shape[0] * 16, trellis.shape[1] * 16
    out = torch.empty((K, N), dtype=torch.float16, device=trellis.device)
    _ext().dequant(trellis.contiguous(), out, k2, cb)
    return out
