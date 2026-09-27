"""Routed EXL3 experts of any storage on CUDA, one grouped launch per projection for a whole layer.

Covers every EXL3 expert storage: codebooks 3inst (0), mcg (1) and mul1 (2), and a bit width per expert matrix
(1 to 8 bits in half-bit steps: every width ExLlamaV3 writes, which is 1..8 and, with mul1, 1.5 / 2.5 / 3.5, plus the
half-integer widths it does not write), mixed freely inside one layer (mixed-K packs such as MiMo-V2.6-Flash
2.50bpw or a SAGE pack) as well as uniform packs. A window's routed (row, slot) pairs go through:

    group            distinct experts of the window (ids ascending) and each one's member (row, slot) pairs
    rot_in           Xg, Xu = fp16((x * suh) @ H) for each pair's expert (gate and up have their own suh)
    grouped (x2)     Zg, Zu = X @ W_q for every distinct expert's rows, per fixed K split (fp32)
    gateup_epilogue  g, u = (sum of splits) @ H * svh; act = SwiGLU; Xd = fp16((act * suh_d) @ H)
    grouped          Zd = Xd @ W_q(down), per fixed K split
    down_epilogue    Y[row, slot] = (sum of splits) @ H * svh_d      (fp32)
    combine          out[row] = sum over slots in order of w[row, slot] * Y[row, slot]

H is the 128-block Hadamard / sqrt(128). Tile settings and splits depend only on the shape, so a row's bits never
depend on the window's other rows. With ``GLM_GATEUP`` / ``GLM_DOWN`` and ``act_mode=ACT_BF16`` on 4-bit mcg
experts every kernel does the same arithmetic in the same order as TensorFold's GLM path
(``families/glm5_next/cuda/exl3.cu``), so Y is bit-identical to it (tests/cuda/test_exl3_experts.py).

Weights are not copied: an expert matrix is read in place from its trellis tensor (``int16 [K/16, N/16, 16 * K]``
as stored in the checkpoint) through a pointer table; only the suh/svh vectors are stacked per layer.
"""

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

# (n tiles a block, warps a block, K splits, tiles in flight) of the grouped GEMV. The GLM family's settings, whose
# arithmetic order this reproduces bit for bit:
GLM_GATEUP = (8, 4, 4, 1)
GLM_DOWN = (8, 4, 1, 1)


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

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
    """One layer's routed experts (or one rank's share of their width): trellis pointer and bit-width tables per
    projection, stacked suh/svh, and the tensors that keep the trellises alive."""

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
    """Build a layer from per-expert ``(trellis, suh, svh)`` triples (trellis int16 [K/16, N/16, 16K] contiguous on
    ``device``; suh [K], svh [N] fp16). Trellises are referenced, not copied; each may have its own bit width."""

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
    """A uniform-bit layer already stacked per projection: trellis int16 [E, K/16, N/16, 16K] (or int32 words
    [E, K/16, N/16, 8K], as the GLM family keeps them) and suh/svh [E, n]."""

    def as16(t):
        return t.view(torch.int16) if t.dtype == torch.int32 else t

    gt, ut, dt = as16(gt), as16(ut), as16(dt)
    E = gt.shape[0]
    return prepare([(gt[e], suh_g[e], svh_g[e]) for e in range(E)], [(ut[e], suh_u[e], svh_u[e]) for e in range(E)],
                   [(dt[e], suh_d[e], svh_d[e]) for e in range(E)], codebook, device=gt.device)


def default_config(K: int, N: int, gateup: bool) -> tuple[int, int, int, int]:
    """Tile setting for a projection of shape K -> N: the GLM family's where it divides, else the largest that
    does. Depends only on the shape (so rows stay independent of the window)."""

    cands = [GLM_GATEUP if gateup else GLM_DOWN, (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2)]
    for nt, w, sk, pf in cands:
        if K % (16 * sk * w) == 0 and N % (16 * nt) == 0:
            return nt, w, sk, pf
    raise ValueError(f"no tile setting divides K={K}, N={N}")


class Scratch:
    """Per-window buffers for up to ``rows`` rows of ``slots`` slots. ``y`` rows of slots whose pick is not a
    routed expert (>= E, e.g. a shared expert's slot) are never written here: the caller fills or zeroes them."""

    def __init__(self, ex: Exl3RoutedExperts, rows: int, slots: int, cfg_gu=None, cfg_d=None, device="cuda") -> None:
        D, I = ex.dims, ex.width
        self.cfg_gu = cfg_gu or default_config(D, I, True)
        self.cfg_d = cfg_d or default_config(I, D, False)
        P = rows * slots
        sk = max(self.cfg_gu[2], self.cfg_d[2])
        self.xg = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xu = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xd = torch.zeros((P, I), dtype=torch.float16, device=device)
        self.z = torch.zeros((2 * sk * P * max(D, I),), dtype=torch.float32, device=device)
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
    """The routed experts of R rows: ``pick`` int32 [>= R, slots] (expert ids; ids >= E are skipped), ``wts`` fp32
    [R, slots] (None: stop after Y), ``x`` bf16 or fp16 [R, D] (row stride may exceed D). Writes and returns ``out``
    fp32 [R, D] = sum_k wts[r, k] * Y[r, k] (Y = s.y viewed [R, slots, D]). No host synchronisation.
    ``group=False`` reuses the grouping of a previous call with the same picks and R."""

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


def dequant(trellis: torch.Tensor, codebook: int | str) -> torch.Tensor:
    """W_q [K, N] fp16 of one matrix through the kernels' own lane decode (what ExLlamaV3's ``reconstruct``
    writes); for tests."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    k2 = k2_of(trellis)
    K, N = trellis.shape[0] * 16, trellis.shape[1] * 16
    out = torch.empty((K, N), dtype=torch.float16, device=trellis.device)
    _ext().dequant(trellis.contiguous(), out, k2, cb)
    return out
