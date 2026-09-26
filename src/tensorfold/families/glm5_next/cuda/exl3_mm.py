"""GLM-5.3-Flash's routed experts in EXL3 on CUDA (``exl3.cu``): the grouped trellis GEMV and the rotations.

A window's routed (row, slot) pairs go through, per layer:

    rot_in           Xg, Xu = fp16((x * suh) @ H) for each pair's expert (gate and up have their own suh)
    grouped          Zg, Zu = X @ W_q for every distinct expert's rows, per fixed K split (fp32)
    gateup_epilogue  g, u = (sum of splits) @ H * svh; act = GLM's limited SwiGLU; Xd = fp16((act * suh_d) @ H)
    grouped          Zd = Xd @ W_q(down), per fixed K split
    down_epilogue    Y[row, slot] = (sum of splits) @ H * svh_d      (this rank's share, fp32)

H is the 128-block Hadamard / sqrt(128) (``exl3.py``). The shared expert is BF16 and fills Y's last slot
elsewhere; ``glue.combine`` then adds the slots in order as before. Splits of K and warps are fixed by the shape,
so a pair's outputs never depend on the window's other rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

# (n tiles a block, warps a block, K splits) for the gate/up (K = model width) and down (K = the rank's expert
# width) matrices; none changes a row's bits across row counts.
GATEUP_CFG = (8, 4, 4)
DOWN_CFG = (8, 4, 1)


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_exl3_v1", sources=[str(here / "exl3.cpp"), str(here / "exl3.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


@dataclass
class Exl3Experts:
    """One layer's routed experts on this rank: trellis words [E, K/16, N/16, 32] int32 and scales per expert."""

    gt: torch.Tensor          # gate trellis [E, D/16, NI/16, 32]
    ut: torch.Tensor          # up trellis
    dt: torch.Tensor          # down trellis [E, NI/16, D/16, 32]
    suh_g: torch.Tensor       # [E, D] fp16
    suh_u: torch.Tensor
    svh_g: torch.Tensor       # [E, NI] fp16 (this rank's columns)
    svh_u: torch.Tensor
    suh_d: torch.Tensor       # [E, NI] fp16 (this rank's rows of down)
    svh_d: torch.Tensor       # [E, D]
    count: int
    width: int                # NI: this rank's share of the expert width
    dims: int                 # D

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.gt, self.ut, self.dt, self.suh_g, self.suh_u,
                                                         self.svh_g, self.svh_u, self.suh_d, self.svh_d))


def words(trellis: torch.Tensor) -> torch.Tensor:
    """A trellis (int16 [..., 64], 4 bits) as the kernels read it: int32 [..., 32], the same bytes."""

    if trellis.dtype != torch.int16 or trellis.shape[-1] != 64:
        raise ValueError("only 4-bit EXL3 trellises (int16 [..., 64]) are supported")
    return trellis.contiguous().view(torch.int32)


class Scratch:
    """Per-window buffers for up to ``rows`` rows of ``slots`` slots (the last slot is the shared expert's)."""

    def __init__(self, rows: int, slots: int, dims: int, width: int, device) -> None:
        P = rows * slots
        sk = max(GATEUP_CFG[2], DOWN_CFG[2])
        self.xg = torch.zeros((P, dims), dtype=torch.float16, device=device)
        self.xu = torch.zeros((P, dims), dtype=torch.float16, device=device)
        self.xd = torch.zeros((P, width), dtype=torch.float16, device=device)
        self.z = torch.zeros((2 * sk * P * max(width, dims),), dtype=torch.float32, device=device)
        self.rows, self.slots = rows, slots


def routed(x: torch.Tensor, pick: torch.Tensor, group, ex: Exl3Experts, s: Scratch, y: torch.Tensor, R: int,
           limit: float) -> None:
    """Y[row * slots + slot] for every routed slot of rows 0..R-1 (fp32, [.., D]); x: the normed rows (bf16)."""

    ext = _ext()
    slots, P = s.slots, s.rows * s.slots
    D, NI = ex.dims, ex.width
    ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots)
    nt, w, sk = GATEUP_CFG
    ext.grouped(s.xg, s.xu, ex.gt, ex.ut, group.ids, group.count, group.members, s.z, 2, D, NI, P, sk, slots, nt, w)
    ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, NI, sk, slots, float(limit))
    nt, w, sk = DOWN_CFG
    ext.grouped(s.xd, s.xd, ex.dt, ex.dt, group.ids, group.count, group.members, s.z, 1, NI, D, P, sk, slots, nt, w)
    ext.down_epilogue(s.z, pick, ex.svh_d, y, R, P, D, sk, slots)
