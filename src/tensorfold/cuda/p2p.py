"""Two ranks on one machine: a row-parallel sum in one launch over a peer mapping, with the rank-ordered sum's bits."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

STAGE_BYTES = 8 << 20       # a rank's staging buffer: 409 fp32 rows of 5120 (decode windows; prompts above use NCCL)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_p2p_v1", sources=[str(here / "p2p.cu")], extra_cuda_cflags=["-O3"], verbose=False)


class PeerSum:
    """This rank's staging and flags, and the peer's, mapped once at startup."""

    def __init__(self, rank: int, group=None) -> None:
        import torch.distributed as dist

        ext = _ext()
        self.rank, self.round = rank, 0
        self.stage = ext.alloc(STAGE_BYTES)
        self.sig = ext.alloc(ext.signal_bytes())
        mine = torch.cat([ext.handle(self.stage), ext.handle(self.sig)]).cuda()
        both = torch.empty((2, mine.numel()), dtype=torch.uint8, device="cuda")
        dist.all_gather_into_tensor(both, mine, group=group)
        peer = both[1 - rank].cpu()
        half = peer.numel() // 2
        self.peer_stage, self.peer_sig = ext.open(peer[:half].contiguous()), ext.open(peer[half:].contiguous())
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def fits(self, local: torch.Tensor) -> bool:
        return local.numel() * local.element_size() <= STAGE_BYTES

    def __call__(self, local: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        local = local.contiguous()
        out = torch.empty(local.shape, dtype=torch.bfloat16, device=local.device)
        self.round += 1
        _ext().p2p_sum(local, self.stage, self.peer_stage, self.sig, self.peer_sig, out, self.rank, self.round)
        return out if dtype == torch.bfloat16 else out.to(dtype)


_PEER: PeerSum | None = None


def install(rank: int, group=None) -> bool:
    """Map the peer for row-parallel sums when both GPUs can reach each other (both ranks call this together)."""

    global _PEER
    import torch.distributed as dist

    me = torch.cuda.current_device()
    others = [d for d in range(torch.cuda.device_count()) if d != me]
    able = int(os.environ.get("TF_TP_P2P", "1") != "0" and any(torch.cuda.can_device_access_peer(me, d)
                                                                  for d in others))
    flags = torch.tensor([able], dtype=torch.int32, device="cuda")
    both = torch.empty(2, dtype=torch.int32, device="cuda")
    dist.all_gather_into_tensor(both, flags, group=group)
    if not bool(both.all()):
        return False
    _PEER = PeerSum(rank, group)
    return True


def peer() -> PeerSum | None:
    return _PEER
