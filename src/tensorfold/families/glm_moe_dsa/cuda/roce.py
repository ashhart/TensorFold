"""Decode-window reductions over RoCE (b12x RoCEnante, in the serving image): one launch, every peer's partial
RDMA-written into pinned host memory, summed in fixed rank order - every rank stores the same bits, as the NCCL
all-gather + rank-order sum does - and capturable in CUDA graphs. Prompt chunks stay on NCCL (large messages)."""

from __future__ import annotations

import os

import torch

MAX_BYTES = 1 << 20              # decode windows: 16 rows x 6144 fp32 = 384 KB; logits argmax rows: tiny


class RoceReduce:
    def __init__(self, rank: int, world: int, master: str, port: int, nccl=None) -> None:
        import torch.distributed as dist
        from b12x.comm.roce import AllReduce

        if os.environ.get("NCCL_SOCKET_IFNAME"):         # gloo would pick the hostname's loopback in a container
            os.environ.setdefault("GLOO_SOCKET_IFNAME", os.environ["NCCL_SOCKET_IFNAME"])
        if not dist.is_initialized():                    # the runtime's setup exchange only (gloo over TCP)
            from datetime import timedelta

            dist.init_process_group("gloo", init_method=f"tcp://{master}:{port}", rank=rank, world_size=world,
                                    timeout=timedelta(seconds=120))
        hca = os.environ.get("NCCL_IB_HCA")
        gid = os.environ.get("NCCL_IB_GID_INDEX")
        self.rt = AllReduce(exchange_group=dist.group.WORLD, device=torch.device("cuda", 0), max_size=MAX_BYTES,
                            max_gather_bytes=MAX_BYTES, hca_names=[hca] if hca else None,
                            gid_index=int(gid) if gid else None)
        self.rt.prepare((torch.float32,), padded_gather=True)
        self.rank, self.world = rank, world
        if nccl is not None:
            self._check(nccl)

    def all_reduce(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.rt.all_reduce(x, out=out)

    def all_gather(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.rt.all_gather(x.reshape(-1), dim=0, out=out.view(-1))

    def _check(self, nccl) -> None:
        """Every rank's sum must carry the same bits (compared through NCCL), else refuse to serve on RoCE."""
        g = torch.Generator(device="cpu").manual_seed(1000 + self.rank)
        x = (torch.randn(3, 6144, generator=g) * 10.0 ** (self.rank - 1)).cuda()
        s = torch.empty_like(x)
        self.all_reduce(x, s)
        allv = torch.empty((self.world, 3, 6144), device="cuda")
        nccl.all_gather(s, allv)
        ref = allv.new_zeros(3, 6144)
        parts = torch.empty((self.world, 3, 6144), device="cuda")
        nccl.all_gather(x, parts)
        ref = parts[0].clone()
        for r in range(1, self.world):
            ref += parts[r]
        same = all(torch.equal(allv[r], allv[0]) for r in range(self.world))
        order = torch.equal(allv[0], ref)
        if not same:
            raise RuntimeError("RoCE all-reduce: ranks hold different bits")
        print(f"[tensorfold] RoCE one-shot reduce ready (ranks bit-equal: {same}; equals the NCCL rank-order sum: "
              f"{order})", flush=True)
