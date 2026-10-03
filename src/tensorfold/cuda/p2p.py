"""Ranks on one machine (two to eight): a row-parallel sum in one launch over peer mappings or shared host memory."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

STAGE_BYTES = 8 << 20       # a rank's staging buffer: 409 fp32 rows of 5120 (decode windows; prompts above use NCCL)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import VOLTA, load

    here = Path(__file__).parent
    return load(name="tensorfold_p2p_v4", sources=[str(here / "p2p.cu")], need=VOLTA, extra_cuda_cflags=["-O3"],
                verbose=False)


def _world(group) -> int:
    import torch.distributed as dist

    return dist.get_world_size(group)


class PeerSum:
    """This rank's staging and flags, and every peer's, mapped once at startup."""

    kind = "P2P (the peer GPUs mapped)"
    backoff = 0                 # ns a waiting block sleeps between flag polls (0: spin)

    def __init__(self, rank: int, group=None) -> None:
        import torch.distributed as dist

        ext = _ext()
        world = _world(group)
        self.rank, self.world = rank, world
        stage, sig = ext.alloc(STAGE_BYTES), ext.alloc(ext.signal_bytes())
        mine = torch.cat([ext.handle(stage), ext.handle(sig)]).cuda()
        every = torch.empty((world, mine.numel()), dtype=torch.uint8, device="cuda")
        dist.all_gather_into_tensor(every, mine, group=group)
        every = every.cpu()
        half = mine.numel() // 2
        stages, sigs = [], []
        for q in range(world):
            if q == rank:
                stages.append(stage)
                sigs.append(sig)
            else:
                stages.append(ext.open(every[q, :half].contiguous()))
                sigs.append(ext.open(every[q, half:].contiguous()))
        self._addresses(stages, sigs)
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def _addresses(self, stages: list[int], sigs: list[int]) -> None:
        self.stages = torch.tensor(stages, dtype=torch.int64)
        self.sigs = torch.tensor(sigs, dtype=torch.int64)
        self.rounds = torch.zeros(_ext().blocks(), dtype=torch.int64, device="cuda")

    def fits(self, local: torch.Tensor) -> bool:
        return local.numel() * local.element_size() <= STAGE_BYTES

    def __call__(self, local: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        local = local.contiguous()
        out = torch.empty(local.shape, dtype=torch.bfloat16, device=local.device)
        _ext().p2p_sum(local, self.stages, self.sigs, out, self.rank, self.rounds, self.backoff)
        return out if dtype == torch.bfloat16 else out.to(dtype)


NAME_BYTES = 128


class HostSum(PeerSum):
    """Every rank's stages and flags in one page-locked ``/dev/shm`` buffer each GPU maps; rank 0 creates and unlinks it."""

    kind = "host memory (the GPUs cannot map each other: the partials go over PCIe through shared page-locked memory)"

    backoff = int(os.environ.get("TF_TP_HOST_SPIN_NS", "500"))   # flags polled over PCIe: gently

    def __init__(self, rank: int, group=None) -> None:
        import mmap
        import uuid

        import torch.distributed as dist

        ext = _ext()
        world = _world(group)
        self.rank, self.world = rank, world
        sig = ext.signal_bytes()
        size = -(-(world * (STAGE_BYTES + sig)) // mmap.PAGESIZE) * mmap.PAGESIZE
        name = torch.zeros(NAME_BYTES, dtype=torch.uint8)
        if rank == 0:
            path = f"/dev/shm/tensorfold-sum-{os.getpid()}-{uuid.uuid4().hex[:12]}"
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, size)                                  # zero-filled: every flag starts below round 1
            raw = path.encode()
            name[:len(raw)] = torch.tensor(list(raw), dtype=torch.uint8)
        every = torch.empty((world, NAME_BYTES), dtype=torch.uint8, device="cuda")
        dist.all_gather_into_tensor(every, name.cuda(), group=group)
        if rank != 0:
            raw = bytes(b for b in every[0].cpu().tolist() if b)
            path = raw.decode()
            fd = os.open(path, os.O_RDWR)
        try:
            self._map = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        import ctypes

        host = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        base = ext.host_map(host, size)                             # this process's device address of the buffer
        self._addresses([base + q * STAGE_BYTES for q in range(world)],
                        [base + world * STAGE_BYTES + q * sig for q in range(world)])
        torch.cuda.synchronize()
        dist.barrier(group=group)
        if rank == 0:
            os.unlink(path)                                         # all mapped: the name is not needed any more


_PEER: PeerSum | None = None


def _same_machine(group=None) -> bool:
    """Whether every rank runs on this machine (one boot): only then can they share host memory."""

    import hashlib
    import socket

    import torch.distributed as dist

    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return False
    digest = hashlib.sha256(f"{socket.gethostname()}/{boot}".encode()).digest()[:8]
    mine = torch.tensor(list(digest), dtype=torch.uint8, device="cuda")
    every = torch.empty((_world(group), 8), dtype=torch.uint8, device="cuda")
    dist.all_gather_into_tensor(every, mine, group=group)
    return bool((every == every[0]).all())


def install(rank: int, group=None) -> bool:
    """Install peer sums when every GPU maps every other, else host-memory sums on one machine; False: NCCL sums."""

    global _PEER
    import torch.distributed as dist

    me = torch.cuda.current_device()
    others = [d for d in range(torch.cuda.device_count()) if d != me]
    on = os.environ.get("TF_TP_P2P", "1") != "0"
    reach = [torch.cuda.can_device_access_peer(me, d) for d in others]
    able = int(on and bool(reach) and (any(reach) if _world(group) == 2 else all(reach)))
    flags = torch.tensor([able, int(on and os.environ.get("TF_TP_HOST_SUM", "1") != "0")], dtype=torch.int32,
                         device="cuda")
    every = torch.empty((_world(group), 2), dtype=torch.int32, device="cuda")
    dist.all_gather_into_tensor(every, flags, group=group)
    if bool(every[:, 0].all()):
        _PEER = PeerSum(rank, group)
        return True
    if bool(every[:, 1].all()) and _same_machine(group):
        _PEER = HostSum(rank, group)
        return True
    return False


def peer() -> PeerSum | None:
    return _PEER
