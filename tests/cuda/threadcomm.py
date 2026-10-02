"""``comm.NCCL`` for ranks that are threads of one process on one GPU, any world size (tests only).

The existing two-rank tests keep their own copy; four-rank tests (full GLM-5.3) use this one, which also has the
all-to-all that the exact reduce-scatter needs.
"""

from __future__ import annotations

import threading

import torch


class Hub:
    def __init__(self, world: int) -> None:
        self.world = world
        self.slots: list = [None] * world
        self.barrier = threading.Barrier(world)


class ThreadComm:
    def __init__(self, hub: Hub, rank: int) -> None:
        self.hub, self.rank, self.world = hub, rank, hub.world

    def _exchange(self, send: torch.Tensor) -> list:
        torch.cuda.current_stream().synchronize()
        self.hub.slots[self.rank] = send
        self.hub.barrier.wait()
        return list(self.hub.slots)

    def _done(self) -> None:
        torch.cuda.current_stream().synchronize()
        self.hub.barrier.wait()

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        slots = self._exchange(send)
        n = send.numel()
        flat = recv.view(-1)
        for r in range(self.world):
            flat[r * n:(r + 1) * n].copy_(slots[r].reshape(-1))
        self._done()

    def reduce_scatter(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv <- block self.rank of the rank-order (fp32) sum of every rank's send [world * n]."""
        if send.numel() != recv.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("reduce_scatter: send must hold world x recv of the same dtype")
        slots = self._exchange(send)
        n = recv.numel()
        acc = slots[0].reshape(-1)[self.rank * n:(self.rank + 1) * n].float().clone()
        for r in range(1, self.world):
            acc += slots[r].reshape(-1)[self.rank * n:(self.rank + 1) * n].float()
        recv.view(-1).copy_(acc)
        self._done()

    def all_to_all(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        if send.shape != recv.shape or send.dim() != 2 or send.shape[0] != self.world or send.dtype != recv.dtype:
            raise ValueError("all_to_all: send and recv must be [world, n] tensors of the same dtype")
        slots = self._exchange(send)
        for j in range(self.world):
            recv[j].copy_(slots[j][self.rank])
        self._done()

    def barrier(self) -> None:
        self.hub.barrier.wait()


class RingThreadComm(ThreadComm):
    """ThreadComm plus an all-reduce (deterministic, rank-order fp32 sum), so the prompt path takes its NCCL-ring
    branches (two-micro-batch overlap); opt-in because it changes which branch older tests exercise."""

    def all_reduce(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """Sum in rank order (fp32), every rank alike."""
        slots = self._exchange(send)
        acc = slots[0].float().clone()
        for r in range(1, self.world):
            acc += slots[r].float()
        recv.copy_(acc.view(recv.shape))
        self._done()


def _load_extensions() -> None:
    """Build/load TensorFold's EXL3 extensions once, in this thread: rank threads importing a JIT module at the same
    moment can see it half-initialized (serving has one rank per process and never races)."""
    try:
        from tensorfold.cuda.exl3 import experts, linear
        linear._ext()
        experts._ext()
    except Exception:  # noqa: BLE001  (tests that do not use EXL3 run without a compiler)
        pass


def run_ranks(fn, world: int, comm_cls=ThreadComm) -> list:
    """fn(rank, comm) on every rank at once (threads); results in rank order."""

    _load_extensions()
    hub = Hub(world)
    results: list = [None] * world
    errors: list = []

    def body(r: int) -> None:
        try:
            with torch.no_grad():
                results[r] = fn(r, comm_cls(hub, r))
        except BaseException as exc:        # noqa: BLE001  (reported below; unblock the other ranks)
            errors.append(exc)
            hub.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(world)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return results
