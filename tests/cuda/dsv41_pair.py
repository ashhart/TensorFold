"""Two DeepSeek-V4.1 ranks as two threads on one GPU: the all-gather is a barrier and device copies in rank order."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import torch


class _Hub:
    def __init__(self) -> None:
        self.slots: list[torch.Tensor | None] = [None, None]
        self.gate = threading.Barrier(2, timeout=600)


class PairComm:
    """``comm.NCCL``'s all_gather for one of two thread ranks; ``barrier`` and ``ready`` have nothing to wait for."""

    world = 2
    store = None

    def __init__(self, hub: _Hub, rank: int) -> None:
        self.hub, self.rank = hub, rank

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        torch.cuda.current_stream().synchronize()          # the other thread reads send from its own stream
        self.hub.slots[self.rank] = send
        self.hub.gate.wait()
        n, flat = send.numel(), recv.view(-1)
        for r in range(self.world):
            flat[r * n:(r + 1) * n].copy_(self.hub.slots[r].reshape(-1))
        torch.cuda.current_stream().synchronize()          # copied before either rank reuses its send
        self.hub.gate.wait()

    def barrier(self) -> None:
        pass

    def ready(self, label: str, **kwargs) -> None:
        pass


def pair() -> tuple[PairComm, PairComm]:
    """Rank 0's and rank 1's communicators, gathering with each other."""

    hub = _Hub()
    return PairComm(hub, 0), PairComm(hub, 1)


def run_pair(fn0: Callable[[PairComm], Any], fn1: Callable[[PairComm], Any],
             comms: tuple[PairComm, PairComm] | None = None) -> tuple[Any, Any]:
    """``fn0(comm0)`` and ``fn1(comm1)`` at once on two threads -> both results; either thread's exception is raised."""

    comms = pair() if comms is None else comms
    results: list[Any] = [None, None]
    errors: list[BaseException | None] = [None, None]

    def body(r: int, fn: Callable[[PairComm], Any]) -> None:
        try:
            with torch.no_grad():                          # grad mode is per thread
                results[r] = fn(comms[r])
        except BaseException as exc:                       # noqa: BLE001  (raised below after both threads end)
            errors[r] = exc
            comms[r].hub.gate.abort()                      # the other rank's gather would wait for this one

    threads = [threading.Thread(target=body, args=(r, fn)) for r, fn in enumerate((fn0, fn1))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [e for e in errors if e is not None]
    if failed:                                             # the cause, not the other rank's broken barrier
        raise next((e for e in failed if not isinstance(e, threading.BrokenBarrierError)), failed[0])
    return results[0], results[1]
