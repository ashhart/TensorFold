"""Rank 0's messages to the other ranks: flat int32 words over the communicator's all-gather, and an idle doorbell."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from datetime import timedelta
from typing import Any, Protocol

from tensorfold.engine.exact_sampling import Sampling

ADMIT, ROUND, DONE, EVICT, STOP, IDLE = 1, 2, 3, 4, 5, 6
SAMPLING_WORDS = 18  # pack_sampling's length


class Link(Protocol):
    """Rank 0 ``send``s each message, every other rank ``recv``s it in order."""

    def send(self, kind: int, ints: Sequence[int]) -> None: ...

    def recv(self) -> tuple[int, list[int]]: ...

    def idle(self) -> None: ...


class LocalLink:
    """One rank: messages go nowhere."""

    def send(self, kind: int, ints: Sequence[int]) -> None:
        return None

    def recv(self) -> tuple[int, list[int]]:
        raise RuntimeError("a one-rank link has nothing to receive")

    def idle(self) -> None:
        return None


class RoundLink:
    """A message is [kind, length, values] in one gather of ``words`` ints, and a second gather when it is longer."""

    def __init__(self, comm: Any, words: int = 256, device: Any = "cuda", *, prefix: str = "tf_lanes") -> None:
        import torch

        if words < 3:
            raise ValueError("a round link message needs at least three words")
        self.comm, self.words, self.device, self.prefix = comm, int(words), device, prefix
        self.torch = torch
        self.head = torch.zeros((self.words,), dtype=torch.int32, device=device)
        self.heads = torch.zeros((comm.world * self.words,), dtype=torch.int32, device=device)
        self.bells = 0
        self.sleeping = False

    def _gather(self, send) -> list[int]:
        out = self.torch.empty((self.comm.world * send.numel(),), dtype=self.torch.int32, device=self.device)
        self.comm.all_gather(send, out)
        return out[: send.numel()].tolist()  # rank 0's part

    def send(self, kind: int, ints: Sequence[int]) -> None:
        if self.sleeping:
            self._ring()
        values = [int(v) for v in ints]
        room = self.words - 2
        first = [int(kind), len(values), *values[:room]]
        self.head.zero_()
        self.head[: len(first)] = self.torch.tensor(first, dtype=self.torch.int32)
        self.comm.all_gather(self.head, self.heads)
        if len(values) > room:
            self._gather(self.torch.tensor(values[room:], dtype=self.torch.int32, device=self.device))

    def recv(self) -> tuple[int, list[int]]:
        while True:
            self.head.zero_()
            self.comm.all_gather(self.head, self.heads)
            got = self.heads[: self.words].tolist()
            kind, n = got[0], got[1]
            room = self.words - 2
            values = got[2 : 2 + min(n, room)]
            if n > room:
                values += self._gather(self.torch.zeros((n - room,), dtype=self.torch.int32, device=self.device))
            if kind != IDLE:
                return kind, values
            self._wait()

    def idle(self) -> None:
        """Nothing to do: the followers sleep on the store, not in a spinning collective, until the next send."""

        if not self.sleeping:
            self.send(IDLE, [])
            self.sleeping = True

    def _ring(self) -> None:
        self.sleeping = False
        store = getattr(self.comm, "store", None)
        if store is not None:
            self.bells += 1
            store.set(f"{self.prefix}/bell/{self.bells}", "1")

    def _wait(self) -> None:
        store = getattr(self.comm, "store", None)
        if store is None:
            return
        self.bells += 1
        while True:
            try:
                store.wait([f"{self.prefix}/bell/{self.bells}"], timedelta(seconds=60))
                return
            except Exception as exc:
                if "timeout" not in str(exc).lower():
                    raise


def _bits(x: float) -> list[int]:
    v = struct.unpack("<Q", struct.pack("<d", float(x)))[0]
    return [(v >> (16 * i)) & 0xFFFF for i in range(4)]


def _value(words: Sequence[int]) -> int:
    return sum(int(w) << (16 * i) for i, w in enumerate(words))


def pack_sampling(s: Sampling | None) -> list[int]:
    """A request's sampling as 18 ints: the seed and the floats cross as their exact bits."""

    if s is None:
        return [0] * SAMPLING_WORDS
    seed = int(s.seed) & ((1 << 64) - 1)
    return [
        1,
        *[(seed >> (16 * i)) & 0xFFFF for i in range(4)],
        int(s.top_k),
        *_bits(s.temperature),
        *_bits(s.top_p),
        *_bits(s.min_p),
    ]


def unpack_sampling(w: Sequence[int]) -> Sampling | None:
    if not w[0]:
        return None
    floats = [struct.unpack("<d", struct.pack("<Q", _value(w[6 + 4 * i : 10 + 4 * i])))[0] for i in range(3)]
    return Sampling(_value(w[1:5]), temperature=floats[0], top_k=int(w[5]), top_p=floats[1], min_p=floats[2])


__all__ = [
    "ADMIT",
    "DONE",
    "EVICT",
    "IDLE",
    "ROUND",
    "SAMPLING_WORDS",
    "STOP",
    "Link",
    "LocalLink",
    "RoundLink",
    "pack_sampling",
    "unpack_sampling",
]
