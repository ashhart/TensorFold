"""The ``Scheduler``'s decoder for Kolibri 1: a prompt chunk a round, then one row of every decoding stream together."""
# Rows are row-invariant, so a stream's reply equals the one it gets alone.

from __future__ import annotations

import time

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.sampling import sample_rows
from tensorfold.cuda.streams import Stream, next_fill

from .forward import PROMPT_CHUNK, Chain, Model

STEP = 1024              # prompt rows a round while other streams decode


class Decoder:
    def __init__(self, model: Model, eos: tuple[int, ...]) -> None:
        self.model, self.eos = model, tuple(eos)
        self.free = list(range(model.slots))
        self.streams: dict[int, Stream] = {}      # decoding
        self.filling: list[Stream] = []           # admitted, prompts still prefilling (oldest first)
        self.slot: dict[int, int] = {}
        self.done_at: dict[int, int] = {}         # prompt rows prefilled
        self.next_id = 0

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def admit(self, s: Stream) -> None:
        if s.constraint is not None:
            raise ValueError("Kolibri 1 on CUDA does not take response_format grammars yet")
        room = self.model.context - len(s.prompt)
        if len(s.prompt) < 1 or room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.model.context}-token "
                             "context (--context)")
        if not self.free:
            raise NoRoom("every stream slot is busy")
        s.count = min(s.count, room)
        s.sid = self.next_id
        self.next_id += 1
        self.slot[s.sid] = self.free.pop(0)
        self.done_at[s.sid] = 0
        self.filling.append(s)

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.eos if s.stop_eos else ()

    def _fill(self) -> list[Stream]:
        s = next_fill(self.filling)
        a = self.done_at[s.sid]
        b = min(len(s.prompt), a + (STEP if self.streams else PROMPT_CHUNK))
        t0 = time.perf_counter()
        try:
            logits = self.model.forward([Chain(self.slot[s.sid], a, s.prompt[a:b])], prompt=True)
            first = sample_rows(logits, [len(s.prompt)], s.sampling)[0] if b == len(s.prompt) else None
        except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
            self.filling.remove(s)
            s.error, s.done = exc, True
            return [s]
        self.done_at[s.sid] = b
        s.prefill_s += time.perf_counter() - t0
        if first is None:
            return []
        self.filling.remove(s)
        self.streams[s.sid] = s
        s.started = time.perf_counter()
        s.take([first], self._ends(s))
        return [s] if s.done else []

    def round(self) -> list[Stream]:
        """A prompt chunk for the next queued prompt, then a row of each decoding stream; returns the finished."""

        done = self._fill() if self.filling else []
        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return done
        positions = [len(s.prompt) + len(s.out) - 1 for s in live]
        logits = self.model.step([s.out[-1] for s in live], positions, [self.slot[s.sid] for s in live])
        for k, s in enumerate(live):
            tok = sample_rows(logits[k:k + 1], [positions[k] + 1], s.sampling)[0]
            s.counted(1)
            s.take([tok], self._ends(s))
        return done + [s for s in live if s.done]

    def finish(self, done: list[Stream]) -> None:
        for s in done:
            self.streams.pop(s.sid, None)
            if s in self.filling:
                self.filling.remove(s)
            slot = self.slot.pop(s.sid, None)
            if slot is not None:
                self.free.append(slot)
                self.free.sort()
            self.done_at.pop(s.sid, None)

    def drop(self) -> list[Stream]:
        gone = [s for s in self.streams.values() if not s.done] + list(self.filling)
        self.streams, self.filling = {}, []
        self.free = list(range(self.model.slots))
        self.slot.clear()
        self.done_at.clear()
        return gone
