"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue."""
# Background requests go last; one decoding yields its lane to a waiting request and re-queues to replay later.

from __future__ import annotations

import itertools
import queue
import threading
from typing import Any, Callable

from .memory_gate import NoRoom
from .streams import Stream


class Waiting(queue.PriorityQueue):
    """(stream, box) pairs in arrival order, background streams after every other."""

    def __init__(self) -> None:
        super().__init__()
        self._order = itertools.count()

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        super().put((1 if item[0].background else 0, next(self._order), item), block, timeout)

    def get(self, block: bool = True, timeout: float | None = None):
        return super().get(block, timeout)[2]

    def foreground(self) -> bool:
        """Whether a foreground request waits."""

        with self.mutex:
            return bool(self.queue) and self.queue[0][0] == 0


class Scheduler:
    def __init__(self, decoder: Any, *, max_streams: int = 4) -> None:
        self.decoder = decoder
        self.max_streams = max_streams
        self.waiting = Waiting()
        self.held: tuple | None = None               # a request waiting for memory, admitted before any other
        self.boxes: dict[int, queue.Queue] = {}
        self.yields = 0                              # background streams that gave up their lane
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, prompt: list[int], count: int, sampling: Any, draft: bool,
               emit: Callable[[list[int]], bool | None], stop_eos: bool = True, *, vision: Any = None,
               constraint: Any = None, background: bool = False, probabilities: Any = None) -> dict:
        """Decode one request; ``emit`` runs on the calling thread and returns True to stop. Returns its stats."""

        box: queue.Queue = queue.Queue()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, stop_eos=stop_eos, vision=vision,
                        constraint=constraint, background=background, probabilities=probabilities)
        cancel = [False]
        stream.emit = lambda new: (box.put(("tokens", new)), cancel[0])[1]
        self.waiting.put((stream, box))
        while True:
            kind, value = box.get()
            if kind == "tokens":
                if not cancel[0] and emit(value):
                    cancel[0] = True                 # the client left: the stream ends after its next round
            elif kind == "error":
                raise value
            else:
                return value

    def _admit(self, first=None) -> list[Stream]:
        done = []
        while self.decoder.live() < self.max_streams:
            if first is not None:
                (stream, box), first = first, None
            elif self.held is not None:
                (stream, box), self.held = self.held, None
            else:
                try:
                    stream, box = self.waiting.get_nowait()
                except queue.Empty:
                    break
            self.boxes[id(stream)] = box
            try:
                self.decoder.admit(stream)
            except NoRoom as exc:
                self.boxes.pop(id(stream))
                if self.decoder.live():              # waits, first in line, until a live stream finishes
                    self.held = (stream, box)
                    break
                box.put(("error", exc))
                continue
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                self.boxes.pop(id(stream)).put(("error", exc))
                continue
            if stream.done:
                done.append(stream)
        return done

    def _yield(self) -> None:
        """Lanes full, a foreground request waiting: the newest background stream (no grammar or images) re-queues."""

        if self.decoder.live() < self.max_streams or not self.waiting.foreground():
            return
        live = list(getattr(self.decoder, "streams", {}).values())
        stream = next((s for s in reversed(live) if s.background and not s.done and s.constraint is None
                       and s.vision is None and len(s.out) < s.count), None)
        if stream is None:
            return
        box = self.boxes.pop(id(stream))
        self.decoder.finish([stream])
        self.yields += 1
        self.waiting.put((stream.continued(), box))

    def _reply(self, s: Stream, kind: str, value: Any) -> None:
        box = self.boxes.pop(id(s), None)            # None: the stream's request has had its reply
        if box is not None:
            box.put((kind, value))

    def _loop(self) -> None:
        while True:
            self._yield()
            idle = not self.decoder.live() and self.held is None
            done = self._admit(self.waiting.get() if idle else None)                  # idle: wait for a request
            try:
                done += self.decoder.round()
            except Exception as exc:                 # noqa: BLE001  (the live requests fail)
                for s in self.decoder.drop():
                    self._reply(s, "error", exc)
            self.decoder.finish(done)
            for s in done:
                self._reply(s, *(("error", s.error) if s.error is not None else ("done", s.stats())))
