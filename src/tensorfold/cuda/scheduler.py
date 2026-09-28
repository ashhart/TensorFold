"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue."""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable

from .streams import Stream


class Scheduler:
    def __init__(self, decoder: Any, *, max_streams: int = 4) -> None:
        self.decoder = decoder
        self.max_streams = max_streams
        self.waiting: queue.Queue = queue.Queue()
        self.boxes: dict[int, queue.Queue] = {}
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, prompt: list[int], count: int, sampling: Any, draft: bool,
               emit: Callable[[list[int]], bool | None], constraint: Any = None) -> dict:
        """Decode one request; ``emit`` runs on the calling thread and returns True to stop. Returns its stats."""

        box: queue.Queue = queue.Queue()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, constraint=constraint)
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
            else:
                try:
                    stream, box = self.waiting.get_nowait()
                except queue.Empty:
                    break
            self.boxes[id(stream)] = box
            try:
                self.decoder.admit(stream)
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                self.boxes.pop(id(stream)).put(("error", exc))
                continue
            if stream.done:
                done.append(stream)
        return done

    def _reply(self, s: Stream, kind: str, value: Any) -> None:
        box = self.boxes.pop(id(s), None)            # None: the stream's request has had its reply
        if box is not None:
            box.put((kind, value))

    def _loop(self) -> None:
        while True:
            done = self._admit(None if self.decoder.live() else self.waiting.get())   # idle: wait for a request
            try:
                done += self.decoder.round()
            except Exception as exc:                 # noqa: BLE001  (the live requests fail)
                for s in self.decoder.drop():
                    self._reply(s, "error", exc)
            self.decoder.finish(done)
            for s in done:
                self._reply(s, *(("error", s.error) if s.error is not None else ("done", s.stats())))
