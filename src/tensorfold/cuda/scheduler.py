"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue."""
# Background requests go last; one decoding yields its lane to a waiting request and re-queues to replay later.

from __future__ import annotations

import itertools
import queue
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

from .memory_gate import NoRoom
from .streams import Stream


class Waiting(queue.PriorityQueue):
    """(stream, box) pairs in arrival order, background streams after every other."""

    def __init__(self) -> None:
        super().__init__()
        self._order = itertools.count()

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        order = item[0].order                         # a re-queued request keeps its place
        if order is None:
            order = item[0].order = next(self._order)
        super().put((1 if item[0].background else 0, order, item), block, timeout)

    def get(self, block: bool = True, timeout: float | None = None):
        return super().get(block, timeout)[2]

    def stop(self) -> None:
        """Wake an idle worker to stop: None comes after every waiting request."""

        super().put((2, next(self._order), None))

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
        self.call_since: float | None = None         # monotonic start of the engine call running now (/health)
        self.last_round: float | None = None         # monotonic end of the last round that returned
        self._quit = False                           # shutdown(): the worker stops between rounds
        if hasattr(decoder, "arrived"):              # a decoder filling prompts lets a new request in between passes
            decoder.arrived = self.waiting.foreground
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def close(self) -> None:
        """Stop the worker once idle and let go of the decoder (its thread held it, and its weights, until now)."""

        self.waiting.stop()
        self.thread.join()
        self.decoder = None

    def shutdown(self, timeout: float = 120.0) -> None:
        """The server stops: the worker refuses what waits, then (between rounds) has the decoder end its live
        streams and wrap up (``decoder.shutdown``: e.g. write kept prompts to disk on both ranks), and returns."""

        self._quit = True
        self.waiting.stop()
        self.thread.join(timeout)

    def _stop(self) -> None:
        """The worker's last act after ``shutdown``: every request waiting or live answered with an error."""

        exc = RuntimeError("the server is shutting down; retry the request after it restarts")
        if self.held is not None:
            self.held[1].put(("error", exc))
            self.held = None
        while True:
            try:
                item = self.waiting.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                item[1].put(("error", exc))
        for s in getattr(self.decoder, "shutdown", list)():
            self._reply(s, "error", exc)

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
            elif self.held is not None and self.held[0].background and self.waiting.foreground():
                self.waiting.put(self.held)          # a held background request does not hold up a foreground one
                self.held = None
                continue
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
                if not stream.background and self._make_way(stream):
                    first = (stream, box)            # background streams gave way: again
                    continue
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

    def _make_way(self, stream: Stream) -> bool:
        """A foreground request found no room: the background streams whose memory would make it (a decoder's
        ``yield_for``, newest first) re-queue to replay later. Whether any did."""

        victims = self.decoder.yield_for(stream) if hasattr(self.decoder, "yield_for") else []
        self._give_way(victims)
        return bool(victims)

    # re-queue to replay later, as GlmScheduler._requeue does in MiaAI-Lab's GLM-5.3-Flash recipe
    # (patch 0030, Apache-2.0)
    def _give_way(self, streams: list[Stream]) -> None:
        """End these background streams and queue their replays (each keeps its place in line)."""

        for s in streams:
            box = self.boxes.pop(id(s))
            self.decoder.finish([s])
            self.yields += 1
            again = s.continued()
            again.order = s.order
            self.waiting.put((again, box))

    def _yield(self) -> None:
        """Lanes full, a foreground request waiting: the newest background stream (no grammar or images) re-queues."""

        if self.decoder.live() < self.max_streams or not self.waiting.foreground():
            return
        live = list(getattr(self.decoder, "streams", {}).values())
        stream = next((s for s in reversed(live) if s.background and not s.done and s.constraint is None
                       and s.vision is None and len(s.out) < s.count), None)
        if stream is None:
            return
        self._give_way([stream])

    def _reply(self, s: Stream, kind: str, value: Any) -> None:
        box = self.boxes.pop(id(s), None)            # None: the stream's request has had its reply
        if box is not None:
            box.put((kind, value))

    @contextmanager
    def _calling(self):
        """Time the engine calls in this block for /health's stall check (never the idle wait for a request)."""

        self.call_since = time.monotonic()
        try:
            yield
        finally:
            self.call_since = None

    def _loop(self) -> None:
        while True:
            if self._quit:                                                            # shutdown()
                self._stop()
                return
            with self._calling():
                self._yield()
            idle = not self.decoder.live() and self.held is None
            if idle and hasattr(self.decoder, "idle"):     # e.g. a TP=2 follower then idles on the CPU, not the GPU
                with self._calling():
                    self.decoder.idle()
            first = self.waiting.get() if idle else None                              # idle: wait for a request
            if self._quit:
                if first is not None:
                    first[1].put(("error", RuntimeError("the server is shutting down; retry the request after it "
                                                        "restarts")))
                self._stop()
                return
            if idle and first is None:
                return                                                                # close()
            with self._calling():
                done = self._admit(first)
            with self._calling():
                try:
                    done += self.decoder.round()
                    self.last_round = time.monotonic()
                except Exception as exc:             # noqa: BLE001  (the live requests fail)
                    for s in self.decoder.drop():
                        self._reply(s, "error", exc)
            with self._calling():
                self.decoder.finish(done)
                yielded = getattr(self.decoder, "yielded", None)
                if yielded:                          # streams that gave their memory to older ones: replay later
                    self._give_way(list(yielded))
            for s in done:
                self._reply(s, *(("error", s.error) if s.error is not None else ("done", s.stats())))
