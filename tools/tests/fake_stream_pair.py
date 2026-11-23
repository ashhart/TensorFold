"""Optional rendezvous for the served-evidence fixture; ordinary conformance streams keep their original pacing."""
import contextlib
import json
import threading
import time


def content(step):
    """Whether this SSE event contains visible text."""
    return isinstance(step, bytes) and step.startswith(b"data: {") and any(
        c.get("delta", {}).get("content") for c in json.loads(step[6:]).get("choices", []))


class StreamPairs:
    """Pair active generated replies; a lone reply never waits for admission."""

    def __init__(self, timeout):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.pending = None

    @contextlib.contextmanager
    def claim(self):
        with self.lock:
            slot = self.pending
            if slot is None:
                slot = [None, None]
                self.pending = slot
            else:
                slot[0] = threading.Barrier(2)
                self.pending = None
        try:
            yield slot
        finally:
            with self.lock:
                barrier = slot[0]
                if self.pending is slot:
                    self.pending = None
            if barrier is not None:
                barrier.abort()

    def meet(self, slot):
        with self.lock:
            barrier = slot[0]
            if barrier is not None and slot[1] is None:
                slot[1] = time.monotonic() + self.timeout
        if barrier is not None:
            try:
                barrier.wait(timeout=max(0.0, slot[1] - time.monotonic()))
            except threading.BrokenBarrierError:
                pass

    def steps(self, steps, slot):
        """Keep every byte and pause; spread visible text across the existing generated-stream pause budget."""
        events = [s for s in steps if isinstance(s, bytes)]
        pauses = [s for s in steps if not isinstance(s, bytes)]
        visible = sum(bool(content(s)) for s in events)
        if pauses and visible > 2:
            paced, slept = [], 0
            for index, step in enumerate(events):
                due = min(index, visible - 2) * (len(pauses) - 1) // (visible - 2)
                due += index >= visible - 1
                paced.extend(pauses[slept:due])
                paced.append(step)
                slept = due
            steps = paced
        for step in steps:
            visible = content(step)
            if visible:
                self.meet(slot)
            yield step
            if visible:
                self.meet(slot)
