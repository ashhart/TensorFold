"""A failed admission answers its own request; the concurrent CUDA scheduler goes on serving the others."""

import importlib
import threading
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

WAIT = 10.0                                   # seconds a reply may take; a stopped worker never replies


class Request:
    """``Scheduler.submit`` on its own thread; ``reply()`` is ("done", stats), ("error", exception) or None."""

    def __init__(self, sched, prompt, count):
        self.got = []
        self.thread = threading.Thread(target=self._go, args=(sched, list(prompt), count), daemon=True)
        self.thread.start()

    def _go(self, sched, prompt, count):
        try:
            self.got.append(("done", sched.submit(prompt, count, None, True, lambda new: False)))
        except Exception as exc:              # noqa: BLE001
            self.got.append(("error", exc))

    def reply(self):
        self.thread.join(WAIT)
        return self.got[0] if self.got else None


def ask(sched, prompt, count):
    return Request(sched, prompt, count).reply()


class St:
    """A committed state: its position only (the stand-in prefill and copies hold no tensors)."""

    def __init__(self, pos=0):
        self.pos, self.limit = pos, 0


@pytest.fixture
def decoders(monkeypatch, allocations):  # noqa: F811
    """``make(world)``: the 27B's ``MultiDecoder`` on CPU stand-ins; ``fail["copy"]`` fails the next prompt-end copy."""

    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    fail = {"copy": False}

    def prefill_state(w, prompt, st, **kw):
        st.pos = len(prompt)

    def kept(st):
        if fail["copy"]:
            fail["copy"] = False
            raise torch.OutOfMemoryError("CUDA out of memory (simulated at the prompt-end copy)")
        return St(st.pos)

    monkeypatch.setattr(multi, "prefill_state", prefill_state)
    monkeypatch.setattr(multi, "first_token", lambda *args: 7)
    monkeypatch.setattr(multi, "kept", kept)
    monkeypatch.setattr(multi, "private", lambda st, rows: St(st.pos))
    monkeypatch.setattr(multi, "State", lambda w: St())
    monkeypatch.setattr(multi, "_share", lambda values, src, device: values)      # two ranks: no NCCL here

    def make(world=1):
        w = SimpleNamespace(config=SimpleNamespace(eos=(0,), vocab=10), norm=SimpleNamespace(device="cpu"),
                            head=SimpleNamespace(n=10))
        dec = multi.MultiDecoder(w, None, allow_copy=False, world=world)
        dec.gate, dec.entered = threading.Event(), threading.Event()      # a round waits at the gate while it is shut
        dec.gate.set()

        def verify(plan, copied=None):                            # each window keeps its pending token, samples 5
            dec.entered.set()
            dec.gate.wait(WAIT)
            return [([pending], [-1]) for _, _, pending, _ in plan], None, None, None, [[5] for _ in plan]

        dec._verify = verify
        dec._commit = lambda *args: None
        return dec

    return make, fail


A, B, C, D, E = ([k] * 5 + list(range(k * 10, k * 10 + 30)) for k in (1, 2, 3, 4, 5))


def test_a_failed_prompt_end_copy_answers_its_request_and_the_worker_goes_on(decoders):
    make, fail = decoders
    dec = make()
    sched = Scheduler(dec, max_streams=4)
    assert ask(sched, A, 1)[0] == "done"
    fail["copy"] = True
    b = ask(sched, B, 64)
    assert b is not None and b[0] == "error" and isinstance(b[1], torch.OutOfMemoryError), b
    c, d = ask(sched, C, 1), ask(sched, D, 3)                     # D decodes two rounds
    assert c is not None and c[0] == "done", c
    assert d is not None and d[0] == "done", d
    assert sched.thread.is_alive()
    assert dec.live() == 0 and not dec.streams
    assert B not in [entry[0] for entry in dec.cache.entries]
    assert [entry[0] for entry in dec.cache.entries] == [A, C, D]


def test_a_failed_admission_leaves_the_live_streams_decoding(decoders):
    make, fail = decoders
    dec = make()
    dec.gate.clear()
    sched = Scheduler(dec, max_streams=4)
    e = Request(sched, E, 3)
    assert dec.entered.wait(WAIT)                                 # E is admitted; its first round waits
    fail["copy"] = True
    b = Request(sched, B, 64)
    deadline = time.monotonic() + WAIT
    while sched.waiting.qsize() == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    dec.gate.set()                                                # B is admitted after E's first round
    got_b, got_e = b.reply(), e.reply()
    assert got_b is not None and got_b[0] == "error" and isinstance(got_b[1], torch.OutOfMemoryError), got_b
    assert got_e is not None and got_e[0] == "done", got_e
    assert sched.thread.is_alive() and not dec.streams


def test_two_ranks_answer_every_later_request_with_the_restart_error(decoders):
    make, fail = decoders
    dec = make(world=2)
    sched = Scheduler(dec, max_streams=4)
    assert ask(sched, A, 1)[0] == "done"
    fail["copy"] = True
    b = ask(sched, B, 64)
    assert b is not None and b[0] == "error" and isinstance(b[1], torch.OutOfMemoryError), b
    assert isinstance(dec.broken, torch.OutOfMemoryError)
    for prompt in (C, D):
        got = ask(sched, prompt, 1)
        assert got is not None and got[0] == "error", got
        assert "restart both" in str(got[1])
    assert sched.thread.is_alive() and not dec.streams


class FailingRound:
    """A decoder whose first round fails; its ``drop()`` also returns a stream no request waits on."""

    def __init__(self):
        self.streams, self.failed = [], False

    def live(self):
        return len(self.streams)

    def admit(self, s):
        self.streams.append(s)
        s.take([1])

    def round(self):
        live = [s for s in self.streams if not s.done]
        if live and not self.failed:
            self.failed = True
            raise RuntimeError("a round failed")
        for s in live:
            s.take([2])
        return [s for s in live if s.done]

    def drop(self):
        live = [s for s in self.streams if not s.done]
        self.streams = [s for s in self.streams if s.done]
        return live + [Stream([9], 1)]

    def finish(self, done):
        gone = {id(s) for s in done}
        self.streams = [s for s in self.streams if id(s) not in gone]


def test_a_dropped_stream_without_a_request_does_not_stop_the_worker():
    dec = FailingRound()
    sched = Scheduler(dec, max_streams=4)
    x = ask(sched, [1, 2, 3], 2)
    assert x is not None and x[0] == "error" and str(x[1]) == "a round failed", x
    y = ask(sched, [4, 5, 6], 2)
    assert y is not None and y[0] == "done", y
    assert sched.thread.is_alive() and not dec.streams

