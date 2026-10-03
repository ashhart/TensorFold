"""priority "background" on CUDA, as on the Mac: a background request comes after the others, one decoding gives its
lane (or the one-at-a-time engine) to a foreground request that waits and decodes on later from its reply with the
same tokens, and a foreground prompt prefills before a background one."""

import importlib
import threading
import time

import pytest

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream
from tensorfold.cuda.turns import Turns
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the modules import)


def _until(check, seconds=10.0):
    end = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.002)


class Lanes:
    """A decoder whose streams write token (position % 97), one a round: a continuation writes what its stream would."""

    def __init__(self):
        self.streams, self.admitted, self.next = {}, [], 0

    def live(self):
        return len(self.streams)

    def admit(self, s):
        s.sid, self.next = self.next, self.next + 1
        self.admitted.append((list(s.prompt), s.background))
        self.streams[s.sid] = s
        s.take([len(s.prompt) % 97])

    def round(self):
        time.sleep(0.005)
        for s in list(self.streams.values()):
            if not s.done:
                s.counted(1)
                s.take([(len(s.prompt) + len(s.out)) % 97])
        return [s for s in self.streams.values() if s.done]

    def finish(self, done):
        for s in done:
            self.streams.pop(s.sid, None)

    def drop(self):
        live, self.streams = list(self.streams.values()), {}
        return live


def test_a_background_stream_gives_its_lane_to_a_foreground_request_and_replays():
    dec = Lanes()
    sched = Scheduler(dec, max_streams=1)
    back, front, stats = [], [], {}
    worker = threading.Thread(target=lambda: stats.update(back=sched.submit(
        [1] * 10, 60, None, True, lambda new: back.extend(new) or False, background=True)))
    worker.start()
    _until(lambda: len(back) >= 5)
    stats["front"] = sched.submit([2] * 7, 5, None, True, lambda new: front.extend(new) or False)
    assert len(back) < 60 and worker.is_alive()               # the foreground finished first
    worker.join()
    assert front == [(7 + i) % 97 for i in range(5)]
    assert back == [(10 + i) % 97 for i in range(60)]         # every token once, as without the yield
    assert sched.yields == 1 and dec.admitted[2] == ([1] * 10, True)      # it replays from its own prompt
    assert stats["back"]["rounds"] > 59                       # the replay's rounds count too (the work was redone)


def test_background_requests_wait_for_the_foreground_ones_queued_with_them():
    dec = Lanes()
    order = []
    gate = threading.Event()
    real = dec.admit
    dec.admit = lambda s: (gate.wait(5), order.append(s.background), real(s))
    sched = Scheduler(dec, max_streams=1)
    threads = [threading.Thread(target=sched.submit, args=([3] * 4, 2, None, True, lambda new: False),
                                kwargs={"background": b}) for b in (True, False, True, False)]
    for t in threads:
        t.start()
        time.sleep(0.02)                                      # arrival order: background first
    gate.set()
    for t in threads:
        t.join()
    assert order[1:] == [False, False, True]                  # after the first, foreground before background


def test_turns_give_the_engine_to_a_waiting_foreground_request_first():
    turns = Turns()
    turns.take(False)
    got = []
    back = threading.Thread(target=lambda: (turns.take(True), got.append("back"), turns.give()))
    back.start()
    time.sleep(0.05)
    front = threading.Thread(target=lambda: (turns.take(False), got.append("front"), turns.give()))
    front.start()
    time.sleep(0.05)
    assert turns.wanted()
    turns.give()
    back.join()
    front.join()
    assert got == ["front", "back"]


class Steady:
    """One request at a time, a token a round (position-keyed, so a continuation writes the same ones)."""

    eos = (0,)
    tp = 1

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.prompts.append(list(prompt))
        for i in range(max_tokens):
            time.sleep(0.004)
            if on_tokens([100 + (len(prompt) + i) % 50]):
                break
        return {"rounds": max_tokens}


def _body(content, count, background):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": count, "return_token_ids": True}
    return {**body, "priority": "background"} if background else body


def test_one_at_a_time_a_background_reply_yields_between_rounds(tmp_path):
    from tests.test_cuda_tool_choice import app_for

    alone = app_for(tmp_path, Steady())
    want = alone.run(_body("tell me a story", 60, True), True, lambda d: True)["stats"]["token_ids"]
    engine = Steady()
    app = app_for(tmp_path, engine)
    got, deltas = {}, []
    worker = threading.Thread(target=lambda: got.update(back=app.run(_body("tell me a story", 60, True), True,
                                                                     lambda d: deltas.append(d) or True)))
    worker.start()
    _until(lambda: len(deltas) >= 5)
    got["front"] = app.run(_body("hi", 3, False), True, lambda d: True)
    assert worker.is_alive()                                  # the foreground reply came first
    worker.join()
    assert got["back"]["stats"]["token_ids"] == want          # the same tokens as alone
    first, front, rest = engine.prompts                       # it stopped, the foreground ran, it replayed
    assert len(front) < len(first) and rest == first


def test_a_session_title_request_is_background(tmp_path):
    from tests.test_cuda_tool_choice import app_for

    engine = Steady()
    app = app_for(tmp_path, engine)
    seen = []
    real = app._turns().take
    app.turns.take = lambda background, cancelled=None: (seen.append(background), real(background, cancelled))[1]
    title = [{"role": "system", "content": "Write a short title for this conversation."},
             {"role": "user", "content": "hello"}]
    for messages in (title, [{"role": "user", "content": "hello"}]):
        app.run({"messages": messages, "max_tokens": 2}, True, lambda d: True)
    assert seen == [True, False]


@pytest.mark.torch
@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_5_moe", "qwen4_exp"])
def test_the_engines_hand_background_to_their_scheduler_only_when_set(allocations, family):  # noqa: F811
    import inspect

    mod = importlib.import_module(f"tensorfold.families.{family}.cuda.engine")
    cls = next(c for c in vars(mod).values() if inspect.isclass(c) and c.__module__ == mod.__name__
               and "generate" in vars(c) and "background" in inspect.signature(c.generate).parameters)
    eng = cls.__new__(cls)
    calls = []
    eng.scheduler = type("S", (), {"submit": lambda self, *a, **kw: calls.append(kw) or {}})()
    for name, value in (("context_window", 4096), ("max_len", 4096), ("depth", 1), ("vision", None)):
        try:
            setattr(eng, name, value)
        except AttributeError:                                # a property of that engine
            pass
    for background in (False, True):
        eng.generate([1, 2, 3], 4, None, lambda new: False, background=background)
    assert ["background" in kw for kw in calls] == [False, True] and calls[1]["background"] is True


@pytest.mark.torch
def test_a_foreground_prompt_prefills_before_a_background_one(allocations):  # noqa: F811
    """A background prompt queued first does not hold back a foreground one's first token."""

    from tests.test_cuda_27b_ignore_eos import scripted_decoder

    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    dec = scripted_decoder(multi)
    back, front = Stream([1, 2, 3], 8, background=True), Stream([4, 5], 8)
    for s in (back, front):
        s.emit = lambda new: False
        dec.admit(s)
    dec.round()
    assert front.out and not back.out                         # the foreground prompt went first
    dec.round()
    assert back.out


@pytest.mark.torch
@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_5_moe"])
def test_a_background_prompt_prefills_a_step_at_a_time(allocations, family):  # noqa: F811
    """A background prompt prefills STEP rows a step even alone, so a foreground prompt arriving waits one step."""

    from types import SimpleNamespace

    multi = importlib.import_module(f"tensorfold.families.{family}.cuda.multi")
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.streams, dec.filling, dec.eos, dec.world = {}, [], (0,), 1
    dec._send = lambda message: None
    steps = []

    def step(s, stop):
        steps.append((s.background, stop))
        s.st.pos = stop
        if stop < len(s.prompt):
            return None
        dec.filling.remove(s)
        return 7                                          # the first token

    dec._step = step
    n = 2 * multi.STEP + 5
    for sid, background in enumerate((True, False)):
        s = Stream([1] * n, 4, background=background)
        s.sid, s.st, s.stops = sid, SimpleNamespace(pos=0), []
        dec.filling.append(s)
        while s in dec.filling:
            dec._fill()
    assert steps == [(True, multi.STEP), (True, 2 * multi.STEP), (True, n), (False, n)]


class Shifting:
    """An engine whose tokens depend on where its run started: a run resumed from prompt + reply writes other tokens
    than the run it continues, so only a replay from the same prompt gives the reply alone."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, count, feed):
        self.prompts.append(list(prompt))
        for i in range(count):
            if feed([1000 + (7 * len(prompt) + i) % 50]):      # a longer prompt shifts every token
                break
        return {"rounds": count}


class Once:
    """A replay gate that cuts once, after ``at`` tokens."""

    replay = True

    def __init__(self, at):
        self.at, self.seen, self.done = at, 0, False

    def cut(self, tokens):
        if self.done or self.seen + len(tokens) < self.at:
            return None
        self.done = True
        return self.at - self.seen, []

    def observe(self, token):
        self.seen += 1


def test_a_yield_replays_from_the_same_prompt_and_sends_each_token_once():
    from tensorfold.engine.call_gate import generate_gated

    alone, sent = Shifting(), []
    generate_gated(alone.generate, [7] * 5, 12, [], lambda new: sent.extend(new) or False)
    want, sent = list(sent), []
    engine = Shifting()
    generate_gated(engine.generate, [7] * 5, 12, [Once(4)], lambda new: sent.extend(new) or False)
    assert sent == want and engine.prompts == [[7] * 5, [7] * 5]


def test_a_replay_that_differs_from_what_it_sent_fails():
    from tensorfold.engine.call_gate import generate_gated

    runs = []

    def generate(prompt, count, feed):
        runs.append(1)
        for i in range(count):
            if feed([len(runs) * 100 + i]):                   # the second run writes other tokens
                break

    with pytest.raises(RuntimeError, match="replay"):
        generate_gated(generate, [1, 2], 8, [Once(3)], lambda new: False)
