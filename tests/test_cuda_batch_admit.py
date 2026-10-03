"""Queued foreground prompts prefill together: one forward takes each one's next step, STEP rows in all while streams
decode and a forward's prompt rows otherwise; background and image prompts go alone; two ranks mirror the batch."""

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.streams import Stream  # noqa: E402
from tests.test_cuda_geometry import allocations  # noqa: E402,F401  (fixture: fake triton, so the module imports)


@pytest.fixture
def multi(allocations):  # noqa: F811
    return importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")


def decoder(multi, prompt_rows=4096, world=1):
    """The 27B's decoder with its prefill steps recorded: a step moves the state, a prompt's end gives token 7."""

    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.w = SimpleNamespace(prompt_rows=prompt_rows)
    dec.streams, dec.filling, dec.eos, dec.world, dec.rank = {}, [], (0,), world, 0
    dec.sent, dec.batches, dec.alone = [], [], []
    dec._send = lambda message: dec.sent.append(list(message))

    def step(s, stop):
        s.st.pos = stop
        if stop < len(s.prompt):
            return None
        dec.filling.remove(s)
        dec.streams[s.sid] = s
        return 7

    def steps(batch):
        dec.batches.append([(s.sid, stop) for s, stop in batch])
        return [step(s, stop) for s, stop in batch]

    def alone(s, stop):
        dec.alone.append((s.sid, stop))
        return step(s, stop)

    dec._steps, dec._step = steps, alone
    return dec


def queue(dec, sizes, background=(), image=(), stops=None):
    out = []
    for sid, n in enumerate(sizes):
        s = Stream([1] * n, 4, background=sid in background)
        s.sid, s.st, s.stops = sid, SimpleNamespace(pos=0), list((stops or {}).get(sid, []))
        s.vision = object() if sid in image else None
        s.emit = lambda new: False
        dec.filling.append(s)
        out.append(s)
    return out


def decoding(dec):
    """A live stream, so prefill steps take STEP rows."""

    dec.streams[99] = SimpleNamespace(done=False)


def test_prompts_queued_together_prefill_in_one_forward(multi):
    dec = decoder(multi)
    ss = queue(dec, [20, 35, 19])
    assert dec._fill() == []                                      # none ends at its first token
    assert dec.batches == [[(0, 20), (1, 35), (2, 19)]] and not dec.alone
    assert all(s.out == [7] for s in ss) and not dec.filling


def test_while_streams_decode_a_batch_takes_step_rows(multi):
    dec = decoder(multi)
    decoding(dec)
    queue(dec, [600, 600, 30])
    dec._fill()
    assert dec.batches == [[(0, 600), (1, multi.STEP - 600)]]     # the second takes the rows left
    dec._fill()
    assert dec.batches[-1] == [(1, 600), (2, 30)]


def test_nothing_decoding_a_batch_fills_one_forward(multi):
    dec = decoder(multi, prompt_rows=100)
    queue(dec, [60, 60, 60])
    dec._fill()
    assert dec.batches == [[(0, 60), (1, 40)]]
    dec._fill()
    assert dec.batches[-1] == [(1, 60), (2, 60)]


def test_a_batch_stops_at_each_prompts_next_kept_state(multi):
    dec = decoder(multi)
    queue(dec, [300, 40], stops={0: [256]})
    dec._fill()
    assert dec.batches == [[(0, 256), (1, 40)]]


def test_background_prompts_go_alone_after_the_foreground(multi):
    dec = decoder(multi)
    queue(dec, [30, 20, 25], background=(0,))
    dec._fill()
    assert dec.batches == [[(1, 20), (2, 25)]] and not dec.alone
    dec._fill()
    assert dec.alone == [(0, 30)]


def test_an_image_prompt_goes_alone_in_its_turn(multi):
    dec = decoder(multi)
    queue(dec, [30, 20, 25], image=(0,))
    dec._fill()
    assert dec.alone == [(0, 30)] and not dec.batches
    dec._fill()
    assert dec.batches == [[(1, 20), (2, 25)]]


def test_one_prompt_takes_the_one_prompt_step(multi):
    dec = decoder(multi)
    decoding(dec)
    queue(dec, [3000])
    dec._fill()
    assert dec.alone == [(0, multi.STEP)] and not dec.batches


def test_batch_off_fills_one_prompt_a_round(multi, monkeypatch):
    monkeypatch.setattr(multi, "BATCH", False)
    dec = decoder(multi)
    queue(dec, [20, 35])
    dec._fill()
    assert dec.alone == [(0, 20)] and not dec.batches


def test_two_ranks_send_the_batch_and_the_follower_runs_it(multi, monkeypatch):
    dec = decoder(multi, world=2)
    queue(dec, [20, 35])
    dec._fill()
    assert dec.sent == [[multi.FILLS, 2, 0, 20, 1, 35]]
    follower = decoder(multi, world=2)
    queue(follower, [20, 35])
    messages = iter([dec.sent[0], []])
    monkeypatch.setattr(multi, "_share", lambda values, src, device: next(messages))
    follower.device = None
    follower.follow()
    assert follower.batches == dec.batches


def test_a_failed_batch_ends_each_of_its_requests_and_the_rest_wait(multi):
    dec = decoder(multi)
    ss = queue(dec, [20, 35, 30], background=(2,))

    def fail(batch):
        raise RuntimeError("simulated")

    dec._steps = fail
    done = dec._fill()
    assert done == ss[:2] and all(s.done and isinstance(s.error, RuntimeError) for s in ss[:2])
    assert dec.filling == [ss[2]]
