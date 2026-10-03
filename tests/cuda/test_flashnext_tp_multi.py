"""Flash Next's concurrent decoder on two ranks (--tp 2 --parallel N), and a lone stream in the one-stream graphs.

Two ranks run in two threads on one GPU, their all-gathers through host memory (test_flashnext_tp's fakes); rank 0
drives a MultiDecoder and tells rank 1 every admission, round and completion over a fake rendezvous store. Checks:
streams decoded together emit what each rank's tensor-parallel serial decoding emits, greedy and sampled (the
gathered candidates and the wider rules' collective path), a client leaving on rank 0 alone, prompts resuming from
kept slots; and on one GPU, a stream decoding alone replays the one-stream graphs with serial decoding's tokens.
"""

import gc
import threading
from contextlib import contextmanager

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_tp import _checkpoint, _run_ranks

from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
from tensorfold.families.qwen4_exp.cuda.multi import Link, MultiDecoder, OutOfStep
from tensorfold.families.qwen4_exp.cuda.weights import load

PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31], [400, 401, 402]]
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=0, top_p=0.9),
             Sampling(seed=9, top_k=50, top_p=0.95), Sampling(seed=11, temperature=0.7, top_k=20, min_p=0.1)]


@contextmanager
def _scheduler(dec):
    """Stop the test's idle worker so it releases its decoder before the next case."""

    from tensorfold.cuda.scheduler import Scheduler

    class StopWorker(Exception):
        pass

    class TestScheduler(Scheduler):
        def _admit(self, first=None):
            if first is not None and first[1] is None:
                raise StopWorker
            return super()._admit(first)

        def _loop(self):
            try:
                super()._loop()
            except StopWorker:
                pass

    sch = TestScheduler(dec, max_streams=4)
    try:
        yield sch
    finally:
        sch.waiting.put((Stream([], 1, None), None))
        sch.thread.join(timeout=10)
        assert not sch.thread.is_alive(), "the scheduler worker did not stop"


class _Store:
    """The TCP store's set / wait / get / delete_key, in memory."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.cv = threading.Condition()

    def set(self, key, value) -> None:
        with self.cv:
            self.data[key] = value.encode() if isinstance(value, str) else value
            self.cv.notify_all()

    def wait(self, keys, timeout=None) -> None:
        with self.cv:
            if not self.cv.wait_for(lambda: all(k in self.data for k in keys), timeout=60):
                raise TimeoutError(keys)

    def get(self, key) -> bytes:
        return self.data[key]

    def delete_key(self, key) -> None:
        with self.cv:
            self.data.pop(key, None)


@pytest.fixture(autouse=True)
def release_decoder_cycles():
    """Tamper closures and captured graph slots can hold decoders until cyclic collection."""

    gc.collect()
    yield
    gc.collect()


@pytest.fixture(scope="module")
def ranks(tmp_path_factory):
    root = tmp_path_factory.mktemp("flashnext_tp_multi")
    _checkpoint(root)
    return [load(root, tp=(r, 2)) for r in range(2)]


def _serial(ranks, prompt, sampling, count, kv_dtype):
    """Each rank's tensor-parallel serial decoding (both ranks emit the same tokens)."""

    engines = [Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype) for w in ranks]
    got = _run_ranks(lambda r, e: serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens, engines)
    assert got[0] == got[1]
    return got[0]


def _two_rank_run(ranks, kv_dtype, drive, slots=4, tamper=None, setup=None, **options):
    """rank 0: ``drive(decoder)`` with its link to rank 1, then stop; rank 1: follow (``tamper(decoder)`` first, to
    put it out of step on purpose). Returns drive's result."""

    decs = [MultiDecoder(w, slots=slots, capacity=1024, depth=3, confidence=0.3,
                         kv_dtype=kv_dtype, graphs=False, **options)
            for w in ranks]
    if setup is not None:
        for dec in decs:
            setup(dec)
    if tamper is not None:
        tamper(decs[1])
    store = _Store()

    def body(r, dec):
        if r == 0:
            dec.link = Link(store)
            try:
                return drive(dec)
            finally:
                dec.link.send(["stop"])
        dec.follow(Link(store))
        return dec

    out = _run_ranks(body, decs)
    follower = out[1]
    assert not follower.streams and not follower.filling
    assert len(follower.free) + len({id(k[1]) for k in follower.kept}) == slots
    return out[0]


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
def test_tp_streams_decoded_together_equal_each_ranks_serial_decoding(ranks, kv_dtype):
    refs = [_serial(ranks, p, s, 20, kv_dtype) for p, s in zip(PROMPTS, SAMPLINGS)]

    def drive(dec):
        streams = []
        for i, (prompt, sampling) in enumerate(zip(PROMPTS, SAMPLINGS)):
            got: list[int] = []
            s = Stream(prompt, 20, sampling, draft=i != 2, emit=lambda new, got=got: got.extend(new))
            if i < 4:
                dec.admit(s)
            streams.append((s, got))
        joined = False
        while dec.live():
            dec.finish(dec.round())
            if not joined:                                    # the fifth joins after the first round, mid-flight
                dec.admit(streams[4][0])
                joined = True
        return streams

    for i, (s, got) in enumerate(_two_rank_run(ranks, kv_dtype, drive, slots=5)):
        assert got == refs[i] and s.out == refs[i], i


def test_tp_a_client_leaving_on_rank_0_alone_keeps_the_ranks_in_step(ranks):
    refs = [_serial(ranks, p, s, 20, "int8") for p, s in zip(PROMPTS[:3], SAMPLINGS[:3])]

    def drive(dec):
        outs = []
        for i, (prompt, sampling) in enumerate(zip(PROMPTS[:3], SAMPLINGS[:3])):
            got: list[int] = []
            # stream 1's client leaves after its fourth token: rank 0 alone ends it, rank 1 hears of it
            emit = (lambda new, got=got: (got.extend(new), len(got) >= 4)[1]) if i == 1 else \
                (lambda new, got=got: got.extend(new))
            s = Stream(prompt, 20, sampling, emit=emit)
            dec.admit(s)
            outs.append((s, got))
        while dec.live():
            dec.finish(dec.round())
        after: list[int] = []                                 # the ranks still agree: a new request decodes exactly
        s = Stream(PROMPTS[3], 20, SAMPLINGS[3], emit=lambda new: after.extend(new))
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        return outs, after

    outs, after = _two_rank_run(ranks, "int8", drive)
    assert outs[0][1] == refs[0] and outs[2][1] == refs[2]
    assert outs[1][1] == refs[1][:len(outs[1][1])] and 4 <= len(outs[1][1]) < 20
    assert after == _serial(ranks, PROMPTS[3], SAMPLINGS[3], 20, "int8")


def test_tp_prompts_resume_from_kept_slots_on_both_ranks(ranks):
    sampling = Sampling(seed=31, top_k=20, top_p=0.95)

    def drive(dec):
        def run(prompt, count):
            s = Stream(list(prompt), count, sampling)
            dec.admit(s)
            while dec.live():
                dec.finish(dec.round())
            return s
        first = run(PROMPTS[1], 12)
        longer = PROMPTS[1] + first.out[:-1] + [42, 43]
        return longer, run(longer, 10)

    longer, warm = _two_rank_run(ranks, "int8", drive)
    assert warm.cached == len(PROMPTS[1]) - 1 and warm.out == _serial(ranks, longer, sampling, 10, "int8")


@pytest.mark.parametrize("sampling", [None, Sampling(seed=5, top_k=20, top_p=0.95)])
@pytest.mark.parametrize("join", [False, True])
@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
def test_a_lone_stream_replays_the_one_stream_graphs_with_serial_tokens(sampling, join, kv_dtype, monkeypatch):
    from test_flashnext_forward import _model

    w = _model()
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype)
    assert dec.solo is not None and dec.free[0] is dec.solo.st
    calls = {"solo": 0}
    real = dec._solo_round
    monkeypatch.setattr(dec, "_solo_round", lambda s: (calls.__setitem__("solo", calls["solo"] + 1), real(s))[1])

    def fresh(prompt, count):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        return serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens

    alone = Stream(PROMPTS[0], 24, sampling)
    dec.admit(alone)
    assert alone.st is dec.solo.st                           # a lone request takes the graphs' slot
    rounds = 0
    other = None
    while dec.live():
        dec.finish(dec.round())
        rounds += 1
        if join and rounds == 3:                         # a second stream joins, decodes beside it, then leaves
            other = Stream(PROMPTS[1], 6, sampling)
            dec.admit(other)
    assert calls["solo"] >= 3 and dec.solo.graphs.captures > 0
    assert alone.out == fresh(PROMPTS[0], 24)
    assert other is None or other.out == fresh(PROMPTS[1], 6)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_a_lone_stream_keeps_the_graph_slot_without_discarding_kept_prefixes(kv_dtype):
    """A lone stream moves keeps out of its graph slot without changing either reply."""

    from test_flashnext_forward import _model

    w = _model()
    sampling = Sampling(seed=17, top_k=20, top_p=0.95)
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype)

    def fresh(prompt, count):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        return serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens

    def run(prompt, count):
        s = Stream(list(prompt), count, sampling)
        dec.admit(s)
        start = s.st
        while dec.live():
            dec.finish(dec.round())
        return s, start

    first, start = run(PROMPTS[0], 12)
    assert start is dec.solo.st and any(k[1] is dec.solo.st for k in dec.kept)     # its prompt end kept there
    first_slot = start
    graphs = dec.solo.graphs
    captured = dict(graphs.main)
    dropped, state_changed = [], dec._state_changed
    dec._state_changed = lambda st: (dropped.append(st), state_changed(st))[1]
    second, start = run(PROMPTS[1], 16)
    assert start is not first_slot and second.st is dec.solo.st is first_slot
    assert first_slot not in dropped
    assert dec.solo.graphs is graphs
    assert captured and all(graphs.main[key] is graph for key, graph in captured.items())
    assert any(k[1] is not first_slot and k[0] == PROMPTS[0][:-1] for k in dec.kept)
    assert first.out == fresh(PROMPTS[0], 12) and second.out == fresh(PROMPTS[1], 16)
    assert not any(k[1] is dec.solo.st and k[0] == PROMPTS[0] for k in dec.kept)
    again, _ = run(PROMPTS[1] + second.out[:-1] + [42], 8)                       # resumes from the old slot's kept end
    assert again.cached == len(PROMPTS[1]) - 1 and again.out == fresh(PROMPTS[1] + second.out[:-1] + [42], 8)


PROMPTS8 = PROMPTS + [[77, 78], [2, 3, 5, 7, 11, 13], [600, 9]]
SAMPLINGS8 = SAMPLINGS + [None, Sampling(seed=21, top_k=20, top_p=0.95), Sampling(seed=23, temperature=0.8, top_k=40)]
_SOLO: dict[int, list[int]] = {}


def _alone(ranks, i, count):
    """Request ``i`` served by itself on two ranks, with drafts (its solo run); kept across the parametrized cases."""

    if i not in _SOLO:
        def drive(dec):
            s = Stream(list(PROMPTS8[i]), count, SAMPLINGS8[i])
            dec.admit(s)
            while dec.live():
                dec.finish(dec.round())
            return s.out
        _SOLO[i] = _two_rank_run(ranks, "int8", drive, slots=8)
    return _SOLO[i]


@pytest.mark.parametrize("users", [1, 2, 4, 8])
def test_tp_every_reply_equals_its_serial_run_and_its_solo_run_at_1_2_4_8_users(ranks, users):
    """Sampling over the split vocabulary per stream: every reply, greedy or sampled, equals by token ids the same
    request's ``"draft": false`` run (serial, no drafts) and its solo run (alone, with drafts)."""

    count = 16

    def drive(dec):
        streams = [Stream(list(p), count, s) for p, s in zip(PROMPTS8[:users], SAMPLINGS8[:users])]
        for s in streams:
            dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        return [s.out for s in streams]

    together = _two_rank_run(ranks, "int8", drive, slots=8)
    for i, got in enumerate(together):
        assert got == _serial(ranks, PROMPTS8[i], SAMPLINGS8[i], count, "int8"), (users, i)
        assert got == _alone(ranks, i, count), (users, i)


@pytest.mark.parametrize("leaves", ["in prefill", "in decode", "in both"])
def test_tp_a_client_leaving_in_prefill_or_decode_leaves_neither_rank_hanging(ranks, leaves):
    """Through the server's Scheduler (its admit / round / finish order): a client that leaves before its first
    token (a disconnect while its prompt prefills) or mid-reply ends on rank 0 alone; rank 1 hears of it before the
    next round, both ranks end with every slot free, the others' replies and a later request stay exact."""

    leave = {"in prefill": {1: 1}, "in decode": {1: 5}, "in both": {1: 1, 2: 5}}[leaves]

    def drive(dec):
        with _scheduler(dec) as sch:
            got: dict[int, list[int]] = {i: [] for i in range(4)}
            errors: list = []

            def client(i):
                def emit(new):
                    got[i].extend(new)
                    return i in leave and len(got[i]) >= leave[i]
                try:
                    sch.submit(list(PROMPTS8[i]), 20, SAMPLINGS8[i], True, emit)
                except Exception as exc:                          # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=client, args=(i,)) for i in range(4)]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=120)
            assert not any(th.is_alive() for th in threads), "a request never finished"
            after: list[int] = []                                 # the ranks still agree: a new request decodes exactly
            sch.submit(list(PROMPTS8[5]), 20, SAMPLINGS8[5], True, lambda new: after.extend(new))
            return got, after, errors

    got, after, errors = _two_rank_run(ranks, "int8", drive, slots=4)
    assert not errors, errors
    for i in range(4):
        ref = _serial(ranks, PROMPTS8[i], SAMPLINGS8[i], 20, "int8")
        if i in leave:
            assert leave[i] <= len(got[i]) < 20 and got[i] == ref[:len(got[i])], (i, got[i])
        else:
            assert got[i] == ref, i
    assert after == _serial(ranks, PROMPTS8[5], SAMPLINGS8[5], 20, "int8")


def test_tp_a_resumed_prompt_equals_the_same_prompt_fresh(ranks):
    """A prompt resumed from a kept slot on both ranks replies, by token ids, as the same prompt prefilled fresh
    (with drafts, on a new decoder) and as its serial run."""

    sampling = Sampling(seed=31, top_k=20, top_p=0.95)

    def run(dec, prompt, count):
        s = Stream(list(prompt), count, sampling)
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        return s

    def resumed(dec):
        first = run(dec, PROMPTS[1], 12)
        longer = PROMPTS[1] + first.out[:-1] + [42, 43]
        return longer, run(dec, longer, 10)

    longer, warm = _two_rank_run(ranks, "int8", resumed)
    fresh = _two_rank_run(ranks, "int8", lambda dec: run(dec, longer, 10))
    assert warm.cached == len(PROMPTS[1]) - 1 and fresh.cached == 0
    assert warm.out == fresh.out == _serial(ranks, longer, sampling, 10, "int8")


def test_tp_a_stream_rank_0_ends_at_its_first_token_is_ended_on_rank_1_before_the_next_round(ranks):
    """The decoder alone, the scheduler's ordering: a client gone by its first token ends the stream at admission on
    rank 0 alone; the next round tells rank 1, the ranks stay in step and the stream beside it stays exact."""

    ref = _serial(ranks, PROMPTS8[0], SAMPLINGS8[0], 20, "int8")

    def drive(dec):
        beside: list[int] = []
        a = Stream(list(PROMPTS8[0]), 20, SAMPLINGS8[0], emit=lambda new: beside.extend(new))
        gone = Stream(list(PROMPTS8[1]), 20, SAMPLINGS8[1], emit=lambda new: True)
        dec.admit(a)
        dec.admit(gone)
        first = dec.round()                                  # admission now queues prefill inside a round
        assert gone.done and len(gone.out) == 1
        done = first + dec.round()                           # the round before the admission's finish, as served
        dec.finish(done)
        while dec.live():
            dec.finish(dec.round())
        return beside

    assert _two_rank_run(ranks, "int8", drive) == ref


def _served(dec, requests):
    """Requests through the server's Scheduler one after another: each reply's tokens, or the error it failed with."""

    out = []
    with _scheduler(dec) as sch:
        for prompt, count, sampling in requests:
            got: list[int] = []
            try:
                sch.submit(list(prompt), count, sampling, True, lambda new, got=got: got.extend(new))
                out.append(got)
            except Exception as exc:                              # noqa: BLE001
                out.append(exc)
    return out


def test_tp_ranks_planning_a_different_round_both_refuse_it_and_serve_on(ranks):
    """Rank 1 made to hold other drafts for stream 0: both ranks find it in the round's check before any of the round's
    work, stream 0 fails with OutOfStep instead of either rank waiting on the other, and the next request is exact."""

    def tamper(dec):
        real = dec.admit

        def admit(s, told=None):
            real(s, told=told)
            if s.sid == 0:
                s.drafts = list(s.drafts) + [7]
        dec.admit = admit

    requests = [(PROMPTS8[0], 20, SAMPLINGS8[0]), (PROMPTS8[1], 20, SAMPLINGS8[1])]
    first, second = _two_rank_run(ranks, "int8", lambda dec: _served(dec, requests), tamper=tamper)
    assert isinstance(first, OutOfStep), first
    assert second == _serial(ranks, PROMPTS8[1], SAMPLINGS8[1], 20, "int8")


def test_tp_rank_1_takes_rank_0s_slot_and_resume_point_or_both_refuse_the_admission(ranks):
    """Rank 1 resumes from the kept prompt end rank 0 names; when it has not got it (its kept prompt altered here),
    the admission fails on both ranks before the prefill, and serving goes on exactly."""

    sampling = Sampling(seed=31, top_k=20, top_p=0.95)

    def drive(dec):
        first = _served(dec, [(PROMPTS[1], 12, sampling)])[0]
        longer = PROMPTS[1] + first[:-1] + [42, 43]
        return first, longer, _served(dec, [(longer, 10, sampling), (PROMPTS8[5], 20, SAMPLINGS8[5])])

    def tamper(dec):                                          # after rank 1 keeps the first prompt, alter it
        real = dec._remember

        def remember(ids, st, snap, tail):
            real([*ids[:-1], ids[-1] + 1], st, snap, tail)
        dec._remember = remember

    _, _, (resumed, after) = _two_rank_run(ranks, "int8", drive, tamper=tamper)
    assert isinstance(resumed, OutOfStep), resumed
    assert after == _serial(ranks, PROMPTS8[5], SAMPLINGS8[5], 20, "int8")
    _, longer2, (resumed2, _) = _two_rank_run(ranks, "int8", drive)      # untouched: the resume goes through
    assert resumed2 == _serial(ranks, longer2, sampling, 10, "int8")


def test_tp_mixed_prompt_pieces_and_cache_growth_follow_the_leaders_width(ranks):
    prompts = [PROMPTS[0], (PROMPTS[1] * 53)[:310], (PROMPTS[3] * 87)[:430]]
    samplings = [SAMPLINGS[0], SAMPLINGS[1], SAMPLINGS[4]]
    refs = [_serial(ranks, p, smp, 20, "bf16") for p, smp in zip(prompts, samplings)]

    def drive(dec):
        dec.round_s, dec.row_s = 1.0, 1.0
        streams = [Stream(p, 20, smp) for p, smp in zip(prompts, samplings)]
        dec.admit(streams[0])
        dec.finish(dec.round())
        for s in streams[1:]:
            dec.admit(s)
            assert s.st.capacity > 256
        while dec.live():
            dec.finish(dec.round())
        return [s.out for s in streams]

    def tamper(dec):
        dec.round_s, dec.row_s = 1000.0, 1.0                  # follower timing must not choose prompt pieces

    assert _two_rank_run(ranks, "bf16", drive, tamper=tamper, prefill_rows=256, share=0.5) == refs


def test_tp_growth_that_does_not_fit_one_rank_refuses_before_model_collectives(ranks):
    long = (PROMPTS[1] * 53)[:310]

    def tamper(dec):
        real = dec.admit
        first = True
        def admit(s, told=None):
            nonlocal first
            saved = dec.memory_gate.live
            if first:
                dec.memory_gate.live = lambda: 0
                first = False
            try:
                real(s, told=told)
            finally:
                dec.memory_gate.live = saved
        dec.admit = admit

    result = _two_rank_run(ranks, "bf16", lambda dec: _served(dec, [(long, 8, None), (PROMPTS[0], 12, None)]),
                           tamper=tamper)
    assert isinstance(result[0], ValueError)
    assert "prompt cannot fit available memory" in str(result[0])
    assert result[1] == _serial(ranks, PROMPTS[0], None, 12, "bf16")


def test_lone_graph_slot_rebinds_after_growth_and_a_smaller_stream_moves_in():
    from test_flashnext_forward import _model

    w = _model()
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, prefill_rows=64)
    sampling = Sampling(seed=29, top_k=20, top_p=0.95)
    for prompt in [(PROMPTS[1] * 47)[:270], PROMPTS[0], (PROMPTS[3] * 103)[:510]]:
        s = Stream(prompt, 20, sampling)
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64)
        assert s.out == serial_decode(e, prefill(e, prompt, sampling), 20, sampling).tokens
        assert dec.solo.graphs.captures > 0


def test_waiting_stream_keeps_its_recurrence_when_other_rounds_reuse_shared_scratch(monkeypatch):
    from test_flashnext_forward import _model
    from tensorfold.families.qwen4_exp.cuda import multi

    monkeypatch.setattr(multi, "FIRST", 8)
    monkeypatch.setattr(multi, "STEP", 16)
    real = MultiDecoder._grow
    def grow(dec, st, rows, *, alone=False):
        blocked = dec.streams.get(1)
        if (getattr(dec, "test_wait", False) and blocked is not None and st is blocked.st
                and rows > st.capacity and not alone):
            return False
        return real(dec, st, rows, alone=alone)
    monkeypatch.setattr(MultiDecoder, "_grow", grow)
    w = _model()
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, prefill_rows=64)
    dec.test_wait = True
    streams = [Stream(PROMPTS[0], 60, stop_eos=False), Stream(PROMPTS[1], 20, stop_eos=False)]
    for s in streams:
        dec.admit(s)
    waits = 0
    while dec.live():
        dec.finish(dec.round())
        if streams[1].waiting:
            waits += 1
        if waits >= 4:
            dec.test_wait = False
    assert waits >= 4
    for s in streams:
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64)
        assert s.out == serial_decode(e, prefill(e, s.prompt, None), s.count, None).tokens


def test_tp_prompt_markers_follow_the_leader_with_different_local_markers(ranks):
    prompt = PROMPTS[0] * 80
    sampling = Sampling(seed=61, top_k=20, top_p=0.95)
    expected = _serial(ranks, prompt, sampling, 8, "int8")

    def drive(dec):
        stream = Stream(prompt, 8, sampling)
        dec.admit(stream)
        while dec.live():
            dec.finish(dec.round())
        assert stream.out == expected
        return [len(ids) for ids, _, _, _ in dec.kept]

    def different(dec):
        dec.points = lambda ids: []

    kept = _two_rank_run(ranks, "int8", drive, tamper=different, points=lambda ids: [256])
    assert 256 in kept and len(prompt) - 1 in kept
