"""ignore_eos on the 27B's CUDA engine: the server hands ``stop_eos`` to engines that take it, and every decode
path of the engine (one GPU, two ranks, concurrent streams) receives it; a stream carries its own end tokens."""

import importlib
import queue
from types import SimpleNamespace

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tensorfold.cuda import server
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import PrefixCache, Stream, accept
from tests.test_cuda_admission import http_server
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the modules import)
from tests.test_cuda_stop_strings import END, Engine, ids, make_app, reply


class StopEosEngine(Engine):
    """Takes ``stop_eos`` in ``generate``, as the 27B's engine does."""

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, stop_eos=True):
        self.stop_eos = stop_eos
        return super().generate(prompt, max_tokens, sampling, on_tokens, draft)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("ignore", [None, False, True])
def test_the_server_hands_stop_eos_only_to_an_engine_that_takes_it(tmp_path, stream, ignore):
    text = "one" + END + "two" + END + "three"
    reply_ids = ids(text)
    taker, other = StopEosEngine(reply_ids), Engine(reply_ids)
    fields = {} if ignore is None else {"ignore_eos": ignore}
    with http_server(make_app(tmp_path, taker)) as port:
        got = [reply(port, True, stream, draft=d, max_tokens=len(reply_ids), **fields) for d in (True, False)]
    with http_server(make_app(tmp_path, other)) as port:       # a TypeError here if stop_eos were passed
        plain = [reply(port, True, stream, draft=d, max_tokens=len(reply_ids), **fields) for d in (True, False)]
    n = len(ids("one")) + 1
    before = [("one", "", "stop", n, server.token_sha(reply_ids[:n]), [])] * 2
    assert plain == before and {c["stop_eos"] for c in other.calls} == {True}
    assert {c["stop_eos"] for c in taker.calls} == {not ignore}
    if ignore:   # end tokens are text, as on the Mac; the reply runs to its limit
        assert got == [(text, "", "length", len(reply_ids), server.token_sha(reply_ids), [])] * 2
    else:
        assert got == before


@pytest.mark.parametrize("stream", [False, True])
def test_ignore_eos_keeps_stop_strings_active_on_the_27b(tmp_path, stream):
    text = "one" + END + "two" + END + "three"
    reply_ids = ids(text)
    with http_server(make_app(tmp_path, StopEosEngine(reply_ids))) as port:
        got = [reply(port, False, stream, draft=d, ignore_eos=True, stop=["hr"]) for d in (True, False)]
    through = len(ids("one" + END + "two" + END + "thr"))
    assert got == [("one" + END + "two" + END + "t", "", "stop", through, server.token_sha(reply_ids[:through]),
                    [])] * 2


def test_accept_and_take_follow_the_end_tokens_they_are_given():
    tokens, parents = [10, 11, 12, 13], [-1, 0, 1, 2]
    assert accept(tokens, parents, [11, 12, 13, 14], room=8, eos=(12,)) == ([0, 1], 12)
    assert accept(tokens, parents, [11, 12, 13, 14], room=8, eos=()) == ([0, 1, 2, 3], 14)
    s = Stream([1], 10)
    s.take([5, 12], ())
    assert not s.done
    s.take([7, 12], (12,))
    assert s.done and s.out == [5, 12, 7, 12]


SCRIPT = [5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 13, 14, 15, 16]      # end token 0 twice, early


def scripted_decoder(multi, script=SCRIPT, eos=(0,)):
    """The 27B's ``MultiDecoder`` with the model replaced by ``script``: a drafted stream's window drafts the next
    three tokens of it; a serial stream's window is one row. Acceptance and ``take`` are the decoder's own."""

    from tensorfold.cuda.streams import PrefixCache

    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.w, dec.draft, dec.max_rows, dec.allow_copy = None, None, 16, False
    dec.context, dec.eos, dec.rank, dec.world, dec.device = 0, tuple(eos), 0, 1, None
    dec.split, dec.drafts, dec.streams, dec.cache = False, False, {}, PrefixCache(8)
    dec.next_id, dec.broken, dec.costs, dec.filling, dec.points = 0, None, None, [], None

    def _queue(s, hit):                                   # a prompt's prefill step: the script's first token
        s.st, s.snap, s.stops = SimpleNamespace(pos=0), None, []
        dec.filling.append(s)

    def _step(s, stop):
        s.context, s.copies = list(s.prompt), None
        dec.filling = [x for x in dec.filling if x is not s]
        dec.streams[s.sid] = s
        return script[0]

    def _verify(plan, copied=None):
        wins, sampled = [], []
        for sid, _, pending, _ in plan:
            s = dec.streams[sid]
            n = len(s.out)
            guesses = script[n:n + 3] if s.draft else []
            wins.append(([pending] + guesses, list(range(-1, len(guesses)))))
            sampled.append([script[n + i] if n + i < len(script) else 99 for i in range(len(guesses) + 1)])
        return wins, None, None, None, sampled

    dec._queue, dec._step, dec._verify = _queue, _step, _verify
    dec._commit = lambda plan, wins, record, taps, starts, paths: [
        dec.streams[item[0]].counted(len(w[0])) for item, w in zip(plan, wins)]
    return dec


@pytest.mark.torch
@pytest.mark.parametrize("script", [SCRIPT, [0] + SCRIPT[1:]], ids=["end-later", "end-first"])
@pytest.mark.parametrize("draft", [True, False])
def test_concurrent_streams_each_stop_at_their_own_end_tokens(allocations, draft, script):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    dec = scripted_decoder(multi, script)
    rounds: list[list[int]] = []
    stopping = Stream([1, 2], 12, draft=draft)
    ignoring = Stream([3, 4], 12, draft=draft, stop_eos=False, emit=lambda new: rounds.append(list(new)))
    for s in (stopping, ignoring):
        dec.admit(s)
    dec.finish([s for s in (stopping, ignoring) if s.done])      # as the scheduler does after admitting
    while dec.live():
        dec.finish(dec.round())
    assert stopping.out == script[:script.index(0) + 1]      # ends at its first end token
    assert ignoring.out == script[:12]                      # decodes past every end token to its count
    # a drafted round accepts through an end token (serial rounds are one token each)
    assert (ignoring.rounds < 11) is draft and any(0 in r[:-1] for r in rounds) is draft


@pytest.mark.torch
def test_the_scheduler_hands_stop_eos_to_its_stream(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    sched = Scheduler(scripted_decoder(multi), max_streams=2)
    got: dict[bool, list[int]] = {True: [], False: []}
    for stop_eos in (True, False):
        sched.submit([1], 9, None, True, lambda new, k=stop_eos: got[k].extend(new) or False, stop_eos=stop_eos)
    assert got == {True: SCRIPT[:3], False: SCRIPT[:9]}
    got_default: list[int] = []
    sched.submit([1], 9, None, True, lambda new: got_default.extend(new) or False)
    assert got_default == SCRIPT[:3]


def bare_engine(engine_mod, **attrs):
    eng = engine_mod.Qwen27Engine.__new__(engine_mod.Qwen27Engine)
    eng.context_window, eng.scheduler, eng.tp, eng.draft, eng.cache = 1000, None, 1, None, PrefixCache(4)
    eng.points = None
    eng.max_rows, eng.allow_copy, eng.eos = 12, True, (0,)
    eng.w = SimpleNamespace(norm=SimpleNamespace(device="cpu"))
    for k, v in attrs.items():
        setattr(eng, k, v)
    return eng


@pytest.mark.torch
@pytest.mark.parametrize("stop_eos", [None, True, False])
@pytest.mark.parametrize("draft", [True, False])
def test_the_engine_passes_stop_eos_to_the_one_gpu_decode(monkeypatch, allocations, stop_eos, draft):  # noqa: F811
    engine_mod = importlib.import_module("tensorfold.families.qwen3_5.cuda.engine")
    decode = importlib.import_module("tensorfold.families.qwen3_5.cuda.decode")
    seen = {}
    monkeypatch.setattr(decode, "prefill", lambda w, prompt, sampling, drafter, state=None, keep_at=None, **kw:
                        (SimpleNamespace(pos=0), 5, (SimpleNamespace(pos=keep_at), None)))
    monkeypatch.setattr(decode, "draft_decode", lambda *a, **kw: seen.update(kw) or SimpleNamespace(
        seconds=0.0, rounds=0, widths=[], drafted_rows=0, accepted_drafts=0))
    eng = bare_engine(engine_mod)
    kw = {} if stop_eos is None else {"stop_eos": stop_eos}
    eng.generate([1, 2, 3], 8, None, lambda new: False, draft=draft, **kw)
    assert seen["stop_eos"] is (stop_eos is not False)


@pytest.mark.torch
@pytest.mark.parametrize("stop_eos", [True, False])
def test_the_engine_passes_stop_eos_to_rank_zero_of_two(monkeypatch, allocations, stop_eos):  # noqa: F811
    engine_mod = importlib.import_module("tensorfold.families.qwen3_5.cuda.engine")
    decode_tp = importlib.import_module("tensorfold.families.qwen3_5.cuda.decode_tp")
    seen, shared = {}, []
    monkeypatch.setattr(decode_tp, "_share", lambda values, rank, device: shared.append(list(values)) or values)
    monkeypatch.setattr(decode_tp, "prefill_tp",
                        lambda w, prompt, sampling, rank, drafter, state=None, keep_at=None, **kw:
                        (SimpleNamespace(pos=0), 5, (SimpleNamespace(pos=keep_at), None)))
    monkeypatch.setattr(decode_tp, "decode_tp", lambda *a, **kw: seen.update(kw) or SimpleNamespace(
        seconds=0.0, rounds=0, widths=[]))
    eng = bare_engine(engine_mod, tp=2)
    eng.generate([1, 2, 3], 8, None, lambda new: False, stop_eos=stop_eos)
    assert seen["stop_eos"] is stop_eos
    # no stop_eos field: rank 1 follows rank 0 (after the sampling words, the image flag)
    assert len(shared[0]) == 5 + decode_tp.SAMPLING_WORDS


@pytest.mark.torch
@pytest.mark.parametrize("stop_eos", [True, False])
def test_the_engine_passes_stop_eos_to_its_scheduler(allocations, stop_eos):  # noqa: F811
    engine_mod = importlib.import_module("tensorfold.families.qwen3_5.cuda.engine")
    calls = queue.Queue()
    sched = SimpleNamespace(submit=lambda *a, **kw: calls.put((a, kw)) or {})
    eng = bare_engine(engine_mod, scheduler=sched)
    eng.generate([1, 2, 3], 8, None, lambda new: False, stop_eos=stop_eos)
    args, kw = calls.get_nowait()
    assert kw == {"stop_eos": stop_eos} and args[:4] == ([1, 2, 3], 8, None, True)


@pytest.mark.torch
def test_the_admit_message_rank_one_reads_is_unchanged(allocations):  # noqa: F811
    """Rank 1 builds its streams from the ADMIT message alone and commits the paths rank 0 sends: end tokens are
    decided on rank 0 only, so the message carries no stop_eos field (after the sampling words, the image flag)."""

    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    dec = scripted_decoder(multi)
    sent = []
    dec.world, dec._send = 2, sent.append
    dec.admit(Stream([1, 2], 12, stop_eos=False))
    assert sent[0][:5] == [multi.ADMIT, 0, 12, 1, 0] and len(sent[0]) == 6 + multi.W
