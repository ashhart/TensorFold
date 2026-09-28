"""ignore_eos on the 27B's CUDA engine: the server hands ``stop_eos`` to engines that take it, and every decode
path of the engine (one GPU, two ranks, concurrent streams) receives it; a stream carries its own end tokens."""

import importlib
import json
import queue
from types import SimpleNamespace

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tensorfold.cuda import server
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream, accept
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


@pytest.mark.parametrize("ignore", [False, True])
def test_each_run_of_a_required_tool_call_gets_stop_eos(tmp_path, ignore):
    """tool_choice "required" stops the engine at its cut and runs it again from the reply: every run hears the
    request's ignore_eos."""

    from tests import test_cuda_tool_choice as tc

    class TakesStopEos(tc.Engine):
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, stop_eos=True):
            self.flags.append(stop_eos)
            return super().generate(prompt, max_tokens, sampling, on_tokens, draft)

    engine = TakesStopEos()
    engine.flags = []
    status, body = tc.ask(tc.app_for(tmp_path, engine), ignore_eos=ignore)
    choice = json.loads(body)["choices"][0]
    assert status == 200 and choice["finish_reason"] == "tool_calls"
    assert [c["function"]["name"] for c in choice["message"]["tool_calls"]] == ["get_weather"]
    assert len(engine.calls) == 2 and engine.flags == [not ignore] * 2


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


class St:
    """A committed state: its position only (the stand-in prefill and copies hold no tensors)."""

    def __init__(self, pos=0):
        self.pos, self.limit = pos, 0


def scripted_decoder(multi, monkeypatch, script=SCRIPT, eos=(0,)):
    """The 27B's ``MultiDecoder`` with the model replaced by ``script`` (on tests/test_cuda_failed_admission.py's
    CPU stand-ins): a prompt's first token is ``script[0]``; a drafted stream's window drafts the next three tokens
    of it; a serial stream's window is one row. Admission, prefill steps, acceptance and ``take`` are the decoder's
    own."""

    def prefill_state(w, prompt, st, **kw):
        st.pos = len(prompt)

    monkeypatch.setattr(multi, "prefill_state", prefill_state)
    monkeypatch.setattr(multi, "first_token", lambda *args: script[0])
    monkeypatch.setattr(multi, "kept", lambda st: St(st.pos))
    monkeypatch.setattr(multi, "private", lambda st, rows: St(st.pos))
    monkeypatch.setattr(multi, "State", lambda w: St())
    model = SimpleNamespace(config=SimpleNamespace(eos=tuple(eos), vocab=128), norm=SimpleNamespace(device="cpu"),
                            head=SimpleNamespace(n=128))
    dec = multi.MultiDecoder(model, None, allow_copy=False)

    def _verify(plan, copied=None):
        wins, sampled = [], []
        for sid, _, pending, _ in plan:
            s = dec.streams[sid]
            n = len(s.out)
            guesses = script[n:n + 3] if s.draft else []
            wins.append(([pending] + guesses, list(range(-1, len(guesses)))))
            sampled.append([script[n + i] if n + i < len(script) else 99 for i in range(len(guesses) + 1)])
        return wins, None, None, None, sampled

    dec._verify = _verify
    dec._commit = lambda plan, wins, record, taps, starts, paths: [
        dec.streams[item[0]].counted(len(win[0])) for item, win in zip(plan, wins)]
    return dec


@pytest.mark.parametrize("script", [SCRIPT, [0] + SCRIPT[1:]], ids=["end-later", "end-first"])
@pytest.mark.parametrize("draft", [True, False])
def test_concurrent_streams_each_stop_at_their_own_end_tokens(monkeypatch, allocations, draft, script):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    dec = scripted_decoder(multi, monkeypatch, script)
    rounds: list[list[int]] = []
    stopping = Stream([1, 2], 12, draft=draft)
    ignoring = Stream([3, 4], 12, draft=draft, stop_eos=False, emit=lambda new: rounds.append(list(new)))
    for s in (stopping, ignoring):
        dec.admit(s)
    while dec.live():            # a first token that is an end token ends its stream in its prefill step
        dec.finish(dec.round())
    assert stopping.out == script[:script.index(0) + 1]      # ends at its first end token
    assert ignoring.out == script[:12]                      # decodes past every end token to its count
    # a drafted round accepts through an end token (serial rounds are one token each)
    assert (ignoring.rounds < 11) is draft and any(0 in r[:-1] for r in rounds) is draft


def test_the_scheduler_hands_stop_eos_to_its_stream(monkeypatch, allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    sched = Scheduler(scripted_decoder(multi, monkeypatch), max_streams=2)
    got: dict[bool, list[int]] = {True: [], False: []}
    for stop_eos in (True, False):
        sched.submit([1], 9, None, True, lambda new, k=stop_eos: got[k].extend(new) or False, stop_eos=stop_eos)
    assert got == {True: SCRIPT[:3], False: SCRIPT[:9]}
    got_default: list[int] = []
    sched.submit([1], 9, None, True, lambda new: got_default.extend(new) or False)
    assert got_default == SCRIPT[:3]


def bare_engine(engine_mod, **attrs):
    from tensorfold.cuda.streams import PrefixCache

    eng = engine_mod.Qwen27Engine.__new__(engine_mod.Qwen27Engine)
    eng.context_window, eng.scheduler, eng.tp, eng.draft = 1000, None, 1, None
    eng.cache, eng.points = PrefixCache(engine_mod.KEEP_ONE), None
    eng.max_rows, eng.allow_copy, eng.eos = 12, True, (0,)
    eng.w = SimpleNamespace(norm=SimpleNamespace(device="cpu"))
    for k, v in attrs.items():
        setattr(eng, k, v)
    return eng


@pytest.mark.parametrize("stop_eos", [None, True, False])
@pytest.mark.parametrize("draft", [True, False])
def test_the_engine_passes_stop_eos_to_the_one_gpu_decode(monkeypatch, allocations, stop_eos, draft):  # noqa: F811
    engine_mod = importlib.import_module("tensorfold.families.qwen3_5.cuda.engine")
    decode = importlib.import_module("tensorfold.families.qwen3_5.cuda.decode")
    seen = {}
    monkeypatch.setattr(decode, "prefill", lambda w, prompt, sampling, drafter, state=None, **kw: (
        SimpleNamespace(pos=0), 5))
    monkeypatch.setattr(decode, "draft_decode", lambda *a, **kw: seen.update(kw) or SimpleNamespace(
        seconds=0.0, rounds=0, widths=[]))
    eng = bare_engine(engine_mod)
    kw = {} if stop_eos is None else {"stop_eos": stop_eos}
    eng.generate([1, 2, 3], 8, None, lambda new: False, draft=draft, **kw)
    assert seen["stop_eos"] is (stop_eos is not False)


@pytest.mark.parametrize("stop_eos", [True, False])
def test_the_engine_passes_stop_eos_to_rank_zero_of_two(monkeypatch, allocations, stop_eos):  # noqa: F811
    engine_mod = importlib.import_module("tensorfold.families.qwen3_5.cuda.engine")
    decode_tp = importlib.import_module("tensorfold.families.qwen3_5.cuda.decode_tp")
    seen, shared = {}, []
    monkeypatch.setattr(decode_tp, "_share", lambda values, rank, device: shared.append(list(values)) or values)
    monkeypatch.setattr(decode_tp, "prefill_tp",
                        lambda w, prompt, sampling, rank, drafter, state=None, **kw: (SimpleNamespace(pos=0), 5))
    monkeypatch.setattr(decode_tp, "decode_tp", lambda *a, **kw: seen.update(kw) or SimpleNamespace(
        seconds=0.0, rounds=0, widths=[]))
    eng = bare_engine(engine_mod, tp=2)
    eng.generate([1, 2, 3], 8, None, lambda new: False, stop_eos=stop_eos)
    assert seen["stop_eos"] is stop_eos
    assert len(shared[0]) == 18                   # the header rank 1 reads is unchanged: it follows rank 0's windows


@pytest.mark.parametrize("stop_eos", [True, False])
def test_the_engine_passes_stop_eos_to_its_scheduler(allocations, stop_eos):  # noqa: F811
    engine_mod = importlib.import_module("tensorfold.families.qwen3_5.cuda.engine")
    calls = queue.Queue()
    sched = SimpleNamespace(submit=lambda *a, **kw: calls.put((a, kw)) or {})
    eng = bare_engine(engine_mod, scheduler=sched)
    eng.generate([1, 2, 3], 8, None, lambda new: False, stop_eos=stop_eos)
    args, kw = calls.get_nowait()
    assert kw == {"stop_eos": stop_eos} and args[:4] == ([1, 2, 3], 8, None, True)


def test_the_admit_message_rank_one_reads_is_unchanged(monkeypatch, allocations):  # noqa: F811
    """Rank 1 builds its streams from the ADMIT message alone and commits the paths rank 0 sends: end tokens are
    decided on rank 0 only, so the message carries no new field."""

    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    dec = scripted_decoder(multi, monkeypatch)
    sent = []
    dec.world, dec._send = 2, sent.append
    dec.admit(Stream([1, 2], 12, stop_eos=False))
    assert sent[0][:5] == [multi.ADMIT, 0, 12, 1, 0] and len(sent[0]) == 19
