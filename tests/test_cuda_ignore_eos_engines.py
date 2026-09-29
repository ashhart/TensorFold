"""ignore_eos on Flash Next's and Nemotron's CUDA engines, as on the 27B (test_cuda_27b_ignore_eos.py): ``generate``
takes ``stop_eos`` and hands it to every decode path, one GPU, ``--parallel`` and two ranks (rank 1 reads the field
rank 0 sends), and a first token that is an end token ends the reply only when end tokens count."""

import importlib
import json
from types import SimpleNamespace

import pytest

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the modules import)

torch = pytest.importorskip("torch")

END = 0


@pytest.fixture
def modules(allocations, monkeypatch):  # noqa: F811
    """The fake triton, its ``jit`` taking options as Nemotron's kernels pass them."""

    import sys

    monkeypatch.setattr(sys.modules["triton"], "jit", lambda fn=None, **kw: fn if fn is not None else (lambda f: f))
    return allocations


def _recording(calls, name):
    def decode(*args, **kw):
        calls.append((name, kw["stop_eos"]))
        return SimpleNamespace(seconds=0.0, rounds=1, tokens_per_second=0.0, drafted=0, accepted=0, widths=[1])

    return decode


def _flash_next(monkeypatch, calls):
    mod = importlib.import_module("tensorfold.families.qwen4_exp.cuda.engine")
    decode = importlib.import_module("tensorfold.families.qwen4_exp.cuda.decode")
    monkeypatch.setattr(decode, "prefill", lambda e, prompt, sampling, **kw: END)       # the first token ends it
    monkeypatch.setattr(decode, "serial_decode", _recording(calls, "serial"))
    monkeypatch.setattr(decode, "mtp_decode", _recording(calls, "mtp"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    eng = mod.FlashNextEngine.__new__(mod.FlashNextEngine)
    eng.e = SimpleNamespace(st=SimpleNamespace(snapshot=lambda: {}), mbuf=None, last_streams=None)
    eng.serial, eng.tp, eng.depth, eng.confidence, eng.scheduler = SimpleNamespace(), 1, 3, 0.0, None
    eng.cache, eng.eos, eng.max_len, eng.served = [], (END,), 1024, 0
    return mod, eng


def _nemotron(monkeypatch, calls):
    mod = importlib.import_module("tensorfold.families.nemotron_h.cuda.app")
    decode = importlib.import_module("tensorfold.families.nemotron_h.cuda.decode")
    pre = SimpleNamespace(pending=END, engine=None, mtp=None, last_hidden=None)
    monkeypatch.setattr(decode, "prefill", lambda *a, **kw: pre)
    monkeypatch.setattr(decode, "serial_decode", _recording(calls, "serial"))
    monkeypatch.setattr(decode, "draft_decode", _recording(calls, "mtp"))
    eng = mod.NemotronEngine.__new__(mod.NemotronEngine)
    eng.e, eng.mtp, eng.serial = SimpleNamespace(max_rows=16), object(), SimpleNamespace()
    eng.tp, eng.drafts, eng.confidence, eng.cache = 1, 3, 0.0, []
    eng.eos, eng.max_len, eng.served = (END,), 1024, 0
    return mod, eng


@pytest.mark.torch
@pytest.mark.parametrize("family", [_flash_next, _nemotron], ids=["flashnext", "nemotron"])
@pytest.mark.parametrize("stop_eos", [True, False])
@pytest.mark.parametrize("draft", [True, False])
def test_every_path_takes_stop_eos(modules, monkeypatch, family, stop_eos, draft):
    calls, heard = [], []
    _, eng = family(monkeypatch, calls)
    eng.generate([5, 6, 7], 12, None, lambda new: heard.append(new) or False, draft=draft, stop_eos=stop_eos)
    assert heard == [[END]]
    # an end token first: the reply ends there unless end tokens are ignored, and then it decodes on without them
    assert calls == ([] if stop_eos else [("mtp" if draft else "serial", False)])


@pytest.mark.torch
def test_flash_next_hands_stop_eos_to_its_scheduler(modules, monkeypatch):
    _, eng = _flash_next(monkeypatch, [])
    got = []
    eng.scheduler = SimpleNamespace(submit=lambda *a, **kw: got.append(kw) or {})
    for stop_eos in (True, False):
        eng.generate([5, 6, 7], 12, None, lambda new: False, stop_eos=stop_eos)
    assert [kw["stop_eos"] for kw in got] == [True, False]


@pytest.mark.torch
@pytest.mark.parametrize("family", [_flash_next, _nemotron], ids=["flashnext", "nemotron"])
@pytest.mark.parametrize("stop_eos", [True, False])
def test_rank_one_reads_the_field_rank_zero_sends(modules, monkeypatch, family, stop_eos):
    mod, eng = family(monkeypatch, [])
    sent = {}
    eng.comm = SimpleNamespace(store=SimpleNamespace(set=lambda key, text: sent.update(text=text)))
    eng._key = lambda n: f"request/{n}"
    got = eng._share([5, 6], 9, None, True, 0, None, stop_eos)
    unpack = getattr(mod, "_unpack", None) or eng._unpack
    assert got[-1] is stop_eos and unpack(sent["text"])[-1] is stop_eos
    body = json.loads(sent["text"])
    body.pop("stop_eos")                                 # a body from before the field: end tokens count
    assert unpack(json.dumps(body))[-1] is True


@pytest.mark.torch
def test_flash_next_streams_carry_their_own_end_tokens(modules):
    from tensorfold.cuda.streams import Stream

    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.eos = (END,)
    assert dec._ends(Stream([1], 4)) == (END,) and dec._ends(Stream([1], 4, stop_eos=False)) == ()
