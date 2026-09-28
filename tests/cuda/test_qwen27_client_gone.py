"""The server's client-gone stop and per-token failures on the 27B decoders (synthetic weights): a request that
stops early sends a prefix of the exact reply, one that fails raises, and the requests beside and after them are
exact."""

import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_qwen27_multi import PROMPTS, SAMPLINGS, _Oracle, _model, _serial  # noqa: E402

from tensorfold.cuda import server  # noqa: E402
from tensorfold.cuda.scheduler import Scheduler  # noqa: E402
from tensorfold.cuda.streams import PrefixCache  # noqa: E402
from tensorfold.families.qwen3_5.cuda.engine import KEEP_ONE, Qwen27Engine  # noqa: E402
from tensorfold.server.cancellation import RequestCancelled  # noqa: E402

COUNT = 20


class _Tok:
    def decode(self, ids, **kwargs):
        return "".join(chr(0x100 + int(t)) for t in ids)          # one printable character a token


def _app(engine):
    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok = engine, "qwen27-synthetic", _Tok()
    app.default_thinking = False
    app.sampling = {"temperature": 0.0, "top_k": 0, "top_p": 1.0}
    app.max_tokens = COUNT
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    return app


def _engine(w, scheduler=None):
    engine = Qwen27Engine.__new__(Qwen27Engine)
    engine.torch, engine.tp, engine.rank, engine.max_rows, engine.allow_copy = torch, 1, 0, 6, True
    engine.w, engine.draft, engine.eos, engine.cache = w, None, tuple(w.config.eos), PrefixCache(KEEP_ONE)
    engine.points = None
    engine.context_window = 4096
    engine.concurrent, engine.multi, engine.scheduler = scheduler is not None, None, scheduler
    return engine


def _run(app, prompt, sampling, count=COUNT, emit=lambda delta: True, cancelled=None):
    body = ({"temperature": 0.0} if sampling is None else
            {"temperature": sampling.temperature, "seed": sampling.seed, "top_k": sampling.top_k,
             "top_p": sampling.top_p})
    prepared = server.PreparedRequest(list(prompt), count, [], False, app.sampling_for(body, list(prompt)))
    return app.run(body, False, emit, prepared=prepared, cancelled=cancelled)


def _after(n):
    """A ``cancelled`` that turns true at its n-th check: the client leaves mid-reply."""

    checks = []
    return lambda: checks.append(1) or len(checks) >= n


def _keeping(sent):
    """An ``emit`` that keeps what the reply sends; ``_Tok`` gives one character a token."""

    def emit(delta):
        sent.append(delta.get("content", ""))
        return True

    return emit


def _ids(sent):
    return [ord(c) - 0x100 for c in "".join(sent)]


def _failing(n):
    """An ``emit`` that fails at its n-th delta, as a broken per-token step does."""

    deltas = []

    def emit(delta):
        deltas.append(delta)
        if len(deltas) >= n:
            raise RuntimeError("emit failed")
        return True

    return emit


def _full(w, prompt, sampling):
    ref = _serial(w, prompt, sampling, COUNT)
    assert len(ref) == COUNT, "the synthetic model ended this reply early; pick another prompt"
    return ref


def test_one_stream_stops_early_and_stays_exact():
    w = _model()
    engine = _engine(w)
    app = _app(engine)
    prompt, sampling = PROMPTS[1], SAMPLINGS[1]
    ref = _full(w, prompt, sampling)
    sent: list[str] = []
    with pytest.raises(RequestCancelled):
        _run(app, prompt, sampling, emit=_keeping(sent), cancelled=_after(3))
    n = len(_ids(sent))
    assert 1 <= n < COUNT and _ids(sent) == ref[:n]
    # the prompt end kept by the stopped request resumes a longer prompt with a fresh prefill's tokens
    longer = prompt + ref[:5] + [42, 43]
    got = _run(app, longer, sampling, count=12)
    assert got["stats"]["cached"] == len(prompt)
    assert got["stats"]["token_sha"] == server.token_sha(_serial(w, longer, sampling, 12))
    with pytest.raises(RuntimeError, match="emit failed"):
        _run(app, prompt, sampling, emit=_failing(3))
    assert _run(app, prompt, sampling)["stats"]["token_sha"] == server.token_sha(ref)


def test_shared_rounds_keep_the_other_streams_exact():
    w = _model()
    refs = {tuple(p): _full(w, p, s) for p, s in zip(PROMPTS[:4], SAMPLINGS)}
    scheduler = Scheduler(_Oracle(w, refs, seed=3), max_streams=3)
    app = _app(_engine(w, scheduler))
    gate = threading.Barrier(3)
    results: dict[int, object] = {}
    sent: list[str] = []

    def go(i, **kw):
        gate.wait(timeout=120)
        try:
            results[i] = _run(app, PROMPTS[i], SAMPLINGS[i], **kw)
        except Exception as exc:                    # noqa: BLE001
            results[i] = exc

    threads = [threading.Thread(target=go, args=(0,), kwargs={"emit": _keeping(sent), "cancelled": _after(3)}),
               threading.Thread(target=go, args=(1,), kwargs={"emit": _failing(3)}),
               threading.Thread(target=go, args=(2,))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads)
    left, failed, beside = results[0], results[1], results[2]
    n = len(_ids(sent))
    assert isinstance(left, RequestCancelled) and 1 <= n < COUNT and _ids(sent) == refs[tuple(PROMPTS[0])][:n]
    assert isinstance(failed, RuntimeError) and str(failed) == "emit failed"
    assert beside["stats"]["token_sha"] == server.token_sha(refs[tuple(PROMPTS[2])])
    after = _run(app, PROMPTS[3], SAMPLINGS[3])
    assert after["stats"]["token_sha"] == server.token_sha(refs[tuple(PROMPTS[3])])
    assert scheduler.decoder.live() == 0
