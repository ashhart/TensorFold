"""Nemotron resends and changed final prompt tokens reuse a state with fresh-prefix bits."""

import pytest

from nemotron_fakes import tiny_weights
from tensorfold.cuda.draft_depth import Costs, DepthRule
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.nemotron_h.cuda.app import NemotronEngine
from tensorfold.families.nemotron_h.cuda.engine import Engine
from tensorfold.families.nemotron_h.cuda.mtp import MTPHead


@pytest.mark.parametrize("sampling", [None, Sampling(23, 1.0, 20, 0.95)])
def test_nemotron_resends_and_thinking_turns_resume(sampling):
    w = tiny_weights(5)
    app = NemotronEngine.__new__(NemotronEngine)
    app._make = lambda: Engine(w, max_len=1024, graphs=False, prefill_rows=16)
    app.e = app._make()
    app.mtp = MTPHead(app.e)
    app.tp, app.rank, app.drafts, app.confidence = 1, 0, 3, 0.3
    app.rules = {m: DepthRule(Costs(tuple(range(18)), 0.5), 3, floor=0.3) for m in (False, True)}
    app.max_len, app.cache, app.serial, app.eos = 1024, [], None, tuple(w.config.eos)
    prompt = list(range(11, 30))

    def ask(tokens, drafted=True):
        out = []
        stats = app.generate(tokens, 8, sampling, lambda new: out.extend(new) or False, draft=drafted)
        return out, stats

    cold, _ = ask(prompt)
    repeated, stats = ask(prompt)
    assert stats["cached"] == len(prompt) - 1
    assert repeated == cold == ask(prompt, False)[0]
    turn = prompt[:-1] + [271, 77, 78]
    resumed, stats = ask(turn)
    assert stats["cached"] == len(prompt) - 1
    assert resumed == ask(turn, False)[0]


@pytest.mark.parametrize("point", [1, 5, 16, 18])
def test_nemotron_cut_keeps_fresh_prefix_bits_without_an_extra_forward(point, monkeypatch):
    import torch
    from tensorfold.families.nemotron_h.cuda.decode import prefill

    w = tiny_weights(5)
    eng = Engine(w, max_len=512, graphs=False, prefill_rows=16)
    head = MTPHead(eng)
    prompt = list(range(11, 30))
    full = prefill(eng, head, prompt, None)
    calls = []
    forward = eng.prefill_chunk

    def count(tokens, **kwargs):
        calls.append(len(tokens))
        return forward(tokens, **kwargs)

    monkeypatch.setattr(eng, "prefill_chunk", count)
    cut = prefill(eng, head, prompt, None, keep_at=point)
    assert calls == [16, 3]
    assert cut.pending == full.pending
    assert torch.equal(cut.last_hidden, full.last_hidden)
    for name in eng.STATE:
        assert torch.equal(cut.engine[name], full.engine[name]), name
    fresh = prefill(eng, head, prompt[:point], None)
    for name in ("ssm", "conv_base"):
        assert torch.equal(cut.kept["engine"][name], fresh.engine[name]), name
    assert cut.kept["engine"]["host"] == fresh.engine["host"]
    assert torch.equal(cut.kept["tail"], fresh.last_hidden)
    assert cut.kept["mtp"]["pos"] == fresh.mtp["pos"]
    for name in ("k", "v"):
        assert torch.equal(cut.kept["mtp"][name][:point - 1], fresh.mtp[name][:point - 1]), name


@pytest.mark.parametrize("sampling", [None, Sampling(23, 1.0, 20, 0.95)])
def test_three_resends_preserve_every_kept_nemotron_state(sampling):
    import torch
    from prefix_checks import same_tokens
    from tensorfold.families.nemotron_h.cuda.decode import prefill, serial_decode

    w = tiny_weights(5)
    app = NemotronEngine.__new__(NemotronEngine)
    app._make = lambda: Engine(w, max_len=1024, graphs=False, prefill_rows=16)
    app.e, app.mtp = app._make(), None
    app.mtp = MTPHead(app.e)
    app.tp, app.rank, app.drafts, app.confidence = 1, 0, 3, 0.3
    app.rules = {m: DepthRule(Costs(tuple(range(18)), 0.5), 3, floor=0.3) for m in (False, True)}
    app.max_len, app.cache, app.serial, app.eos = 1024, [], None, ()
    ref = app._make()
    ref_head = MTPHead(ref)
    system = list(range(11, 20))
    prompt = system + list(range(30, 50))
    turn = prompt[:-1] + [271, 77, 78]
    different = system + [301, 302, 303, 304]
    for step, tokens in enumerate((prompt, prompt, prompt, turn, different)):
        actual = []
        stats = app.generate(tokens, 8, sampling, lambda new: actual.extend(new) or False, stop_eos=False)
        if step in (1, 2, 3):
            assert stats["cached"] == len(prompt) - 1
        if step == 4:
            assert stats["cached"] == 0
        full = prefill(ref, ref_head, tokens, sampling)
        same_tokens(actual, serial_decode(ref, full, 8, sampling).tokens)
        for ids, kept in app.cache:
            fresh = prefill(ref, ref_head, ids, sampling)
            for name in ("ssm", "conv_base"):
                assert torch.equal(kept["engine"][name], fresh.engine[name]), (step, name)
            assert kept["engine"]["host"] == fresh.engine["host"]
            assert torch.equal(kept["tail"], fresh.last_hidden)
            pos = kept["mtp"]["pos"]
            assert pos == fresh.mtp["pos"] == len(ids) - 1
            for name in ("k", "v"):
                assert torch.equal(kept["mtp"][name][:pos], fresh.mtp[name][:pos]), (step, name)
