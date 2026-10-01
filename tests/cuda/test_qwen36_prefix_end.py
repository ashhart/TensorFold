"""Identical resends and a changed final prompt token resume the retained prefix exactly."""

import pytest
import torch

from test_qwen36_moe import _model, _serial
from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen3_5_moe.cuda import multi
from tensorfold.families.qwen3_5_moe.cuda import decode


@pytest.mark.parametrize("sampling", [None, Sampling(23, 1.0, 20, 0.95)])
def test_resend_and_changed_last_prompt_token_resume(monkeypatch, sampling):
    monkeypatch.setattr(multi, "STEP", 3)
    w, head = _model()
    decoder = multi.MultiDecoder(w, head, depth=3, confidence=0.3, keep=8)
    prompt = [9, 10, 11, 12, 13, 14, 15, 16, 17]

    def run(tokens):
        stream = Stream(tokens, 8, sampling, background=True)
        decoder.admit(stream)
        while decoder.live():
            decoder.finish(decoder.round())
        return stream

    cold = run(prompt)
    same = run(prompt)
    assert same.cached == len(prompt) - 1
    assert same.out == cold.out == _serial(w, prompt, sampling, 8)
    turn = prompt[:-1] + [91, 92, 93]
    resumed = run(turn)
    assert resumed.cached == len(prompt) - 1
    assert resumed.out == _serial(w, turn, sampling, 8)


@pytest.mark.parametrize("point", [1, 5, 22, 23])
def test_cut_snapshot_is_a_fresh_prefix_without_an_extra_forward(monkeypatch, point):
    w, head = _model()
    prompt, saved, calls = list(range(20, 43)), {}, []
    real = decode.prefill_chunk

    def counted(*args, **kwargs):
        calls.append(int(args[1].shape[0]))
        return real(*args, **kwargs)

    monkeypatch.setattr(decode, "prefill_chunk", counted)
    _, _, first, carry = decode.prefill(w, head, prompt, None, keep_at=point,
        keep=lambda p, st, mc, held: saved.update(state=st, cache=mc, held=held))
    assert calls == [b - a for a, b in decode.chunks(0, len(prompt))]
    st, mc, _, short = decode.prefill(w, head, prompt[:point], None)
    assert saved["state"].pos == point and saved["cache"].pos == mc.pos == point - 1
    for name in ("rec", "conv"):
        assert all(torch.equal(a, b) for a, b in zip(getattr(saved["state"], name), getattr(st, name))
                   if a is not None)
    assert torch.equal(saved["held"], short.states)
    assert torch.equal(saved["cache"].k[:mc.pos], mc.k[:mc.pos])
    assert torch.equal(saved["cache"].v[:mc.pos], mc.v[:mc.pos])
    _, _, fresh_first, fresh_carry = decode.prefill(w, head, prompt, None)
    assert first == fresh_first and torch.equal(carry.states, fresh_carry.states)


@pytest.mark.parametrize("sampling", [None, Sampling(23, 1.0, 20, 0.95)])
@pytest.mark.parametrize("concurrent", [False, True])
def test_three_resends_keep_fresh_state_after_each_decode(monkeypatch, sampling, concurrent):
    from prefix_checks import same_tokens
    from tensorfold.cuda.streams import PrefixCache
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    w, head = _model()
    system = [1 + i % 63 for i in range(256)]
    prompt = system + [80 + i % 80 for i in range(300)]
    turn = prompt[:-1] + [191, 192, 193]
    different = system + [201] * 300
    if concurrent:
        monkeypatch.setattr(multi, "STEP", 127)
        engine = multi.MultiDecoder(w, head, depth=3, confidence=0.3, keep=8, points=lambda _: [len(system)])
    else:
        engine = Qwen36Engine.__new__(Qwen36Engine)
        engine.w, engine.head, engine.graphs, engine.scheduler = w, head, None, None
        engine.context_window, engine.depth, engine.confidence = 1024, 3, 0.3
        engine.points, engine.cache, engine.eos = lambda _: [len(system)], PrefixCache(8), ()

    for step, tokens in enumerate((prompt, prompt, prompt, turn, different)):
        if concurrent:
            stream = Stream(tokens, 8, sampling, background=True, stop_eos=False)
            engine.admit(stream)
            while engine.live():
                engine.finish(engine.round())
            actual, cached = stream.out, stream.cached
        else:
            actual = []
            stats = engine.generate(tokens, 8, sampling, lambda new: actual.extend(new) or False, stop_eos=False)
            cached = stats["cached"]
        if step in (1, 2, 3):
            assert cached == len(prompt) - 1
        if step == 4:
            assert cached == len(system)
        from tensorfold.families.qwen3_5.cuda.decode import draft_decode, prefill as serial_prefill

        state, first = serial_prefill(w, tokens, sampling)
        fresh_tokens = draft_decode(w, state, tokens, first, 8, sampling, None,
                                    allow_copy=False, stop_eos=False).tokens
        same_tokens(actual, fresh_tokens)
        for ids, kept, (cache, held) in engine.cache.entries:
            fresh, mc, _, carry = decode.prefill(w, head, ids, sampling)
            assert kept.pos == len(ids)
            for name in ("rec", "conv"):
                assert all(torch.equal(a, b) for a, b in zip(getattr(kept, name), getattr(fresh, name))
                           if a is not None), (step, name)
            assert cache.pos == mc.pos == len(ids) - 1
            assert torch.equal(cache.k[:mc.pos], mc.k[:mc.pos]), step
            assert torch.equal(cache.v[:mc.pos], mc.v[:mc.pos]), step
            assert torch.equal(held, carry.states), step
