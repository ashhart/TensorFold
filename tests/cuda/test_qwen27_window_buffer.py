"""One stream over the host-RAM tier: every state's keys and values live in one window-sized buffer, and a
conversation that is not the GPU's own comes back from host RAM into it.

On the toy model of ``test_qwen27_prompt_end_cache`` (the 27B's layer pattern):

- conversations served round-robin resume each later turn from host RAM (``cached``), end their prefills in a fresh
  prefill's state, logits and first token, and reply as a fresh engine and the serial reference do;
- every state the GPU cache holds, and every one a prefill kept, is on the window buffer: no attention buffer of its
  own, whatever the order of hits, misses, restores and ``"draft": false`` requests;
- a ``"draft": false`` request takes the buffer too, and the conversation it displaced comes back exactly;
- a conversation sharing a system prompt with the GPU's own resumes from the longest match, on the GPU or in RAM.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_qwen27_prompt_end_cache import (END_THINK, NL, NL2, THINK, _assert_same_state, _engine,  # noqa: E402
                                          _generate, _model, _prefill, _prompt, _same_bits)

from tensorfold.cuda.host_tier import HostTier  # noqa: E402
from tensorfold.cuda.sampling import sample_rows  # noqa: E402
from tensorfold.cuda.streams import PrefixCache  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5.cuda import decode  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import clone_state  # noqa: E402
from tensorfold.families.qwen3_5.cuda.engine import KEEP_ONE  # noqa: E402


@pytest.fixture(scope="module")
def w():
    return _model()


def _windowed(w, keep: int = KEEP_ONE, *, points=None, pin=None):
    """The one-stream engine over a host tier with the window buffer, as ``--ram-tier-gib`` builds it."""

    engine = _engine(w)
    engine.tier = HostTier(1 << 30, "cuda", chunk=1 << 20, pin=pin)
    engine.cache = PrefixCache(keep, on_evict=engine.tier.put)
    engine.points = points
    c, rows = w.config, engine.context_window + engine.max_rows
    engine.window = [None if layer.linear else
                     tuple(torch.empty((rows, c.kv_heads, c.head_dim), dtype=torch.bfloat16, device="cuda")
                           for _ in range(2)) for layer in w.layers]
    return engine


def _on_window(engine, st) -> bool:
    return all(kv is None or (kv[0] is engine.window[i][0] and kv[1] is engine.window[i][1])
               for i, kv in enumerate(st.kv))


def _next(prompt: list[int], seed: int) -> list[int]:
    """The next turn's prompt: this one to its last token, an empty reasoning block, a user message, a new
    generation prompt (it extends ``prompt[:-1]``, the prompt-end entry)."""

    return prompt[:-1] + [NL2, END_THINK, NL2] + _prompt(7, seed) + [THINK, NL]


def _conversations(count: int, turns: int, base: int = 120, seed: int = 200) -> list[list[list[int]]]:
    out = []
    for c in range(count):
        prompts = [_prompt(base + 3 * c, seed + 17 * c) + [THINK, NL]]
        for t in range(1, turns):
            prompts.append(_next(prompts[-1], seed + 17 * c + t))
        out.append(prompts)
    return out


def _spied(monkeypatch, engine, prompt, sampling, **kwargs):
    """``_generate`` with the state the prefill ended in and the logits it sampled the first token from."""

    captured, sampled = [], []
    real_prefill = decode.prefill

    def spy_prefill(*args, **kw):
        out = real_prefill(*args, **kw)
        captured.append(clone_state(out[0]))            # the decode then commits into out[0]
        return out

    def spy_sample(logits, positions, samp):
        sampled.append(logits.clone())
        return sample_rows(logits, positions, samp)

    monkeypatch.setattr(decode, "prefill", spy_prefill)
    monkeypatch.setattr(decode, "sample_rows", spy_sample)
    try:
        reply, stats = _generate(engine, prompt, sampling, **kwargs)
    finally:
        monkeypatch.setattr(decode, "prefill", real_prefill)
        monkeypatch.setattr(decode, "sample_rows", sample_rows)
    return reply, stats, captured[0], sampled[0]


@pytest.mark.parametrize("sampling", [None, Sampling(4321)])
def test_round_robin_conversations_resume_from_ram_into_the_window_buffer(monkeypatch, w, sampling):
    engine = _windowed(w)
    convs = _conversations(3, 3)
    for t in range(3):
        for c, prompts in enumerate(convs):
            prompt = prompts[t]
            reply, stats, st, logits = _spied(monkeypatch, engine, prompt, sampling)
            assert stats["cached"] == (len(prompts[t - 1]) - 1 if t else 0), (c, t)
            assert _on_window(engine, st)
            (fresh, fresh_pending), fresh_logits = _prefill(monkeypatch, w, prompt, sampling)
            _assert_same_state(st, fresh)
            assert _same_bits(logits, fresh_logits) and reply[0] == fresh_pending
            assert reply == _generate(_engine(w), prompt, sampling)[0] == \
                _generate(_engine(w), prompt, sampling, draft=False)[0]
            assert engine.cache.entries and all(_on_window(engine, e[1]) for e in engine.cache.entries)
            assert all(e[0] == prompt[:len(e[0])] for e in engine.cache.entries)     # the GPU holds one path
    ids = [e.ids for e in engine.tier.entries]
    assert all(prompts[2][:-1] in ids for prompts in convs[:2])
    for i in ids:                                        # what the tier holds is what a fresh prefill leaves
        _, back, _ = engine.tier.take(i + [0])
        _assert_same_state(back, _prefill(monkeypatch, w, i)[0][0])


def test_a_serial_request_takes_the_window_and_the_conversation_comes_back(monkeypatch, w):
    engine = _windowed(w)
    (first, second), other = _conversations(1, 2)[0], _prompt(90, seed=301) + [THINK, NL]
    _generate(engine, first, None)
    assert _generate(engine, other, None, draft=False)[0] == _generate(_engine(w), other, None, draft=False)[0]
    assert not engine.cache.entries and [e.ids for e in engine.tier.entries] == [first[:-1]]
    reply, stats, st, _ = _spied(monkeypatch, engine, second, None)
    assert stats["cached"] == len(first) - 1 and _on_window(engine, st)
    _assert_same_state(st, _prefill(monkeypatch, w, second)[0][0])
    assert reply == _generate(_engine(w), second, None, draft=False)[0]


@pytest.mark.parametrize("pin", [True, False])
def test_conversations_under_one_system_prompt_resume_the_longest_match(monkeypatch, w, pin):
    """States kept at the system prompt's end: a second conversation resumes from the GPU's (the first one's prompt
    end goes to RAM first); the first's next turn comes back from RAM, longer than the GPU's system prompt state."""

    system = _prompt(300, seed=310)
    points = lambda prompt: [len(system)] if len(prompt) > len(system) else []      # noqa: E731
    engine = _windowed(w, points=points, pin=pin)
    first = system + _prompt(300, seed=311) + [THINK, NL]
    other = system + _prompt(300, seed=312) + [THINK, NL]
    _generate(engine, first, None)
    assert [ids for ids, _, _ in engine.cache.entries] == [system, first[:-1]]
    _, stats, st, _ = _spied(monkeypatch, engine, other, None)
    assert stats["cached"] == len(system) and _on_window(engine, st)
    assert [e.ids for e in engine.tier.entries] == [first[:-1]]
    second = _next(first, 313)
    reply, stats, st, _ = _spied(monkeypatch, engine, second, None)
    assert stats["cached"] == len(first) - 1 and _on_window(engine, st)
    _assert_same_state(st, _prefill(monkeypatch, w, second)[0][0])
    assert reply == _generate(_engine(w), second, None, draft=False)[0]
    for ids, entry, _ in engine.cache.entries:
        assert _on_window(engine, entry)
        _assert_same_state(entry, _prefill(monkeypatch, w, ids)[0][0])
