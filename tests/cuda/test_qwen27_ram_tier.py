"""The host-RAM tier (``--ram-tier-gib``) on a GPU: prompt states the GPU cache lets go come back bit for bit.

On the toy model of ``test_qwen27_prompt_end_cache`` (the 27B's layer pattern), with pinned and pageable chunks:

- an entry spilled to host memory and taken back holds the bytes it left with (attention rows below ``pos`` in
  buffers of the old row count, every DeltaNet state), though a prefill overwrote the GPU buffer it shared meanwhile;
- an entry spilled after its prefix copies only its own rows, and comes back into given buffers bit for bit (rows
  past ``pos`` untouched), a prefill continuing there as a fresh one;
- a drafter snapshot comes back with the same bits and drafts the same block;
- a one-entry GPU cache spills a conversation when another arrives; its next turn restores it and ends in a fresh
  prefill's state, logits and first token, and its reply is a fresh engine's and the serial reference's;
- a conversation dropped under a shared system prompt (the next one's prefill writes into the buffer they share)
  comes back the same way;
- the concurrent decoder admits a returning conversation from the tier and decodes the serial reference's tokens.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_qwen27_dflash2_blocks import H, _drafter  # noqa: E402
from test_qwen27_prompt_end_cache import (END_THINK, NL, NL2, THINK, _assert_same_state, _engine,  # noqa: E402
                                          _generate, _model, _prefill, _prompt, _same_bits, _turns)

from tensorfold.cuda.host_tier import HostTier  # noqa: E402
from tensorfold.cuda.sampling import sample_rows  # noqa: E402
from tensorfold.cuda.streams import PrefixCache, Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5.cuda import decode  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import clone_state, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen3_5.cuda.engine import KEEP_ONE  # noqa: E402
from tensorfold.families.qwen3_5.cuda.multi import MultiDecoder  # noqa: E402


@pytest.fixture(scope="module")
def w():
    return _model()


def _tier(pin=None) -> HostTier:
    return HostTier(1 << 30, "cuda", chunk=1 << 20, pin=pin)


def _tiered(w, keep: int, *, pin=None, points=None):
    """The one-stream engine with a GPU cache of ``keep`` entries over a host tier."""

    engine = _engine(w)
    engine.tier = _tier(pin)
    engine.cache = PrefixCache(keep, on_evict=engine.tier.put)
    engine.points = points
    return engine


def _rows(st) -> list[int]:
    return [kv[0].shape[0] for kv in st.kv if kv is not None]


@pytest.mark.parametrize("pin", [True, False])
def test_a_spilled_entry_comes_back_bit_for_bit_after_its_buffer_is_overwritten(w, pin):
    """The entry at 200 shares its key/value buffers with the state kept at 100; a prefill resumed from 100 (after the
    fence a resume runs) rewrites rows 100-199 there, and the entry still comes back as a fresh prefill of 200."""

    prompt = _prompt(300, seed=81)
    at_100 = []
    _, _, (kept, _) = prefill(w, prompt, None, stops=[100], keep=lambda p, st, snap: at_100.append(st), keep_at=200)
    (short,) = at_100
    assert all(a is None or a[0].data_ptr() == b[0].data_ptr() for a, b in zip(short.kv, kept.kv))
    ref = prefill(w, prompt[:200], None)[0]
    rows = _rows(kept)
    tier = _tier(pin)
    assert tier.put(prompt[:200], kept, None)
    tier.fence()
    prefill(w, prompt[:100] + _prompt(150, seed=82), None, state=short)
    assert not all(_same_bits(a[0][:200], b[0][:200]) for a, b in zip(kept.kv, ref.kv) if a is not None)
    ids, back, snap = tier.take(prompt[:201])
    assert ids == prompt[:200] and snap is None and not tier.entries
    _assert_same_state(back, ref)
    assert back.limit == kept.limit and _rows(back) == rows
    assert all(a is None or a[0].data_ptr() != b[0].data_ptr() for a, b in zip(back.kv, kept.kv))
    assert all(a is None or a.data_ptr() != b.data_ptr() for a, b in zip(back.rec + back.conv, kept.rec + kept.conv))


@pytest.mark.parametrize("pin", [True, False])
def test_a_restore_into_given_buffers_fills_their_rows_below_pos_bit_for_bit(w, pin):
    """The state at 100 spilled, then the entry at 200 sharing its buffers (only rows 100-199 copied); after a prefill
    overwrote those buffers, the entry comes back into buffers of 256 rows: rows below 200 a fresh prefill's, the rest
    untouched, the state holding those very buffers, and a prefill resumed there ends as a fresh one."""

    prompt = _prompt(300, seed=84)
    at_100 = []
    _, _, (kept, _) = prefill(w, prompt, None, stops=[100], keep=lambda p, st, snap: at_100.append(st), keep_at=200)
    (short,) = at_100
    tier = _tier(pin)
    assert tier.put(prompt[:100], short, None)
    before = tier.to_host
    assert tier.put(prompt[:200], kept, None)
    row = sum(t[0].numel() * t.element_size() for kv in kept.kv if kv is not None for t in kv)
    own = sum(t.numel() * t.element_size() for t in kept.conv + kept.rec if t is not None)
    assert tier.to_host - before == own + 100 * row
    tier.fence()
    prefill(w, prompt[:100] + _prompt(150, seed=85), None, state=short)
    into = [None if kv is None else tuple(torch.full((256, *t.shape[1:]), 7.0, dtype=t.dtype, device="cuda")
                                          for t in kv) for kv in kept.kv]
    tail = [None if kv is None else tuple(t[200:].clone() for t in kv) for kv in into]
    ids, back, _ = tier.take(prompt[:201], into=into)
    assert ids == prompt[:200] and back.limit == kept.limit and all(a is b for a, b in zip(back.kv, into))
    _assert_same_state(back, prefill(w, prompt[:200], None)[0])
    assert all(kv is None or all(_same_bits(t[200:], u) for t, u in zip(kv, rest)) for kv, rest in zip(back.kv, tail))
    _assert_same_state(prefill(w, prompt[:250], None, state=back)[0], prefill(w, prompt[:250], None)[0])
    _, back, _ = tier.take(prompt[:101])
    _assert_same_state(back, prefill(w, prompt[:100], None)[0])


def test_a_drafter_snapshot_comes_back_and_drafts_the_same_block(tmp_path, w):
    d = _drafter(tmp_path)
    gen = torch.Generator(device="cuda").manual_seed(8)
    d.restore(([None] * d.layers, [None] * d.layers, 0, 0))
    d.add_taps((torch.randn(90, 5 * H, generator=gen, device="cuda") * 0.5).bfloat16())     # past the 63-row window
    snap = d.snapshot()
    prompt = _prompt(50, seed=83)
    tier = _tier()
    assert tier.put(prompt, prefill(w, prompt, None)[0], snap)
    _, back, back_snap = tier.take(prompt + [1])
    _assert_same_state(back, prefill(w, prompt, None)[0])
    assert back_snap[2:] == snap[2:] and isinstance(back_snap[0], list)
    assert all(_same_bits(a, b) for a, b in zip(back_snap[0] + back_snap[1], snap[0] + snap[1]))
    sampling = Sampling(1234, 1.0, 20, 0.95)
    blocks = []
    for which in (snap, back_snap):
        d.restore(which)
        block = d.launch_block(7, 11)
        blocks.append(([t[block[1]:block[1] + block[2]].clone() for t in block[0][:2]],
                       d.finish_tree(block, 50, 11, sampling)))
    (tensors_a, tree_a), (tensors_b, tree_b) = blocks
    assert all(torch.equal(a, b) for a, b in zip(tensors_a, tensors_b)) and tree_a == tree_b


@pytest.mark.parametrize("sampling", [None, Sampling(4321)])
def test_a_conversation_evicted_to_the_tier_resumes_its_next_turn_exactly(monkeypatch, w, sampling):
    """One GPU entry: another conversation's prompt end evicts the first's to the tier; the first's next turn restores
    it (``cached``), its prefill ends in a fresh prefill's state, logits and first token, and its reply is a fresh
    engine's and the serial reference's."""

    first, second = _turns(126, seed=90)
    other = _prompt(90, seed=91)
    engine = _tiered(w, keep=1)
    _generate(engine, first, sampling)
    _generate(engine, other, sampling)
    assert [e.ids for e in engine.tier.entries] == [first[:-1]]
    captured, sampled = [], []
    real_prefill = decode.prefill

    def spy_prefill(*args, **kwargs):
        out = real_prefill(*args, **kwargs)
        captured.append((clone_state(out[0]), *out[1:]))     # the decode then commits into out[0]
        return out

    def spy_sample(logits, positions, samp):
        sampled.append((logits.clone(), list(positions)))
        return sample_rows(logits, positions, samp)

    monkeypatch.setattr(decode, "prefill", spy_prefill)
    monkeypatch.setattr(decode, "sample_rows", spy_sample)
    reply, stats = _generate(engine, second, sampling)
    monkeypatch.setattr(decode, "prefill", real_prefill)
    monkeypatch.setattr(decode, "sample_rows", sample_rows)
    assert stats["cached"] == len(first) - 1
    assert [e.ids for e in engine.tier.entries] == [other[:-1], first[:-1]]   # the second's prompt end evicted it again
    ((st, pending, _),) = captured
    logits, positions = sampled[0]
    assert positions == [len(second)]
    (fresh, fresh_pending), fresh_logits = _prefill(monkeypatch, w, second, sampling)
    assert _same_bits(logits, fresh_logits) and pending == fresh_pending == reply[0]
    _assert_same_state(st, fresh)
    assert reply == _generate(_engine(w), second, sampling)[0] == _generate(_engine(w), second, sampling,
                                                                           draft=False)[0]
    for ids, entry, _ in engine.cache.entries:
        _assert_same_state(entry, _prefill(monkeypatch, w, ids)[0][0])
    for ids in [e.ids for e in engine.tier.entries]:
        _assert_same_state(engine.tier.take(ids + [0])[1], _prefill(monkeypatch, w, ids)[0][0])


@pytest.mark.parametrize("pin", [True, False])
def test_a_conversation_dropped_under_a_shared_system_prompt_comes_back_exactly(monkeypatch, w, pin):
    """States kept at the system prompt's end: the second conversation resumes there and writes into the buffer the
    first's prompt end shares, so that entry is dropped to the tier first; the first's next turn resumes from it."""

    system = _prompt(300, seed=92)
    head = system + _prompt(300, seed=93)
    first = head + [THINK, NL]
    second = head + [THINK, NL2, END_THINK, NL2] + _prompt(7, seed=94) + [THINK, NL]
    other = system + _prompt(300, seed=95) + [THINK, NL]
    points = lambda prompt: [len(system)] if len(prompt) > len(system) else []      # noqa: E731
    engine = _tiered(w, keep=KEEP_ONE, pin=pin, points=points)
    _generate(engine, first, None)
    assert [ids for ids, _, _ in engine.cache.entries] == [system, first[:-1]]
    _, stats = _generate(engine, other, None)
    assert stats["cached"] == len(system) and [e.ids for e in engine.tier.entries] == [first[:-1]]
    reply, stats = _generate(engine, second, None)
    assert stats["cached"] == len(first) - 1 and not engine.tier.entries
    assert reply == _generate(_engine(w), second, None)[0] == _generate(_engine(w), second, None, draft=False)[0]
    for ids, entry, _ in engine.cache.entries:
        _assert_same_state(entry, _prefill(monkeypatch, w, ids)[0][0])


@pytest.mark.parametrize("sampling", [None, Sampling(77)])
def test_the_concurrent_decoder_resumes_a_returning_conversation_from_the_tier(w, sampling):
    tier = _tier()
    dec = MultiDecoder(w, None, keep=1, context=1 << 14, tier=tier)
    first, second = _turns(140, seed=96)
    out = {}
    for prompt in (first, _prompt(80, seed=97), second):
        got: list[int] = []
        s = Stream(list(prompt), 16, sampling, emit=lambda new, got=got: got.extend(new))
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        out[tuple(prompt)] = (got, s.cached)
    st, first_token = prefill(w, second, sampling)
    want = serial_decode(w, st, first_token, 16, sampling).tokens
    assert out[tuple(second)] == (want, len(first) - 1)
    assert [ids for ids, _, _ in dec.cache.entries] == [second[:-1]]
    for ids, entry, _ in dec.cache.entries:
        _assert_same_state(entry, prefill(w, ids, None)[0])
    for ids in [e.ids for e in tier.entries]:
        _, back, _ = tier.take(ids + [0])
        _assert_same_state(back, prefill(w, ids, None)[0])
