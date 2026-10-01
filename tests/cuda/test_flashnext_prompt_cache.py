"""Flash Next message-start snapshots and fork lanes preserve fresh-prefill and serial bits."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import numpy as np  # noqa: E402

from test_flashnext_forward import V, _model, _rows  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import (  # noqa: E402
    Engine, entry_end, prefill, serial_decode)
from tensorfold.families.qwen4_exp.cuda.engine import KEEP_SERIAL, FlashNextEngine  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import State  # noqa: E402

NL, NL2, THINK, END_THINK = 198, 271, 300, 301     # stand-ins for Qwen's "\n", "\n\n", the think tags
CHUNK = 16                                          # the toy engines' prompt chunk


def _prompt(n: int, seed: int = 5) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (n,), generator=gen).tolist()


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))


def _same_snap(a: dict, b: dict) -> bool:
    return (a["pos"] == b["pos"] and a["mtp_len"] == b["mtp_len"] and _same_bits(a["rec"], b["rec"])
            and _same_bits(a["conv"], b["conv"]) and _same_bits(a["ple_tail"], b["ple_tail"])
            and (a["ple_history"] is None) == (b["ple_history"] is None)
            and (a["ple_history"] is None or np.array_equal(a["ple_history"], b["ple_history"])))


def _assert_same_state(a: State, b: State) -> None:
    assert a.pos == b.pos and a.mtp_len == b.mtp_len
    assert _same_snap(a.snapshot(), b.snapshot())
    for la, lb in zip(a.kc, b.kc):
        assert _same_bits(la.k[:a.pos], lb.k[:b.pos]) and _same_bits(la.v[:a.pos], lb.v[:b.pos])
    for ia, ib in zip(a.ikc, b.ikc):
        assert _same_bits(ia[:a.pos], ib[:b.pos])
    for ka, kb in zip(_rows(a.mtp_kc, a.mtp_len), _rows(b.mtp_kc, b.mtp_len)):
        assert _same_bits(ka, kb)
    assert _same_bits(a.mtp_ikc[:a.mtp_len], b.mtp_ikc[:b.mtp_len])


@pytest.fixture(scope="module")
def w():
    return _model()


def _engine(w, sampling) -> tuple[Engine, list[dict]]:
    """Build a toy-width engine over one state."""

    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    return e, []


def _run(e: Engine, prompt, sampling, *, stops=(), keep=None, resume=None):
    return prefill(e, prompt, sampling, stops=stops, keep=keep, resume=resume)


@pytest.mark.parametrize("n", [8, 17, 33, 64, 130])
@pytest.mark.parametrize("cuts", [[], [7], [3, 9], [CHUNK - 1, CHUNK, CHUNK + 1], [1], [64]])
def test_stops_leave_the_prefill_and_the_kept_states_fresh(w, n, cuts):
    """Arbitrary stop points preserve the final state and each saved prefix bit for bit."""

    sampling = Sampling(seed=31, top_k=20, top_p=0.95)
    prompt = _prompt(n, seed=n)
    base = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    first_ref = _run(base, prompt, sampling)
    ref = base.st.clone()
    stops = sorted(p for p in cuts if 0 < p < n)
    keeps: dict[int, tuple[dict, torch.Tensor | None]] = {}

    def keep(p, snap, tail):
        keeps[p] = (snap, tail)

    mine = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    first = _run(mine, prompt, sampling, stops=stops, keep=keep)
    assert first == first_ref
    _assert_same_state(mine.st, ref)
    assert set(keeps) == set(stops)
    for p, (snap, tail) in keeps.items():
        fresh = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
        _run(fresh, prompt[:p], sampling)
        assert _same_snap(snap, fresh.st.snapshot()), p
        assert tail is not None and _same_bits(tail, fresh.last_streams), p


@pytest.mark.parametrize("sampling", [None, Sampling(seed=99, top_k=20, top_p=0.95)])
@pytest.mark.parametrize("base", [16, 40, 130])
def test_a_turn_sent_back_without_its_reasoning_resumes_from_one_token_early(w, sampling, base):
    """A rendered follow-up resumes from the prompt entry and matches fresh prefill and decode."""

    head = _prompt(base, seed=base)
    first = head + [THINK, NL]
    second = head + [THINK, NL2, END_THINK, NL2] + _prompt(7, seed=3) + [THINK, NL]
    assert second[:len(first) - 1] == first[:-1] and second[len(first) - 1] != first[-1]
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    _run(e, first, sampling)
    keeps: list[tuple[int, dict, torch.Tensor | None]] = []

    def keep(p, snap, tail):
        keeps.append((p, snap, tail))

    _run(e, first, sampling, stops=[entry_end(first)], keep=keep)
    ((p, snap, tail),) = keeps
    assert p == len(first) - 1
    ref = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    _run(ref, first[:-1], sampling)
    assert _same_snap(snap, ref.st.snapshot()) and _same_bits(tail, ref.last_streams)
    got = _run(e, second, sampling, resume={"state": snap, "tail": tail})      # restore keeps e's rows below pos
    fresh = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    want = _run(fresh, second, sampling)
    assert got == want
    _assert_same_state(e.st, fresh.st)
    assert serial_decode(e, got, 12, sampling, stop_eos=False).tokens == \
           serial_decode(fresh, want, 12, sampling, stop_eos=False).tokens


@pytest.mark.parametrize("fork", [64, 129])
@pytest.mark.parametrize("gap", [1, 2, 17])
def test_a_fork_at_an_earlier_message_resumes_from_a_mid_prompt_keep(w, fork, gap):
    """An earlier divergence resumes its shared message prefix and leaves a fresh state."""

    sampling = Sampling(seed=17, top_k=20, top_p=0.95)
    head = _prompt(fork, seed=fork)
    one = head + _prompt(9, seed=1) + [NL]
    two = head + _prompt(gap, seed=2) + [NL]                      # parts ways right after the shared block
    assert one[:fork] == two[:fork]
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    keeps: dict[int, tuple[dict, torch.Tensor | None]] = {}

    def keep(p, snap, tail):
        keeps[p] = (snap, tail)

    _run(e, one, sampling, stops=[fork], keep=keep)
    snap, tail = keeps[fork]
    got = _run(e, two, sampling, resume={"state": snap, "tail": tail})       # restore keeps e's rows below pos
    fresh = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    want = _run(fresh, two, sampling)
    assert got == want
    _assert_same_state(e.st, fresh.st)


def _engine_shim(w, points=None) -> FlashNextEngine:
    """Build a serial engine over toy weights without loading a checkpoint."""

    engine = object.__new__(FlashNextEngine)
    engine.tp, engine.rank, engine.depth, engine.confidence = 1, 0, 1, 0.3
    engine.max_len, engine.eos = 1024, tuple(w.cfg.eos)
    engine.concurrent, engine.multi, engine.scheduler = False, None, None
    engine.e = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    engine.points = points
    engine.cache = []
    engine.serial = None
    return engine


def _generate(engine, prompt, sampling, max_tokens=12, **kwargs) -> tuple[list[int], dict]:
    out: list[int] = []

    def on_tokens(new):
        out.extend(new)
        return False

    stats = engine.generate(list(prompt), max_tokens, sampling, on_tokens, **kwargs)
    return out, stats


@pytest.mark.parametrize("sampling", [None, Sampling(seed=4321, top_k=20, top_p=0.95)])
@pytest.mark.parametrize("base", [40, 130])
def test_the_engine_resumes_the_next_turn_exactly(w, sampling, base):
    """The engine keeps a valid prefix chain and matches a fresh serial reply."""

    head = _prompt(base, seed=base)
    first = head + [THINK, NL]
    second = head + [THINK, NL2, END_THINK, NL2] + _prompt(7, seed=6) + [THINK, NL]
    engine = _engine_shim(w)
    reply, stats = _generate(engine, first, sampling)
    assert [ids for ids, _ in engine.cache] == [first[:-1]]
    reply, stats = _generate(engine, second, sampling)
    assert stats["cached"] == len(first) - 1
    assert reply == _generate(_engine_shim(w), second, sampling)[0]
    assert reply == _generate(_engine_shim(w), second, sampling, draft=False)[0]
    assert [ids for ids, _ in engine.cache] == [first[:-1], second[:-1]]
    assert len(engine.cache) <= KEEP_SERIAL
    fresh = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    _run(fresh, second[:-1], sampling)
    for ids, snap in engine.cache:
        mine = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
        _run(mine, ids, sampling)
        assert _same_snap(snap["state"], mine.st.snapshot()) and _same_bits(snap["tail"], mine.last_streams)


@pytest.mark.parametrize("points", [None, lambda ids: [len(ids) // 2]])
def test_the_concurrent_decoder_resumes_forks_and_next_turns(w, points):
    """Forks preserve longer chains in spare lanes and resume earlier entries when no lane is spare."""

    sampling = Sampling(seed=77, top_k=20, top_p=0.95)
    shared = _prompt(600, seed=9)                                  # past MIN_GAP, so the points' keep survives
    one = shared + [NL] + _prompt(9, seed=1)
    nxt = one + [NL2] + _prompt(5, seed=3)                         # the next turn: extends the first's prompt
    fork = shared + [NL] + _prompt(9, seed=2)                      # parts ways at the shared block's end
    other = shared + [NL] + _prompt(9, seed=4)                    # and another one, when every lane is held

    def fresh(prompt):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
        first = prefill(e, prompt, sampling)
        return serial_decode(e, first, 14, sampling, stop_eos=False).tokens

    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, points=points)

    def run(prompt):
        got: list[int] = []
        s = Stream(list(prompt), 14, sampling, emit=lambda new, got=got: got.extend(new))
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        return s, got

    s_one, out_one = run(one)
    assert out_one == fresh(one) and s_one.cached == 0
    s_nxt, out_nxt = run(nxt)
    assert out_nxt == fresh(nxt) and s_nxt.cached == len(one) - 1
    s_fork, out_fork = run(fork)
    assert out_fork == fresh(fork) and s_fork.cached == 0          # a fork: a free lane, the chain stays whole
    s_other, out_other = run(other)
    assert out_other == fresh(other)
    assert s_other.cached == (len(one) // 2 if points is not None else 0)   # no lane spare: the mid entry
    per_slot: dict[int, list[int]] = {}
    for ids, slot, _, _ in dec.kept:
        per_slot.setdefault(id(slot), []).append(ids)
    for ids_list in per_slot.values():                             # every slot's entries stay a prefix chain
        assert ids_list == sorted(ids_list, key=len)
        for shorter, longer in zip(ids_list, ids_list[1:]):
            assert longer[:len(shorter)] == shorter


@pytest.mark.parametrize("sampling", [None, Sampling(seed=51, top_k=20, top_p=0.95)])
def test_three_resends_then_an_earlier_fork_keep_fresh_state_bits(w, sampling):
    shared = _prompt(300, seed=8)
    prompt = shared + _prompt(25, seed=12)
    engine = _engine_shim(w, points=lambda ids: [len(shared)])
    for turn in range(3):
        got, stats = _generate(engine, prompt, sampling)
        assert stats["cached"] == (len(prompt) - 1 if turn else 0)
        assert got == _generate(_engine_shim(w), prompt, sampling, draft=False)[0]
        for ids, snap in engine.cache:
            fresh = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
            prefill(fresh, ids, sampling)
            assert _same_snap(snap["state"], fresh.st.snapshot())
            assert _same_bits(snap["tail"], fresh.last_streams)
    fork = shared + _prompt(31, seed=13)
    got, stats = _generate(engine, fork, sampling)
    assert stats["cached"] == len(shared)
    assert got == _generate(_engine_shim(w), fork, sampling, draft=False)[0]


@pytest.mark.parametrize("keep_at", [7, 12])
def test_explicit_kept_point_and_message_stops_are_all_retained(w, keep_at):
    prompt, saved = _prompt(33), {}
    engine = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    first = prefill(engine, prompt, None, keep_at=keep_at, stops=[3, 9],
                    keep=lambda p, snap, tail: saved.update({p: {"state": snap, "tail": tail}}))
    assert set(saved) == {3, 9, keep_at}
    assert engine.kept["state"]["pos"] == keep_at
    fresh = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    assert first == prefill(fresh, prompt, None)
    _assert_same_state(engine.st, fresh.st)
    for p, snap in saved.items():
        prefill(fresh, prompt[:p], None)
        assert _same_snap(snap["state"], fresh.st.snapshot())
        assert _same_bits(snap["tail"], fresh.last_streams)


def test_filling_requests_do_not_retain_evicted_snapshot_references(w):
    prompt = _prompt(700)
    dec = MultiDecoder(w, slots=1, capacity=1024, depth=1, prefill_rows=300, points=lambda ids: [300], keep=1)
    stream = Stream(prompt, 4, stop_eos=False)
    dec.admit(stream)
    dec._pass()
    assert stream in dec.filling and len(dec.kept) == 1
    assert dec.fills[stream.sid][3] is None
    while dec.live():
        dec.finish(dec.round())
    assert len(dec.kept) == 1
