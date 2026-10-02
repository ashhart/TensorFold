"""--parallel on the tiny synthetic checkpoint of test_glm_engine (with its DFlash2 drafter and MTP head): a round's
batched verify (one forward over every stream's rows: segmented KDA and DSA kernels) gives each stream's logits, taps,
hidden rows and committed state the bits of its window run alone; concurrent requests through the real kernels
(DFlash2 and MTP drafts, serial ones), each reply bit-identical to its request served alone with ``"draft": false`` by
a --parallel 1 engine; a conversation's next turn resumes from its kept extent; a cancelled request ends at a round
boundary; a stream's extent moved mid-reply (the pool's relocation, a device copy of every plane) keeps its bits; a
stream past the dense limit (DSA's sparse rows, the indexer's planes at a non-zero base) beside short ones; grammar
requests beside plain ones; and a second decoder following rank 0's messages (as rank 1 does) ends in the same pool,
slots and drafter contexts; two ranks in one process make the same collectives.

One GPU plays rank 0 of two (``_TwoCopies``). The engines build in admission, so they need the GPU's memory free of a
serving model (tests/test_glm_multi.py covers the protocol, placement and rank agreement on CPU)."""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from test_glm_engine import _checkpoint, _drafter, _generate, _TwoCopies  # noqa: E402

KEPT_GIB = "0.5"                  # TF_GLM_CACHE_GIB: kept prompts' rows (the pool past the window) and states


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm_multi")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return path


def _engine(path, parallel: int, mtp: str = "1", **kw):
    """``mtp``: TF_GLM_MTP ("1": the head beside DFlash2, so MTP policies draft on it; "": the default)."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TF_GLM_CACHE_GIB", KEPT_GIB)
        if mtp:
            mp.setenv("TF_GLM_MTP", mtp)
        else:
            mp.delenv("TF_GLM_MTP", raising=False)
        return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies(),
                         parallel=parallel, **kw)


@pytest.fixture(scope="module")
def solo(paths):
    e = _engine(paths, 1)
    assert e.multi is None and not e.concurrent
    return e


@pytest.fixture(scope="module")
def multi(paths):
    e = _engine(paths, 4)
    assert e.multi is not None and e.concurrent and e.scheduler is not None
    assert e.w.mtp is not None and e.multi.mtp     # TF_GLM_MTP=1: the head beside DFlash2, MTP policies draft on it
    assert e.multi.pool.rows % 2048 == 0 and e.multi.pool.rows >= 3 * 2048, e.multi.pool.rows
    return e


def test_parallel_leaves_the_mtp_head_out_beside_dflash2_by_default(paths, solo):
    """--parallel without TF_GLM_MTP: concurrent streams draft with DFlash2 and the head stays out (its weights,
    cache rows and buffers neither loaded nor estimated); an MTP policy drafts on DFlash2 instead, the same reply."""

    e = _engine(paths, 2, mtp="")
    try:
        assert e.w.mtp is None and not e.mtp_on and not e.multi.mtp
        reply, stats = _generate(e, _prompt(80, 90), Sampling(3, 1.0, 20, 0.95), policy="3", tokens=30)
        assert reply == _serial(solo, _prompt(80, 90), Sampling(3, 1.0, 20, 0.95), 30)
        assert stats["drafted"] > 0
    finally:
        e.scheduler = None
        del e
        torch.cuda.empty_cache()


def _serial(solo, prompt, sampling, tokens):
    solo.cache.clear()
    solo.live = []
    return _generate(solo, prompt, sampling, draft=False, tokens=tokens)[0]


def _prompt(n: int, seed: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(0, 1000, size=n)]


def _concurrent(engine, reqs, delays=None):
    """Each request from its own thread (policy and stop_eos are the calling thread's, as in the server)."""

    out: dict[int, tuple] = {}
    errors = []

    def run(i, prompt, sampling, policy, draft, tokens):
        try:
            if delays:
                threading.Event().wait(delays[i])
            out[i] = _generate(engine, prompt, sampling, draft=draft, policy=policy, tokens=tokens)
        except Exception as exc:                  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i, *r)) for i, r in enumerate(reqs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    assert not errors, errors
    return out


REQS = [(_prompt(37, 1), Sampling(1234, 1.0, 20, 0.95), "f3", True, 40),
        (_prompt(150, 2), None, None, True, 36),
        (_prompt(90, 3), Sampling(7, 1.0, 0, 0.9, 0.02), "fc5:0.3", True, 30),
        (_prompt(260, 4), Sampling(99, 0.7, 20, 0.95, 0.1), None, False, 28)]
MTP_REQS = [(_prompt(120, 5), Sampling(55, 1.0, 20, 0.95), "3", True, 40),        # MTP drafts on the head
            (_prompt(70, 6), None, "c3:0.35", True, 36),
            (_prompt(200, 7), Sampling(56, 1.0, 0, 0.9), "a", True, 30),
            (_prompt(45, 8), None, "f3", True, 30)]                                # beside a DFlash2 stream


@pytest.mark.parametrize("reqs", [REQS, MTP_REQS], ids=["dflash2", "mtp"])
def test_concurrent_requests_equal_their_serial_replies(multi, solo, reqs):
    got = _concurrent(multi, reqs)
    for i, (prompt, sampling, _, _, tokens) in enumerate(reqs):
        reply, stats = got[i]
        assert reply == _serial(solo, prompt, sampling, tokens), i
        assert stats["parallel"] == 4
    assert any(got[i][1]["drafted"] for i in range(3))              # drafted windows of several rows
    if reqs is REQS:
        assert got[3][1]["drafted"] == 0                             # the serial reference, one row a round
    h = multi.multi.health()
    assert h["streams"] == {"decoding": 0, "filling": 0, "paused": 0} and h["pool_tokens"] == multi.multi.pool.rows


@pytest.mark.parametrize("policy", [None, "3"], ids=["dflash2", "mtp"])
def test_staggered_arrivals_and_a_resumed_turn(multi, solo, policy):
    first, _ = _generate(multi, _prompt(300, 10), None, tokens=20, policy=policy)
    turn = _prompt(300, 10) + first + _prompt(20, 11)
    reqs = [(turn, Sampling(5, 1.0, 20, 0.95), policy, True, 30),
            (_prompt(500, 12), None, "f3", True, 40),
            (_prompt(64, 13), Sampling(6, 1.0, 20, 0.95), None, True, 25)]
    got = _concurrent(multi, reqs, delays=[0.0, 0.05, 0.3])
    assert got[0][1]["cached"] == 299                                  # the first turn's prompt, kept in the pool
    for i, (prompt, sampling, _, _, tokens) in enumerate(reqs):
        assert got[i][0] == _serial(solo, prompt, sampling, tokens), i


def test_a_cancelled_request_ends_at_a_round_boundary(multi, solo):
    out: list[int] = []
    done = {}

    def stopper():
        multi.request.policy, multi.request.stop_eos = None, False
        done["stats"] = multi.generate(_prompt(80, 20), 400, None, lambda new: (out.extend(new), len(out) >= 8)[1])

    other = {}

    def runner():
        other["r"] = _generate(multi, _prompt(120, 21), Sampling(3, 1.0, 20, 0.95), tokens=60)

    ts = [threading.Thread(target=stopper), threading.Thread(target=runner)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=600)
    assert 8 <= len(out) < 8 + 2 * 16                               # stopped within a round or two of the stop
    assert out == _serial(solo, _prompt(80, 20), None, 400)[:len(out)]
    assert other["r"][0] == _serial(solo, _prompt(120, 21), Sampling(3, 1.0, 20, 0.95), 60)
    assert not multi.multi.lanes


def test_a_moved_extent_keeps_its_bits(multi, solo):
    """Drive the decoder directly: a stream's extent is relocated (the pool's MOVE: every plane's rows copied) after
    its prompt and again mid-reply; the reply is the serial one."""

    from tensorfold.cuda.streams import Stream
    from tensorfold.families.glm5_next.cuda.engine import encode_policy
    from tensorfold.families.glm5_next.cuda.multi import MOVE

    d = multi.multi
    prompt, sampling = _prompt(700, 30), Sampling(8, 1.0, 20, 0.95)
    s = Stream(prompt, 80, sampling, draft=True, stop_eos=False)
    s.glm = {"code": encode_policy("f3"), "spec": "f3"}
    got: list[int] = []
    s.emit = lambda new: got.extend(new)
    d.admit(s)
    lane = d.lanes[s.sid]
    moves = 0
    while d.lanes:
        done = d.round()
        if lane.decoding and moves < 2 and len(s.out) >= 1 + 30 * moves:
            x = lane.extent
            base = d.pool.place(x.size)                   # the lowest free rows (x's own are taken)
            assert base is not None and base != x.base
            d._emit(MOVE, [x.eid, base, x.size])
            d._move(x, base, x.size)
            assert lane.st.base == base and lane.st.kc[0].data_ptr() == d.arena.planes[0].tensor[base].data_ptr()
            moves += 1
        d.finish(done)
    assert moves == 2
    assert got == _serial(solo, prompt, sampling, 80)


def test_a_mirror_decoder_follows_rank_0(paths, solo):
    """Rank 0's decoder and a second one applying its messages (rank 1's ``apply``) over two engines: after
    concurrent requests with kept prompts, both hold the same pool, kept prompts, arena rows, slots and contexts."""

    from tensorfold.cuda.streams import Stream
    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    a, b = _engine(paths, 3), _engine(paths, 3)
    b.costs = a.costs                                      # two real ranks gather one cost table at startup
    try:
        sent = []
        a._share = lambda values: (sent.append(list(values)), list(values))[1]
        b._share = lambda values: sent.pop(0)
        da, db = a.multi, b.multi
        reqs = [(_prompt(200, 40), Sampling(1, 1.0, 20, 0.95)), (_prompt(90, 41), None),
                (_prompt(200, 40) + _prompt(30, 42), Sampling(2, 1.0, 0, 0.9))]
        streams = []
        for i, (prompt, sampling) in enumerate(reqs):
            s = Stream(prompt, 30, sampling, draft=True, stop_eos=False)
            s.glm = {"code": encode_policy("fc5:0.3"), "spec": "fc5:0.3"}
            s.got = []
            s.emit = s.got.extend
            streams.append(s)
        da.admit(streams[0])
        da.admit(streams[1])
        it = 0
        while da.lanes or it < 4:
            if it == 3:
                da.admit(streams[2])                       # arrives mid-reply, shares the first prompt
            done = da.round()
            da.finish(done)
            while sent:
                db.follow(once=True)
            it += 1
        for s, (prompt, sampling) in zip(streams, reqs):
            assert s.got == _serial(solo, prompt, sampling, 30)
        assert streams[2].cached == 199                   # copied: the first stream still wrote in that extent
        torch.cuda.synchronize()
        assert [(x.base, x.size, x.owner) for x in da.pool.extents] == [(x.base, x.size, x.owner)
                                                                        for x in db.pool.extents]
        assert [c.ids for c in da.kept] == [c.ids for c in db.kept]
        for p, q in zip(da.arena.planes, db.arena.planes):
            assert torch.equal(p.tensor, q.tensor)
        assert torch.equal(a.e.slots.rec, b.e.slots.rec) and torch.equal(a.e.slots.conv, b.e.slots.conv)
        assert torch.equal(da.drafts.kc[0], db.drafts.kc[0])
        assert db.idle and not db.lanes
    finally:
        for e in (a, b):
            e.scheduler = None
        del a, b
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def long_pair(paths):
    """--context 2600 (past the dense limit: the indexer's planes, sparse rows): a --parallel 4 engine and a solo."""

    return _engine(paths, 4, context=2600), _engine(paths, 1, context=2600)


def test_a_long_stream_beside_short_ones(long_pair):
    multi, solo = long_pair
    assert multi.multi.pool.rows >= 3 * 2048 and multi.e.caches.index is not None
    reqs = [(_prompt(2200, 50), Sampling(9, 1.0, 20, 0.95), None, True, 40),
            (_prompt(100, 51), None, "f3", True, 40),
            (_prompt(1800, 52), None, None, True, 60)]
    got = _concurrent(multi, reqs, delays=[0.0, 0.1, 0.2])
    for i, (prompt, sampling, _, _, tokens) in enumerate(reqs):
        assert got[i][0] == _serial(solo, prompt, sampling, tokens), i


def _decoding_lanes(engine, prompts):
    """Streams admitted and prefilled (their first token sampled) through the engine's decoder, not yet decoding."""

    from tensorfold.cuda.streams import Stream
    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    d = engine.multi
    streams = []
    for i, prompt in enumerate(prompts):
        s = Stream(prompt, 1000, Sampling(40 + i, 1.0, 20, 0.95), draft=True, stop_eos=False)
        s.glm = {"code": encode_policy("f3"), "spec": "f3"}
        s.emit = lambda new: None
        d.admit(s)
        streams.append(s)
    while any(not l.decoding for l in d.lanes.values()):
        lane = next(l for l in d.lanes.values() if not l.decoding)
        first = d._fill(lane, d._stop(lane))
        if first is not None:
            lane.s.take([first])
    return [d.lanes[s.sid] for s in streams]


def _snapshot(engine, lanes):
    d = engine.multi
    return ([p.tensor.clone() for p in d.arena.planes], engine.e.slots.rec.clone(), engine.e.slots.conv.clone(),
            [(l.st.pos, list(l.st.cur)) for l in lanes])


def _restore(engine, lanes, snap):
    d = engine.multi
    planes, rec, conv, pos = snap
    for p, t in zip(d.arena.planes, planes):
        p.tensor.copy_(t)
    engine.e.slots.rec.copy_(rec)
    engine.e.slots.conv.copy_(conv)
    for l, (n, cur) in zip(lanes, pos):
        l.st.set_pos(n)
        l.st.cur = list(cur)


def _run_window(verify, lanes, windows, keeps):
    from tensorfold.families.glm5_next.cuda.verify import Segment

    segs = [Segment(l.st, w, hidden=True) for l, w in zip(lanes, windows)]
    v = verify.forward(segs)
    logits = [x.clone() for x in v.logits]
    taps = [t.clone() for t in v.taps]
    hidden = [h.clone() for h in v.hidden]
    verify.commit(segs, keeps)
    torch.cuda.synchronize()
    return logits, taps, hidden


@pytest.mark.parametrize("rows", [(1, 1), (4, 2, 7, 1), (16, 16), (8, 8, 8, 8), (13, 1, 9, 9), (3,)],
                         ids=lambda r: "-".join(map(str, r)))
@pytest.mark.parametrize("which", ["dense", "long"])
@pytest.mark.parametrize("graphs", [True, False], ids=["graphs", "eager"])
def test_batched_windows_equal_serial_ones(multi, long_pair, which, rows, graphs):
    """The same windows through SerialVerify and BatchedVerify (its captured graph of that size, or eager) from the
    same states: every segment's logits and taps bit for bit, and after the commits (some keeping every row, some
    fewer) the same KDA states, conv windows and arena rows (every plane of every extent)."""

    from tensorfold.families.glm5_next.cuda.verify import BatchedVerify, SerialVerify

    engine = multi if which == "dense" else long_pair[0]
    d = engine.multi
    lengths = [37, 300, 2100, 777] if which == "long" else [37, 300, 555, 777]
    lanes = _decoding_lanes(engine, [_prompt(n, 60 + i) for i, n in enumerate(lengths[:len(rows)])])
    try:
        rng = np.random.default_rng(sum(rows))
        for lane in lanes:                                 # room for the widest window
            assert d._grow(lane.extent, lane.st.pos + 32, protect=[lane.extent])
        windows = [[lane.s.out[-1]] + [int(t) for t in rng.integers(0, 1000, size=n - 1)]
                   for lane, n in zip(lanes, rows)]
        keeps = [max(1, n - k) for k, n in enumerate(rows)]
        serial = SerialVerify(engine.e, taps=True)
        batched = d.verify if graphs else BatchedVerify(engine.e, taps=engine.drafter.tap_layers, rows=32)
        assert isinstance(batched, BatchedVerify) and bool(batched.graphs) == graphs
        before = dict(batched.replays)
        snap = _snapshot(engine, lanes)
        want = _run_window(serial, lanes, windows, keeps)
        after = _snapshot(engine, lanes)
        _restore(engine, lanes, snap)
        got = _run_window(batched, lanes, windows, keeps)
        assert batched.replays["graph" if graphs else "eager"] == before["graph" if graphs else "eager"] + 1
        for k in range(len(rows)):
            assert torch.equal(got[0][k].view(torch.int16), want[0][k].view(torch.int16)), ("logits", k)
            assert torch.equal(got[1][k], want[1][k]), ("taps", k)
            assert torch.equal(got[2][k], want[2][k]), ("hidden rows (the MTP head's input)", k)
        now = _snapshot(engine, lanes)
        assert now[3] == after[3]
        assert torch.equal(now[1], after[1]) and torch.equal(now[2], after[2])
        for p, (a, b) in enumerate(zip(now[0], after[0])):
            assert torch.equal(a, b), ("plane", p)
    finally:
        for lane in lanes:
            lane.s.done = True
        d.finish([lane.s for lane in lanes])


# -- grammar requests beside plain ones ------------------------------------------------------------------------------
def _grammar():
    """A fresh constraint of the toy JSON-like grammar (tests/cuda/toy_grammar.py) and its Grammars."""

    from tensorfold.engine.grammar import Spec
    from toy_grammar import RULE, toy

    from test_glm_engine import V

    grammars, compiled = toy(V)
    return grammars, lambda: grammars.constraint(compiled, spec=Spec("grammar", RULE))


def _gen(engine, prompt, sampling, *, draft=True, policy=None, tokens=24, constraint=None):
    out: list[int] = []
    engine.request.policy = policy
    engine.request.stop_eos = False
    stats = engine.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=draft,
                            constraint=constraint)
    return out, stats


def _solo(solo, prompt, sampling, tokens, **kw):
    solo.cache.clear()
    solo.live = []
    return _gen(solo, prompt, sampling, draft=False, tokens=tokens, **kw)[0]


def _spelled(tokens) -> str:
    from toy_grammar import spell

    text = ""
    for t in tokens:
        if t == 0:
            break
        text += spell(t)
    return text


def test_grammar_requests_beside_plain_ones(multi, solo):
    """response_format grammars (each stream masks and walks its own) and plain requests at once: each reply is its
    request's alone with draft: false, and the grammar replies follow the grammar."""

    import re

    _, fresh = _grammar()
    reqs = [dict(prompt=_prompt(60, 82), sampling=Sampling(12, 1.0, 20, 0.95), grammar=True, tokens=40),
            dict(prompt=_prompt(200, 83), sampling=None, grammar=True, tokens=40, policy="fc5:0.3"),
            dict(prompt=_prompt(120, 80), sampling=Sampling(11, 0.8, 20, 0.95), tokens=30, policy="3"),
            dict(prompt=_prompt(90, 84), sampling=Sampling(13, 1.0, 0, 0.9), tokens=30)]
    out: dict[int, tuple] = {}
    errors = []

    def run(i, r):
        try:
            threading.Event().wait(0.05 * i)
            out[i] = _gen(multi, r["prompt"], r["sampling"], policy=r.get("policy"), tokens=r["tokens"],
                          constraint=fresh() if r.get("grammar") else None)
        except Exception as exc:                  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i, r)) for i, r in enumerate(reqs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    assert not errors, errors
    for i, r in enumerate(reqs):
        want = _solo(solo, r["prompt"], r["sampling"], r["tokens"], constraint=fresh() if r.get("grammar") else None)
        assert out[i][0] == want, i
    for i in (0, 1):                               # the grammar held: a JSON-like prefix of the rule
        assert re.fullmatch(r"\{([a-h]+:[0-9]+,)*([a-h]+(:[0-9]*)?)?\}?", _spelled(out[i][0])), _spelled(out[i][0])


def test_a_grammar_request_through_the_mirror(paths, solo):
    """A grammar request and a plain one through rank 0's decoder and a mirror applying its messages (rank 1
    compiles the grammar from ADMIT): the replies are the solo ones and both decoders end in the same bits."""

    from tensorfold.cuda.streams import Stream
    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    grammars, fresh = _grammar()
    a, b = _engine(paths, 2), _engine(paths, 2)
    b.costs = a.costs
    b.grammars = grammars                          # rank 1's compiler (grammar.compiler reads it off the owner)
    try:
        sent = []
        a._share = lambda values: (sent.append(list(values)), list(values))[1]
        b._share = lambda values: sent.pop(0)
        reqs = [(_prompt(80, 86), Sampling(21, 1.0, 20, 0.95), True), (_prompt(150, 87), None, False)]
        streams = []
        for prompt, sampling, shaped in reqs:
            s = Stream(prompt, 40, sampling, draft=True, stop_eos=False, constraint=fresh() if shaped else None)
            s.glm = {"code": encode_policy("f3"), "spec": "f3"}
            s.got = []
            s.emit = s.got.extend
            a.multi.admit(s)
            streams.append(s)
        while a.multi.lanes:
            done = a.multi.round()
            a.multi.finish(done)
            while sent:
                b.multi.follow(once=True)
        for s, (prompt, sampling, shaped) in zip(streams, reqs):
            assert s.got == _solo(solo, prompt, sampling, 40, constraint=fresh() if shaped else None)
        torch.cuda.synchronize()
        for p, q in zip(a.multi.arena.planes, b.multi.arena.planes):
            assert torch.equal(p.tensor, q.tensor)
        assert torch.equal(a.e.slots.rec, b.e.slots.rec) and b.multi.idle
    finally:
        for e in (a, b):
            e.scheduler = None
        del a, b
        torch.cuda.empty_cache()



# -- speed settings (multi_tune) -------------------------------------------------------------------------------------
def test_a_lone_stream_replays_the_one_stream_graphs(multi, solo):
    """TF_GLM_MULTI_LONE=1: a stream outside the graphs' home goes on alone after its neighbour ends: it moves home
    (slot 0, the pool's first rows) and its windows replay the one-stream graphs; a stream joining it runs batched
    again. The replies are the serial ones."""

    d, e = multi.multi, multi.e
    assert not d.tune.lone and d.lone_rows == 6         # off by default (TF_GLM_MULTI_LONE=1); the graphs' rows
    d.tune.lone = True                                  # one rank here: no collective depends on it
    try:
        _lone_stream(d, e, solo)
    finally:
        d.tune.lone = False


def _lone_stream(d, e, solo):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    def stream(prompt, count, sampling):
        s = Stream(prompt, count, sampling, draft=True, stop_eos=False)
        s.glm = {"code": encode_policy("f3"), "spec": "f3"}
        s.got = []
        s.emit = s.got.extend
        d.admit(s)
        return s

    a = stream(_prompt(1990, 120), 6, None)                       # a whole block before b
    b = stream(_prompt(400, 121), 90, Sampling(31, 1.0, 20, 0.95))
    assert d.lanes[b.sid].slot == 1 and d.lanes[b.sid].extent.base > 0
    c, solo_before = None, e.replays["main"] + e.replays["sparse"]
    while d.lanes:
        done = d.round()
        d.finish(done)
        if a.done and c is None and len(b.out) > 50:
            assert e.graphed(d.lanes[b.sid].st)                   # home: slot 0, base 0
            assert e.replays["main"] + e.replays["sparse"] > solo_before
            c = stream(_prompt(120, 122), 30, None)
    assert b.got == _serial(solo, _prompt(400, 121), Sampling(31, 1.0, 20, 0.95), 90)
    assert c.got == _serial(solo, _prompt(120, 122), None, 30)
    assert a.got == _serial(solo, _prompt(1990, 120), None, 6)


def test_every_speed_setting_keeps_the_serial_replies(paths, solo, capfd):
    """TF_GLM_MULTI_SAMPLER=packed, TF_GLM_MULTI_DEPTH=joint (the batched window timed by rows at startup),
    TF_GLM_MULTI_ASYNC=1 and TF_GLM_MULTI_PROFILE on one engine: concurrent top-k, nucleus, min_p and greedy
    requests are their serial replies; the profile reports both paths."""

    with pytest.MonkeyPatch.context() as mp:
        for k, v in (("TF_GLM_MULTI_SAMPLER", "packed"), ("TF_GLM_MULTI_DEPTH", "joint"), ("TF_GLM_MULTI_ASYNC", "1"),
                     ("TF_GLM_MULTI_PROFILE", "4")):
            mp.setenv(k, v)
        e = _engine(paths, 3)
    try:
        d = e.multi
        assert d.tune.sampler == "packed" and d.tune.depth == "joint" and d.tune.async_msg
        assert len(d.row_ms) == 32 and all(ms > 0 for ms in d.row_ms)
        reqs = [(_prompt(50, 130), Sampling(41, 1.0, 0, 0.95), "fnc7:0.3", True, 40),
                (_prompt(150, 131), Sampling(42, 1.0, 20, 0.95), None, True, 40),
                (_prompt(90, 132), Sampling(43, 0.7, 0, 1.0, 0.05), "fc5:0.3", True, 30),
                (_prompt(70, 133), None, "f3", True, 35)]
        got = _concurrent(e, reqs, delays=[0.0, 0.0, 0.05, 1.0])
        for i, (prompt, sampling, _, _, tokens) in enumerate(reqs):
            assert got[i][0] == _serial(solo, prompt, sampling, tokens), i
        lone, _ = _generate(e, _prompt(60, 134), Sampling(44, 1.0, 0, 0.95), tokens=40)
        assert lone == _serial(solo, _prompt(60, 134), Sampling(44, 1.0, 0, 0.95), 40)
        out = capfd.readouterr().out
        assert "one-stream graphs vs batched graphs" in out and "multi profile (batched)" in out
    finally:
        e.scheduler = None
        del e
        torch.cuda.empty_cache()


def _collect() -> None:
    """Free a dropped engine's graphs now: collected later, during another engine's capture, their reset is an
    operation the capture forbids."""

    import gc

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


class _Recording(_TwoCopies):
    """``_TwoCopies`` that records every collective in order: all-gather sizes and dtypes, barriers, and each replay
    of a CUDA graph that captured all-gathers (``_graph_replays``): a graph's gathers run again at every replay
    without calling the communicator, so a rank replaying graphs the other does not is a mismatch too."""

    capturing = None                               # the graph being captured, if any
    gathers: dict = {}                             # id(graph) -> all-gathers it captured
    active = None                                  # the recorder of the engine being built

    def __init__(self) -> None:
        self.calls: list = []

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        self.calls.append((send.numel(), str(send.dtype)))
        if _Recording.capturing is not None:
            _Recording.gathers[id(_Recording.capturing)] = _Recording.gathers.get(id(_Recording.capturing), 0) + 1
        super().all_gather(send, recv)

    def barrier(self) -> None:
        self.calls.append("barrier")
        super().barrier()


@pytest.mark.parametrize("env", [{}, {"TF_GLM_MULTI_PROFILE": "5", "TF_GLM_MULTI_DEPTH": "joint"}],
                         ids=["default", "profile+joint"])
def test_both_ranks_make_the_same_startup_collectives(paths, env):
    """Rank 0 and rank 1 engines (--parallel 3) built with every setting alike make the same collectives in the same
    order at startup: rank 0 alone timing graphs that all-gather (TF_GLM_MULTI_PROFILE's path comparison once did)
    leaves rank 1 waiting on the doorbell and rank 0 inside an all-gather forever."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    graph = torch.cuda.CUDAGraph
    begin, end, replay = graph.capture_begin, graph.capture_end, graph.replay

    def capture_begin(self, *a, **k):
        _Recording.capturing = self
        _Recording.gathers[id(self)] = 0
        return begin(self, *a, **k)

    def capture_end(self, *a, **k):
        out = end(self, *a, **k)
        _Recording.capturing = None
        return out

    def replayed(self, *a, **k):
        n = _Recording.gathers.get(id(self), 0)
        if n and _Recording.active is not None:
            _Recording.active.calls.append(("graph replay", n))
        return replay(self, *a, **k)

    calls = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(graph, "capture_begin", capture_begin)
        mp.setattr(graph, "capture_end", capture_end)
        mp.setattr(graph, "replay", replayed)
        mp.setenv("TF_GLM_CACHE_GIB", KEPT_GIB)
        for k, v in env.items():
            mp.setenv(k, v)
        for rank in (0, 1):
            comm = _Recording()
            _Recording.active = comm
            e = GlmEngine(paths / "model", rank=rank, master="", port=0, drafter=paths / "dflash2", comm=comm,
                          parallel=3)
            assert (e.scheduler is not None) == (rank == 0) and e.multi is not None
            calls.append(list(comm.calls))
            _Recording.active = None
            e.scheduler = None
            del e
            _collect()
    assert any(c[0] == "graph replay" for c in calls[0])        # the startup replays graphs that gather
    assert len(calls[0]) == len(calls[1]) and calls[0] == calls[1]


# -- two ranks in one process: real rendezvous all-gathers ------------------------------------------------------------
class _Store:
    """The rendezvous TCP store's surface the idle doorbell uses (set / wait / delete_key), over threading events."""

    def __init__(self) -> None:
        import threading

        self.lock, self.keys = threading.Lock(), {}

    def _event(self, key):
        import threading

        with self.lock:
            return self.keys.setdefault(key, threading.Event())

    def set(self, key, value) -> None:
        self._event(key).set()

    def wait(self, keys, timeout) -> None:
        for key in keys:
            if not self._event(key).wait(timeout.total_seconds()):
                raise RuntimeError("store wait timeout")

    def delete_key(self, key) -> None:
        with self.lock:
            self.keys.pop(key, None)


class _Pair:
    """Two ranks' communicators in one process: each all-gather meets the other rank's in a barrier (a mismatch in
    count or size fails, a missing call times out instead of hanging). Both engines compute the same rows (each
    was built as rank 0), so every rank's words are the same and the gathers' results are _TwoCopies'."""

    def __init__(self, timeout: float = 120.0) -> None:
        import threading

        self.barrier = threading.Barrier(2, timeout=timeout)
        self.slots: list = [None, None]
        self.log: list[list] = [[], []]
        self.store = _Store()

    def comm(self, rank: int):
        pair = self

        class Comm:
            world = 2
            store = pair.store

            def __init__(self) -> None:
                self.rank = rank

            def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
                # the buffers' 16-byte alignment rides along: a transport that chooses by it (RoceComm did) must
                # see the same on both ranks
                shape = (send.numel(), send.dtype, send.data_ptr() % 16 == 0, recv.data_ptr() % 16 == 0)
                pair.slots[rank] = (*shape, send.detach().reshape(-1).clone())
                pair.log[rank].append(shape)
                pair.barrier.wait()
                mine, other = pair.slots[rank], pair.slots[1 - rank]
                if mine[:4] != other[:4]:
                    pair.barrier.abort()
                    raise RuntimeError(f"rank {rank}: all-gather of {mine[:4]} met {other[:4]} (count, dtype, "
                                       f"send aligned, receive aligned)")
                n = mine[0]
                flat = recv.view(-1)
                flat[:n].copy_(pair.slots[0][4])
                flat[n:2 * n].copy_(pair.slots[1][4])
                pair.barrier.wait()

            def barrier(self) -> None:
                torch.cuda.synchronize()

        return Comm()


def _two_ranks(paths, env: dict):
    """Two --parallel 2 engines joined by a ``_Pair``: rank 0's scheduler serves, rank 1's decoder follows in a
    thread (its errors collected)."""

    import threading

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TF_GLM_CACHE_GIB", KEPT_GIB)
        for k, v in env.items():
            mp.setenv(k, v)
        a, b = _engine(paths, 2), _engine(paths, 2)
    b.costs = a.costs
    b.scheduler = None
    pair = _Pair()
    for rank, e in enumerate((a, b)):
        c = pair.comm(rank)
        e.comm = e.w.comm = c
        e.rank = e.multi.rank = rank
    errors = []

    def follow():
        try:
            b.multi.follow()
        except Exception as exc:                  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=follow, daemon=True)
    t.start()
    return a, b, pair, errors


FLAGS = {"TF_GLM_MULTI_LONE": "1", "TF_GLM_MULTI_SAMPLER": "packed", "TF_GLM_MULTI_ASYNC": "1"}


@pytest.mark.parametrize("flags", [dict(FLAGS), {"TF_GLM_MULTI_LONE": "1"}, {"TF_GLM_MULTI_ASYNC": "1"},
                                   {"TF_GLM_MULTI_SAMPLER": "packed"}, {}],
                         ids=["all", "lone", "async", "packed", "none"])
def test_two_ranks_a_second_request_joins_a_lone_stream(paths, solo, flags, monkeypatch):
    """A stream decoding alone (on the one-stream graphs with TF_GLM_MULTI_LONE=1), then a second request arriving
    mid-reply with a prompt of two 1,024-row chunks, on two ranks whose all-gathers meet for real and must agree in
    count, dtype and buffer alignment (a transport may choose by the buffer's address: the async message once went
    from a view 4 bytes past its allocation). Both requests finish with their serial replies; rank 1 raises nothing
    and ends idle."""

    import threading

    for k, v in flags.items():
        monkeypatch.setenv(k, v)
    a, b, pair, errors = _two_ranks(paths, flags)
    try:
        first_sampling = Sampling(51, 1.0, 0, 0.95)
        out: dict = {}

        def client(name, prompt, sampling, tokens, wait=None):
            if wait is not None:
                wait.wait(120)
            out[name] = _generate(a, prompt, sampling, tokens=tokens)

        started = threading.Event()
        long_prompt = _prompt(1900, 141)
        t1 = threading.Thread(target=client, args=("a", _prompt(60, 140), first_sampling, 160))
        t2 = threading.Thread(target=client, args=("b", long_prompt, Sampling(52, 1.0, 0, 0.95), 60, started))
        t1.start()
        t2.start()
        for _ in range(600):                        # the second arrives once the first decodes alone
            if a.multi.lanes and any(l.decoding and len(l.s.out) > 20 for l in a.multi.lanes.values()):
                break
            threading.Event().wait(0.05)
        started.set()
        t1.join(300)
        t2.join(300)
        assert not t1.is_alive() and not t2.is_alive(), "the requests hung"
        assert not errors, errors
        assert out["a"][0] == _serial(solo, _prompt(60, 140), first_sampling, 160)
        assert out["b"][0] == _serial(solo, long_prompt, Sampling(52, 1.0, 0, 0.95), 60)
        for _ in range(200):
            if b.multi.idle and not b.multi.lanes:
                break
            threading.Event().wait(0.05)
        assert b.multi.idle and not b.multi.lanes
    finally:
        a.scheduler = None
        pair.barrier.abort()
        del a, b
        _collect()
