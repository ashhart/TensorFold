"""Full GLM-5.3 concurrent streams with DFlash2 drafts (Phase B: multi.GlmMultiDecoder + dflash.MultiDrafter) on real
weights (TF_GLM53_CKPT, layers 0-3 + the MTP layer) and a DFlash2 drafter (TF_GLM53_DFLASH_TEST):

1. the multi-stream drafter's block pass gives each stream the drafts its solo drafter gives (one stream and three);
2. every concurrent stream's reply - DFlash2, auto and MTP requests mixed, greedy and seeded sampled, staggered, one
   prompt past index_topk - equals the same request alone, for 2, 3 and 4 streams, with verify windows past 16 rows
   (confidence 0: every DFlash2 stream drafts 7, 4 x 8 = 32 rows); a lone DFlash2 stream takes the one-stream path's
   rounds;
3. the same with CUDA graphs (one thread as rank 0 of four, the drafter's passes and every window shape replayed);
4. Glm53Engine(parallel=3) with TF_GLM53_DFLASH serves "dflash" requests concurrently as it serves them alone.

The 4-layer model taps layers 0-3 (slots 0-3; the drafter's last two taps stay zero), so drafts are poor but real.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import test_glm_moe_dsa_multi as base
from threadcomm import run_ranks


@pytest.fixture(autouse=True)
def _prompt_rows_4096(monkeypatch):
    """Four rank threads share one GPU here: 8192-row prompt buffers four times over exhaust a 128 GB node."""
    from tensorfold.families.glm_moe_dsa.cuda import runner

    monkeypatch.setattr(runner, "PROMPT_ROWS", min(runner.PROMPT_ROWS, 4096))

DFLASH = os.environ.get("TF_GLM53_DFLASH_TEST", "")
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not base.CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint"),
              pytest.mark.skipif(not DFLASH, reason="set TF_GLM53_DFLASH_TEST to a GLM-5.3 DFlash2 checkpoint")]

K = base.K
CAPACITY = base.CAPACITY


@pytest.fixture(autouse=True)
def _full_depth(monkeypatch):
    monkeypatch.setenv("TF_GLM53_DFLASH_CONFIDENCE", "0")       # every DFlash2 round drafts its full depth
    monkeypatch.setenv("TF_GLM53_DFLASH_DEPTH", "7")


def _weights(rank, world, comm):
    w = base._weights(rank, world, comm)
    n = len(json.loads((Path(DFLASH) / "config.json").read_text())["dflash_config"]["target_layer_ids"])
    w.tap_slot = {i: i for i in range(len(base.LAYERS))}
    w.tap_slot.update({1000 + i: i for i in range(len(base.LAYERS), n)})      # no such layers: those taps stay zero
    return w


def _drafter(w):
    from tensorfold.families.glm_moe_dsa.cuda.dflash import GlmDrafter

    return GlmDrafter(DFLASH, w, capacity=CAPACITY + 16)


def _requests():
    """(name, prompt, tokens, sampling, mode)."""
    reqs = base._requests()
    modes = {"a": "dflash", "b": "dflash", "c": "auto", "d": "dflash", "e": "normed/normed"}
    out = {n: (*v, modes[n]) for n, v in reqs.items()}
    out["f"] = (base._tokens(60, 6), 26, None, "auto")
    return out


SCENARIOS = [
    [("a", 0), ("b", 0)],
    [("c", 0), ("f", 1), ("e", 3)],
    [("d", 0), ("b", 0), ("a", 1), ("f", 2)],
    [("a", 0), ("b", 0), ("f", 0), ("d", 0)],
]


def _solo(w, dr, reqs, graphs):
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner

    sample = base._sample_fn()
    run = Runner(w, CAPACITY + K + 1, K, graphs=graphs)
    run.drafter = dr
    out, stats = {}, {}
    for name, (prompt, n, s, mode) in reqs.items():
        fn = None if s is None else (lambda lg, pos, s=s: sample(lg, pos, s))
        st = run.generate(prompt, n, fn, lambda t: False, lambda new: None, K, mode, sampling=s)
        out[name], stats[name] = st["out"], st
    del run
    torch.cuda.empty_cache()
    return out, stats


def _drive(dec, reqs, scenario):
    from tensorfold.cuda.streams import Stream

    got = {name: [] for name, _ in scenario}
    streams = {}
    arrivals = sorted(scenario, key=lambda x: x[1])
    rnd, i = 0, 0
    while i < len(arrivals) or dec.live():
        while i < len(arrivals) and arrivals[i][1] <= rnd:
            name = arrivals[i][0]
            prompt, n, s, mode = reqs[name]
            draft = mode if mode in ("dflash", "auto") else True
            st = Stream(list(prompt), n, s, draft=draft, stop_eos=False)
            st.emit = lambda new, name=name: got[name].extend(new)
            dec.admit(st)
            streams[name] = st
            i += 1
        dec.finish(dec.round())
        rnd += 1
    return got, {n: (s.rounds, getattr(s, "arms", "")) for n, s in streams.items()}


def _multi(rank, world, comm, w, dr, reqs, graphs):
    from tensorfold.families.glm_moe_dsa.cuda.multi import GlmMultiDecoder
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner

    run = Runner(w, CAPACITY + K + 1, K, graphs=graphs, slots=4)
    run.drafter = dr
    dec = GlmMultiDecoder(run, rank=rank, world=world, comm=comm, limit=CAPACITY, eos=tuple(w.cfg.eos_token_ids),
                          sample=base._sample_fn())
    assert dec.rows == 32 and dec.frows == 8
    if graphs:
        dec.prewarm()

    def phase(work):
        out = None
        if rank:
            dec.follow()
        else:
            out = work()
            dec.stop()
        torch.cuda.synchronize()
        comm.barrier()
        return out

    lone = phase(lambda: _drive(dec, reqs, [("a", 0)]))
    outs = phase(lambda: [_drive(dec, reqs, sc) for sc in SCENARIOS])
    return None if rank else (lone, outs, dec.widest, dec.free)


def _check(solo, stats, res):
    (lone_got, lone_counts), outs, widest, free = res
    assert lone_got["a"] == solo["a"]
    rounds, _ = lone_counts["a"]
    assert rounds == stats["a"]["rounds"], (rounds, stats["a"]["rounds"])      # the one-stream path's rounds
    for sc, (got, counts) in zip(SCENARIOS, outs):
        for name, _ in sc:
            assert got[name] == solo[name], f"{len(sc)} streams, request {name}: {got[name]} != alone {solo[name]}"
            if name in ("c", "f"):                       # auto: both arms ran
                assert "m" in counts[name][1] and "f" in counts[name][1], (name, counts[name])
    assert widest > 16, f"no verify window wider than 16 rows (widest {widest})"
    assert sorted(free) == [0, 1, 2, 3], free


def test_multi_drafter_matches_solo():
    """dflash.MultiDrafter: one stream's drafts equal its solo GlmDrafter's (the same arithmetic); three streams in
    one block pass give each its solo drafts too."""
    from tensorfold.families.glm_moe_dsa.cuda.dflash import MultiDrafter

    with torch.no_grad():
        comm = base._Alone()
        w = _weights(0, 4, comm)
        dr = _drafter(w)
        md = MultiDrafter(dr, 4)
        D = len(dr.tap_layers) * w.cfg.hidden_size
        g = torch.Generator(device="cuda").manual_seed(3)
        ctx = {s: (torch.randn((L, D), generator=g, device="cuda") * 2).bfloat16() for s, L in ((0, 70), (2, 131),
                                                                                                (3, 9))}
        pend = {0: 1234, 2: 777, 3: 42}
        solo = {}
        for s, taps in ctx.items():
            dr.reset()
            dr.add_taps(taps)
            solo[s] = dr.propose(pend[s], 7, None, 0.0)
            md.reset(s)
            md.commit([(s, taps)])
        for s in ctx:
            assert md.propose([(s, pend[s], 7, None, 0.0)])[0] == solo[s], f"one stream (slot {s}) != solo"
        got = md.propose([(s, pend[s], 7, None, 0.0) for s in ctx])
        assert got == [solo[s] for s in ctx], f"three streams {got} != solo {solo}"
        assert all(len(v) == 7 for v in solo.values())
        # a context update of several streams at once == one at a time
        more = {s: (torch.randn((5 + s, D), generator=g, device="cuda")).bfloat16() for s in ctx}
        md.commit([(s, t) for s, t in more.items()])
        for s in ctx:
            dr.reset()
            dr.add_taps(torch.cat([ctx[s], more[s]]))
            assert md.propose([(s, pend[s], 7, None, 0.0)])[0] == dr.propose(pend[s], 7, None, 0.0)


def test_dflash_streams_equal_alone_four_ranks():
    reqs = _requests()

    def run(rank, comm):
        w = _weights(rank, 4, comm)
        dr = _drafter(w)
        solo, stats = _solo(w, dr, reqs, graphs=False)
        res = _multi(rank, 4, comm, w, dr, reqs, graphs=False)
        return solo, stats, res

    results = run_ranks(run, 4)
    solo, stats, res = results[0]
    for r in range(1, 4):
        assert results[r][0] == solo, "ranks disagree on the lone replies"
    _check(solo, stats, res)


def test_dflash_streams_equal_alone_graphs():
    """CUDA graphs on both sides (one thread as rank 0 of four): the drafter's passes and every window replayed."""
    reqs = _requests()
    comm = base._Alone()
    with torch.no_grad():
        w = _weights(0, 4, comm)
        dr = _drafter(w)
        dr.capture()
        solo, stats = _solo(w, dr, reqs, graphs=True)
        res = _multi(0, 1, comm, w, dr, reqs, graphs=True)
    assert len(set(map(tuple, solo.values()))) == len(solo), "degenerate replies: the check would be weak"
    _check(solo, stats, res)


def _engine_serve(rank, comm, reqs):
    import threading

    from tensorfold.families.glm_moe_dsa.cuda.engine import Glm53Engine

    e = Glm53Engine(base.CKPT, rank=rank, master="", port=0, context=CAPACITY, comm=comm, layers=len(base.LAYERS),
                    mtp_drafts=K, parallel=3)
    if rank:
        e.follow()
        return None
    names = ["a", "b", "f"]
    together: dict = {}

    def one(name, box):
        prompt, n, s, mode = reqs[name]
        got = []
        st = e.generate(prompt, n, s, got.extend, stop_eos=False, mtp_mode=mode)
        box[name] = (got, st)

    threads = [threading.Thread(target=one, args=(nm, together)) for nm in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    alone: dict = {}
    for nm in names:
        one(nm, alone)
    widest = e.multi.widest
    e.close()
    return together, alone, widest


def test_engine_parallel_dflash(monkeypatch):
    """Glm53Engine(parallel=3) with TF_GLM53_DFLASH: "dflash" / "auto" requests from three threads reply exactly as
    each does alone."""
    if os.environ.get("TF_GLM53_GRAPHS", "1") != "0":
        pytest.skip("thread ranks cannot capture graphs: run with TF_GLM53_GRAPHS=0")
    monkeypatch.setenv("TF_GLM53_DFLASH", DFLASH)
    reqs = _requests()
    together, alone, widest = run_ranks(lambda r, c: _engine_serve(r, c, reqs), 4)[0]
    for nm, (got, st) in together.items():
        assert got == alone[nm][0], nm
        assert len(got) == reqs[nm][1]
        assert st["sha256"] == alone[nm][1]["sha256"]
    assert widest > 16, widest
