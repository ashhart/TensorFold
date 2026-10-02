"""Full GLM-5.3 concurrent streams (multi.GlmMultiDecoder, Phase A) on real weights (TF_GLM53_CKPT), layers 0-3 + the
MTP layer: every concurrent stream's reply - greedy, and sampled with fixed seeds - is token-identical to the same
request run alone through the one-stream Runner, for 2, 3 and 4 streams with different prompt lengths (one past
index_topk, so sparse rows share windows with dense ones) and staggered arrival; many admit/finish cycles leave the
GPU memory where it was.

Four ranks as threads of one GPU (eager: the thread communicator cannot be captured), and one rank with CUDA graphs
(the per-row tables copied in before each replay).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

CKPT = os.environ.get("TF_GLM53_CKPT", "")
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint")]

from threadcomm import run_ranks  # noqa: E402

LAYERS = (0, 1, 2, 3)
CAPACITY = 2400                      # per stream: room for a prompt past index_topk (2048) and its reply
K = 2                                # MTP drafts: 4 streams x 3 rows = 12 rows a window


def _weights(rank, world, comm):
    from tensorfold.families.glm_moe_dsa.config import Config
    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.weights import RankReader, load_layer, load_mtp

    cfg = Config.from_dict(json.loads((Path(CKPT) / "config.json").read_text()))
    r = RankReader(CKPT, rank, world)
    embed, norm, head = (r.get(t, "cuda") for t in ("model.embed_tokens.weight", "model.norm.weight",
                                                     "lm_head.weight"))
    w = fused.Weights(cfg, rank, world, comm, embed, norm, head, [load_layer(r, cfg, i) for i in LAYERS],
                      load_mtp(r, cfg))
    return w


def _tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 150000, (n,), generator=g).tolist()


def _sample_fn():
    from tensorfold.families.glm_moe_dsa.cuda.engine import Glm53Engine

    return lambda lg, pos, s: Glm53Engine._sample(None, lg, pos, s)


def _requests():
    """(name, prompt, tokens, sampling): different lengths, one past index_topk; greedy and seeded sampled."""
    from tensorfold.engine.exact_sampling import Sampling

    return {
        "a": (_tokens(40, 1), 28, None),
        "b": (_tokens(150, 2), 24, Sampling(99, 0.9, 20, 0.95, 0.0)),
        "c": (_tokens(700, 3), 20, None),
        "d": (_tokens(2100, 4), 22, Sampling(1234, 0.8, 20, 0.95, 0.0)),
        "e": (_tokens(9, 5), 30, Sampling(7, 1.0, 0, 1.0, 0.0)),
    }


# scenarios: (request, round it arrives at)
SCENARIOS = [
    [("a", 0), ("b", 3)],
    [("c", 0), ("e", 1), ("a", 5)],
    [("d", 0), ("b", 0), ("e", 2), ("c", 4)],
]


def _solo(w, reqs, graphs):
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner

    sample = _sample_fn()
    run = Runner(w, CAPACITY + K + 1, K, graphs=graphs)
    out = {}
    for name, (prompt, n, s) in reqs.items():
        fn = None if s is None else (lambda lg, pos, s=s: sample(lg, pos, s))
        st = run.generate(prompt, n, fn, lambda t: False, lambda new: None, K, None, sampling=s)
        out[name] = st["out"]
    del run
    torch.cuda.empty_cache()
    return out


def _drive(dec, reqs, scenario):
    """Rank 0: admit each request at its round, run rounds until all finish; returns each request's tokens."""
    from tensorfold.cuda.streams import Stream

    got = {name: [] for name, _ in scenario}
    arrivals = sorted(scenario, key=lambda x: x[1])
    rnd, i = 0, 0
    while i < len(arrivals) or dec.live():
        while i < len(arrivals) and arrivals[i][1] <= rnd:
            name = arrivals[i][0]
            prompt, n, s = reqs[name]
            st = Stream(list(prompt), n, s, draft=True, stop_eos=False)
            st.emit = lambda new, name=name: got[name].extend(new)
            dec.admit(st)
            i += 1
        dec.finish(dec.round())
        rnd += 1
    return got


def _cycles(dec, count=24, live=4):
    """Many short requests through the slots (admit, a few rounds, finish)."""
    from tensorfold.cuda.streams import Stream

    pending = [(_tokens(5 + 13 * j % 60, 100 + j), 3 + j % 5) for j in range(count)]
    while pending or dec.live():
        while pending and dec.live() < live:
            prompt, n = pending.pop(0)
            st = Stream(prompt, n, None, draft=True, stop_eos=False)
            st.emit = lambda new: None
            dec.admit(st)
        dec.finish(dec.round())


def _multi(rank, world, comm, w, reqs, graphs):
    from tensorfold.families.glm_moe_dsa.cuda.multi import GlmMultiDecoder
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner

    run = Runner(w, CAPACITY + K + 1, K, graphs=graphs, slots=4)
    dec = GlmMultiDecoder(run, rank=rank, world=world, comm=comm, limit=CAPACITY, eos=tuple(w.cfg.eos_token_ids),
                          sample=_sample_fn())
    if graphs:
        dec.prewarm()                                   # the engine's startup captures (scratch cache values)

    def phase(work):
        """Rank 0 drives ``work`` while the others follow; then every rank stops at a barrier, quiet."""
        out = None
        if rank:
            dec.follow()
        else:
            out = work()
            dec.stop()
        torch.cuda.synchronize()
        comm.barrier()
        return out

    outs = phase(lambda: [_drive(dec, reqs, sc) for sc in SCENARIOS])
    phase(lambda: _cycles(dec))                         # every shape the cycles use (graphs, allocator)
    before = torch.cuda.memory_allocated()
    comm.barrier()
    phase(lambda: _cycles(dec))
    after = torch.cuda.memory_allocated()
    comm.barrier()
    return None if rank else (outs, before, after, dec.free)


def _check(solo, res):
    outs, before, after, free = res
    for sc, got in zip(SCENARIOS, outs):
        for name, _ in sc:
            assert got[name] == solo[name], f"{len(sc)} streams, request {name}: {got[name]} != alone {solo[name]}"
    assert after <= before, f"GPU memory grew over admit/finish cycles: {before} -> {after}"
    assert sorted(free) == [0, 1, 2, 3], free


def test_concurrent_streams_equal_alone_four_ranks():
    reqs = _requests()

    def run(rank, comm):
        w = _weights(rank, 4, comm)
        solo = _solo(w, reqs, graphs=False)
        res = _multi(rank, 4, comm, w, reqs, graphs=False)
        return solo, res

    results = run_ranks(run, 4)
    solo, res = results[0]
    for r in range(1, 4):
        assert results[r][0] == solo, "ranks disagree on the lone replies"
    _check(solo, res)


class _Alone:
    """One thread as rank 0 of four, the other ranks' partials zero: device copies only, so graphs capture it (the
    model is rank 0's quarter of each layer - not GLM-5.3's numbers, but the same code paths on both sides)."""
    world, rank = 4, 0

    def all_gather(self, send, recv):
        r = recv.view(self.world, -1)
        r.zero_()
        r[0].copy_(send.reshape(-1))

    def all_to_all(self, send, recv):
        recv.zero_()
        recv[0].copy_(send[0])

    def barrier(self):
        pass


def test_concurrent_streams_equal_alone_graphs():
    """CUDA graphs on both sides (one thread, rank 0's quarter): replays with the per-row tables copied in give each
    stream its lone reply."""
    reqs = _requests()
    comm = _Alone()
    with torch.no_grad():
        w = _weights(0, 4, comm)
        solo = _solo(w, reqs, graphs=True)
        res = _multi(0, 1, comm, w, reqs, graphs=True)
    assert len(set(map(tuple, solo.values()))) == len(solo), "degenerate replies: the check would be weak"
    _check(solo, res)


def _engine_serve(rank, comm, reqs):
    import threading

    from tensorfold.families.glm_moe_dsa.cuda.engine import Glm53Engine

    e = Glm53Engine(CKPT, rank=rank, master="", port=0, context=CAPACITY, comm=comm, layers=len(LAYERS),
                    mtp_drafts=K, parallel=3)
    if rank:
        e.follow()                                   # until rank 0's close()
        return None
    names = ["a", "b", "e"]
    together: dict = {}

    def one(name, box):
        prompt, n, s = reqs[name]
        got = []
        st = e.generate(prompt, n, s, got.extend, stop_eos=False)
        box[name] = (got, st)

    threads = [threading.Thread(target=one, args=(nm, together)) for nm in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    alone: dict = {}
    for nm in names:
        one(nm, alone)
    e.close()
    return together, alone


def test_engine_parallel_scheduler():
    """Glm53Engine(parallel=3): three requests from three threads at once (the Scheduler's rounds, followers on
    ranks 1-3) reply exactly as each does alone; stats carry the reply hash."""
    if os.environ.get("TF_GLM53_GRAPHS", "1") != "0":
        pytest.skip("thread ranks cannot capture graphs: run with TF_GLM53_GRAPHS=0")
    reqs = _requests()
    together, alone = run_ranks(lambda r, c: _engine_serve(r, c, reqs), 4)[0]
    for nm, (got, st) in together.items():
        assert got == alone[nm][0], nm
        assert len(got) == reqs[nm][1]
        assert st["sha256"] == alone[nm][1]["sha256"]


# ----------------------------------------------------------------------------------- fills between layers ---
FILL_ROWS = 512                      # prompt chunks here: the 2100-token prompt in 5 chunks, the 1200 one in 3
ILV = [("p", 0), ("q", 0), ("long", 2), ("mid", 3)]   # two prompts arrive while two streams decode


def _ilv_requests():
    from tensorfold.engine.exact_sampling import Sampling

    return {
        "long": (_tokens(2100, 21), 10, None),
        "mid": (_tokens(1200, 22), 8, Sampling(5, 0.8, 20, 0.95, 0.0)),
        "p": (_tokens(30, 23), 60, None),
        "q": (_tokens(64, 24), 60, Sampling(11, 0.9, 20, 0.95, 0.0)),
    }


def _ilv(rank, world, comm, w, reqs, graphs, fill_layers):
    """Lone replies (chunks of FILL_ROWS), then the ILV scenario with fills fill_layers layers a step."""
    from tensorfold.families.glm_moe_dsa.cuda.multi import GlmMultiDecoder
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner

    sample = _sample_fn()
    solo = {}
    if rank == 0 or not graphs:
        run = Runner(w, CAPACITY + K + 1, K, graphs=graphs)
        run.prompt_rows = FILL_ROWS
        for name, (prompt, n, s) in reqs.items():
            fn = None if s is None else (lambda lg, pos, s=s: sample(lg, pos, s))
            solo[name] = run.generate(prompt, n, fn, lambda t: False, lambda new: None, K, None, sampling=s)["out"]
        del run
        torch.cuda.empty_cache()
    run = Runner(w, CAPACITY + K + 1, K, graphs=graphs, slots=4)
    run.prompt_rows = FILL_ROWS
    dec = GlmMultiDecoder(run, rank=rank, world=world, comm=comm, limit=CAPACITY, eos=tuple(w.cfg.eos_token_ids),
                          sample=sample)
    dec.fill_layers = fill_layers
    got = None
    if rank:
        dec.follow()
    else:
        got = _drive(dec, reqs, ILV)
        dec.stop()
    torch.cuda.synchronize()
    comm.barrier()
    return solo, got, dec.mid_rounds, run.overlap


def _ilv_check(solo, got, mid, overlap, want_overlap, least):
    assert overlap == want_overlap, "the fill path under test did not run"
    for name, _ in ILV:
        assert got[name] == solo[name], f"request {name} filled between decode rounds: {got[name]} != {solo[name]}"
    assert mid >= least, f"only {mid} decode rounds ran while a prompt chunk was paused between layers"


class _Reduce:
    """A thread communicator with an all-reduce (rank-order bf16 sum), so prompt chunks take the two-micro-batch path
    whose reductions overlap compute on a comm stream (and are in flight when a fill pauses)."""

    def __init__(self, comm):
        self.c, self.rank, self.world = comm, comm.rank, comm.world

    def all_gather(self, send, recv):
        self.c.all_gather(send, recv)

    def all_to_all(self, send, recv):
        self.c.all_to_all(send, recv)

    def barrier(self):
        self.c.barrier()

    def all_reduce(self, send, recv):
        slots = self.c._exchange(send)
        recv.copy_(slots[0])
        for r in range(1, self.world):
            recv.add_(slots[r])
        self.c._done()


def test_fill_between_layers_four_ranks():
    """Four ranks, followers mirroring FILL layer ranges, chunks as two overlapped micro-batches paused one layer at
    a time while two streams decode: every reply equals its lone reply."""
    reqs = _ilv_requests()

    def run(rank, comm):
        c = _Reduce(comm)
        return _ilv(rank, 4, c, _weights(rank, 4, c), reqs, graphs=False, fill_layers=1)

    results = run_ranks(run, 4)
    solo, got, mid, overlap = results[0]
    for r in range(1, 4):
        assert results[r][0] == solo, "ranks disagree on the lone replies"
    _ilv_check(solo, got, mid, overlap, True, 10)


def test_fill_between_layers_graphs():
    """One thread (rank 0's quarter), decode rounds replayed as CUDA graphs between fill steps of two layers."""
    reqs = _ilv_requests()
    with torch.no_grad():
        comm = _Alone()
        solo, got, mid, overlap = _ilv(0, 1, comm, _weights(0, 4, comm), reqs, graphs=True, fill_layers=2)
    assert len(set(map(tuple, solo.values()))) == len(solo), "degenerate replies: the check would be weak"
    _ilv_check(solo, got, mid, overlap, False, 5)


def test_fill_between_layers_sp_four_ranks(monkeypatch):
    """As test_fill_between_layers_four_ranks with sequence-parallel prompt chunks (TF_GLM53_PROMPT_SP=1): the
    paused chunks run compute_prompt_sp one layer a step, lone replies use it whole; every reply equals its lone
    reply."""
    from tensorfold.families.glm_moe_dsa.cuda import fused

    monkeypatch.setattr(fused, "PROMPT_SP", True)
    real = fused.compute_prompt_sp
    calls = {"whole": 0, "paused": 0}

    def spy(*a, **k):
        calls["whole" if k.get("layers") is None else "paused"] += 1
        return real(*a, **k)

    monkeypatch.setattr(fused, "compute_prompt_sp", spy)
    reqs = _ilv_requests()

    def run(rank, comm):
        c = _Reduce(comm)
        return _ilv(rank, 4, c, _weights(rank, 4, c), reqs, graphs=False, fill_layers=1)

    results = run_ranks(run, 4)
    solo, got, mid, overlap = results[0]
    for r in range(1, 4):
        assert results[r][0] == solo, "ranks disagree on the lone replies"
    assert calls["whole"] and calls["paused"], f"sequence-parallel chunks did not run whole and paused: {calls}"
    _ilv_check(solo, got, mid, overlap, True, 10)
