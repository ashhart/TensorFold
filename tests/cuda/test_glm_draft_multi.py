"""DFlash2 for several streams (dflash2_multi) against one solo Drafter per stream, bit for bit: 1-4 streams at
different context lengths (the ring wrapped, flat buffers too) in batched block passes and batched context updates,
eager and in CUDA graphs, and two ranks (threads) eager; the 4-bit matmuls' rows independent of the row count at the
synthetic and the real drafter's shapes; batched vs sequential timings (printed with -s).

Small: the one-layer synthetic drafter of test_glm_engine, a few MiB of caches; the real-shape matmuls use random
weights, one matrix at a time."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_glm_engine import DRAFT, D, V, _drafter  # noqa: E402

# per-rank (N, K) of the real drafter (hidden 4096, 32 / 8 heads of 128, MLP 12,288, 5 taps; two ranks, then one)
# and of a vocabulary head's slice, plus the synthetic drafter's
REAL = [(1024, 4096), (3072, 4096), (1024, 4096), (4096, 2048), (12288, 4096), (4096, 6144), (4096, 20480),
        (16384, 4096), (6144, 4096), (2048, 4096), (4096, 4096), (24576, 4096), (4096, 12288)]
SYNTH = [(128, 512), (1536, 512), (512, 512), (512, 1024), (512, 256), (1024, 512), (V, 512)]


@pytest.mark.parametrize("n,k", REAL + SYNTH)
def test_q4_rows_do_not_depend_on_the_row_count(n, k):
    """A block pass's matmuls take 8 rows alone, S x 8 batched (to 32); updates up to 64 stacked rows; the head 7
    rows alone, S x 7 batched: every row's bits (bf16 and fp32 outputs) are the same at every row count."""

    from tensorfold.families.glm5_next.cuda import qmm

    g = torch.Generator(device="cuda").manual_seed(n * 7 + k)
    q = qmm.quantize4((torch.randn((n, k), generator=g, device="cuda") * 0.03).bfloat16())
    x = torch.randn((64, k), generator=g, device="cuda").bfloat16()
    for f32 in (False, True):
        full = qmm.matmul(x, q, f32=f32)
        for r in (1, 7, 8, 9, 14, 16, 17, 21, 24, 28, 32, 33, 40, 48, 56, 63):
            assert torch.equal(qmm.matmul(x[:r], q, f32=f32), full[:r]), (r, f32)
        for a, b in ((8, 16), (16, 24), (24, 32), (7, 14), (21, 28)):   # a later block alone
            assert torch.equal(qmm.matmul(x[a:b].clone(), q, f32=f32), full[a:b]), (a, b, f32)
    del q, x
    torch.cuda.empty_cache()


def _weights():
    from tensorfold.families.glm5_next.cuda import qmm

    g = torch.Generator(device="cuda").manual_seed(5)
    embed = (torch.randn((V, D), generator=g, device="cuda") * 0.05).bfloat16()
    head = qmm.quantize4((torch.randn((V, D), generator=g, device="cuda") * 0.03).bfloat16())
    return SimpleNamespace(device=torch.device("cuda"), rank=0, world=1, comm=None, embed=embed, head=head,
                           draft_head=None, vocab_offset=0)


STREAMS = 4


@pytest.fixture(scope="module", params=[(2048, True), (49, True), (2048, False)],
                ids=["window2047-ring", "window48-ring", "window2047-flat"])
def rig(request, tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.dflash2 import Drafter
    from tensorfold.families.glm5_next.cuda.dflash2_multi import MultiDrafter

    window, ring = request.param
    path = tmp_path_factory.mktemp(f"dmulti{window}{ring}")
    _drafter(path)
    (path / "config.json").write_text(json.dumps(dict(DRAFT, sliding_window=window)))
    w = _weights()
    solos = [Drafter(path, w, capacity=9000, ring=ring) for _ in range(STREAMS)]
    base = Drafter(path, w, capacity=9000, ring=ring)
    for s in solos:                          # the same weights (quantized alike): the drafts' only other input
        assert torch.equal(s.fc.weight, base.fc.weight) and torch.equal(s.layers[0].gu.weight, base.layers[0].gu.weight)
    multi = MultiDrafter(base, streams=STREAMS)
    assert multi.t_max == 64 and multi.cap == base.cap
    yield solos, multi
    del solos, multi, base
    torch.cuda.empty_cache()


def _drive(rig, seed: int, rounds: int = 60) -> int:
    """Four streams prompted to different lengths (one past the ring several times, one short, one alone in a batch
    commit of every prompt), then rounds of random stream subsets in random order: each stream's candidates, logits
    and selector rows, and its sampled / greedy chains, equal to its solo drafter's; kept rows committed batched on
    one side, one stream at a time on the other. A kept state taken, run past and restored on the way."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm5_next.cuda import decode
    from tensorfold.families.glm5_next.cuda.dflash2_multi import DraftRequest

    solos, multi = rig
    ctxs = multi.contexts
    rng = np.random.default_rng(seed)
    width = solos[0].tap_in.shape[1]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    compared = 0

    def taps(n: int) -> torch.Tensor:
        return (torch.randn((n, width), generator=gen, device="cuda") * 0.5).bfloat16()

    def check_ends() -> None:
        for s, c in zip(solos, ctxs):
            assert s.context_end == c.context_end
            assert int(c.pos_dev.item()) == c.context_end == int(s.pos_dev.item())

    for s in solos:
        s.reset()
    multi.reset()
    # prompts: stream 0 3,481 rows in chunks, stream 1 5,300 (the ring wraps twice), stream 2 17, stream 3 700;
    # streams 0 and 1 one at a time (ctx.add_taps), then all four in one batched commit
    for i, chunks in ((0, (300, 64, 1000, 17, 2100)), (1, (2048, 2048, 1204))):
        for n in chunks:
            t = taps(n)
            solos[i].add_taps(t)
            ctxs[i].add_taps(t)
    items = [(ctxs[i], taps(n)) for i, n in ((2, 17), (3, 700), (0, 5), (1, 90))]
    for c, t in items:
        solos[c.slot].add_taps(t)
    multi.commit(items)
    check_ends()

    def one_round() -> None:
        nonlocal compared
        S = int(rng.integers(1, STREAMS + 1))
        pick = [int(i) for i in rng.permutation(STREAMS)[:S]]
        pend = [int(rng.integers(0, 1000)) for _ in pick]
        depth = [int(rng.integers(1, solos[0].block)) for _ in pick]
        got = multi.candidates([(ctxs[i], p, d) for i, p, d in zip(pick, pend, depth)])
        for i, p, d, g in zip(pick, pend, depth, got):
            want = solos[i].candidates(p, d)
            assert all(np.array_equal(a, b) for a, b in zip(want, g)), (i, solos[i].context_end, compared)
            compared += 1
        # chains through propose: sampled (keyed noise) or greedy, with or without a stop rule
        reqs, want = [], []
        for i, p in zip(pick, pend):
            samp = Sampling(seed=int(rng.integers(1 << 30)), temperature=1.0) if rng.random() < 0.7 else None
            conf = float(rng.choice([0.0, 0.05, 0.3]))
            reqs.append(DraftRequest(ctxs[i], p, 7, samp, conf))
            want.append(solos[i].propose(p, 7, samp, conf))
        assert multi.propose(reqs) == want
        # kept rows: the drafted streams and sometimes another, 1..8 rows each
        keep = pick + [int(i) for i in range(STREAMS) if i not in pick and rng.random() < 0.3]
        items = [(ctxs[i], taps(int(rng.integers(1, 9)))) for i in keep]
        for c, t in items:
            solos[c.slot].add_taps(t)
        multi.commit(items)
        check_ends()

    for _ in range(rounds):
        one_round()
    # a kept state of stream 1 (ring: its window copied), the stream run on past its ring, then restored
    e = SimpleNamespace(st=SimpleNamespace(cur=[], rec=[torch.zeros(1, device="cuda")],
                                           conv=torch.zeros(1, device="cuda"), mtp_len=0, mtp_drafted=0,
                                           set_pos=lambda n: None, set_mtp_len=lambda n: None))
    n1 = solos[1].context_end
    snaps = [decode.take_snapshot(e, list(range(n1)), None, mtp=False, drafter=d) for d in (solos[1], ctxs[1])]
    t = taps(2500)
    solos[1].add_taps(t)
    multi.commit([(ctxs[1], t)])
    for _ in range(rounds // 3):
        one_round()
    extra = ctxs[1].context_end - n1
    for d, s in zip((solos[1], ctxs[1]), snaps):
        decode.restore(e, s, d)
    check_ends()
    assert extra > 2500
    for _ in range(rounds):
        one_round()
    return compared


def test_streams_draft_their_solo_bits_eager(rig):
    assert _drive(rig, 1) > 200


def test_streams_draft_their_solo_bits_in_cuda_graphs(rig):
    solos, multi = rig
    for s in solos:
        s.capture()
    multi.capture()
    assert sorted(multi.block_graphs) == [1, 2, 3, 4] and sorted(multi.tap_graphs) == list(range(8, 65, 8))
    try:
        assert _drive(rig, 2) > 200
    finally:
        for s in solos:
            s.block_graph, s.tap_graphs = None, {}
        multi.drop_graphs()


def test_batched_vs_sequential_timing(rig):
    """Timings only (printed): one batched block pass of S streams vs S solo passes, eager and graphs."""

    solos, multi = rig
    for s in solos:
        s.reset()
    multi.reset()
    t = (torch.randn((3000, solos[0].tap_in.shape[1]), device="cuda") * 0.5).bfloat16()
    for s, c in zip(solos, multi.contexts):
        s.add_taps(t)
        c.add_taps(t)

    def clock(fn, reps=30) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    rows = []
    for graphs in (False, True):
        if graphs:
            for s in solos:
                s.capture()
            multi.capture()
        for S in range(1, STREAMS + 1):
            seq = clock(lambda: [solos[i].candidates(5, 7) for i in range(S)])
            bat = clock(lambda: multi.candidates([(multi.contexts[i], 5, 7) for i in range(S)]))
            rows.append(("graphs" if graphs else "eager", S, seq, bat))
    for s in solos:
        s.block_graph, s.tap_graphs = None, {}
    multi.drop_graphs()
    print("\nblock pass ms (synthetic drafter): mode S sequential batched")
    for r in rows:
        print(f"  {r[0]:6s} {r[1]} {r[2]:7.3f} {r[3]:7.3f}")


def test_two_ranks_draft_their_solo_bits(tmp_path):
    """Two ranks as threads of one process (test_flashnext_tp's thread all-gather), eager: the heads, MLP and
    vocabulary split over the ranks, row-parallel partials and candidates all-gathered in batched passes; each rank's
    merged candidates equal its solo drafters', and the two ranks' equal each other."""

    import threading

    from test_flashnext_tp import _Hub, _ThreadComm

    from tensorfold.families.glm5_next.cuda import qmm
    from tensorfold.families.glm5_next.cuda.dflash2 import Drafter
    from tensorfold.families.glm5_next.cuda.dflash2_multi import MultiDrafter

    _drafter(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(DRAFT))
    g = torch.Generator(device="cuda").manual_seed(5)
    embed = (torch.randn((V, D), generator=g, device="cuda") * 0.05).bfloat16()
    head = (torch.randn((V, D), generator=g, device="cuda") * 0.03).bfloat16()
    hub = _Hub(2)
    ranks = []
    for r in range(2):
        w = SimpleNamespace(device=torch.device("cuda"), rank=r, world=2, comm=_ThreadComm(hub, r), embed=embed,
                            head=qmm.quantize4(head[r * V // 2:(r + 1) * V // 2].contiguous()), draft_head=None,
                            vocab_offset=r * V // 2)
        solos = [Drafter(tmp_path, w, capacity=6000, ring=True) for _ in range(3)]
        multi = MultiDrafter(Drafter(tmp_path, w, capacity=6000, ring=True), streams=3)
        assert multi.gathered == 2 and multi.d.heads == 4
        ranks.append((solos, multi))
    results: list = [None, None]
    errors: list = []

    def body(r: int) -> None:
        try:
            with torch.no_grad():
                solos, multi = ranks[r]
                rng = np.random.default_rng(7)             # the same draws on both ranks
                gen = torch.Generator(device="cuda").manual_seed(7)
                width = solos[0].tap_in.shape[1]
                seen = []
                for i, n in enumerate((2500, 300, 40)):
                    t = (torch.randn((n, width), generator=gen, device="cuda") * 0.5).bfloat16()
                    solos[i].add_taps(t)
                    multi.contexts[i].add_taps(t)
                for _ in range(25):
                    S = int(rng.integers(1, 4))
                    pick = [int(i) for i in rng.permutation(3)[:S]]
                    pend = [int(rng.integers(0, V)) for _ in pick]
                    got = multi.candidates([(multi.contexts[i], p, 7) for i, p in zip(pick, pend)])
                    for i, p, c in zip(pick, pend, got):
                        want = solos[i].candidates(p, 7)
                        assert all(np.array_equal(a, b) for a, b in zip(want, c)), (r, i)
                        seen.append(c)
                    items = []
                    for i in pick:
                        t = (torch.randn((int(rng.integers(1, 9)), width), generator=gen, device="cuda") * 0.5
                             ).bfloat16()
                        solos[i].add_taps(t)
                        items.append((multi.contexts[i], t))
                    multi.commit(items)
                results[r] = seen
        except BaseException as exc:        # noqa: BLE001
            errors.append(exc)
            hub.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    assert len(results[0]) > 40
    for a, b in zip(*results):
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
