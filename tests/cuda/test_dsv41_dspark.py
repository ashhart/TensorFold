"""DSpark drafting: block logits within the model-level gate of the reference's DSpark driver fed the same taps (tiny
checkpoint and the pack's mtp stages), greedy drafts agreeing with the reference's on real weights, two thread ranks
drafting alike, absorb writing committed positions only, the Markov chain stopping at d, and graph replays equal to
eager runs.

Top-1 counts only rows the reference decides between its fp32 and mirror modes (as the forward's test does), and a row
those modes route apart is held to the nearer. The pack's check needs ``TF_DSV41_MODEL``; its taps come from the whole
backbone, loaded a few layers at a time.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_tiny
from dsv41_pair import pair, run_pair
from dsv41_ref_weights import RefWeights
from dsv41_reference import Mode, State

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import BLOCK, MAX_ROWS, buffers, dspark, engram, loader
from tensorfold.families.deepseek_v41.cuda import forward as F
from tensorfold.families.deepseek_v41.engram_table import Reader
from tensorfold.families.glm5_next.cuda import glue

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
MODEL = os.environ.get("TF_DSV41_MODEL", "")
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="set TF_DSV41_MODEL to the checkpoint")
CAP = 1024
PROMPT_ROWS = 256
TOP1, KL, REL_L2 = 0.99, 1e-3, 0.05     # the model-level gate, reduced model
DECIDED = 0.25                          # least share of rows whose top-1 the reference decides
AGREE = 0.95                            # greedy drafts equal to the reference's on real weights
# prompt chunks (ints) then decode rounds (rows, keep); every commit is absorbed and followed by a block
PLAN = [100, 60, (1, 1), (6, 3), (4, 4), (2, 1), (5, 5), (3, 2), (6, 6), (6, 1), 40, (5, 4), (3, 3)]
SAMPLINGS = [None, Sampling(seed=3, temperature=1.0, top_k=0, top_p=1.0),
             Sampling(seed=5, temperature=1.0, top_k=20, top_p=0.95)]
TEXT = ("The river runs through a wide valley before it reaches the sea. In spring the snow on the mountains melts "
        "and the water rises quickly, carrying soil and stones down to the plains. Farmers along its banks have "
        "learned to plant late, after the flood has passed, and to build their houses on higher ground. Over many "
        "centuries the river has changed its course several times, leaving behind curved lakes and wide fields of "
        "rich earth. Today a series of dams controls the flow, producing electricity for the cities and storing "
        "water for the dry summer months. Engineers measure the level every hour and release water when the "
        "reservoirs are full. Fish that once swam upstream to lay their eggs now climb special ladders built beside "
        "each dam. Scientists count them every year to see whether the population is recovering, and the numbers "
        "have slowly improved since the ladders were added. Visitors come to watch the fish jump, to walk along the "
        "paths beside the water, and to learn how a river can be both a source of danger and a source of life.")


class Eng:
    """One rank's sequence and drafter: weights, state, prompt-chunk and decode buffers, DSpark scratch."""

    def __init__(self, w, hasher=None, reader=None, prompt_rows: int = PROMPT_ROWS) -> None:
        self.w, self.hasher, self.reader = w, hasher, reader
        self.st = buffers.State(w.cfg, CAP, "cuda")
        self.pbuf = buffers.Buffers(w.cfg, w.world, prompt_rows, CAP, prefill=True, device="cuda")
        self.dbuf = buffers.Buffers(w.cfg, w.world, MAX_ROWS, CAP, device="cuda")
        self.dwork = dspark.Work(w.cfg, w.world, "cuda")

    def forward(self, ids, prompt: bool, keep: int) -> tuple[buffers.Buffers, int]:
        """A target forward of ``ids`` keeping ``keep`` rows -> (its buffers, tap rows of committed positions)."""

        b = self.pbuf if prompt else self.dbuf
        R = F.stage(self.w, self.st, b, ids, self.hasher, self.reader)
        F.compute(self.w, self.st, b, R, prompt=prompt, head_rows=0 if prompt else R)
        F.commit(self.w, self.st, b, R, keep)
        return b, min(keep, b.taps.shape[0])


def _drive(e: Eng, plan, seed: int, sampling=None, d: int = BLOCK, threshold=None) -> list[tuple]:
    """``plan``'s forwards, each commit absorbed, then a block for a random pending token -> events
    ("absorb", taps, start) and ("propose", y, p, drafts, confidences, block logits)."""

    g = torch.Generator().manual_seed(seed)
    vocab, events = e.w.cfg.vocab_size, []
    e.st.reset()
    for step in plan:
        prompt = isinstance(step, int)
        rows, keep = (step, step) if prompt else step
        b, n = e.forward(torch.randint(2, vocab, (rows,), generator=g).tolist(), prompt, keep)
        events.append(("absorb", b.taps[:n].clone(), e.st.pos - n))
        dspark.absorb(e, b, n, prompt=prompt)
        y, p = int(torch.randint(2, vocab, (1,), generator=g)), e.st.pos - 1
        drafts, conf = dspark.propose(e, y, p, sampling, d, threshold)
        events.append(("propose", y, p, drafts, conf, e.dbuf.dlog.clone()))
    return events


def _replay(ref: RefWeights, mode: Mode, events) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """The reference's DSpark on the same taps and pending tokens, its Markov chain fed TF's drafts -> (block
    logits, confidences, its own greedy pick of each row)."""

    r = ref.reference(mode, State())
    logits, confs = [], []
    for ev in events:
        if ev[0] == "absorb":
            r.absorb(ev[1].float().cpu().flatten(1), ev[2])
            continue
        _, y, p, drafts, _, _ = ev
        r.state.pos = p + 1
        _, lg, cf = r.propose(y, lambda row, at, p=p, drafts=drafts: drafts[at - p - 2])
        logits.append(lg)
        confs.append(cf)
    logits = torch.cat(logits)
    return logits, torch.cat(confs), logits.argmax(-1).tolist()


def _ours(events) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    props = [ev for ev in events if ev[0] == "propose"]
    return (torch.cat([ev[5].cpu() for ev in props]), torch.tensor([c for ev in props for c in ev[4]]),
            [t for ev in props for t in ev[3]])


def _kl(p: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((p * (torch.log_softmax(x, -1) - torch.log_softmax(y, -1))).sum(-1).mean())


def _t2(got: torch.Tensor, mirror: torch.Tensor, fp32: torch.Tensor, what: str) -> torch.Tensor:
    """The model-level gate against the reference in mirror mode, its fp32 mode giving its own spread: top-1 counts only
    rows it decides (margin above twice that spread), and a row whose two modes are farther apart than the gate (a
    routing flip) is held to the nearer one -> the rows held to fp32."""

    g, x, f = got.double(), mirror.double(), fp32.double()
    flip = (f - x).norm(dim=-1) > REL_L2 * x.norm(dim=-1)
    held = flip & ((g - f).norm(dim=-1) < (g - x).norm(dim=-1))
    spread = 2 * (x - f).abs().amax(-1)
    x = torch.where(held[:, None], f, x)
    pick, top = g.argmax(-1), x.argmax(-1)
    near = (x.gather(-1, top[:, None]) - x.gather(-1, pick[:, None]))[:, 0] <= spread
    gap = x.topk(2, -1).values
    decided = float((gap[:, 0] - gap[:, 1] > spread).double().mean())
    top1 = float(((pick == top) | near).double().mean())
    kl, rel = _kl(torch.softmax(x, -1), x, g), float((g - x).norm() / x.norm())
    print(f"{what}: top-1 {float((pick == top).double().mean()):.4f} ({top1:.4f} counted, {decided:.2f} of rows "
          f"decided), mean KL {kl:.3g}, rel-L2 {rel:.3g}; {int(flip.sum())} rows the modes route apart, "
          f"{int(held.sum())} held to fp32")
    assert decided >= DECIDED, f"{what}: only {decided} of rows decided"
    assert top1 >= TOP1 and kl <= KL and rel <= REL_L2, f"{what}: top-1 {top1}, KL {kl}, rel-L2 {rel}"
    return held


def _confidence(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    rel = float((got.double() - want.double()).norm() / want.double().norm())
    print(f"{what}: confidence rel-L2 {rel:.3g}")
    assert rel <= REL_L2, f"{what}: confidence rel-L2 {rel}"


@pytest.fixture(scope="module")
def ref(tiny_dir):
    return RefWeights(tiny_dir)


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    return loader.load(tiny_dir, Config.read(tiny_dir), 0, 1, None, capacity=CAP)


@pytest.fixture(scope="module")
def eng(tiny, ref):
    return Eng(tiny, ref.hasher, ref.reader)


def test_policy_counts_drafts():
    logit = math.log(0.9 / 0.1)                         # sigmoid 0.9
    assert dspark.policy([5.0] * 5, 3) == 3 and dspark.policy([], 4) == 4
    assert dspark.policy([logit] * 5, 5, 0.8) == 2      # 0.9, 0.81, 0.729
    assert dspark.policy([logit] * 5, 5, 0.7) == 3
    assert dspark.policy([logit] * 5, 2, 0.1) == 2      # d_fixed caps it
    assert dspark.policy([-50.0, 9.0], 5, 0.5) == 1     # at least one draft
    assert dspark.policy([logit, logit], 5, 0.5) == 2   # a prefix: no row yet below the threshold
    assert dspark.policy([-1000.0], 5, 0.5) == 1 and dspark.policy([1000.0] * 5, 5, 0.99) == 5


def test_block_logits_follow_the_reference(eng, ref):
    events = _drive(eng, PLAN, 0)
    got, conf, drafts = _ours(events)
    mirror, mconf, _ = _replay(ref, Mode("mirror", world=1), events)
    fp32, fconf, _ = _replay(ref, Mode("fp32", world=1), events)
    assert got.shape == mirror.shape == (len(PLAN) * BLOCK, eng.w.cfg.vocab_size)
    assert drafts == got.argmax(-1).tolist(), "greedy drafts are the rows' top-1, lower id on ties"
    held = _t2(got, mirror, fp32, f"{len(PLAN)} blocks")
    _confidence(conf, torch.where(held, fconf, mconf), f"{len(PLAN)} blocks")


def test_absorb_writes_committed_positions_only(eng):
    w, st, cfg = eng.w, eng.st, eng.w.cfg
    stages, win = [sw.index for sw in w.dspark.stages], cfg.sliding_window
    g = torch.Generator().manual_seed(7)
    st.reset()
    with pytest.raises(ValueError, match="absorb"):
        dspark.absorb(eng, eng.pbuf, 1, prompt=True)    # nothing committed yet
    for step in [150, (6, 4), (3, 1), (6, 6), 30, (2, 2)]:
        prompt = isinstance(step, int)
        rows, keep = (step, step) if prompt else step
        b, n = eng.forward(torch.randint(2, cfg.vocab_size, (rows,), generator=g).tolist(), prompt, keep)
        before = st.rings.clone()
        dspark.absorb(eng, b, n, prompt=prompt)
        changed = (st.rings != before).any(-1)
        assert not changed[:cfg.num_hidden_layers].any(), "absorb wrote a target ring"
        for L in stages:
            assert set(changed[L].nonzero()[:, 0].tolist()) <= {q % win for q in range(st.pos - n, st.pos)}
            for j, q in enumerate(range(st.pos - n, st.pos)):
                assert torch.equal(st.rings[L, q % win], b.kvw[L, j]), f"stage ring {L}, position {q}"
        rings, pos = st.rings.clone(), st.pos
        dspark.propose(eng, 5, st.pos - 1, None, BLOCK)
        assert torch.equal(st.rings, rings) and st.pos == pos == int(st.pos_dev), "a block wrote state"
    with pytest.raises(ValueError, match="absorb"):
        dspark.absorb(eng, eng.dbuf, MAX_ROWS + 1, prompt=False)    # more rows than the window's taps
    with pytest.raises(ValueError, match="propose"):
        dspark.propose(eng, 5, st.pos, None, 3)
    with pytest.raises(ValueError, match="propose"):
        dspark.propose(eng, 5, st.pos - 1, None, BLOCK + 1)


def test_markov_steps_run_exactly_d_times(eng, monkeypatch):
    calls = {"step": 0, "input": 0, "draw": 0}

    def counted(name, fn):
        def run(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)
        return run

    monkeypatch.setattr(dspark, "markov_step", counted("step", dspark.markov_step))
    monkeypatch.setattr(dspark, "markov_input", counted("input", dspark.markov_input))
    monkeypatch.setattr(dspark.sample, "draft_rows", counted("draw", dspark.sample.draft_rows))
    _drive(eng, [150], 11)
    y, p = 9, eng.st.pos - 1

    def propose(d, threshold=None):
        for k in calls:
            calls[k] = 0
        return dspark.propose(eng, y, p, None, d, threshold)

    full, conf = propose(BLOCK)
    assert len(full) == len(conf) == BLOCK
    for d in range(1, BLOCK + 1):
        drafts, c = propose(d)
        assert drafts == full[:d] and c == conf[:d] and calls == {"step": d, "input": d, "draw": d}
    prods = [math.prod(1 / (1 + math.exp(-x)) for x in conf[:i + 1]) for i in range(BLOCK)]
    print("confidence logits", [f"{x:.3g}" for x in conf], "products", [f"{x:.3g}" for x in prods])
    cases = [(1, (1 + prods[0]) / 2)] + [(k, (prods[k - 1] + prods[k]) / 2) for k in range(1, BLOCK)]
    tested = 0
    for want, threshold in [*cases, (BLOCK, prods[-1])]:
        if want < BLOCK and not prods[want] < threshold <= (prods[want - 1] if want > 1 else 1.0):
            continue                                    # equal products: no threshold between them
        tested += 1
        drafts, c = propose(BLOCK, threshold)
        assert dspark.policy(conf, BLOCK, threshold) == want
        assert drafts == full[:want] and c == conf[:want] and calls["step"] == calls["draw"] == want
        assert calls["input"] == min(want + 1, BLOCK), "the chain reads one confidence past its last draft"
    assert tested >= 3


def test_two_ranks_draft_alike_and_repeat(tiny_dir, tiny, ref):
    comms = pair()
    engs = [Eng(loader.load(tiny_dir, tiny.cfg, r, 2, comms[r], capacity=CAP), ref.hasher, Reader(ref.reader.layout))
            for r in range(2)]
    plan = PLAN[:9]
    for sampling in SAMPLINGS:
        for d, threshold in ((3, None), (BLOCK, 0.3)):
            def rank(r, sampling=sampling, d=d, threshold=threshold):
                def go(_):
                    return [ev[3:5] for ev in _drive(engs[r], plan, 2, sampling, d, threshold) if ev[0] == "propose"]
                return go

            runs = [run_pair(rank(0), rank(1), comms) for _ in range(2)]
            what = f"{sampling}, d {d}, threshold {threshold}"
            assert runs[0][0] == runs[0][1], f"{what}: the ranks drafted differently"
            assert runs[1] == runs[0], f"{what}: a repeated run drafted differently"
            assert all(0 < len(dr) <= d for dr, _ in runs[0][0])


def _capture(fn) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _same(eager: list[torch.Tensor], graph: torch.cuda.CUDAGraph, outs: list[torch.Tensor], what: str) -> None:
    """``graph`` replayed rewrites ``outs`` with the bits the eager run left in ``eager``."""

    for x in outs:
        x.fill_(float("nan") if x.is_floating_point() else -1)
    graph.replay()
    assert all(torch.equal(x, y) for x, y in zip(eager, outs)), what


def test_graph_replays_equal_eager(eng):
    w, st, b, k = eng.w, eng.st, eng.dbuf, eng.dwork
    stages = [sw.index for sw in w.dspark.stages]
    _drive(eng, [150], 13)
    g = torch.Generator().manual_seed(13)

    def window() -> None:
        eng.forward(torch.randint(2, w.cfg.vocab_size, (MAX_ROWS,), generator=g).tolist(), False, MAX_ROWS)

    window()
    graphs = {n: _capture(lambda n=n: dspark.absorb(eng, b, n, prompt=False)) for n in range(1, MAX_ROWS + 1)}
    k.bids[:1].fill_(17)
    blk = _capture(lambda: dspark.block(eng))
    ins = [_capture(lambda i=i: dspark.markov_input(eng, i)) for i in range(BLOCK)]
    steps = [_capture(lambda i=i: dspark.markov_step(eng, i)) for i in range(BLOCK)]
    for _ in range(2):                                   # the second pass at another position
        window()
        rings = st.rings.clone()
        for n in range(1, MAX_ROWS + 1):
            st.rings.copy_(rings)
            dspark.absorb(eng, b, n, prompt=False)
            eager = st.rings.clone()
            st.rings.copy_(rings)
            for L in stages:
                st.rings[L, [q % w.cfg.sliding_window for q in range(st.pos - n, st.pos)]] = float("nan")
            graphs[n].replay()
            assert torch.equal(st.rings, eager), f"absorb {n} at {st.pos}"
        dspark.block(eng)
        _same([b.dlog.clone(), b.hidden[:BLOCK].clone()], blk, [b.dlog, b.hidden[:BLOCK]], f"block at {st.pos}")
        k.mids.copy_(torch.tensor([17, 40, 3, 900, 41], dtype=torch.int32))
        for i in range(BLOCK):
            dspark.markov_input(eng, i)
            outs = [b.me[i], b.conf[i:i + 1]]
            _same([x.clone() for x in outs], ins[i], outs, f"Markov input {i}")
            before = b.dlog[i].clone()
            dspark.markov_step(eng, i)
            eager = b.dlog[i].clone()
            b.dlog[i].copy_(before)
            steps[i].replay()
            assert torch.equal(b.dlog[i], eager), f"Markov step {i} at {st.pos}"


# -- the pack -----------------------------------------------------------------------------------------------------

def _pack_taps(cfg: Config, rw: RefWeights, ids, group: int = 4) -> torch.Tensor:
    """The DSpark taps of every row of ``ids`` (one prompt chunk at position 0) from the whole backbone, loaded
    ``group`` layers at a time so one GPU holds it: bf16 [len(ids), taps, D] on the host."""

    R = len(ids)
    taps = torch.empty((R, len(cfg.dspark_target_layer_ids), cfg.hidden_size), dtype=torch.bfloat16, device="cuda")

    def tap(b, rows, slot):                             # every row, where the forward keeps the last 128
        glue.stream_mean(b.X[:rows].view(rows, -1), b.hidden[:rows])
        taps[:, slot].copy_(b.hidden[:rows])

    st = buffers.State(cfg, CAP, "cuda")
    b = buffers.Buffers(cfg, 1, R, CAP, prefill=True, device="cuda")
    w = loader.load(MODEL, cfg, 0, 1, None, layers=[], dspark=False, capacity=CAP)
    F.stage(w, st, b, ids, rw.hasher, rw.reader)
    glue.embed(b.ids[:R], w.embed, cfg.hidden_size, cfg.hc_mult, b.X[:R])     # forward.compute's start
    b.pre_in[:R].zero_()
    b.pre_in[:R, 0].fill_(1.0)
    e = engram.exchange(engram.dequant_rows(b.eraw[:R], b.eloc[:R]), None, b.egat, b.eng)
    del w
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(F, "_tap", tap)
        for first in range(0, cfg.num_hidden_layers, group):
            w = loader.load(MODEL, cfg, 0, 1, None, layers=range(first, first + group), dspark=False, capacity=CAP)
            for lw in w.layers:
                F.layer(lw, w, st, b, R, e, True)
            del w
            torch.cuda.empty_cache()
    return taps.cpu()


@needs_model
def test_pack_blocks_follow_the_reference_and_greedy_drafts_agree():
    from tokenizers import Tokenizer

    ids = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json")).encode(TEXT, add_special_tokens=False).ids
    cfg = Config.read(MODEL)
    rw = RefWeights(MODEL, experts=48)
    taps = _pack_taps(cfg, rw, ids)
    w = loader.load(MODEL, cfg, 0, 1, None, layers=[], capacity=CAP)
    e = Eng(w, prompt_rows=cfg.sliding_window)
    start = len(ids) - 49                               # 48 blocks, the ring full from the first
    assert start > cfg.sliding_window
    events = []
    for a in range(0, start, cfg.sliding_window):
        n = min(cfg.sliding_window, start - a)
        e.st.set_pos(a + n)
        e.pbuf.taps[:n].copy_(taps[a:a + n])
        events.append(("absorb", taps[a:a + n], a))
        dspark.absorb(e, e.pbuf, n, prompt=True)
    for p in range(start - 1, len(ids) - 2):
        drafts, conf = dspark.propose(e, ids[p + 1], p, None, BLOCK)
        events.append(("propose", ids[p + 1], p, drafts, conf, e.dbuf.dlog.clone()))
        e.st.set_pos(p + 2)
        e.dbuf.taps[:1].copy_(taps[p + 1:p + 2])
        events.append(("absorb", taps[p + 1:p + 2], p + 1))
        dspark.absorb(e, e.dbuf, 1, prompt=False)
    del w, e
    torch.cuda.empty_cache()
    got, conf, drafts = _ours(events)
    want, wconf, picks = _replay(rw, Mode("mirror", world=1), events)
    fp32, fconf, _ = _replay(rw, Mode("fp32", world=1), events)
    held = _t2(got, want, fp32, f"{len(ids)}-token text, 48 blocks")
    _confidence(conf, torch.where(held, fconf, wconf), "48 blocks")
    agree = sum(a == b for a, b in zip(drafts, picks)) / len(drafts)
    print(f"greedy drafts equal to the reference's (its chain fed ours): {agree:.4f} of {len(drafts)}")
    assert agree >= AGREE
