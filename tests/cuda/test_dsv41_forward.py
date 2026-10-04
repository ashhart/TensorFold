"""The target forward on the tiny checkpoint: every block and the head within the model-level gate of the reference port
fed the same stream, the whole model's logits as close to the reference's as its own fp32 mode is, two thread ranks
deterministic and in agreement, a prompt row's head independent of its chunk, rings rebuilt from committed rows, graph
replays equal to eager runs.

The tiny's random weights amplify roundings from block to block: the reference's own fp32 and mirror modes disagree on
top-1 where margins are small, so end-to-end top-1 counts only rows whose top-1 is decided.
"""

from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_score as score
import dsv41_tiny
from dsv41_pair import pair, run_pair
from dsv41_ref_weights import RefWeights
from dsv41_reference import Mode, State, hc_pre, rms_norm

from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import MAX_ROWS, PREFILL_ROWS, buffers, loader
from tensorfold.families.deepseek_v41.cuda import forward as F
from tensorfold.families.deepseek_v41.engram_table import Reader

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
CAP = 2560
MIRROR = Mode("mirror", world=1)
TOP1 = 0.99                             # reduced model: top-1 agreement
KL = 1e-3                               # mean KL(reference || TF)
REL_L2 = 0.05                           # T2: logits rel-L2
DECIDED = 0.25                          # least share of rows whose top-1 the tiny reference decides
BLOCK_L2 = 4 * 2**-8                    # T2, one block: residual stream rel-L2 and cosine
COS = 0.9999
# (rows, keep, prompt): chunks of every size class, then windows of every R with kept prefixes and rejected drafts
PLAN = ([(100, 100, True), (60, 60, True)] + [(R, k, False) for R, k in
        ((1, 1), (6, 3), (4, 4), (2, 1), (5, 5), (3, 2), (6, 6), (2, 2), (6, 1), (5, 4))]
        + [(40, 40, True)] + [(R, k, False) for R, k in ((3, 3), (6, 2), (1, 1), (4, 3), (6, 6), (2, 2), (6, 6),
                                                         (5, 4))])
POSITIONS = sum(k for _, k, _ in PLAN)
FED_PLAN = [(150, 150, True), (1, 1, False), (6, 6, False), (2, 2, False), (40, 40, True), (5, 5, False),
            (3, 3, False), (4, 4, False)]
PAIR_PLAN = [(150, 150, True)] + [(R, k, False) for R, k in ((1, 1), (4, 2), (6, 6), (3, 1), (5, 3), (2, 2))] + \
    [(20, 20, True), (6, 4, False)]


def _windows(plan, seed: int, vocab: int) -> list[list[int]]:
    """Each forward's tokens: committed positions take the plan's ids, rows past ``keep`` random drafts."""

    g = torch.Generator().manual_seed(seed)
    return [torch.randint(2, vocab, (rows,), generator=g).tolist() for rows, _, _ in plan]


def _committed(plan, windows) -> list[int]:
    return [t for (_, keep, _), win in zip(plan, windows) for t in win[:keep]]


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class Eng:
    """One rank's sequence: weights, state, a prompt chunk's and a decode window's buffers, Engram host side."""

    def __init__(self, w, hasher, reader, capacity: int = CAP, prompt_rows: int = PREFILL_ROWS) -> None:
        self.w, self.hasher, self.reader = w, hasher, reader
        self.st = buffers.State(w.cfg, capacity, "cuda")
        self.pbuf = buffers.Buffers(w.cfg, w.world, prompt_rows, capacity, prefill=True, device="cuda")
        self.dbuf = buffers.Buffers(w.cfg, w.world, MAX_ROWS, capacity, device="cuda")

    def buf(self, prompt: bool) -> buffers.Buffers:
        return self.pbuf if prompt else self.dbuf

    def compute(self, tokens, prompt: bool) -> torch.Tensor:
        """Every row's logits [R, V / world] of one forward at ``st.pos``, nothing committed."""

        b = self.buf(prompt)
        R = F.stage(self.w, self.st, b, tokens, self.hasher, self.reader)
        if not prompt:
            return F.compute(self.w, self.st, b, R, prompt=False, head_rows=R).clone()
        F.compute(self.w, self.st, b, R, prompt=True, head_rows=0)
        out = torch.empty((R, b.logits.shape[1]), dtype=torch.float32, device="cuda")
        for a in range(0, R, score.ROWS):
            n = min(score.ROWS, R - a)
            F.head(self.w, b, a, n, out[a:a + n], prompt=True)
        return out

    def run(self, plan, windows) -> list[torch.Tensor]:
        """The plan's forwards and commits -> the kept rows' logits of each."""

        out = []
        for (rows, keep, prompt), win in zip(plan, windows):
            out.append(self.compute(win, prompt)[:keep])
            F.commit(self.w, self.st, self.buf(prompt), rows, keep)
        return out


def _t2(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    g, x = got.double().cpu(), want.double().cpu()
    top1 = float((g.argmax(-1) == x.argmax(-1)).double().mean())
    p = torch.softmax(x, -1)
    kl = float((p * (torch.log_softmax(x, -1) - torch.log_softmax(g, -1))).sum(-1).mean())
    rel = float((g - x).norm() / x.norm())
    print(f"{what}: top-1 {top1:.4f}, mean KL {kl:.3g}, rel-L2 {rel:.3g}")
    assert top1 >= TOP1 and kl <= KL and rel <= REL_L2, f"{what}: top-1 {top1}, KL {kl}, rel-L2 {rel}"


def _kl(p: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((p * (torch.log_softmax(x, -1) - torch.log_softmax(y, -1))).sum(-1).mean())


def _follows(got: torch.Tensor, mirror: torch.Tensor, fp32: torch.Tensor, what: str) -> None:
    """Whole-model logits against the reference in mirror mode: top-1 agreement where the reference's margin
    exceeds twice its own fp32-vs-mirror spread on the row (at least DECIDED of rows), KL and rel-L2 no larger
    than fp32 mode's."""

    g, x, f = got.double().cpu(), mirror.double().cpu(), fp32.double().cpu()
    pick, want = g.argmax(-1), x.argmax(-1)
    margin = (x.gather(-1, want[:, None]) - x.gather(-1, pick[:, None]))[:, 0]
    spread = 2 * (x - f).abs().amax(-1)
    top1 = float(((pick == want) | (margin <= spread)).double().mean())
    gap = x.topk(2, -1).values
    decided = float((gap[:, 0] - gap[:, 1] > spread).double().mean())
    p = torch.softmax(x, -1)
    kl, kl_ref = _kl(p, x, g), _kl(p, x, f)
    rel, rel_ref = float((g - x).norm() / x.norm()), float((f - x).norm() / x.norm())
    print(f"{what}: top-1 {float((pick == want).double().mean()):.4f} ({top1:.4f} where {decided:.2f} decided), "
          f"KL {kl:.3g}, rel-L2 {rel:.3g}; the reference's fp32 mode: top-1 "
          f"{float((f.argmax(-1) == want).double().mean()):.4f}, KL {kl_ref:.3g}, rel-L2 {rel_ref:.3g}")
    assert decided >= DECIDED, f"{what}: only {decided} of rows decided"
    assert top1 >= TOP1 and kl <= kl_ref and rel <= rel_ref, f"{what}: top-1 {top1}, KL {kl}, rel-L2 {rel}"


@pytest.fixture(scope="module")
def ref(tiny_dir):
    return RefWeights(tiny_dir)


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    cfg = Config.read(tiny_dir)
    return loader.load(tiny_dir, cfg, 0, 1, None, dspark=False, capacity=CAP)


def _reference(ref, ids) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference model's logits of ``ids`` in mirror and fp32 modes (one rank)."""

    return tuple(ref.model(ids, Mode(m, world=1), all_logits=True)[0] for m in ("mirror", "fp32"))


@pytest.fixture(scope="module")
def plan_logits(tiny, ref):
    windows = _windows(PLAN, 0, tiny.cfg.vocab_size)
    return windows, *_reference(ref, _committed(PLAN, windows))


def test_blocks_and_head_follow_the_reference(tiny, ref, monkeypatch):
    w, cfg = tiny, tiny.cfg
    e = Eng(w, ref.hasher, ref.reader)
    entering = []
    block = F.layer

    def fed(lw, w_, st, b, R, *args):
        entering.append((b.X[:R].float().cpu(), b.pre_in[:R].cpu()))
        return block(lw, w_, st, b, R, *args)

    monkeypatch.setattr(F, "layer", fed)
    state = State()
    for (rows, _, prompt), ids in zip(FED_PLAN, _windows(FED_PLAN, 5, cfg.vocab_size)):
        entering.clear()
        logits = e.compute(ids, prompt)
        b = e.buf(prompt)
        entering.append((b.X[:rows].float().cpu(), b.pre_in[:rows].cpu()))
        where, worst = f"{rows} {'prompt' if prompt else 'decode'} rows at {state.pos}", 0.0
        for L in range(cfg.num_hidden_layers):
            if cfg.roles[L].tap is not None:            # the last min(R, 128) rows' mean copy, summed in order
                n = min(rows, b.taps.shape[0])
                x = entering[L][0][rows - n:]
                mean = ((((x[:, 0] + x[:, 1]) + x[:, 2]) + x[:, 3]) / 4).to(torch.bfloat16)
                assert torch.equal(b.taps[:n, cfg.roles[L].tap].cpu(), mean), f"tap of layer {L}, {where}"
            X, pre = ref.layer(L, *entering[L], state, MIRROR, tokens=ids)
            got = entering[L + 1][0].double().flatten()
            x = X.double().flatten()
            rel, cos = float((got - x).norm() / x.norm()), float(got @ x / (got.norm() * x.norm()))
            assert rel <= BLOCK_L2 and cos >= COS, f"layer {L}, {where}: rel-L2 {rel:.3g}, cosine {cos:.6f}"
            worst = max(worst, rel)
        print(f"blocks, {where}: worst rel-L2 {worst:.3g}")
        X, pre = entering[-1]
        h = rms_norm(hc_pre(X, pre, MIRROR), ref("norm.weight"), cfg.rms_norm_eps, MIRROR)
        _t2(logits, h @ ref("lm_head.weight").T, f"head, {where}")
        F.commit(w, e.st, b, rows, rows)
        state.pos += rows
        state.tokens += ids


def test_prompt_logits_follow_the_reference(tiny, ref, plan_logits):
    windows, mirror, fp32 = plan_logits
    e = Eng(tiny, ref.hasher, ref.reader)
    got = score.prompt_logits(e, _committed(PLAN, windows), rows=96)
    assert got.shape == (POSITIONS, tiny.cfg.vocab_size) and got.dtype == torch.float32
    assert e.st.pos == 0
    _follows(got, mirror, fp32, f"{POSITIONS} prompt rows")


def test_chunks_and_windows_follow_the_reference(tiny, ref, plan_logits):
    windows, mirror, fp32 = plan_logits
    e = Eng(tiny, ref.hasher, ref.reader)
    got = torch.cat(e.run(PLAN, windows))
    assert e.st.pos == POSITIONS
    _follows(got, mirror, fp32, f"{POSITIONS} kept rows of chunks and windows")


class _Recorder:
    """A rank's communicator that hashes every gather it receives."""

    def __init__(self, comm) -> None:
        self.comm, self.rank, self.world, self.log = comm, comm.rank, comm.world, []

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        self.comm.all_gather(send, recv)
        self.log.append(_sha(recv))


def test_two_ranks_are_deterministic_and_agree(tiny_dir, tiny, ref):
    cfg = tiny.cfg
    comms = pair()
    recs = [_Recorder(c) for c in comms]
    engs = [Eng(loader.load(tiny_dir, cfg, r, 2, recs[r], dspark=False, capacity=CAP), ref.hasher,
                Reader(ref.reader.layout)) for r in range(2)]
    windows = _windows(PAIR_PLAN, 1, cfg.vocab_size)

    def rank(r):
        def go(_):
            e = engs[r]
            e.st.reset()
            for buf in (e.pbuf, e.dbuf):
                buf.exl3.guard_left = 0         # the overflow guard's own gather is tested in test_dsv41_moe
            recs[r].log = []
            logits = e.run(PAIR_PLAN, windows)
            return [_sha(x) for x in logits], logits, recs[r].log
        return go

    runs = [run_pair(rank(0), rank(1), comms) for _ in range(3)]
    for (a, b) in runs:
        assert a[2] == b[2], "the ranks received different gathers"
        assert len(a[2]) == len(PAIR_PLAN) * (1 + 2 + 2 * cfg.num_hidden_layers)
    for later in runs[1:]:
        for r in range(2):
            assert later[r][0] == runs[0][r][0], f"rank {r}: per-step logits differ between runs"
            assert later[r][2] == runs[0][r][2], f"rank {r}: gathers differ between runs"
    whole = torch.cat([torch.cat([x, y], -1) for x, y in zip(runs[0][0][1], runs[0][1][1])])
    _follows(whole, *_reference(ref, _committed(PAIR_PLAN, windows)), "two ranks")


def test_last_row_head_alone_equals_inside_a_full_chunk(tiny, ref):
    n = PREFILL_ROWS
    ids = _windows([(n, n, True)], 2, tiny.cfg.vocab_size)[0]
    e = Eng(tiny, ref.hasher, ref.reader)
    b = e.pbuf
    R = F.stage(tiny, e.st, b, ids, ref.hasher, ref.reader)
    whole = F.compute(tiny, e.st, b, R, prompt=True, head_rows=1).clone()
    e.st.reset()
    R = F.stage(tiny, e.st, b, ids[:-1], ref.hasher, ref.reader)
    assert F.compute(tiny, e.st, b, R, prompt=True, head_rows=0) is None
    F.commit(tiny, e.st, b, R, R)
    R = F.stage(tiny, e.st, b, ids[-1:], ref.hasher, ref.reader)
    alone = F.compute(tiny, e.st, b, R, prompt=True, head_rows=1)
    assert whole.shape == alone.shape == (1, tiny.cfg.vocab_size)
    assert torch.equal(whole, alone)


def test_commit_rebuilds_rings_from_committed_rows(tiny, ref):
    cfg, w = tiny.cfg, tiny
    plan = [(300, 300, True)] + PLAN[2:12] + [(40, 40, True), (129, 129, True)] + PLAN[13:]
    windows = _windows(plan, 3, cfg.vocab_size)
    e = Eng(w, ref.hasher, ref.reader)
    st, layers, win = e.st, cfg.num_hidden_layers, cfg.sliding_window
    kept: list[torch.Tensor] = []                   # [layers, head_dim] of each committed position
    tokens: list[int] = []
    for (rows, keep, prompt), ids in zip(plan, windows):
        b = e.buf(prompt)
        e.compute(ids, prompt)
        kept += list(b.kvw[:layers, :keep].transpose(0, 1).clone())
        cmp = b.cmp[:, keep - 1].clone()
        pos = st.pos
        F.commit(w, st, b, rows, keep)
        tokens += ids[:keep]
        assert st.pos == pos + keep == len(kept) and int(st.pos_dev) == st.pos
        assert st.history == tokens[-(cfg.engram_max_ngram_size - 1):]
        want = torch.zeros_like(st.rings[:layers])
        for p in range(max(0, st.pos - win), st.pos):
            want[:, p % win] = kept[p]
        assert torch.equal(st.rings[:layers], want), f"rings after keeping {keep} of {rows} at {pos}"
        if st.pos % 2:
            assert bool((st.tail_valid == 1).all()) and torch.equal(st.tail, cmp)
        else:
            assert not bool(st.tail_valid.any())


@pytest.mark.parametrize("R", list(range(1, MAX_ROWS + 1)))
def test_graph_replays_equal_eager(tiny, ref, R):
    w, steps, layers = tiny, 4, tiny.cfg.num_hidden_layers
    e = Eng(w, ref.hasher, ref.reader)
    st, b = e.st, e.dbuf
    windows = _windows([(150, 150, True)] + [(R, R, False)] * (steps + 1), 4 + R, w.cfg.vocab_size)
    e.compute(windows[0], True)
    F.commit(w, st, e.pbuf, 150, 150)
    F.stage(w, st, b, windows[1], e.hasher, e.reader)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        F.compute(w, st, b, R, prompt=False, head_rows=R)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = F.compute(w, st, b, R, prompt=False, head_rows=R)
    for k, ids in enumerate(windows[1:1 + steps]):
        keep = 1 + k % R
        F.stage(w, st, b, ids, e.hasher, e.reader)
        eager = F.compute(w, st, b, R, prompt=False, head_rows=R).clone()
        state = [b.kvw[:layers, :R].clone(), b.taps[:R].clone(), *(x.clone() for x in st.row_views(st.pos + R))]
        wrote = [c[layer][st.pos // r:(st.pos + R) // r] for layer, r in st.ratio.items() for c in (st.comp, st.index_k)]
        for x in [out, b.kvw[:layers, :R], b.taps[:R], *wrote]:
            x.fill_(float("nan"))           # the replay must rewrite everything the eager forward did
        graph.replay()
        assert torch.equal(out, eager), f"logits: {R} rows at {st.pos}"
        again = [b.kvw[:layers, :R], b.taps[:R], *st.row_views(st.pos + R)]
        assert all(torch.equal(x, y) for x, y in zip(state, again)), f"state: {R} rows at {st.pos}"
        F.commit(w, st, b, R, keep)
