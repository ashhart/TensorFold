"""Decode rounds: DSpark rounds emit exactly the serial reply (every fixed draft count and the confidence policy, greedy
and keyed draws, one rank and two thread ranks) with serial's logits bits on every kept row, a window's rows equal each
row run alone, serial replies repeat, scripted drafts reach every window size and kept count, drafting does not depend
on a resume or an earlier request, graphs replay the eager rounds, and long contexts cross the indexer's transitions.

The tiny prompts put decode across the tiny config's transitions (16 visible entries, 32-entry candidate pools, the
128-slot ring). The pack's check needs ``TF_DSV41_MODEL`` (layers 0-7, DSpark tapping their last three).
"""

from __future__ import annotations

import hashlib
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

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import BLOCK, MAX_ROWS, loader, snapshot
from tensorfold.families.deepseek_v41.cuda import decode as D
from tensorfold.families.deepseek_v41.cuda import forward as F
from tensorfold.families.deepseek_v41.cuda import prefill as P
from tensorfold.families.deepseek_v41.engram_table import Reader

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
MODEL = os.environ.get("TF_DSV41_MODEL", "")
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="set TF_DSV41_MODEL to the checkpoint")
CAP = 1024
LONG = 5000                             # tiny prompt: ~600 visible entries a row, past every transition
REPLY = 40
CONF = 0.3
POLICIES = [(d, None) for d in range(1, BLOCK + 1)] + [(BLOCK, CONF)]
GRID = [(t, k, p) for t in (0.0, 1.0) for k in (0, 20) for p in (0.95, 1.0)]
SEEDS = (0, 1, 2)
PROMPTS = {0: 5, 1: 100, 2: 600}        # decode crosses 16/32 entries, the ring's wrap, then runs at full lists
KEYED = Sampling(seed=7, temperature=1.0, top_k=20, top_p=0.95)
SCHEDULE = [(R, k) for R in range(1, MAX_ROWS + 1) for k in range(1, R + 1)]     # (window rows, rows kept)
PACK_PROMPT, PACK_REPLY = 400, 256


def _ids(seed: int, n: int, vocab: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, vocab, (n,), generator=g).tolist()


def _sha(tokens) -> str:
    """The server's ``token_sha``."""

    return hashlib.sha256(",".join(str(int(t)) for t in tokens).encode()).hexdigest()[:12]


def _bits(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _engine(w, ref, capacity: int = CAP, graphs: bool = False, reader=None) -> D.Engine:
    return D.Engine(w, capacity, graphs=graphs, hasher=ref.hasher, reader=reader or ref.reader)


def _request(e: D.Engine, prompt, count: int, sampling, policy=None, *, stop_eos: bool = False,
             **prefill) -> D.DecodeResult:
    """Prefill, then serial rounds (``policy`` None) or DSpark rounds with (drafts, confidence)."""

    first = P.prefill(e, prompt, sampling, **prefill)
    got: list[int] = []
    if policy is None:
        res = D.serial_decode(e, first, count, sampling, stop_eos=stop_eos, on_tokens=got.extend)
    else:
        res = D.dspark_decode(e, first, count, sampling, drafts=policy[0], confidence=policy[1], stop_eos=stop_eos,
                              on_tokens=got.extend)
    assert got == res.tokens and res.tokens[0] == first, "on_tokens gets the first token, then each round's"
    return res


def _counts(res: D.DecodeResult, policy) -> None:
    """The stats add up, and fixed policies draft min(d, tokens still to emit - 1) a round."""

    assert res.rounds == len(res.depths) == len(res.keeps) == len(res.clocks)
    assert res.drafted == sum(res.depths) and res.accepted == sum(k - 1 for k in res.keeps)
    assert all(1 <= k <= d + 1 for d, k in zip(res.depths, res.keeps))
    assert 1 + sum(res.keeps) >= len(res.tokens) and res.tokens_per_round == (len(res.tokens) - 1) / res.rounds
    if policy is not None and policy[1] is None:
        done = 1
        for d, k in zip(res.depths, res.keeps):
            assert d == min(policy[0], len(res.tokens) - done - 1)
            done += k
    stats = res.stats()
    assert set(stats["stages_ms"]) == {*D.HOST_PARTS, "device"}


def _recorded(e: D.Engine, *args, **kw) -> tuple[D.DecodeResult, list, list]:
    """``_request`` with every verify forward's logits (each row's bits) and every proposal (drafts, their confidences,
    the block's logits) kept."""

    forwards, proposals = [], []
    forward, propose = e.forward, e.propose

    def fwd(tokens):
        out = forward(tokens)
        forwards.append([_bits(row) for row in out])
        return out

    def prop(*a):
        drafts, conf = propose(*a)
        proposals.append((drafts, conf, _bits(e.dbuf.dlog)))
        return drafts, conf

    e.forward, e.propose = fwd, prop
    try:
        return _request(e, *args, **kw), forwards, proposals
    finally:
        del e.forward, e.propose


def _logit_rows(e: D.Engine, *args, **kw) -> tuple[D.DecodeResult, list[str]]:
    """``_request`` -> (its result, the bits of each kept row's verify logits, in position order)."""

    rows: list[list[str]] = []
    forward = e.forward

    def fwd(tokens):
        out = forward(tokens)
        rows.append([_bits(row) for row in out])
        return out

    e.forward = fwd
    try:
        res = _request(e, *args, **kw)
    finally:
        del e.forward
    keeps = res.keeps or [1] * len(rows)
    return res, [bits for window, k in zip(rows, keeps) for bits in window[:k]]


@pytest.fixture(scope="module")
def ref(tiny_dir):
    return RefWeights(tiny_dir)


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    return loader.load(tiny_dir, Config.read(tiny_dir), 0, 1, None, capacity=LONG + 2 * REPLY)


@pytest.fixture(scope="module")
def geng(tiny, ref):
    e = _engine(tiny, ref, graphs=True)
    assert e.st.pos == 0 and len(e.graphs.verify) == MAX_ROWS and len(e.graphs.steps) == BLOCK
    return e


def test_accept_keeps_leading_matches():
    assert D.accept([5], []) == 1
    assert D.accept([5, 6, 7], [5, 6]) == 3
    assert D.accept([5, 6, 7], [5, 9]) == 2
    assert D.accept([5, 6, 7], [4, 6]) == 1
    assert D.accept([5, 1, 7], [5, 1], ends=(1,)) == 2      # stops right after the end token is sampled
    assert D.accept([1, 6], [1], ends=(1,)) == 1
    assert D.accept([1, 6], [1]) == 2


def test_engine_refuses_bad_settings(tiny, ref, geng):
    with pytest.raises(ValueError, match="hasher"):
        D.Engine(tiny, CAP)
    with pytest.raises(ValueError, match="drafts"):
        D.dspark_decode(geng, 3, 10, None, drafts=BLOCK + 1)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("temperature,top_k,top_p", GRID)
def test_drafted_equals_serial(geng, seed, temperature, top_k, top_p):
    sampling = Sampling(seed=seed, temperature=temperature, top_k=top_k, top_p=top_p)
    prompt = _ids(seed, PROMPTS[seed], geng.w.cfg.vocab_size)
    want, want_rows = _logit_rows(geng, prompt, REPLY, sampling)
    assert len(want.tokens) == REPLY and want.rounds == REPLY - 1 and want.drafted == 0
    _counts(want, None)
    for policy in POLICIES:
        got, rows = _logit_rows(geng, prompt, REPLY, sampling, policy)
        assert got.tokens == want.tokens and _sha(got.tokens) == _sha(want.tokens), f"{policy}, {sampling}"
        assert rows == want_rows[:len(rows)], f"a kept row's logits differ from serial: {policy}, {sampling}"
        _counts(got, policy)


@pytest.mark.parametrize("seed", SEEDS)
def test_serial_replies_repeat(geng, seed):
    prompt = _ids(seed, PROMPTS[seed], geng.w.cfg.vocab_size)
    for sampling in (None, Sampling(seed=seed, temperature=1.0, top_k=0, top_p=0.95)):
        runs = [_sha(_request(geng, prompt, REPLY, sampling).tokens) for _ in range(3)]
        assert runs[0] == runs[1] == runs[2], f"{sampling}: {runs}"


@pytest.mark.parametrize("seed", SEEDS)
def test_two_ranks_draft_the_serial_reply(tiny_dir, tiny, ref, seed):
    comms = pair()
    engs = [_engine(loader.load(tiny_dir, tiny.cfg, r, 2, comms[r], capacity=CAP), ref,
                    reader=Reader(ref.reader.layout)) for r in range(2)]
    prompt = _ids(seed, PROMPTS[seed], tiny.cfg.vocab_size)

    def rank(r):
        def go(_):
            out = []
            for t, k, p in GRID:
                sampling = Sampling(seed=seed, temperature=t, top_k=k, top_p=p)
                serial = [_request(engs[r], prompt, REPLY, sampling).tokens for _ in range(3 if t else 1)]
                drafted = [_request(engs[r], prompt, REPLY, sampling, policy) for policy in POLICIES]
                out.append((serial, [(x.tokens, x.drafted, x.accepted, x.keeps) for x in drafted]))
            return out
        return go

    got0, got1 = run_pair(rank(0), rank(1), comms)
    assert got0 == got1, "the ranks decoded differently"
    for (t, k, p), (serial, drafted) in zip(GRID, got0):
        assert all(s == serial[0] for s in serial), f"serial replies differ across runs: {t}, {k}, {p}"
        for policy, (tokens, *_) in zip(POLICIES, drafted):
            assert tokens == serial[0], f"two ranks, {policy}, temperature {t}, top_k {k}, top_p {p}"


def _scripted(e: D.Engine, reply: list[int], start: int, plan):
    """A drafter proposing the serial ``reply`` (``reply[0]`` at position ``start``) for windows of plan's rows,
    the draft at row k (k < rows) changed so exactly k rows are kept."""

    vocab = e.w.cfg.vocab_size

    def propose(y, p, sampling, d, threshold=None):
        j = p + 1 - start
        assert reply[j] == y, "the drafter sees the serial reply's pending token"
        R, k = next(plan)
        drafts = list(reply[j + 1:j + R])
        assert len(drafts) == R - 1 <= d
        if k < R:
            drafts[k - 1] = (drafts[k - 1] + 1) % vocab
        return drafts, [0.0] * len(drafts)

    return propose


def test_scripted_drafts_reach_every_window_and_keep(geng):
    e = geng
    prompt = _ids(4, 30, e.w.cfg.vocab_size)
    count = 1 + sum(k for _, k in SCHEDULE)             # the schedule emits exactly the reply
    serial, want_rows = _logit_rows(e, prompt, count, KEYED)
    want = serial.tokens
    e.propose = _scripted(e, want, len(prompt), iter(SCHEDULE))
    try:
        got, rows = _logit_rows(e, prompt, count, KEYED, (BLOCK, None))
    finally:
        del e.propose
    assert got.tokens == want
    assert rows == want_rows, "every kept row's logits, rows 0 to 5 of every window size, equal serial's bits"
    assert [(d + 1, k) for d, k in zip(got.depths, got.keeps)] == SCHEDULE
    assert got.drafted == sum(R - 1 for R, _ in SCHEDULE) and got.accepted == sum(k - 1 for _, k in SCHEDULE)
    assert got.tokens_per_round == (count - 1) / len(SCHEDULE)


def test_scripted_drafts_stop_at_an_end_token(geng):
    e = geng
    prompt = _ids(5, 30, e.w.cfg.vocab_size)
    reply = _request(e, prompt, 60, KEYED).tokens
    m = next(i for i in range(8, len(reply)) if reply[i] not in reply[:i])
    e.eos = (reply[m],)
    try:
        want = _request(e, prompt, 60, KEYED, stop_eos=True)
        assert want.tokens == reply[:m + 1]
        e.propose = _scripted(e, reply, len(prompt), iter([(MAX_ROWS, MAX_ROWS)] * 60))
        got = _request(e, prompt, 60, KEYED, (BLOCK, None), stop_eos=True)
        assert got.tokens == reply[:m + 1] and got.keeps[-1] == 1 + (m - 1) % MAX_ROWS
        del e.propose
        assert _request(e, prompt, 60, KEYED, (3, None), stop_eos=True).tokens == reply[:m + 1]
    finally:
        e.eos = (e.w.cfg.eos_token_id,)
        e.__dict__.pop("propose", None)


@pytest.mark.parametrize("policy", [(3, None), (BLOCK, CONF)])
def test_drafting_does_not_depend_on_a_resume_or_an_earlier_request(geng, policy):
    e, vocab = geng, geng.w.cfg.vocab_size
    prompt = _ids(8, 301, vocab)

    def run(**kw):
        res, forwards, proposals = _recorded(e, prompt, REPLY, KEYED, policy, **kw)
        return res.tokens, res.drafted, res.accepted, res.tokens_per_round, res.keeps, forwards, proposals

    solo = run()
    _request(e, _ids(9, 700, vocab), REPLY, None, (2, None))
    assert run() == solo, "after an unrelated request"
    for K in (37, 128, 129, 300):                       # odd and even, either side of the ring's wrap
        kept: list[snapshot.Snapshot] = []
        P.prefill(e, prompt[:K + 1], KEYED, keep_at=K, keep=kept.append)
        snap = kept.pop()
        arena = torch.empty((snapshot.row_bytes(e, snap),), dtype=torch.uint8, device="cuda")
        snapshot.save_rows(e, snap, arena)
        _request(e, _ids(K, 700, vocab), REPLY, KEYED, (BLOCK, None))     # overwrites every live row
        snapshot.load_rows(e, snap)
        snap.rows = None
        assert run(resume=snap) == solo, f"resumed at {K}"


@pytest.mark.parametrize("start", [5, 30, 62, 125, 600, 1010])     # the 16/32-entry transitions, the ring's wrap
def test_a_window_row_equals_the_row_run_alone(geng, start):
    """Each row of an R-row verify forward, every layer through the head, gets the bits of the same row run alone as
    a 1-row window after the rows before it were committed: logits, window KV and DSpark taps."""

    e, vocab, layers = geng, geng.w.cfg.vocab_size, geng.w.cfg.num_hidden_layers     # kvw's later rows are DSpark's
    prompt = _ids(start, start, vocab)
    tokens = [P.prefill(e, prompt, None)] + _ids(start + 1, MAX_ROWS - 1, vocab)
    alone = []
    for t in tokens:
        logits = e.forward([t])
        alone.append((_bits(logits[0]), _bits(e.dbuf.kvw[:layers, 0]), _bits(e.dbuf.taps[0])))
        F.commit(e.w, e.st, e.dbuf, 1, 1)
    for R in range(2, MAX_ROWS + 1):
        P.prefill(e, prompt, None)
        logits = e.forward(tokens[:R])
        got = [(_bits(logits[r]), _bits(e.dbuf.kvw[:layers, r]), _bits(e.dbuf.taps[r])) for r in range(R)]
        assert got == alone[:R], f"a {R}-row window at {start}"


def test_graphs_replay_the_eager_rounds(tiny, ref, geng):
    eager = _engine(tiny, ref)
    before = dict(geng.replays)
    for seed, n in ((10, 5), (11, 120)):
        prompt = _ids(seed, n, tiny.cfg.vocab_size)
        for policy in (None, (BLOCK, None), (BLOCK, CONF), (2, None)):
            runs = [_recorded(x, prompt, REPLY, KEYED, policy) for x in (eager, geng)]
            (a, fa, pa), (b, fb, pb) = runs
            assert a.tokens == b.tokens and a.keeps == b.keeps, f"{n}-token prompt, {policy}"
            assert fa == fb, f"verify logits, {n}-token prompt, {policy}"
            assert pa == pb, f"proposals, {n}-token prompt, {policy}"
    assert eager.replays["graph"] == 0 and geng.replays["eager"] == before["eager"]
    assert geng.replays["graph"] > before["graph"]


def test_long_context_drafted_equals_serial(tiny, ref):
    vocab, cap = tiny.cfg.vocab_size, LONG + 2 * REPLY
    prompt = _ids(12, LONG, vocab)
    e = _engine(tiny, ref, capacity=cap, graphs=True)
    want = _request(e, prompt, REPLY, KEYED)
    for policy in ((3, None), (BLOCK, CONF)):
        assert _request(e, prompt, REPLY, KEYED, policy).tokens == want.tokens, f"{LONG}-token prompt, {policy}"
    del e
    eager = _engine(tiny, ref, capacity=cap)
    assert _request(eager, prompt, REPLY, KEYED, (BLOCK, None)).tokens == want.tokens, "eager, long prompt"


# -- the pack -----------------------------------------------------------------------------------------------------

@needs_model
def test_pack_reduced_model_drafts_the_serial_reply():
    from tokenizers import Tokenizer

    cfg = Config.read(MODEL, 8)
    rw = RefWeights(MODEL, 8)
    text = (Path(MODEL) / "inference" / "model.py").read_text()
    ids = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json")).encode(text, add_special_tokens=False).ids
    prompt = [cfg.bos_token_id] + ids[:PACK_PROMPT - 1]
    cap = PACK_PROMPT + PACK_REPLY + MAX_ROWS
    w = loader.load(MODEL, cfg, 0, 1, None, capacity=cap)
    e = D.Engine(w, cap, graphs=True, hasher=rw.hasher, reader=rw.reader)
    for sampling in (None, Sampling(seed=1, temperature=1.0, top_k=20, top_p=0.95)):
        want = _request(e, prompt, PACK_REPLY, sampling)
        for policy in ((3, None), (BLOCK, CONF)):
            got = _request(e, prompt, PACK_REPLY, sampling, policy)
            assert got.tokens == want.tokens, f"pack, {sampling}, {policy}"
            stats = got.stats()
            print(f"pack, 8 layers, {sampling}, {policy}: {PACK_REPLY} tokens equal to serial "
                  f"({_sha(got.tokens)}), {stats['tokens_per_round']} tokens a round, "
                  f"{stats['accepted']}/{stats['drafted']} drafts kept, host ms a round {stats['stages_ms']}")
