"""Prompt prefill and kept snapshots: any chunking, and a resume from a snapshot, ends in the whole prompt's bits
(rings with the DSpark stages, compressed and index-K caches, compressor tails, first-token logits, the next block's
drafts); a request's state does not depend on the one before it; saved rows round-trip; Engram reads that lag the
device change nothing, and a pinned half is refilled only after the device has copied it.

State is compared where it is defined: ring slots of the last ``window`` committed positions, cache entries of
completed groups, tails while valid. The pack's check needs ``TF_DSV41_MODEL`` (layers 0-7 and the DSpark stages).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_tiny
from dsv41_ref_weights import RefWeights

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import (
    BLOCK,
    MAX_ROWS,
    PREFILL_ROWS,
    buffers,
    dspark,
    loader,
    sample,
    snapshot,
)
from tensorfold.families.deepseek_v41.cuda import forward as F
from tensorfold.families.deepseek_v41.cuda import prefill as P

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
MODEL = os.environ.get("TF_DSV41_MODEL", "")
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="set TF_DSV41_MODEL to the checkpoint")
CAP = 2560
LONG = 2100                             # past one 2048-row chunk
SAMPLING = Sampling(seed=3, temperature=1.0, top_k=20, top_p=0.95)
REDUCED, REDUCED_TOKENS = 8, 3000       # the pack: layers 0-7, ~3k prompt tokens


class Eng:
    """One rank's sequence: weights, state, prompt-chunk buffers of ``rows`` rows, decode buffers, DSpark scratch."""

    def __init__(self, w, hasher, reader, rows: int = PREFILL_ROWS, capacity: int = CAP) -> None:
        self.w, self.hasher, self.reader = w, hasher, reader
        self.st = buffers.State(w.cfg, capacity, "cuda")
        self.pbuf = buffers.Buffers(w.cfg, w.world, rows, capacity, prefill=True, device="cuda")
        self.dbuf = buffers.Buffers(w.cfg, w.world, MAX_ROWS, capacity, device="cuda")
        self.dwork = dspark.Work(w.cfg, w.world, "cuda")


def _ids(seed: int, n: int, vocab: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, vocab, (n,), generator=g).tolist()


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.detach().contiguous().view(torch.uint8).cpu()


def _ring_rows(rings: torch.Tensor, n: int) -> torch.Tensor:
    """Every layer's ring slots of positions max(0, n - window) .. n - 1, in position order."""

    win = rings.shape[1]
    return rings[:, [q % win for q in range(max(0, n - win), n)]]


def _state(e: Eng) -> dict:
    """The defined committed state of ``e``: position, token tail, rings, completed cache entries, valid tails."""

    st = e.st
    n, valid = st.pos, st.tail_valid.bool()
    out = {"pos": torch.tensor([n, int(st.pos_dev)]), "history": torch.tensor(st.history, dtype=torch.int64),
           "rings": _ring_rows(st.rings, n), "tail_valid": st.tail_valid, "tail": st.tail[valid]}
    for layer, r in st.ratio.items():
        out[f"comp {layer}"], out[f"index_k {layer}"] = st.comp[layer][:n // r], st.index_k[layer][:n // r]
    return {k: _bits(v) for k, v in out.items()}


def _snap(s: snapshot.Snapshot) -> dict:
    valid = s.tail_valid.bool()
    return {"ids": torch.tensor(s.ids), "rings": _bits(_ring_rows(s.rings, len(s.ids))),
            "tail_valid": _bits(s.tail_valid), "tail": _bits(s.tail[valid])}


def _run(e: Eng, prompt, sampling=SAMPLING, **kw) -> dict:
    """Prefill ``prompt``, then the next block's drafts -> everything a later step reads, as bits."""

    first = P.prefill(e, prompt, sampling, **kw)
    out = {"first": torch.tensor([first]), "logits": _bits(e.pbuf.logits[0]), **_state(e)}
    drafts, conf = dspark.propose(e, first, len(prompt) - 1, sampling, BLOCK)
    out.update(drafts=torch.tensor(drafts), conf=torch.tensor(conf), dlog=_bits(e.dbuf.dlog))
    return out


def _equal(got: dict, want: dict, what: str) -> None:
    assert got.keys() == want.keys(), what
    differ = [k for k in want if not torch.equal(got[k], want[k])]
    assert not differ, f"{what}: {differ} differ"


@pytest.fixture(scope="module")
def ref(tiny_dir):
    return RefWeights(tiny_dir)


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    return loader.load(tiny_dir, Config.read(tiny_dir), 0, 1, None, capacity=CAP)


@pytest.fixture(scope="module")
def eng(tiny, ref):
    return Eng(tiny, ref.hasher, ref.reader)


@pytest.fixture(scope="module")
def whole(tiny, ref):
    """The LONG-token prompt as one chunk -> (prompt, its result, the snapshot at LONG - 1 of a one-chunk prefix)."""

    prompt = _ids(1, LONG, tiny.cfg.vocab_size)
    e = Eng(tiny, ref.hasher, ref.reader, rows=LONG)
    assert P.chunks(0, LONG, None, e.pbuf.rows) == [(0, LONG)]
    P.prefill(e, prompt[:-1], None)
    kept = _snap(snapshot.take(e, prompt[:-1]))
    return prompt, _run(e, prompt), kept


def test_chunks_end_at_the_keep_point():
    assert P.chunks(0, 1, 1, 2048) == [(0, 1)]
    assert P.chunks(0, 1, None, 2048) == [(0, 1)]
    assert P.chunks(0, 5000, 4999, 2048) == [(0, 2048), (2048, 4096), (4096, 4999), (4999, 5000)]
    assert P.chunks(0, 10, None, 4) == [(0, 4), (4, 8), (8, 10)]
    assert P.chunks(3, 10, 3, 4) == [(3, 7), (7, 10)]
    assert P.chunks(0, 5, 5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert P.chunks(2048, 2050, 2049, 2048) == [(2048, 2049), (2049, 2050)]


@pytest.mark.parametrize("rows", [1, 7, 128, 129, 2048])
def test_chunked_equals_whole(tiny, ref, whole, rows):
    prompt, want, want_kept = whole
    e = Eng(tiny, ref.hasher, ref.reader, rows=rows)
    kept: list[snapshot.Snapshot] = []
    _equal(_run(e, prompt, keep_at=LONG - 1, keep=kept.append), want, f"{rows}-row chunks")
    assert len(kept) == 1
    _equal(_snap(kept[0]), want_kept, f"{rows}-row chunks' snapshot at {LONG - 1}")


def test_one_token_prompt(eng):
    w, st, b = eng.w, eng.st, eng.pbuf
    prompt = [77]
    kept: list[snapshot.Snapshot] = []
    got = _run(eng, prompt, keep_at=1, keep=kept.append)
    assert len(kept) == 1 and kept[0].ids == prompt and st.pos == 1
    _equal(_run(eng, prompt), got, "one token without a keep point")
    st.reset()                                          # by hand: one forward, commit, absorb
    F.stage(w, st, b, prompt, eng.hasher, eng.reader)
    logits = F.compute(w, st, b, 1, prompt=True, head_rows=1)
    assert torch.equal(_bits(logits[0]), got["logits"])
    (first,), _ = sample.target_rows(w, logits, [1], SAMPLING)
    (other,), _ = sample.target_rows(w, logits, [0], SAMPLING)
    assert got["first"].item() == first != other, "the first token is drawn with position 1's key"
    F.commit(w, st, b, 1, 1)
    dspark.absorb(eng, b, 1, prompt=True)
    by_hand = _state(eng)
    _equal(by_hand, {k: got[k] for k in by_hand}, "one token by hand")
    with pytest.raises(ValueError, match="at least one"):
        P.prefill(eng, [], None)


@pytest.mark.parametrize("K", [37, 100, 127, 128, 129, 300, 2047, 2048, 2049])
def test_resumed_equals_fresh(eng, K):
    vocab = eng.w.cfg.vocab_size
    before = _ids(K, K + 1, vocab)
    prompt = before[:K] + _ids(K + 1, 50, vocab)
    kept: list[snapshot.Snapshot] = []
    P.prefill(eng, before, SAMPLING, keep_at=K, keep=kept.append)
    snap = kept.pop()
    assert snap.ids == before[:K]
    arena = torch.empty((snapshot.row_bytes(eng, snap),), dtype=torch.uint8, device="cuda")
    snapshot.save_rows(eng, snap, arena)
    P.prefill(eng, _ids(K + 2, len(prompt) + 20, vocab), None)     # another conversation overwrites every row
    snapshot.load_rows(eng, snap)
    snap.rows = None
    resumed = _run(eng, prompt, resume=snap, keep_at=len(prompt) - 1, keep=kept.append)
    fresh = _run(eng, prompt, keep_at=len(prompt) - 1, keep=kept.append)
    _equal(resumed, fresh, f"resumed at {K}")
    _equal(_snap(kept[0]), _snap(kept[1]), f"the resumed prompt's snapshot, resumed at {K}")


def test_state_after_a_request_is_independent_of_the_one_before(eng):
    vocab = eng.w.cfg.vocab_size
    for n in (90, 300):                                 # inside and past one ring
        prompt = _ids(n, n, vocab)
        runs = []
        for before in (600, 1000):
            P.prefill(eng, _ids(before, before, vocab), SAMPLING)
            runs.append(_run(eng, prompt))
        _equal(runs[0], runs[1], f"{n}-token prompt after another request")


def test_saved_rows_round_trip(eng):
    st = eng.st
    prompt = _ids(5, 600, eng.w.cfg.vocab_size)
    kept: list[snapshot.Snapshot] = []
    P.prefill(eng, prompt, None, keep_at=301, keep=kept.append)
    snap = kept[0]
    P.prefill(eng, prompt[:301], None)                  # the live state at the keep point again
    need, state = snapshot.row_bytes(eng, snap), snapshot.state_bytes(eng)
    assert need == snapshot._span(st.row_views(301))[1] and need % snapshot.ALIGN == 0
    arena = torch.empty((state + need + snapshot.ALIGN,), dtype=torch.uint8, device="cuda")
    held = snapshot.take(eng, prompt[:301], arena[:state])
    for x in (held.rings, held.tail, held.tail_valid):
        assert x.untyped_storage().data_ptr() == arena.untyped_storage().data_ptr(), "a copy outside the arena"
    _equal(_snap(held), _snap(snap), "a snapshot taken into the arena")
    want = [_bits(v) for v in st.row_views(301)]
    with pytest.raises(ValueError, match="bytes"):
        snapshot.save_rows(eng, snap, arena[state:state + need - 1])
    with pytest.raises(ValueError, match="saved rows"):
        snapshot.load_rows(eng, snap)
    snapshot.save_rows(eng, held, arena[state:state + need])
    assert held.nbytes == need and snapshot.snapshot_bytes(held) == state + need
    for v in st.row_views(301):
        v.fill_(float("nan"))
    snapshot.load_rows(eng, held)
    assert all(torch.equal(_bits(v), x) for v, x in zip(st.row_views(301), want))
    with pytest.raises(ValueError, match="300 ids at 301"):
        snapshot.take(eng, prompt[:300])


def test_three_resends_keep_every_state(eng):
    vocab = eng.w.cfg.vocab_size
    prompt = _ids(9, 401, vocab)
    kept: list[snapshot.Snapshot] = []
    want = _run(eng, prompt, keep_at=400, keep=kept.append)
    snap = kept[0]
    held = _snap(snap)
    arena = torch.empty((snapshot.row_bytes(eng, snap),), dtype=torch.uint8, device="cuda")
    for k in range(3):
        snapshot.save_rows(eng, snap, arena)
        P.prefill(eng, _ids(20 + k, 700, vocab), None)
        snapshot.load_rows(eng, snap)
        snap.rows = None
        _equal(_run(eng, prompt, resume=snap, keep_at=400, keep=kept.append), want, f"resend {k + 1}")
        assert kept[-1] is snap and len(kept) == k + 2
        _equal(_snap(snap), held, f"the snapshot after resend {k + 1}")
    with pytest.raises(ValueError, match="strict prefix"):
        P.prefill(eng, prompt[:400], None, resume=snap)


def test_kept_snapshot_lands_in_the_given_space(eng):
    prompt = _ids(6, 300, eng.w.cfg.vocab_size)
    space = torch.empty((snapshot.state_bytes(eng),), dtype=torch.uint8, device="cuda")
    kept: list[snapshot.Snapshot] = []
    P.prefill(eng, prompt, None, keep_at=200, keep=kept.append, space=space)
    for x in (kept[0].rings, kept[0].tail, kept[0].tail_valid):
        assert x.untyped_storage().data_ptr() == space.untyped_storage().data_ptr(), "a copy outside the space"
    P.prefill(eng, prompt[:200], None)
    _equal(_snap(kept[0]), _snap(snapshot.take(eng, prompt[:200])), "a snapshot kept into the space")


class _Slow:
    """The Engram reader with every chunk's read held back ``delay`` seconds on its own thread."""

    def __init__(self, reader, delay: float) -> None:
        self.reader, self.delay = reader, delay
        self.layout, self.wrow = reader.layout, reader.wrow
        self.pool = ThreadPoolExecutor(1)

    def advise(self, ids) -> None:
        self.reader.advise(ids)

    def gather_async(self, ids, out_w, out_s):
        def late():
            time.sleep(self.delay)
            self.reader.gather(ids, out_w, out_s)
        return self.pool.submit(late)


def test_slow_reads_change_nothing(tiny, ref):
    prompt = _ids(11, 700, tiny.cfg.vocab_size)
    e = Eng(tiny, ref.hasher, ref.reader, rows=129)
    want = _run(e, prompt, keep_at=699, keep=lambda s: None)
    e.reader = _Slow(ref.reader, 0.05)
    _equal(_run(e, prompt, keep_at=699, keep=lambda s: None), want, "reads held back")


def test_a_pinned_half_is_refilled_only_after_its_copy(tiny, ref):
    e = Eng(tiny, ref.hasher, ref.reader, rows=129)
    b, reader = e.pbuf, ref.reader
    prompt = _ids(12, 3 * 129, tiny.cfg.vocab_size)
    e.st.reset()
    rows = P._Rows(e, prompt, 0, P.chunks(0, len(prompt), None, 129))
    want = np.empty((rows._ids(0).size, b.eraw.shape[-1]), dtype=np.uint8)
    reader.gather(rows._ids(0), want[:, :reader.wrow], want[:, reader.wrow:])
    rows.read(0)
    torch.cuda._sleep(2_000_000_000)                    # the device ~1 s behind: chunk 0's copy waits
    rows.land(0)
    rows.read(2)                                        # the same pinned half, for chunk 2
    rows.drain()
    torch.cuda.synchronize()
    assert np.array_equal(b.eraw[:129].cpu().numpy().reshape(want.shape), want), "chunk 0 got another chunk's rows"


# -- the pack -----------------------------------------------------------------------------------------------------

@needs_model
def test_pack_reduced_model_chunked_and_resumed_equal_whole():
    from tokenizers import Tokenizer

    cfg = Config.read(MODEL, REDUCED)
    rw = RefWeights(MODEL, REDUCED)
    text = (Path(MODEL) / "inference" / "model.py").read_text()
    tok = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json"))
    prompt = [cfg.bos_token_id] + tok.encode(text, add_special_tokens=False).ids[:REDUCED_TOKENS - 1]
    assert len(prompt) == REDUCED_TOKENS
    cap = REDUCED_TOKENS + MAX_ROWS
    w = loader.load(MODEL, cfg, 0, 1, None, capacity=cap)
    assert w.dspark is not None and w.engram, "the pack check covers the DSpark rings and Engram reads"
    e = Eng(w, rw.hasher, rw.reader, rows=REDUCED_TOKENS, capacity=cap)
    want = _run(e, prompt)
    del e
    torch.cuda.empty_cache()
    e = Eng(w, rw.hasher, rw.reader, capacity=cap)
    kept: list[snapshot.Snapshot] = []
    _equal(_run(e, prompt, keep_at=REDUCED_TOKENS - 1, keep=kept.append), want, "2048-row chunks")
    for K in (101, 1000, 2049):
        P.prefill(e, prompt[:K + 1], SAMPLING, keep_at=K, keep=kept.append)
        snap = kept[-1]
        arena = torch.empty((snapshot.row_bytes(e, snap),), dtype=torch.uint8, device="cuda")
        snapshot.save_rows(e, snap, arena)
        P.prefill(e, prompt[::-1], None)
        snapshot.load_rows(e, snap)
        snap.rows = None
        _equal(_run(e, prompt, resume=snap), want, f"resumed at {K}")
        print(f"pack, {REDUCED} layers, {REDUCED_TOKENS} tokens: resumed at {K} equals the whole prompt")
