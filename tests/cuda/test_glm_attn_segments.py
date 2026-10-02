"""Segmented DSA decode windows (latent.seg_*, sparse.seg_*, segments.SegRows): several streams' rows back to back,
each stream's latents, index keys and pooled keys in its own extent of shared arenas. Every segment's outputs and
cache writes must equal, bit for bit, that segment run alone through the single-stream path on its extent (the
eager path and the CUDA-graph path, dense, straddling the dense limit and sparse up to 200k tokens), and one
captured graph per window size must serve any mix of positions."""

from __future__ import annotations

import os
import random

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import triton  # noqa: E402

from tensorfold.families.glm5_next.cuda import forward, latent, qmm, segments, sparse  # noqa: E402

DEV = "cuda"
H, L = 32, 512                 # a rank's heads, the latent width
IH, ID = 32, 128               # indexer heads and width
SCALE = 256 ** -0.5
LIMIT = sparse.SPARSE_FROM
EXTENTS = (204800, 131072, 65536, 8192)          # four streams' extents (multiples of 2,048)
BASES = tuple(sum(EXTENTS[:i]) for i in range(len(EXTENTS)))
TOTAL = sum(EXTENTS)
MAXR = 32


def _rand(gen, *shape, scale=1.0, dtype=torch.bfloat16):
    return (torch.randn(shape, generator=gen) * scale).to(dtype).to(DEV)


class Arena:
    def __init__(self, gen, index: bool = True):
        self.lc = _rand(gen, TOTAL, L)
        self.index = index
        if index:
            self.ik = _rand(gen, TOTAL, ID)
            self.ig = _rand(gen, TOTAL, ID, scale=2.0)
            self.pk = _rand(gen, TOTAL // 4 + 2, ID)

    def clone(self) -> "Arena":
        other = Arena.__new__(Arena)
        other.index = self.index
        other.lc = self.lc.clone()
        if self.index:
            other.ik, other.ig, other.pk = self.ik.clone(), self.ig.clone(), self.pk.clone()
        return other

    def tensors(self):
        return (self.lc, self.ik, self.ig, self.pk) if self.index else (self.lc,)

    def stream(self, si: int):
        b, e = BASES[si], EXTENTS[si]
        lc = self.lc[b:b + e]
        if not self.index:
            return lc, None
        return lc, (self.ik[b:b + e], self.ig[b:b + e], self.pk[b // 4:b // 4 + e // 4 + 2])


class Inputs:
    """A window's per-row inputs: latents, index key rows (k_raw | head weights), gates, absorbed and index queries."""

    def __init__(self, gen, R: int):
        self.lat = _rand(gen, R, L)
        self.ikr = _rand(gen, R, ID + IH)
        self.igr = _rand(gen, R, ID, dtype=torch.float32)
        self.qa = _rand(gen, R, H, L, scale=0.3)
        self.qi = _rand(gen, R, IH * ID)

    def rows(self, lo: int, hi: int) -> "Inputs":
        other = Inputs.__new__(Inputs)
        for k, v in vars(self).items():
            setattr(other, k, v[lo:hi])
        return other

    def copy_(self, src: "Inputs") -> None:
        for k, v in vars(self).items():
            v.copy_(getattr(src, k))


def _params(gen):
    return _rand(gen, ID), _rand(gen, ID), _rand(gen, 4, ID)         # LayerNorm weight, bias; ape [4, 128]


_POS: dict[int, torch.Tensor] = {}


def _solo(ar: Arena, si: int, pos: int, x: Inputs, params, s, mode: str) -> torch.Tensor:
    """One segment through today's single-stream DSA path (forward._dsa_latent minus the projections) on its
    extent: ``eager`` (host position) or ``graph`` (the captured graphs' arguments: every chunk dense, or every row
    sparse with the pool bucket)."""

    n = x.lat.shape[0]
    lc, index = ar.stream(si)
    pos_dev = _POS.get(pos)
    if pos_dev is None:                  # made once, outside any capture
        pos_dev = _POS[pos] = torch.tensor([pos], dtype=torch.int32, device=DEV)
    latent.latent_write(x.lat, lc, pos_dev)
    if index is not None:
        ik, ig, pk = index
        sparse.index_update(x.ikr[:, :ID], x.igr, *params, ik, ig, pk, pos_dev)
    if mode == "eager":
        host_pos, nch, sparse_np = pos, triton.cdiv(pos + n, latent.CHUNK), None
    else:
        host_pos, nch = None, None
        if pos >= LIMIT:
            sparse_np = sparse.pool_bucket(pos, n, index[2].shape[0] - 2)
        else:
            assert pos + n <= LIMIT, "no graph for a window across the dense limit"
            sparse_np = None
    all_sparse = sparse_np is not None or (host_pos is not None and host_pos >= LIMIT)
    sparse_rows = index is not None and (all_sparse or (host_pos is not None and host_pos + n - 1 >= LIMIT))
    ol = torch.full((n, H, L), float("nan"), dtype=torch.bfloat16, device=DEV)
    if not all_sparse:                   # every row dense first (forward._dsa_latent); sparse rows are redone below
        forward.dense_attention(x.qa, lc, pos_dev, s, scale=SCALE, nch=min(nch or s.nch, s.nch), out=ol)
    if sparse_rows:
        pk = index[2]
        tokens, counts = sparse.select_tokens(x.qi, x.ikr[:, ID:], pk, host_pos, n, pk.shape[0] - 2, pos_dev,
                                              bucket=sparse_np)
        latent.sparse_attention(x.qa, lc, tokens, counts, ol, SCALE)
    return ol


class Seg:
    """The segmented path's static buffers (one set, as an engine keeps them)."""

    def __init__(self, index: bool = True):
        self.rows = segments.SegRows(MAXR, DEV)
        self.sel = segments.SelectScratch(MAXR, max(EXTENTS), DEV)
        self.s = latent.LatentScratch(MAXR, H, latent.chunks_for(2560 + MAXR), DEV)
        self.index = index

    def run(self, ar: Arena, x: Inputs, params, out: torch.Tensor, select: bool = True, hb=None) -> torch.Tensor:
        R = x.lat.shape[0]
        latent.seg_latent_write(x.lat, ar.lc, self.rows)
        tokens = counts = None
        if self.index:
            sparse.seg_index_update(x.ikr[:, :ID], x.igr, *params, ar.ik, ar.ig, ar.pk, self.rows)
            if select:
                tokens, counts = sparse.seg_select_tokens(x.qi, x.ikr[:, ID:], ar.pk, self.rows, self.sel)
        latent.seg_attention(x.qa, ar.lc, self.rows, tokens, counts, self.s, scale=SCALE, out=out[:R], hb=hb)
        return out[:R]


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _check_window(segs, seed: int = 0, index: bool = True, zero_pools: bool = False, hb=None):
    """segs: [(stream, pos, rows)]. Runs the segmented window once and every segment alone (eager, and the graph
    path where one exists), comparing outputs and every cache byte."""

    gen = torch.Generator().manual_seed(seed)
    ar = Arena(gen, index)
    if zero_pools:                       # every pooled key 0: all scores tie, the selection keeps the lowest pools
        ar.pk.zero_()
    params = _params(gen) if index else None
    R = sum(n for _, _, n in segs)
    x = Inputs(gen, R)
    ref = ar.clone()
    seg = Seg(index)
    seg.rows.set([(BASES[si], pos, n) for si, pos, n in segs], sparse_from=LIMIT if index else None)
    out = torch.full((MAXR, H, L), float("nan"), dtype=torch.bfloat16, device=DEV)
    got = seg.run(ar, x, params, out, hb=hb)
    if index:                            # sparse rows (and only they) selected 2,048 + tail tokens
        want_sparse = torch.tensor([p + i >= LIMIT for _, p, n in segs for i in range(n)], device=DEV)
        counts = seg.sel.counts[:R]
        assert torch.equal(counts > 0, want_sparse)
        assert bool((counts[want_sparse] >= sparse.TOPK_POOLS * sparse.POOL).all())
    solo_s = latent.LatentScratch(16, H, latent.chunks_for(2560 + 16), DEV)
    r0 = 0
    for si, pos, n in segs:
        xs = x.rows(r0, r0 + n)
        want = _solo(ref, si, pos, xs, params, solo_s, "eager")
        assert not torch.isnan(want.float()).any()
        assert _same(got[r0:r0 + n], want), f"segment {si} at {pos} ({n} rows): eager"
        straddles = pos < LIMIT < pos + n
        if index and not straddles:
            again = _solo(ref, si, pos, xs, params, solo_s, "graph")     # rewrites the same cache bytes
            assert _same(got[r0:r0 + n], again), f"segment {si} at {pos} ({n} rows): graph path"
        r0 += n
    torch.cuda.synchronize()
    for a, b in zip(ar.tensors(), ref.tensors()):
        assert _same(a, b), "cache writes differ"
    return seg, ar, x, got


# -- the whole DSA block on the synthetic checkpoint (projections included; first, while memory is free) ---------

@pytest.fixture(scope="module")
def engine_long(tmp_path_factory):
    from test_glm_engine import _checkpoint, _TwoCopies

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    import gc

    gc.collect()
    torch.cuda.empty_cache()             # the arenas above: the engine's startup budget reads MemAvailable
    path = tmp_path_factory.mktemp("glm_long_seg")
    _checkpoint(path)
    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), context=2600)


@pytest.mark.parametrize("segs", [[(0, 5, 3), (1, 2040, 16), (2, 2600 - 13, 13)],
                                  [(2, 2051, 1), (0, 2047, 4)],
                                  [(1, 700, 16), (0, 2500, 16)]], ids=["mixed", "edges", "32 rows"])
def test_dsa_segments_equal_dsa_block(engine_long, segs):
    """forward.dsa_segments on a window of several streams (each in its own 4,096-token extent of the arenas)
    gives each segment's rows the partials, cache rows and pooled keys of forward.dsa_block run on that segment
    alone (2 index heads and a 128-wide latent here: the kernels take the widths from the tensors)."""

    from tensorfold.families.glm5_next.cuda import qmm as q

    e = engine_long.e
    w = e.w
    layer = next(l for l in w.layers if l.kind == "dsa")
    c = w.cfg
    ext, streams = 4096, 3
    gen = torch.Generator().manual_seed(len(segs))
    lw, idim = c.kv_lora, c.index_dim
    lc = _rand(gen, streams * ext, lw)
    ik, ig = _rand(gen, streams * ext, idim), _rand(gen, streams * ext, idim)
    pk = _rand(gen, streams * ext // 4 + 2, idim, scale=0.1)
    R = sum(n for _, _, n in segs)
    x = _rand(gen, R, c.hidden)
    b = forward.Buffers(w, MAXR, 2600)
    rows = segments.SegRows(MAXR, DEV)
    sel = segments.SelectScratch(MAXR, ext, DEV)
    arenas = (lc, ik, ig, pk)
    ref = tuple(t.clone() for t in arenas)
    rows.set([(si * ext, p, n) for si, p, n in segs])
    b.normed[:R].copy_(x)
    q.group_sums(b.normed[:R], b.xs[:R])
    got = forward.dsa_segments(layer, w, lc, b, R, rows, sel, (ik, ig, pk)).clone()
    r0 = 0
    for si, p, n in segs:
        base = si * ext
        b.normed[:n].copy_(x[r0:r0 + n])
        q.group_sums(b.normed[:n], b.xs[:n])
        pos_dev = torch.tensor([p], dtype=torch.int32, device=DEV)
        index = (ref[1][base:base + ext], ref[2][base:base + ext], ref[3][base // 4:base // 4 + ext // 4 + 2])
        want = forward.dsa_block(layer, w, ref[0][base:base + ext], None, pos_dev, b, n, triton.cdiv(p + n, 512),
                                 index, host_pos=p)
        assert torch.equal(got[:, r0:r0 + n].contiguous().view(torch.int32), want.contiguous().view(torch.int32)), \
            (si, p, n)
        r0 += n
    for a, bb in zip(arenas, ref):
        assert _same(a, bb)


# -- bit identity ---------------------------------------------------------------------------------------------------

WINDOWS = {
    "one row, short": [(0, 5, 1)],
    "one segment, 16 rows": [(1, 700, 16)],
    "four short": [(3, 0, 3), (1, 511, 8), (0, 1024, 16), (2, 1500, 5)],
    "straddling": [(0, 2040, 16), (1, 2047, 4), (2, 2048, 4), (3, 2035, 8)],
    "limit edges": [(2, 2050, 1), (1, 2051, 1), (0, 2047, 5), (3, 2044, 7)],
    "mixed": [(3, 3, 3), (1, 700, 8), (0, 2040, 16), (2, 5000, 5)],
    "long": [(0, 200000, 4), (1, 131000, 16), (2, 2051, 2), (3, 8000, 10)],
    "long, 16 + 16": [(1, 99999, 16), (0, 150001, 16)],
    "single long": [(0, 204790, 10)],
}


@pytest.mark.parametrize("name", list(WINDOWS))
def test_segments_match_solo(name):
    _check_window(WINDOWS[name], seed=len(name))


@pytest.mark.parametrize("hb", [latent.HB, latent.HB_WIDE])
@pytest.mark.parametrize("name", ["four short", "mixed", "long"])
def test_either_head_block(name, hb):
    """seg_head_block's two tile heights (16 below SEG_WIDE_ROWS, 32 from it) both give the solo kernels' bits."""
    _check_window(WINDOWS[name], seed=3 + hb, hb=hb)


def test_segments_without_index():
    """A short-context engine (no indexer): every row dense, no selection."""
    _check_window([(3, 0, 3), (1, 511, 8), (0, 2040, 11), (2, 1500, 5)], seed=7, index=False)


def test_score_ties_keep_the_lower_pools():
    _check_window([(0, 60000, 4), (1, 2051, 3), (2, 3000, 16), (3, 2040, 9)], seed=11, zero_pools=True)


def _random_windows(n: int, seed: int):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = rng.randint(1, 4)
        streams = rng.sample(range(4), k)
        rows = [rng.randint(1, 16) for _ in streams]
        while sum(rows) > MAXR:
            rows[rows.index(max(rows))] -= 1
        segs = []
        for si, r in zip(streams, rows):
            kind = rng.choice(("short", "edge", "long"))
            top = EXTENTS[si] - r
            pos = (rng.randint(0, LIMIT - 1) if kind == "short" else
                   rng.randint(LIMIT - 20, LIMIT + 4) if kind == "edge" else rng.randint(LIMIT, top))
            segs.append((si, min(pos, top), r))
        out.append(segs)
    return out


@pytest.mark.parametrize("i, segs", list(enumerate(_random_windows(12, 1234))))
def test_random_windows(i, segs):
    _check_window(segs, seed=100 + i)


def test_one_graph_per_window_size():
    """A graph captured for R rows with one mix of segments replays any other mix of R rows (positions, splits,
    dense / sparse) with the eager segmented path's bits (which equal the solo paths', above)."""

    R = 20
    mixes = [[(0, 3000, 8), (1, 2040, 12)],
             [(3, 5, 1), (2, 2049, 3), (0, 180000, 16)],
             [(1, 100000, 10), (0, 100, 10)],
             [(2, 60000, 5), (3, 8000, 5), (0, 2051, 5), (1, 2046, 5)],
             [(0, 204780, 20)]]
    gen = torch.Generator().manual_seed(5)
    ar = Arena(gen)
    params = _params(gen)
    seg = Seg()
    static = Inputs(gen, R)
    out = torch.full((MAXR, H, L), float("nan"), dtype=torch.bfloat16, device=DEV)
    seg.rows.set([(BASES[si], p, n) for si, p, n in mixes[0]])
    for _ in range(2):
        seg.run(ar, static, params, out)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        seg.run(ar, static, params, out)
    for k, mix in enumerate(mixes):
        x = Inputs(gen, R)
        before = ar.clone()
        seg.rows.set([(BASES[si], p, n) for si, p, n in mix])
        static.copy_(x)
        g.replay()
        torch.cuda.synchronize()
        replayed = out[:R].clone()
        eager_out = torch.full_like(out, float("nan"))
        want = seg.run(before, x, params, eager_out)
        torch.cuda.synchronize()
        assert not torch.isnan(want.float()).any()
        assert _same(replayed, want), f"mix {k}"
        for a, b in zip(ar.tensors(), before.tensors()):
            assert _same(a, b), f"mix {k}: cache writes"
        del before


# -- the selection's kernels: split across chunk programs, or one program a row at other block sizes -------------

def _scores_for(kind: str, R: int, NP: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    s = torch.randn((R, NP), generator=gen)
    if kind == "ties":                   # a few values, +0 and -0 among them: the lower pools must win the ties
        s = torch.randint(-3, 5, (R, NP), generator=gen).float() * 0.25
        s[s == 0] = torch.where(torch.rand(int((s == 0).sum()), generator=gen) < 0.5, 0.0, -0.0)
    elif kind == "tail":                 # visible pools first, -inf past them (what the scores kernels write)
        for r in range(R):
            vis = int(torch.randint(513, NP + 1, (1,), generator=gen))
            s[r, vis:] = float("-inf")
    elif kind == "relu":                 # sums of relus: many exact zeros, the rest positive
        s = torch.relu(s) * (torch.rand((R, NP), generator=gen) < 0.3)
    return s.to(DEV)


@pytest.mark.parametrize("kind", ["random", "ties", "tail", "relu"])
@pytest.mark.parametrize("NP", [1024, 5000, 8192, 50000, 51200, 65536])
def test_select_kernels_agree(kind, NP):
    """select_split (the segmented windows' selection), top_pools and _select_rows keep the reference's pools
    (``_top_pools``: a stable sort's), for decode windows and prompt-sized blocks of rows."""

    for R in (5, 70):
        scores = _scores_for(kind, R, NP, NP + R)
        want = sparse._top_pools(scores, sparse.TOPK_POOLS)
        old = torch.empty_like(want)
        sparse._select_rows[(R,)](scores, old, NP, scores, K=sparse.TOPK_POOLS, BLOCK=1024, VISIBLE=False,
                                  num_warps=4)
        assert torch.equal(old, want)
        assert torch.equal(sparse.top_pools(scores, sparse.TOPK_POOLS), want)
        assert torch.equal(sparse.select_split(scores, sparse.TOPK_POOLS), want)


@pytest.mark.parametrize("name", ["mixed", "long", "long, 16 + 16"])
def test_segmented_select_either_kernel(name):
    """seg_select_tokens' split selection and its one-program-a-row kernel give the same tokens and counts."""

    gen = torch.Generator().manual_seed(21)
    ar = Arena(gen)
    segs = WINDOWS[name]
    R = sum(n for _, _, n in segs)
    x = Inputs(gen, R)
    seg = Seg()
    seg.rows.set([(BASES[si], p, n) for si, p, n in segs])
    outs = []
    for split in (True, False):
        sparse.SEG_SPLIT = split
        try:
            seg.sel.tokens.fill_(-7)
            t, c = sparse.seg_select_tokens(x.qi, x.ikr[:, ID:], ar.pk, seg.rows, seg.sel)
        finally:
            sparse.SEG_SPLIT = True
        sp = c > 0
        outs.append((t[sp].clone(), c.clone()))
    assert torch.equal(outs[0][1], outs[1][1]) and torch.equal(outs[0][0], outs[1][0])


# -- the other per-row pieces of a DSA window: absorb / expand, and the projections past 16 rows ----------------

SPLITS = ((0, 4), (4, 20), (20, 21), (21, 32))


@pytest.mark.parametrize("kind", ["bf16", "q4"])
def test_absorb_expand_rows_alone(kind):
    """absorb_q / expand_v of a 32-row window equal each segment's rows run alone (row_block differs past 16)."""

    gen = torch.Generator().manual_seed(3)
    D = 256
    k_rows, v_rows = _rand(gen, H * D, L, scale=L ** -0.5), _rand(gen, H * D, L, scale=L ** -0.5)
    if kind == "bf16":
        a = latent.AbsorbW.from_rows(k_rows, v_rows, H)
    else:
        a = latent.AbsorbQ4(qmm.to_mlx(qmm.quantize4(k_rows)), qmm.to_mlx(qmm.quantize4(v_rows)), H)
    q = _rand(gen, MAXR, H, D)
    ol = _rand(gen, MAXR, H, L, scale=0.3)
    qa = latent.absorb_q(q, a, torch.empty((MAXR, H, L), dtype=torch.bfloat16, device=DEV))
    ov = latent.expand_v(ol, a, torch.empty((MAXR, H, D), dtype=torch.bfloat16, device=DEV))
    for lo, hi in SPLITS:
        pa = latent.absorb_q(q[lo:hi], a, torch.empty((hi - lo, H, L), dtype=torch.bfloat16, device=DEV))
        pv = latent.expand_v(ol[lo:hi], a, torch.empty((hi - lo, H, D), dtype=torch.bfloat16, device=DEV))
        assert _same(pa, qa[lo:hi]), (kind, lo, hi, "absorb")
        assert _same(pv, ov[lo:hi]), (kind, lo, hi, "expand")


@pytest.mark.parametrize("kind", ["q4", "b16"])
@pytest.mark.parametrize("n, k", [(4096, 4096), (2560, 6144), (4096, 1024)])
def test_matmul_rows_alone(kind, n, k):
    """qmm.matmul of a 32-row window (no decode tile variant past 16 rows) equals its segments' rows alone."""

    gen = torch.Generator().manual_seed(n + k)
    w = _rand(gen, n, k, scale=k ** -0.5)
    q = qmm.quantize4(w) if kind == "q4" else qmm.make_b16(w)
    x = _rand(gen, MAXR, k)
    for f32 in (False, True):
        whole = qmm.matmul(x, q, f32=f32)
        for lo, hi in SPLITS:
            part = qmm.matmul(x[lo:hi], q, f32=f32)
            if f32:
                assert torch.equal(part.view(torch.int32), whole[lo:hi].view(torch.int32)), (kind, lo, hi)
            else:
                assert _same(part, whole[lo:hi]), (kind, lo, hi)


# -- timing (TF_SEG_BENCH=1 pytest -s) ------------------------------------------------------------------------------

def _time(fn, iters: int = 50) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000.0


def _graphed(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


@pytest.mark.skipif(os.environ.get("TF_SEG_BENCH") != "1", reason="timing: TF_SEG_BENCH=1")
def test_timing_vs_solo():
    """One DSA layer's cache writes, selection and attention (no projections): the segmented window vs its
    segments' solo launches back to back, both as CUDA graphs (as decode runs them) and eager."""

    gen = torch.Generator().manual_seed(9)
    ar = Arena(gen)
    params = _params(gen)
    seg = Seg()
    solo_s = latent.LatentScratch(16, H, latent.chunks_for(2560 + 16), DEV)
    cases = {
        "4 x 1 row, dense": [(0, 1000, 1), (1, 1500, 1), (2, 500, 1), (3, 1900, 1)],
        "4 x 4 rows, dense": [(0, 1000, 4), (1, 1500, 4), (2, 500, 4), (3, 1900, 4)],
        "4 x 1 row, 8k-60k": [(0, 60000, 1), (1, 30000, 1), (2, 16000, 1), (3, 8000, 1)],
        "4 x 4 rows, 8k-60k": [(0, 60000, 4), (1, 30000, 4), (2, 16000, 4), (3, 8000, 4)],
        "4 x 8 rows, 8k-60k": [(0, 60000, 8), (1, 30000, 8), (2, 16000, 8), (3, 8000, 8)],
        "4 x 4 rows, 8k-200k": [(0, 200000, 4), (1, 130000, 4), (2, 65000, 4), (3, 8000, 4)],
        "2 x 16 rows, 100k-200k": [(0, 200000, 16), (1, 100000, 16)],
        "1 x 4 rows, 200k": [(0, 200000, 4)],
    }
    out = torch.empty((MAXR, H, L), dtype=torch.bfloat16, device=DEV)
    lines = []
    for name, segs in cases.items():
        R = sum(n for _, _, n in segs)
        x = Inputs(gen, R)
        seg.rows.set([(BASES[si], p, n) for si, p, n in segs])
        parts, r0 = [], 0
        for si, p, n in segs:
            parts.append((si, p, x.rows(r0, r0 + n)))
            r0 += n

        def run_seg():
            seg.run(ar, x, params, out)

        def run_solo(mode):
            def go():
                for si, p, xs in parts:
                    _solo(ar, si, p, xs, params, solo_s, mode)
            return go

        t_seg = _time(_graphed(run_seg))
        t_solo = _time(_graphed(run_solo("graph")))
        e_seg = _time(run_seg)
        e_solo = _time(run_solo("eager"))
        lines.append(f"{name:26s} R={R:2d}  graph: seg {t_seg:7.1f} us  solo {t_solo:7.1f} us ({t_solo / t_seg:4.2f}x)"
                     f"   eager: seg {e_seg:7.1f} us  solo {e_solo:7.1f} us")
    print("\n" + "\n".join(lines))
