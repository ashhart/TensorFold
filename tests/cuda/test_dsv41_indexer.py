"""DeepSeek-V4.1's indexer: scores (the op-level bound against fp64), block and entry selections equal to a reference
selection on the kernel's own scores (zero tolerance), rows that never follow the window or row block, graphs, real
layers."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_reference as ref

from tensorfold.cuda.nvfp4.linear import Mx8Linear
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import MAX_ROWS, indexer, mx8, quant, rope
from tensorfold.families.deepseek_v41.cuda.weights import IdxW
from tensorfold.families.glm5_next.cuda.qmm import make_b16

MODEL = os.environ.get("TF_DSV41_MODEL")
DEV = "cuda"
CFG = Config.read(Path(__file__).parents[1] / "fixtures" / "deepseek_v41")
H, D = CFG.index_n_heads, CFG.index_head_dim
K, BLOCKS, BLK = CFG.index_topk, CFG.candidate_topk_blocks, CFG.candidate_block_size
# (ratio, first row's position): visible 511/512/513, ratio-2 positions 1023/1024/1025, pools of 16,384/16,385
WINDOWS = [(1, 0), (2, 0), (1, 508), (2, 1021), (1, 16381), (1, 16387), (2, 33000)]


def _pos(p: int) -> torch.Tensor:
    return torch.tensor([p], dtype=torch.int32, device=DEV)


def _fp4(shape: tuple[int, ...], gen: torch.Generator, scale: float = 1.0) -> torch.Tensor:
    return quant.fp4_qdq_1x32_e8m0((torch.randn(shape, generator=gen, device=DEV) * scale).to(torch.bfloat16))


def _inputs(rows: int, entries: int, seed: int, ties: bool = False):
    """FP4-grid index queries and keys with fp32 weights; ``ties``: keys from 6 rows, queries 0 and 3 zero with row 0's
    weights negative (whole score rows of -0 and of +0)."""

    g = torch.Generator(device=DEV).manual_seed(seed)
    q = _fp4((rows, H, D), g)
    w = torch.randn((rows, H), generator=g, device=DEV) * 0.05
    if ties:
        k = _fp4((6, D), g)[torch.randint(0, 6, (entries,), generator=g, device=DEV)]
        q[::3] = 0
        w[0] = -w[0].abs()
    else:
        k = _fp4((entries, D), g)
    return q.contiguous(), w, k.contiguous()


class Out:
    """One window's outputs (scores of ``buf`` rows), poisoned so an unwritten value shows."""

    def __init__(self, rows: int, width: int, buf: int | None = None) -> None:
        def i32(*shape):
            return torch.full(shape, -7, dtype=torch.int32, device=DEV)

        self.s = torch.full((buf or rows, width), float("nan"), device=DEV)
        self.cand, self.cand_n = i32(rows, BLOCKS), i32(rows)
        self.lists, self.n = i32(rows, K), i32(rows)

    def list(self, r: int) -> torch.Tensor:
        return self.lists[r, :int(self.n[r])].cpu()

    def blocks(self, r: int) -> torch.Tensor:
        return self.cand[r, :int(self.cand_n[r])].cpu()


def _run(q, w, k, ratio, pos, *, source=False, within=None, pos_host=None, buf_rows=None) -> Out:
    o = Out(q.shape[0], k.shape[0], buf_rows)
    indexer.select(CFG, q, w, k, ratio, _pos(pos), o.s, o.lists, o.n, pos=pos_host,
                   source=(o.cand, o.cand_n) if source else None, within=within)
    if not source:
        o.cand_n = o.cand_n[:0]
    return o


def _canon(s: torch.Tensor) -> torch.Tensor:
    return torch.where(s == 0, torch.zeros_like(s), s)


def _ref_topk(s: torch.Tensor, k: int) -> torch.Tensor:
    """GLM's ``_top_pools`` rule: the k best, ties to the lower entry (-0 as +0), ascending."""

    return torch.sort(_canon(s.cpu()), descending=True, stable=True).indices[:k].sort().values.int()


def _ref_blocks(s: torch.Tensor, vis: int) -> torch.Tensor:
    """The reference port's ``select_candidate_blocks`` on a row's visible scores, as ascending block ids."""

    if vis == 0:
        return torch.zeros(0, dtype=torch.int32)
    keep = ref.select_candidate_blocks(_canon(s[:vis].cpu())[None], torch.tensor([vis]), BLOCKS, BLK)[0]
    return torch.unique(torch.nonzero(keep)[:, 0] // BLK).int()


def _dense(o: Out, r: int, vis: int, within) -> torch.Tensor:
    """Row r's scores by entry: -inf outside ``within``'s blocks (M:573-575)."""

    if within is None:
        return o.s[r, :vis].cpu()
    blocks = within[0][r, :int(within[1][r])].long().cpu()
    e = (blocks[:, None] * BLK + torch.arange(BLK)).flatten()
    out = torch.full((vis,), -math.inf)
    ok = e < vis
    out[e[ok]] = o.s[r].cpu()[torch.arange(len(e))[ok]]                         # slot c holds entry e[c]
    return out


def _check_selection(o: Out, ratio: int, pos: int, within=None, source: bool = False) -> None:
    for r in range(o.lists.shape[0]):
        vis = (pos + r + 1) // ratio
        s = _dense(o, r, vis, within)
        assert int(o.n[r]) == min(K, vis), (r, vis)
        assert torch.equal(o.list(r), _ref_topk(s, min(K, vis))), (r, vis)
        if source:
            assert int(o.cand_n[r]) == min(BLOCKS, -(-vis // BLK)), (r, vis)
            assert torch.equal(o.blocks(r), _ref_blocks(s, vis)), (r, vis)


def _fp64(q, w, k, ratio: int, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact scores and the op-level bound 8 sqrt(H D) 2^-24 sum_h |w_h| sum_d |q_hd k_td| per (row, entry), invisible
    -inf."""

    dots = torch.einsum("rhd,td->rht", q.double(), k.double())
    s = (dots.relu() * w.double()[..., None]).sum(1)
    mag = (torch.einsum("rhd,td->rht", q.double().abs(), k.double().abs()) * w.double().abs()[..., None]).sum(1)
    vis = (pos + torch.arange(q.shape[0], device=DEV) + 1) // ratio
    s = s.masked_fill(torch.arange(k.shape[0], device=DEV) >= vis[:, None], -math.inf)
    return s, 8 * math.sqrt(H * D) * 2.0**-24 * mag


def _near_ties(got: torch.Tensor, want: torch.Tensor, s64: torch.Tensor, bound: torch.Tensor) -> float:
    """The selection rule: entries in one list only score within 4x the bounds of the k-th fp64 score.
    Returns the smallest k-th/(k+1)-th fp64 margin over the bound (inf when every entry is taken)."""

    k = len(want)
    if not k:
        return math.inf
    order = torch.sort(s64, descending=True, stable=True)
    kth = order.values[k - 1]
    for t in set(got.tolist()) ^ set(want.tolist()):
        assert abs(float(s64[t] - kth)) <= 4 * float(bound[t] + bound[order.indices[k - 1]]), t
    if k >= len(s64) or not math.isfinite(float(order.values[k])):
        return math.inf
    return float((order.values[k - 1] - order.values[k]) / bound[order.indices[k - 1]].clamp_min(1e-300))


@pytest.mark.parametrize("ratio,pos", [(1, 3990), (2, 7990), (1, 16381)])
def test_scores_track_fp64_and_selections_differ_only_at_near_ties(ratio, pos):
    q, w, k = _inputs(MAX_ROWS, (pos + MAX_ROWS) // ratio + 5, 1)
    o = _run(q, w, k, ratio, pos, source=ratio == 1)
    s64, bound = _fp64(q, w, k, ratio, pos)
    margins = []
    for r in range(MAX_ROWS):
        vis = (pos + r + 1) // ratio
        got = o.s[r, :vis].double()
        assert float(((got - s64[r, :vis]).abs() / bound[r, :vis].clamp_min(1e-300)).max()) <= 1, r
        assert torch.isneginf(o.s[r, vis:(pos + MAX_ROWS) // ratio]).all(), r
        margins.append(_near_ties(o.list(r), _ref_topk(s64[r, :vis], min(K, vis)), s64[r, :vis].cpu(),
                                  bound[r, :vis].cpu()))
    print(f"scores ratio {ratio} pos {pos}: k-th margin / fp32 bound per row {[f'{m:.3g}' for m in margins]}")


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("ratio,pos", WINDOWS)
def test_selections_equal_the_reference_on_the_kernels_scores(ratio, pos, ties):
    q, w, k = _inputs(MAX_ROWS, (pos + MAX_ROWS) // ratio + 3, 2, ties)
    o = _run(q, w, k, ratio, pos, source=True)
    _check_selection(o, ratio, pos, source=True)
    if ties:
        for r, negative in ((0, True), (3, False)):                                 # rows of -0 and of +0
            zero = o.s[r, :(pos + r + 1) // ratio]
            assert (zero == 0).all() and (torch.signbit(zero) == negative).all(), r
            assert torch.equal(o.list(r), torch.arange(int(o.n[r]), dtype=torch.int32)), r


def test_signed_zeros_and_equal_scores_go_to_the_lower_entry():
    g = torch.Generator().manual_seed(3)
    vis = 20000
    rows = torch.zeros((4, vis))
    rows[0] = torch.where(torch.rand(vis, generator=g) < 0.5, -0.0, 0.0)                       # all zeros
    rows[1] = torch.randint(-2, 3, (vis,), generator=g).float() * 0.5                            # five values
    rows[1][torch.rand(vis, generator=g) < 0.3] = -0.0
    rows[2] = torch.where(torch.rand(vis, generator=g) < 0.97, -0.0, 1.0)                      # few positives
    rows[3] = -torch.rand(vis, generator=g)                                                      # all negative
    s = rows.to(DEV)
    for ratio, pos in ((1, vis - 4), (1, 600), (1, 16385), (2, 2 * vis - 8)):
        lists = torch.full((4, K), -7, dtype=torch.int32, device=DEV)
        n = torch.full((4,), -7, dtype=torch.int32, device=DEV)
        cand = torch.full((4, BLOCKS), -7, dtype=torch.int32, device=DEV)
        cand_n = torch.full((4,), -7, dtype=torch.int32, device=DEV)
        indexer.topk(s, ratio, _pos(pos), lists, n)
        indexer.candidates(s, ratio, _pos(pos), cand, cand_n, block=BLK)
        for r in range(4):
            v = (pos + r + 1) // ratio
            assert torch.equal(lists[r, :int(n[r])].cpu(), _ref_topk(rows[r, :v], min(K, v))), (pos, r)
            assert torch.equal(cand[r, :int(cand_n[r])].cpu(), _ref_blocks(rows[r], v)), (pos, r)
        assert torch.equal(lists[0, :int(n[0])].cpu(), torch.arange(int(n[0]), dtype=torch.int32))


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("pos", [600, 16381, 20000])
def test_reindex_layers_score_only_the_candidate_blocks(pos, ties):
    q, w, k = _inputs(MAX_ROWS, pos + MAX_ROWS + 3, 4, ties)
    src = _run(q, w, k, 1, pos, source=True)
    qi, wi, _ = _inputs(MAX_ROWS, 1, 5, ties)
    within = (src.cand, src.cand_n)
    full = _run(qi, wi, k, 1, pos)
    o = _run(qi, wi, k, 1, pos, within=within)
    _check_selection(o, 1, pos, within=within)
    for r in range(MAX_ROWS):
        vis = pos + r + 1
        e = (src.blocks(r).long()[:, None] * BLK + torch.arange(BLK)).flatten()
        ok = e < vis
        assert torch.equal(_dense(o, r, vis, within)[e[ok]], full.s[r, e[ok]].cpu()), r     # unrestricted bits
        assert set(o.list(r).tolist()) <= set(e[ok].tolist()), r
        if vis > BLOCKS * BLK:
            assert len(e[ok]) < vis, r                                                   # a real restriction


def _rows_equal(a: Out, ra: int, b: Out, rb: int, vis: int, wa=None, wb=None) -> None:
    da, db = _dense(a, ra, vis, wa), _dense(b, rb, vis, wb)
    assert torch.equal(da.view(torch.int32), db.view(torch.int32))
    assert torch.equal(a.list(ra), b.list(rb))
    if a.cand_n.numel():
        assert torch.equal(a.blocks(ra), b.blocks(rb))


@pytest.mark.parametrize("ratio,pos", [(1, 508), (2, 1021), (1, 16381)])
def test_rows_do_not_depend_on_the_window(ratio, pos):
    q, w, k = _inputs(MAX_ROWS, (pos + MAX_ROWS) // ratio + 3, 6)
    full = _run(q, w, k, ratio, pos, source=ratio == 1)
    qi, wi, _ = _inputs(MAX_ROWS, 1, 7)
    rfull = _run(qi, wi, k, 1, pos, within=(full.cand, full.cand_n)) if ratio == 1 else None
    for n in range(1, MAX_ROWS + 1):
        for a in (0, MAX_ROWS - n):
            part = _run(q[a:a + n], w[a:a + n], k, ratio, pos + a, source=ratio == 1)
            for r in range(n):
                _rows_equal(part, r, full, a + r, (pos + a + r + 1) // ratio)
            if rfull is not None:
                within = (full.cand[a:a + n], full.cand_n[a:a + n])
                rpart = _run(qi[a:a + n], wi[a:a + n], k, 1, pos + a, within=within)
                for r in range(n):
                    _rows_equal(rpart, r, rfull, a + r, pos + a + r + 1, within, (full.cand, full.cand_n))


@pytest.mark.parametrize("ratio", [1, 2])
def test_prompt_row_blocks_and_windows_give_the_same_rows(ratio):
    pos, rows = 16300, 300
    q, w, k = _inputs(rows, (pos + rows) // ratio + 3, 8)
    whole = _run(q, w, k, ratio, pos, source=ratio == 1, pos_host=pos)
    for buf in (16, 48, 128):
        blocked = _run(q, w, k, ratio, pos, source=ratio == 1, pos_host=pos, buf_rows=buf)
        assert torch.equal(blocked.lists, whole.lists) and torch.equal(blocked.n, whole.n), buf
        assert torch.equal(blocked.cand_n, whole.cand_n), buf
    for a in (0, 137, rows - MAX_ROWS):
        win = _run(q[a:a + MAX_ROWS], w[a:a + MAX_ROWS], k, ratio, pos + a, source=ratio == 1)
        for r in range(MAX_ROWS):
            assert torch.equal(win.list(r), whole.list(a + r)), (a, r)
            if ratio == 1:
                assert torch.equal(win.blocks(r), whole.blocks(a + r)), (a, r)
    _check_selection(whole, ratio, pos, source=ratio == 1)


def _idx_weights(seed: int) -> IdxW:
    g = torch.Generator(device=DEV).manual_seed(seed)
    w8 = torch.randint(0, 256, (H * D, CFG.q_lora_rank), generator=g, dtype=torch.uint8, device=DEV)
    w8[(w8 & 0x7F) >= 0x70] = 0x30                                                  # finite, below 2^7
    s = torch.randint(118, 126, (H * D, CFG.q_lora_rank // 32), generator=g, dtype=torch.uint8, device=DEV)
    wproj = (torch.randn((H, CFG.hidden_size), generator=g, device=DEV) * 0.02).to(torch.bfloat16)
    return IdxW(Mx8Linear.from_checkpoint(w8.view(torch.float8_e4m3fn), s), make_b16(wproj))


def _activations(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return ((torch.randn((rows, CFG.q_lora_rank), generator=g, device=DEV)).to(torch.bfloat16),
            (torch.randn((rows, CFG.hidden_size), generator=g, device=DEV)).to(torch.bfloat16))


@pytest.fixture(scope="module")
def table() -> torch.Tensor:
    return rope.tables(CFG, "yarn", 40000, DEV)


def test_index_q_and_weights(table):
    iw = _idx_weights(9)
    qr, xa = _activations(MAX_ROWS, 10)
    pos = _pos(16381)
    qI = indexer.index_q(iw, qr, table, pos, torch.empty((MAX_ROWS, H, D), dtype=torch.bfloat16, device=DEV))
    want = mx8.mm(iw.wq_b, qr).view(MAX_ROWS, H, D).clone()
    assert torch.equal(qI, quant.fp4_qdq_1x32_e8m0(rope.apply(want, pos, table)))
    wI = indexer.index_weights(iw, xa, torch.empty((MAX_ROWS, H), device=DEV))
    exact = xa.double() @ iw.wproj.weight.double().t() * (D**-0.5 * H**-0.5)
    bound = 8 * math.sqrt(xa.shape[1]) * 2.0**-24 * (xa.double().abs() @ iw.wproj.weight.double().abs().t()) * \
        (D**-0.5 * H**-0.5)
    assert float(((wI.double() - exact).abs() / bound).max()) <= 1
    for n in range(1, MAX_ROWS + 1):                                                   # rows alone equal rows inside
        a = MAX_ROWS - n
        alone = indexer.index_q(iw, qr[a:], table, _pos(16381 + a), torch.empty((n, H, D), dtype=torch.bfloat16,
                                                                                 device=DEV))
        assert torch.equal(alone, qI[a:]), n
        assert torch.equal(indexer.index_weights(iw, xa[a:], torch.empty((n, H), device=DEV)), wI[a:]), n


def test_graph_replays_equal_eager(table):
    iw, iw2 = _idx_weights(11), _idx_weights(12)
    _, _, k = _inputs(1, 20000, 13)
    pos = _pos(0)
    for R in range(1, MAX_ROWS + 1):
        qr, xa = _activations(R, 14)
        qI = torch.empty((R, H, D), dtype=torch.bfloat16, device=DEV)
        wI = torch.empty((R, H), device=DEV)
        src, rei = Out(R, k.shape[0]), Out(R, k.shape[0])

        def step(qr=qr, xa=xa, qI=qI, wI=wI, src=src, rei=rei):
            for w, o, source, within in ((iw, src, (src.cand, src.cand_n), None),
                                         (iw2, rei, None, (src.cand, src.cand_n))):
                indexer.index_q(w, qr, table, pos, qI)
                indexer.index_weights(w, xa, wI)
                indexer.select(CFG, qI, wI, k, 1, pos, o.s, o.lists, o.n, source=source, within=within)

        pos.fill_(17000)
        step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for p, seed in ((16380, 15), (19990, 16), (300, 17)):
            nq, nx = _activations(R, seed)
            qr.copy_(nq)
            xa.copy_(nx)
            pos.fill_(p)
            graph.replay()
            torch.cuda.synchronize()
            got = [t.clone() for t in (src.lists, src.n, src.cand, src.cand_n, rei.lists, rei.n)]
            step()
            torch.cuda.synchronize()
            for i, t in enumerate((src.lists, src.n, src.cand, src.cand_n, rei.lists, rei.n)):
                assert torch.equal(got[i], t), (R, p, i)
            _check_selection(rei, 1, p, within=(src.cand, src.cand_n))


def _swaps_within(mine: set, theirs: set, score: torch.Tensor, delta: float) -> None:
    """Two top sets of one row: an item only the reference took outscores (its fp64) one only TF took by at most twice
    the largest change between the two score sets, so the difference is the scores', never the selection's."""

    if theirs - mine:
        gap = max(float(score[t]) for t in theirs - mine) - min(float(score[t]) for t in mine - theirs)
        assert gap <= 2 * delta, (gap, delta)


@pytest.mark.skipif(not MODEL, reason="set TF_DSV41_MODEL to the checkpoint")
def test_real_layers_track_the_reference_indexer():
    """Layers 2, 20 and 24 on the checkpoint's weights: scores within the op-level bound on TF's own index Q, exact
    selection on them, and the reference port's mirror-mode Indexer (its own Q, bf16 weights) differing only within the
    measured score change."""

    pytest.importorskip("safetensors")
    from dsv41_layouts import mx8_from_block
    from dsv41_ref_pack import RefPack

    pack = RefPack(MODEL)
    cfg = Config.read(MODEL)
    model = ref.Reference(cfg, lambda name: pack.dense_fp32(name.removesuffix(".weight")), mode=ref.MIRROR)
    g = torch.Generator().manual_seed(18)
    n, tf_cand, equal_cand = 8, None, []
    for layer, S in ((2, 2048), (20, 16400), (24, 16400)):
        p, role = f"layers.{layer}", cfg.roles[layer]
        ratio, pos = role.ratio, S - n
        x = torch.randn((S, cfg.hidden_size), generator=g).to(torch.bfloat16).float()
        xa = ref.rms_norm(x, model.W(f"{p}.attn_norm.weight"), cfg.rms_norm_eps, ref.MIRROR)
        if role.kv_src == layer:
            model.compress(layer, xa, 0)
        rows = xa[pos:]
        qr = ref.rms_norm(ref.MIRROR.bf16(ref.linear(rows, model.W(f"{p}.attn.wq_a.weight"), ref.MIRROR)),
                          model.W(f"{p}.attn.q_norm.weight"), cfg.rms_norm_eps, ref.MIRROR)
        want = model.indexer(layer, rows, qr, pos)
        # TF on the same keys, rows and query latents
        iw = IdxW(mx8_from_block(pack.tensor(f"{p}.attn.indexer.wq_b.weight").cuda(),
                                 pack.tensor(f"{p}.attn.indexer.wq_b.scale").cuda()),
                  make_b16(pack.tensor(f"{p}.attn.indexer.weights_proj.weight").cuda()))
        keys = model.state.index_k[role.kv_src].to(torch.bfloat16).cuda()
        qI = indexer.index_q(iw, qr.to(torch.bfloat16).cuda(), rope.tables(cfg, "yarn", S, DEV), _pos(pos),
                             torch.empty((n, H, D), dtype=torch.bfloat16, device=DEV))
        wI = indexer.index_weights(iw, rows.to(torch.bfloat16).cuda(), torch.empty((n, H), device=DEV))
        within = tf_cand if role.uses_candidates else None
        o = _run(qI, wI, keys, ratio, pos, source=role.candidate_source, within=within)
        _check_selection(o, ratio, pos, within=within, source=role.candidate_source)
        s64, bound = _fp64(qI, wI, keys, ratio, pos)
        # the reference's scores from its own index Q and bf16 weights (Reference.indexer's formulas, in fp64)
        qb = ref.MIRROR.bf16(ref.linear(qr, model.W(f"{p}.attn.indexer.wq_b.weight"), ref.MIRROR))
        qref = ref.fp4_act_quant(ref.rope(qb.unflatten(-1, (H, D)), model.cis(role.rope, torch.arange(pos, S)),
                                          ref.MIRROR), 32)
        wref = ref.MIRROR.bf16(rows @ model.W(f"{p}.attn.indexer.weights_proj.weight").T) * (D**-0.5 * H**-0.5)
        sref = _fp64(qref.cuda(), wref.cuda(), keys, ratio, pos)[0].cpu()
        t1, overlap = 0.0, []
        for r in range(n):
            vis = (pos + r + 1) // ratio
            dense = _dense(o, r, vis, within).double()
            live = torch.isfinite(dense)
            t1 = max(t1, float(((dense - s64[r, :vis].cpu()).abs()[live] / bound[r, :vis].cpu()[live]).max()))
            delta = float((dense - sref[r, :vis]).abs()[live].max())
            mine, theirs = set(o.list(r).tolist()), set(want[r].tolist())
            if within is None or equal_cand[r]:         # the same candidate pool on both sides
                _swaps_within(mine, theirs, sref[r, :vis], delta)
            overlap.append(len(mine & theirs) / max(len(theirs), 1))
            if role.candidate_source:
                blocks = torch.unique(torch.nonzero(model.state.cand[r])[:, 0] // BLK).int()
                bmax = torch.nn.functional.pad(sref[r, :vis], (0, -vis % BLK), value=-math.inf).view(-1, BLK).amax(1)
                _swaps_within(set(o.blocks(r).tolist()), set(blocks.tolist()), bmax, delta)
                equal_cand.append(torch.equal(o.blocks(r), blocks))
        assert t1 <= 1, (layer, t1)
        if role.candidate_source:
            tf_cand = (o.cand, o.cand_n)
        print(f"layer {layer}: index Q off the reference {float((qI.cpu().float() != qref).double().mean()):.2e}, "
              f"bound {t1:.3f}, top-k overlap {min(overlap):.4f}..{max(overlap):.4f}, "
              f"candidate rows equal the reference {sum(equal_cand)}/{len(equal_cand)}")
