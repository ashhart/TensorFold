"""The attention sublayer: the model-level gate against the reference port in mirror mode (tiny checkpoint; real layers
with ``TF_DSV41_MODEL``), decode rows bit-equal alone and in windows, prompt rows bit-equal however the prompt is
chunked, graph replays equal to eager.

Each test feeds every layer the same streams and runs the layers in order, so reuse layers read their index layer's
lists of the same forward. The commit here (ring rows, compressor tails, position) is the forward's, written out.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_tiny
from dsv41_ref_pack import RefPack
from dsv41_ref_weights import RefWeights
from dsv41_reference import Mode, Reference, hc_pre, rms_norm

from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import MAX_ROWS, PREFILL_ROWS, buffers, compressor, loader, rope
from tensorfold.families.deepseek_v41.cuda import attention as attn

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
MODEL = os.environ.get("TF_DSV41_MODEL", "")
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="set TF_DSV41_MODEL to the checkpoint")
MIRROR = Mode("mirror", world=1)        # one rank: wo_b sums its whole input dim
REL_L2 = 4 * 2**-8                      # four bf16 half-ulps
COS = 0.9999
TINY_CAP = 1024
TINY_PROMPT = 320                       # prompt-chunk rows of the tiny runs
PICKS = 0.01                            # index picks that may differ from the reference's (near-ties)
REF_ROWS = 256                          # rows the reference indexer scores at once (its scores are [rows, heads, T])
# (rows, prompt) forwards: windows at odd and even starts, ring wrap, contexts 40 to 600
TINY_STEPS = [(40, True), (5, False), (1, False), (1, False), (2, False), (150, True), (6, False), (1, False),
              (255, True), (139, True), (3, False), (4, False)]
REAL_LAYERS = (0, 2, 3, 20, 21, 24)
REAL_CAP = 17_024
DECODE = [(5, False), (1, False), (1, False), (2, False)]     # from an even start: rows at even and odd positions
REAL_STEPS = ([(100, True)] + DECODE + [(991, True)] + DECODE + [(PREFILL_ROWS, True)] * 7
              + [(17_000 - 1109 - 7 * PREFILL_ROWS, True)] + DECODE)


def _inputs(cfg: Config, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Streams bf16 [n, hc, D] and an attention pre fp32 [n, hc] in (0.25, 1), as sigmoid + eps gives."""

    g = torch.Generator().manual_seed(seed)
    X = torch.randn((n, cfg.hc_mult, cfg.hidden_size), generator=g).to(torch.bfloat16)
    return X, 0.25 + 0.75 * torch.rand((n, cfg.hc_mult), generator=g)


def _embedded(cfg: Config, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The checkpoint's embedding rows of random ids, a different id stream in each copy."""

    g = torch.Generator().manual_seed(seed)
    embed = RefPack(MODEL).tensor("embed.weight")
    X = embed[torch.randint(0, cfg.vocab_size, (n, cfg.hc_mult), generator=g)]
    del embed
    return X, 0.25 + 0.75 * torch.rand((n, cfg.hc_mult), generator=g)


class Run:
    """TF's attention for ``layers`` over one sequence: forwards of a decode window or prompt chunk, then commits."""

    def __init__(self, w, layers, capacity: int, prompt_rows: int) -> None:
        self.w, self.cfg, self.layers = w, w.cfg, layers
        self.state = buffers.State(w.cfg, capacity, "cuda")
        self.dec = buffers.Buffers(w.cfg, 1, MAX_ROWS, capacity, device="cuda")
        self.chunk = buffers.Buffers(w.cfg, 1, prompt_rows, capacity, prefill=True, device="cuda")

    def buf(self, prompt: bool) -> buffers.Buffers:
        return self.chunk if prompt else self.dec

    def compute(self, X: torch.Tensor, pre: torch.Tensor, prompt: bool) -> list[torch.Tensor]:
        """Every layer's fp32 part [R, D] for rows at ``state.pos``.., nothing committed."""

        b, R = self.buf(prompt), len(X)
        b.X[:R].copy_(X)
        b.pre_in[:R].copy_(pre)
        out, self.lists = [], {}
        for lw in self.layers:
            out.append(attn.attention(lw, self.w, self.state, b, R, prompt).clone())
            if lw.role.ratio and lw.role.idx_src == lw.index:       # each index layer's lists, for the reference
                n, lists = b.list_n[:R].tolist(), b.lists[:R].cpu().long()
                self.lists[lw.index] = [lists[r, :n[r]] for r in range(R)]
        return out

    def commit(self, prompt: bool, keep: int) -> None:
        """The kept rows' window KV into ring slots (pos + r) % window (the last window rows), tails, position."""

        st, b, win = self.state, self.buf(prompt), self.cfg.sliding_window
        n = min(keep, win)
        slots = (st.pos + torch.arange(keep - n, keep, device="cuda")) % win
        for lw in self.layers:
            st.rings[lw.index, slots] = b.kvw[lw.index, keep - n:keep]
        compressor.commit_tail(st, b, st.pos, keep)
        st.set_pos(st.pos + keep)

    def forward(self, X: torch.Tensor, pre: torch.Tensor, prompt: bool) -> list[torch.Tensor]:
        out = self.compute(X.cuda(), pre.cuda(), prompt)
        self.commit(prompt, len(X))
        return out

    def snapshot(self) -> tuple:
        st = self.state
        return (st.pos, st.rings.clone(), {k: v.clone() for k, v in st.comp.items()},
                {k: v.clone() for k, v in st.index_k.items()}, st.tail.clone(), st.tail_valid.clone())

    def restore(self, snap: tuple) -> None:
        st = self.state
        pos, rings, comp, index_k, tail, valid = snap
        st.rings.copy_(rings)
        for k in comp:
            st.comp[k].copy_(comp[k])
            st.index_k[k].copy_(index_k[k])
        st.tail.copy_(tail)
        st.tail_valid.copy_(valid)
        st.set_pos(pos)


class Ref:
    """The reference port's attention for the same layers, on TF's RoPE tables, in row blocks of REF_ROWS.

    Its indexers attend TF's lists: a near-tie picked the other way moves a row by a whole entry's share, which says
    nothing about the rest of the layer. How many of the reference's own picks differ is counted instead.
    """

    def __init__(self, cfg: Config, weight, tables: dict[str, torch.Tensor], layers) -> None:
        self.r = Reference(cfg, weight, mode=MIRROR)
        self.r._tables.update({kind: torch.view_as_complex(t.cpu()) for kind, t in tables.items()})
        self.cfg, self.layers = cfg, [lw.index for lw in layers]
        self.own, self.r.indexer = self.r.indexer, self._indexer
        self.given: dict[int, list[torch.Tensor]] = {}
        self.differ = self.picked = 0

    def _indexer(self, layer: int, x: torch.Tensor, qr: torch.Tensor, start: int) -> list[torch.Tensor]:
        own = self.own(layer, x, qr, start)
        given = self.given[layer][start - self.r.state.pos:][:len(x)]
        for a, b in zip(own, given):
            self.differ += len(set(a.tolist()) ^ set(b.tolist()))
            self.picked += len(a)
        return given

    def forward(self, X: torch.Tensor, pre: torch.Tensor, lists: dict[int, list[torch.Tensor]]) -> list[torch.Tensor]:
        r, outs, self.given = self.r, [[] for _ in self.layers], lists
        for a in range(0, len(X), REF_ROWS):
            Xa, pa = X[a:a + REF_ROWS].float(), pre[a:a + REF_ROWS].float()
            for i, L in enumerate(self.layers):
                xa = rms_norm(hc_pre(Xa, pa, MIRROR), r.W(f"layers.{L}.attn_norm.weight"), self.cfg.rms_norm_eps,
                              MIRROR)
                outs[i].append(r.attention(L, xa, r.state.pos + a))
        r.state.pos += len(X)
        return [torch.cat(o) for o in outs]


def _t2(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    g, x = got.double().cpu().flatten(), want.double().flatten()
    rel = float((g - x).norm() / x.norm())
    cos = float(g @ x / (g.norm() * x.norm()))
    assert rel <= REL_L2 and cos >= COS, f"{what}: rel-L2 {rel:.3g}, cosine {cos:.6f}"


def _against_reference(run: Run, ref: Ref, steps, X: torch.Tensor, pre: torch.Tensor, prompt_rows: int) -> None:
    at = 0
    for rows, prompt in steps:
        for a in range(at, at + rows, prompt_rows if prompt else rows):
            n = min(prompt_rows if prompt else rows, at + rows - a)
            got = run.forward(X[a:a + n], pre[a:a + n], prompt)
            want = ref.forward(X[a:a + n], pre[a:a + n], run.lists)
            for lw, g, x in zip(run.layers, got, want):
                _t2(g, x, f"layer {lw.index} ({lw.role.mode}), {n} {'prompt' if prompt else 'decode'} rows at {a}")
        at += rows
    print(f"index picks differing from the reference's: {ref.differ} of {ref.picked}")
    assert ref.differ <= PICKS * ref.picked


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    cfg = Config.read(tiny_dir)
    return loader.load(tiny_dir, cfg, 0, 1, None, dspark=False, capacity=TINY_CAP), tiny_dir


def test_tiny_layers_follow_the_reference(tiny):
    w, path = tiny
    n = sum(r for r, _ in TINY_STEPS)
    X, pre = _inputs(w.cfg, n, 0)
    run = Run(w, w.layers, TINY_CAP, TINY_PROMPT)
    _against_reference(run, Ref(w.cfg, RefWeights(path), w.rope, w.layers), TINY_STEPS, X, pre, TINY_PROMPT)


# odd and even starts: index visible 15-17 (ratio 1) and 15-17 (ratio 2, q 30-35), the 32-entry candidate pool,
# ring wrap, a long context
@pytest.mark.parametrize("start", [13, 30, 125, 400])
def test_decode_rows_alone_equal_rows_in_windows(tiny, start):
    w, _ = tiny
    X, pre = _inputs(w.cfg, start + MAX_ROWS, 1)
    run = Run(w, w.layers, TINY_CAP, TINY_PROMPT)
    for a in range(0, start, TINY_PROMPT):
        run.forward(X[a:min(start, a + TINY_PROMPT)], pre[a:min(start, a + TINY_PROMPT)], True)
    snap = run.snapshot()
    alone = [run.forward(X[start + r:start + r + 1], pre[start + r:start + r + 1], False) for r in range(MAX_ROWS)]
    for R in range(2, MAX_ROWS + 1):
        run.restore(snap)
        window = run.forward(X[start:start + R], pre[start:start + R], False)
        for i, lw in enumerate(w.layers):
            for r in range(R):
                assert torch.equal(window[i][r], alone[r][i][0]), f"layer {lw.index}: row {r} of {R} at {start}"


@pytest.mark.parametrize("chunks", [[1, 7, 64, 128, 100], [129, 171], [256, 44], [2, 298]])
def test_prompt_rows_equal_across_chunkings(tiny, chunks):
    w, _ = tiny
    n = sum(chunks)
    X, pre = _inputs(w.cfg, n, 2)

    def run_in(sizes):
        run = Run(w, w.layers, TINY_CAP, n)
        outs, at = [], 0
        for size in sizes:
            outs.append(run.forward(X[at:at + size], pre[at:at + size], True))
            at += size
        st = run.state
        return [torch.cat(o) for o in zip(*outs)], [st.rings, *st.row_views(n), st.tail * st.tail_valid[:, None, None]]

    whole, whole_state = run_in([n])
    split, split_state = run_in(chunks)
    for lw, a, b in zip(w.layers, whole, split):
        assert torch.equal(a, b), f"layer {lw.index}: chunks {chunks}"
    for i, (a, b) in enumerate(zip(whole_state, split_state)):
        assert torch.equal(a, b), f"state tensor {i}: chunks {chunks}"


@pytest.mark.parametrize("R", [1, 4, 6])
def test_graph_replays_equal_eager(tiny, R):
    w, _ = tiny
    start, steps = 150, 5
    X, pre = _inputs(w.cfg, start + steps * R, 3)
    run = Run(w, w.layers, TINY_CAP, TINY_PROMPT)
    run.forward(X[:start], pre[:start], True)
    b, st = run.dec, run.state
    outs = [torch.empty((R, w.cfg.hidden_size), dtype=torch.float32, device="cuda") for _ in w.layers]

    def step():
        for o, lw in zip(outs, w.layers):
            o.copy_(attn.attention(lw, w, st, b, R, False))

    snap = run.snapshot()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    run.restore(snap)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for k in range(steps):
        a = start + k * R
        snap = run.snapshot()
        eager = run.compute(X[a:a + R].cuda(), pre[a:a + R].cuda(), False)
        run.restore(snap)
        b.X[:R].copy_(X[a:a + R])
        b.pre_in[:R].copy_(pre[a:a + R])
        graph.replay()
        for lw, e, g in zip(w.layers, eager, outs):
            assert torch.equal(e, g), f"layer {lw.index}: {R} rows at {a}"
        run.commit(False, R)


def _real_attention(cfg: Config, layers, capacity: int):
    """Only the attention weights of ``layers`` (whole tensors), and RoPE tables of ``capacity`` slots."""

    dev = torch.device("cuda", torch.cuda.current_device())
    pack = loader._Pack(MODEL, 0, 1, dev)
    try:
        b = loader._Build(pack, cfg)
        lws = [SimpleNamespace(index=L, role=cfg.roles[L], attn_norm=b.t(f"layers.{L}.attn_norm.weight"),
                               attn=b.attn(f"layers.{L}.", L)) for L in layers]
    finally:
        pack.close()
    return SimpleNamespace(cfg=cfg, rope={k: rope.tables(cfg, k, capacity, dev) for k in rope.KINDS}), lws


@needs_model
def test_real_layers_follow_the_reference_to_17000():
    cfg = Config.read(MODEL)
    w, lws = _real_attention(cfg, REAL_LAYERS, REAL_CAP)
    n = sum(r for r, _ in REAL_STEPS)
    X, pre = _embedded(cfg, n, 4)
    run = Run(w, lws, REAL_CAP, PREFILL_ROWS)
    _against_reference(run, Ref(cfg, RefWeights(MODEL), w.rope, lws), REAL_STEPS, X, pre, PREFILL_ROWS)
