"""Compressor, index-K and compressed entries: the op-level bound against the reference port, bit-equal however rows are
split.

The reference runs on the CPU with the device's RoPE table, so the tables add no difference of their own.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_reference as ref
import dsv41_tiny

from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import buffers, compressor, rope
from tensorfold.families.deepseek_v41.cuda.weights import CompW, IdxW
from tensorfold.families.glm5_next.cuda.qmm import make_b16

MODEL = os.environ.get("TF_DSV41_MODEL")
TINY = Config.from_dict(dsv41_tiny.config())
CAPACITY = 4096
N = 41                              # positions of the split tests: an odd count leaves the last group open


def _names(layer: int) -> dict[str, str]:
    p = f"layers.{layer}.attn."
    return {"wkv": p + "compressor.wkv.weight", "wgate": p + "compressor.wgate.weight",
            "norm": p + "compressor.norm.weight", "wk": p + "indexer.wk.weight", "k_norm": p + "indexer.k_norm.weight"}


def _random_weights(cfg: Config, seed: int = 0) -> dict[str, torch.Tensor]:
    """bf16 compressor and index-K weights of every KV source, scaled so pooled latents and scores are O(1)."""

    gen = torch.Generator().manual_seed(seed)
    D, hd, dk = cfg.hidden_size, cfg.head_dim, cfg.index_head_dim
    out = {}
    for layer in cfg.kv_source_layer_ids:
        n = _names(layer)
        out[n["wkv"]] = torch.randn((hd, D), generator=gen) * D**-0.5
        if cfg.roles[layer].ratio == 2:
            out[n["wgate"]] = torch.randn((hd, D), generator=gen) * 2 * D**-0.5
        out[n["norm"]] = 1 + 0.5 * torch.randn(hd, generator=gen)
        out[n["wk"]] = torch.randn((dk, hd), generator=gen) * hd**-0.5
        out[n["k_norm"]] = 1 + 0.5 * torch.randn(dk, generator=gen)
    return {k: v.to(torch.bfloat16) for k, v in out.items()}


def _layer(cfg: Config, layer: int, w: dict[str, torch.Tensor]) -> SimpleNamespace:
    """The LayerW fields ``compress`` reads, on the device."""

    n, ratio = _names(layer), cfg.roles[layer].ratio
    kv = torch.cat([w[n["wkv"]], w[n["wgate"]]]) if ratio == 2 else w[n["wkv"]]
    comp = CompW(ratio, make_b16(kv.cuda()), w[n["norm"]].cuda())
    idx = IdxW(None, None, make_b16(w[n["wk"]].cuda()), w[n["k_norm"]].cuda())
    return SimpleNamespace(index=layer, attn=SimpleNamespace(comp=comp, idx=idx))


class Run:
    """One sequence through the compressor of ``layers``: forwards of R rows, each followed by its commit."""

    def __init__(self, cfg: Config, layers: list[int], w: dict[str, torch.Tensor], rows: int = 64,
                 capacity: int = CAPACITY) -> None:
        self.cfg, self.layers = cfg, [_layer(cfg, i, w) for i in layers]
        self.state = buffers.State(cfg, capacity, "cuda")
        self.buf = buffers.Buffers(cfg, 1, rows, capacity, prefill=rows > 6, device="cuda")
        self.table = rope.tables(cfg, "yarn", capacity, "cuda")

    def compress(self, xa: torch.Tensor) -> None:
        for lw in self.layers:
            compressor.compress(lw, xa, self.state, self.buf, self.table, self.cfg.rms_norm_eps)

    def forward(self, xa: torch.Tensor, keep: int | None = None) -> None:
        keep = len(xa) if keep is None else keep
        self.compress(xa.cuda())
        compressor.commit_tail(self.state, self.buf, self.state.pos, keep)
        self.state.set_pos(self.state.pos + keep)

    def entries(self, n: int) -> list[torch.Tensor]:
        """Every source's entries and keys of groups complete within the first ``n`` positions, then the tail."""

        st = self.state
        out = [c[layer][:n // st.ratio[layer]].clone() for layer in st.ratio for c in (st.comp, st.index_k)]
        valid = st.tail_valid.clone()
        return [*out, valid, st.tail * valid[:, None, None]]


def _inputs(cfg: Config, n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn((n, cfg.hidden_size), generator=gen).to(torch.bfloat16)


def _same(a: list[torch.Tensor], b: list[torch.Tensor], what: str) -> None:
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x, y), f"{what}: tensor {i} differs"


def _t1_qdq(got: torch.Tensor, want: torch.Tensor, group: int, what: str) -> None:
    """QDQ'd values: at most 0.5% differ, each by less than its group's largest value (one grid step at most)."""

    g, w = got.float().cpu(), want.float()
    differ = g != w
    frac = float(differ.double().mean())
    assert frac <= 0.005, f"{what}: {frac:.4%} of the values differ"
    amax = w.unflatten(-1, (-1, group)).abs().amax(-1, keepdim=True).expand(*w.shape[:-1], -1, group).flatten(-2)
    assert bool(((g - w).abs() <= amax)[differ].all()), f"{what}: a value moved by more than one grid step"


def _reference(cfg: Config, w: dict[str, torch.Tensor], table: torch.Tensor, start: int) -> ref.Reference:
    """The mirror port with TF's table, its caches holding ``start // ratio`` placeholder entries."""

    r = ref.Reference(cfg, lambda name: w[name], mode=ref.MIRROR)
    r._tables["yarn"] = torch.view_as_complex(table.cpu())
    for layer, ratio in ((i, cfg.roles[i].ratio) for i in cfg.kv_source_layer_ids):
        r.state.comp[layer] = torch.zeros((start // ratio, cfg.head_dim))
        r.state.index_k[layer] = torch.zeros((start // ratio, cfg.index_head_dim))
    return r


def _against_reference(cfg: Config, w: dict[str, torch.Tensor], xa: torch.Tensor, start: int, chunks: list[int],
                       capacity: int) -> None:
    run = Run(cfg, list(cfg.kv_source_layer_ids), w, rows=max(chunks), capacity=capacity)
    run.state.set_pos(start)
    r = _reference(cfg, w, run.table, start)
    at = 0
    for n in chunks:
        run.forward(xa[at:at + n])
        for layer in cfg.kv_source_layer_ids:
            r.compress(layer, xa[at:at + n].float(), start + at)
        at += n
    for layer in cfg.kv_source_layer_ids:
        ratio = cfg.roles[layer].ratio
        js = slice(start // ratio, (start + at) // ratio)
        assert len(r.state.comp[layer]) == js.stop
        _t1_qdq(run.state.comp[layer][js], r.state.comp[layer][js], 16, f"layer {layer} entries")
        _t1_qdq(run.state.index_k[layer][js], r.state.index_k[layer][js], 32, f"layer {layer} index-K")


def test_tiny_matches_the_reference_key_path():
    w = _random_weights(TINY)
    xa = _inputs(TINY, 96, 1)
    _against_reference(TINY, w, xa, 1000, [7, 25, 1, 63], CAPACITY)


@pytest.mark.skipif(not MODEL, reason="set TF_DSV41_MODEL to the checkpoint")
def test_real_layers_2_and_20_match_the_reference_key_path():
    from dsv41_ref_pack import RefPack

    pack = RefPack(MODEL)
    cfg = Config.read(MODEL)
    assert {2, 20} <= set(cfg.kv_source_layer_ids)
    w = {}
    for layer in cfg.kv_source_layer_ids:
        for name in _names(layer).values():
            if pack.has(name):
                w[name] = pack.tensor(name)
    norm = pack.tensor("layers.2.attn_norm.weight").float()
    xa = ref.rms_norm(_inputs(cfg, 64, 2).float(), norm, cfg.rms_norm_eps, ref.MIRROR).to(torch.bfloat16)
    start = 65536
    _against_reference(cfg, w, xa, start, [5, 1, 58], start + 128)


@pytest.fixture(scope="module")
def tiny():
    return _random_weights(TINY, 3), _inputs(TINY, N, 4)


def _whole(w, xa) -> list[torch.Tensor]:
    run = Run(TINY, list(TINY.kv_source_layer_ids), w)
    run.forward(xa)
    return run.entries(N)


def _serial(w, xa, upto: int = N) -> Run:
    run = Run(TINY, list(TINY.kv_source_layer_ids), w, rows=6)
    for i in range(upto):
        run.forward(xa[i:i + 1])
    return run


def test_one_forward_equals_serial_decode_and_prompt_chunks(tiny):
    w, xa = tiny
    whole = _whole(w, xa)
    _same(_serial(w, xa).entries(N), whole, "serial")
    for cuts in ([7, 8, 21], [1, 2, 3, 30], [20, 21]):       # chunk boundaries at odd and even positions
        run = Run(TINY, list(TINY.kv_source_layer_ids), w)
        for a, b in zip([0, *cuts], [*cuts, N]):
            run.forward(xa[a:b])
        _same(run.entries(N), whole, f"chunks {cuts}")


@pytest.mark.parametrize("rows", range(2, 7))
@pytest.mark.parametrize("start", [10, 11])
def test_a_window_equals_serial_rows(tiny, rows, start):
    w, xa = tiny
    run = _serial(w, xa, start)
    run.forward(xa[start:start + rows])
    _same(run.entries(start + rows), _serial(w, xa, start + rows).entries(start + rows), f"R={rows} at {start}")


def test_rejected_rows_are_rewritten(tiny):
    """Windows keep a random prefix; rows past it carry other inputs, and their entries are rewritten later."""

    w, xa = tiny
    gen = torch.Generator().manual_seed(5)
    junk = _inputs(TINY, 6, 6)
    run = Run(TINY, list(TINY.kv_source_layer_ids), w, rows=6)
    seen = set()
    while run.state.pos < N:
        pos = run.state.pos
        rows = min(int(torch.randint(1, 7, (1,), generator=gen)), N - pos)
        keep = int(torch.randint(1, rows + 1, (1,), generator=gen))
        x = torch.cat([xa[pos:pos + keep], junk[:rows - keep]])
        seen.add((rows, keep, pos % 2))
        run.forward(x, keep)
    assert len(seen) > 12
    _same(run.entries(N), _whole(w, xa), "random keeps")


def test_a_ratio_2_source_keeps_its_own_index_k_at_even_positions(tiny):
    """At an even position layer 2 writes nothing; its keys never take the ratio-1 source's."""

    w, xa = tiny
    (r2, r1) = sorted(TINY.kv_source_layer_ids, key=lambda i: -TINY.roles[i].ratio)
    both, alone = Run(TINY, [r2, r1], w, rows=6), Run(TINY, [r2], w, rows=6)
    for i in range(N):
        before = (both.state.comp[r2].clone(), both.state.index_k[r2].clone())
        both.forward(xa[i:i + 1])
        alone.forward(xa[i:i + 1])
        if i % 2 == 0:
            assert torch.equal(both.state.comp[r2], before[0]) and torch.equal(both.state.index_k[r2], before[1])
    keys = both.state.index_k[r2][:N // 2]
    assert torch.equal(keys, alone.state.index_k[r2][:N // 2])
    assert not (keys[:, None] == both.state.index_k[r1][None, :N]).all(-1).any()


def _live(st) -> list[torch.Tensor]:
    return [*st.comp.values(), *st.index_k.values(), st.tail, st.tail_valid]


@pytest.mark.parametrize("rows", [1, 4, 6])
def test_cuda_graph_replay_equals_eager(tiny, rows):
    """One capture replays at odd and even positions with the tail read on the device."""

    w, xa = tiny
    probe, saved = _serial(w, xa, 12), {}
    for p in (12, 13):
        if p == 13:
            probe.forward(xa[12:13])
        saved[p] = [t.clone() for t in _live(probe.state)]
    run = Run(TINY, list(TINY.kv_source_layer_ids), w, rows=6)
    st, live = run.state, _live(run.state)
    x = xa[20:20 + rows].cuda()

    def restore(p: int) -> None:
        for t, s in zip(live, saved[p]):
            t.copy_(s)
        run.buf.lat.zero_()
        run.buf.kI.zero_()
        st.set_pos(p)

    restore(13)
    run.compress(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run.compress(x)
    for p in (12, 13):
        restore(p)
        run.compress(x)
        eager = [t.clone() for t in (*live, run.buf.lat[:rows], run.buf.kI[:rows])]
        restore(p)
        graph.replay()
        torch.cuda.synchronize()
        _same([t.clone() for t in (*live, run.buf.lat[:rows], run.buf.kI[:rows])], eager, f"graph at {p}")
