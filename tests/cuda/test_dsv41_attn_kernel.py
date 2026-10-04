"""Sparse attention: the op-level bound against the reference port's sparse_attn in mirror mode, bit-equal rows, graphs.

Ring slots, ``kvw`` rows and cache entries a row must not read hold NaN, so a wrong read cannot pass.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_reference as ref

from tensorfold.families.deepseek_v41.cuda import attn_kernel as ak
from tensorfold.families.deepseek_v41.cuda import buffers

D, WIN, TOPK = 512, 128, 512
NAN = float("nan")


def _t1(out: torch.Tensor, rows: list[tuple[torch.Tensor, torch.Tensor]], sink: torch.Tensor, what: str) -> None:
    """The op-level bound for attention against the reference port's mirror-mode sparse_attn over each row's (q [H, D],
    entries [n, D]).

    Both round every P to bf16 (2^-9 relative), the kernel against a running max and the reference against the row's
    max, and both round the output once: |out - ref| <= 2^-8 (sum P |v| / den + |ref|) per element. Neither may sit
    further from the fp64 attention than the other beyond 1%.
    """

    want, exact, mag = [], [], []
    for q, e in rows:
        want.append(ref.sparse_attn(q.float(), e.float(), sink, D**-0.5, ref.MIRROR))
        s = (q.double() @ e.double().T) * D**-0.5
        p = torch.exp(s - s.amax(1, keepdim=True))
        den = p.sum(1, keepdim=True) + torch.exp(sink.double()[:, None] - s.amax(1, keepdim=True))
        exact.append(p @ e.double() / den)
        mag.append(p @ e.double().abs() / den)
    want, exact, mag = torch.stack(want).to(torch.bfloat16).double(), torch.stack(exact), torch.stack(mag)
    got = out.double()
    worst = float(((got - want).abs() / (2**-8 * (mag + want.abs()))).max())
    assert worst <= 1, f"{what}: {worst:.3f} of the bound"
    err, err_ref = ((x - exact).pow(2).mean().sqrt() for x in (got, want))
    assert err <= 1.01 * err_ref, f"{what}: rms error {float(err):.3g} vs the reference's {float(err_ref):.3g}"


class Seq:
    """One layer's KV for positions 0..n-1, its compressed entries and every row's ascending list."""

    def __init__(self, n: int, ratio: int, heads: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.ratio, self.heads = ratio, heads
        self.kv = torch.randn((n, D), generator=g).to(torch.bfloat16)
        self.q = (2 * torch.randn((n, heads, D), generator=g)).to(torch.bfloat16)
        self.comp = torch.randn((n // ratio + 1, D), generator=g).to(torch.bfloat16)
        self.sink = torch.randn(heads, generator=g)
        self.lists = torch.full((n, TOPK), -1, dtype=torch.int32)
        self.counts = torch.zeros(n, dtype=torch.int32)
        for i in range(n):
            visible = (i + 1) // ratio
            pick = torch.randperm(visible, generator=g)[:min(TOPK, visible)].sort().values
            self.lists[i, :len(pick)] = pick.int()
            self.counts[i] = len(pick)

    def ring(self, pos: int) -> torch.Tensor:
        """The ring after committing positions < pos; slots holding no position in [pos-128, pos) are NaN."""

        ring = torch.full((WIN, D), NAN).to(torch.bfloat16)
        for w in range(max(0, pos - WIN), pos):
            ring[w % WIN] = self.kv[w]
        return ring

    def comp_seen(self, rows: range) -> torch.Tensor:
        """The compressed cache with entries none of ``rows`` lists poisoned."""

        comp = torch.full_like(self.comp, NAN)
        for i in rows:
            j = self.lists[i, :self.counts[i]].long()
            comp[j] = self.comp[j]
        return comp

    def run(self, pos: int, rows: int, prompt: bool, start: int | None = None, kvw_rows: int | None = None
            ) -> torch.Tensor:
        """Attention for query rows ``start..start+rows-1`` of a forward beginning at ``pos`` (kvw: pos.. committed
        after)."""

        start = pos if start is None else start
        kvw_rows = start + rows - pos if kvw_rows is None else kvw_rows
        sel = range(start, start + rows)
        kvw = torch.full((kvw_rows + 3, D), NAN).to(torch.bfloat16)
        kvw[:kvw_rows] = self.kv[pos:pos + kvw_rows]
        out = torch.empty((rows, self.heads, D), dtype=torch.bfloat16, device="cuda")
        part = torch.full((rows, self.heads, ak.chunks(TOPK), D + 2), NAN, device="cuda")
        ak.attention(self.q[start:start + rows].cuda(), self.ring(pos).cuda(), kvw.cuda(),
                     torch.tensor([pos], dtype=torch.int32, device="cuda"),
                     torch.arange(start, start + rows, dtype=torch.int32, device="cuda"), self.comp_seen(sel).cuda(),
                     self.lists[start:start + rows].cuda(), self.counts[start:start + rows].cuda(), self.sink.cuda(),
                     out, prompt=prompt, part=part)
        return out.cpu()

    def reference(self, pos: int, rows: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """(q, entries) per row as the reference port assembles them: Reference._window over the committed ring, then
        the list."""

        st = SimpleNamespace(ring={0: self.ring(pos).float()})
        ns = SimpleNamespace(cfg=SimpleNamespace(sliding_window=WIN, head_dim=D), state=st)
        kv = self.kv[pos:pos + rows].float()
        out = []
        for i in range(rows):
            e = ref.Reference._window(ns, 0, kv, pos, pos + i)
            out.append((self.q[pos + i], torch.cat([e, self.comp[self.lists[pos + i, :self.counts[pos + i]].long()]])))
        return out


@pytest.mark.parametrize("prompt", [False, True])
@pytest.mark.parametrize("heads", [32, 2])
@pytest.mark.parametrize("pos,ratio", [(0, 2), (5, 1), (130, 2), (1000, 1), (1100, 1)])
def test_matches_reference_t1(pos, ratio, heads, prompt):
    seq = Seq(pos + 6, ratio, heads, seed=pos + heads)
    _t1(seq.run(pos, 6, prompt), seq.reference(pos, 6), seq.sink, f"pos {pos} ratio {ratio} heads {heads} {prompt}")


@pytest.mark.parametrize("prompt", [False, True])
@pytest.mark.parametrize("pos", [0, 5, 127, 128, 129, 1000])
def test_row_alone_equals_row_in_window(pos, prompt):
    """Row r of an R-row window equals the same row as a one-row forward at pos + r (earlier rows committed)."""

    seq = Seq(pos + 6, 1 + pos % 2, 32, seed=pos)
    alone = [seq.run(pos + r, 1, prompt)[0] for r in range(6)]
    for R in range(2, 7):
        window = seq.run(pos, R, prompt)
        for r in range(R):
            assert torch.equal(window[r], alone[r]), (pos, R, r)


@pytest.mark.parametrize("prompt", [False, True])
def test_ring_kvw_split_does_not_change_bits(prompt):
    """One row at anchor 300 with its window split between ring and kvw anywhere."""

    seq = Seq(301, 1, 32, seed=7)
    outs = [seq.run(pos, 1, prompt, start=300) for pos in (150, 173, 174, 200, 299, 300)]
    for pos, o in zip((173, 174, 200, 299, 300), outs[1:]):
        assert torch.equal(o, outs[0]), pos


def test_prompt_rows_equal_across_chunk_sizes():
    seq = Seq(300, 2, 32, seed=11)
    whole = seq.run(0, 300, True)
    for size in (1, 7, 64, 128, 129):
        parts = [seq.run(b, min(size, 300 - b), True) for b in range(0, 300, size)]
        assert torch.equal(torch.cat(parts), whole), size


@pytest.mark.parametrize("p", [0, 5, 126, 127, 128, 500])
def test_dspark_block_attends_window_at_p_and_block_rows(p):
    """Five rows anchored at p (pos = p + 1, no kvw) plus the block: min(p+1, 128) + 5 entries, as the reference
    assembles."""

    seq = Seq(p + 1, 1, 32, seed=p)
    g = torch.Generator().manual_seed(p)
    bkv = torch.randn((5, D), generator=g).to(torch.bfloat16)
    q = (2 * torch.randn((5, 32, D), generator=g)).to(torch.bfloat16)
    ring = seq.ring(p + 1)
    pos = torch.tensor([p + 1], dtype=torch.int32, device="cuda")
    anchors = torch.full((5,), p, dtype=torch.int32, device="cuda")
    lists = torch.arange(5, dtype=torch.int32, device="cuda").repeat(5, 1)
    counts = torch.full((5,), 5, dtype=torch.int32, device="cuda")
    part = torch.empty((5, 32, 3, D + 2), device="cuda")
    out = torch.empty((5, 32, D), dtype=torch.bfloat16, device="cuda")
    ak.attention(q.cuda(), ring.cuda(), None, pos, anchors, bkv.cuda(), lists, counts, seq.sink.cuda(), out,
                 prompt=False, part=part)
    ns = SimpleNamespace(cfg=SimpleNamespace(sliding_window=WIN, head_dim=D),
                         state=SimpleNamespace(ring={0: ring.float()}))
    e = torch.cat([ref.Reference._window(ns, 0, bkv.float(), p + 1, p), bkv.float()])
    assert len(e) == min(p + 1, WIN) + 5
    _t1(out.cpu(), [(q[i], e) for i in range(5)], seq.sink, f"dspark p {p}")
    # the same entries as one explicit list with no window give the same bits: nothing else was read
    flat = torch.cat([seq.kv[max(0, p - WIN + 1):p + 1], bkv]).cuda()
    n = len(flat)
    explicit = torch.empty_like(out)
    ak.attention(q.cuda(), ring.cuda(), None, pos, torch.full_like(anchors, -1), flat,
                 torch.arange(n, dtype=torch.int32, device="cuda").repeat(5, 1), torch.full_like(counts, n),
                 seq.sink.cuda(), explicit, prompt=False, part=part)
    assert torch.equal(explicit, out)


def test_buffers_partials_fit_a_full_list():
    assert buffers.ATTN_CHUNK == ak.CHUNK
    assert -(-(WIN + TOPK) // buffers.ATTN_CHUNK) == ak.chunks(TOPK) == 5


@pytest.mark.parametrize("prompt", [False, True])
def test_empty_list_gives_zeros(prompt):
    q = torch.randn((3, 32, D), device="cuda").to(torch.bfloat16)
    ring = torch.full((WIN, D), NAN, device="cuda").to(torch.bfloat16)
    out = torch.full_like(q, NAN)
    part = torch.empty((3, 32, ak.chunks(4), D + 2), device="cuda")
    none = torch.full((3,), -1, dtype=torch.int32, device="cuda")
    ak.attention(q, ring, None, torch.zeros(1, dtype=torch.int32, device="cuda"), none, ring,
                 torch.zeros((3, 4), dtype=torch.int32, device="cuda"), torch.zeros(3, dtype=torch.int32, device="cuda"),
                 torch.zeros(32, device="cuda"), out, prompt=prompt, part=part)
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize("prompt", [False, True])
def test_cuda_graph_replay_equals_eager(prompt):
    """Captured at one position, replayed at another: positions, anchors and lists are read on the device."""

    seq = Seq(1010, 1, 32, seed=3)
    R = 4
    ring, kvw = seq.ring(600).cuda(), seq.kv[600:600 + R].cuda()
    q, comp = seq.q[600:600 + R].cuda(), seq.comp.cuda()
    pos = torch.tensor([600], dtype=torch.int32, device="cuda")
    anchors = torch.arange(600, 600 + R, dtype=torch.int32, device="cuda")
    lists, counts = seq.lists[600:600 + R].cuda(), seq.counts[600:600 + R].cuda()
    sink, out = seq.sink.cuda(), torch.empty((R, 32, D), dtype=torch.bfloat16, device="cuda")
    part = torch.empty((R, 32, ak.chunks(TOPK), D + 2), device="cuda")

    def step():
        ak.attention(q, ring, kvw, pos, anchors, comp, lists, counts, sink, out, prompt=prompt, part=part)

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for t, v in ((ring, seq.ring(1000)), (kvw, seq.kv[1000:1000 + R]), (q, seq.q[1000:1000 + R]),
                 (pos, torch.tensor([1000])), (anchors, torch.arange(1000, 1000 + R)),
                 (lists, seq.lists[1000:1000 + R]), (counts, seq.counts[1000:1000 + R])):
        t.copy_(v)
    graph.replay()
    replayed = out.clone()
    out.zero_()
    step()
    assert torch.equal(replayed, out)
    assert torch.equal(replayed.cpu(), seq.run(1000, R, prompt))
