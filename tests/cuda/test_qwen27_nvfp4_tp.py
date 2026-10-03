"""Qwen3.8-27B NVFP4 on two ranks: shards are the checkpoint bytes, partials stay fp32, windows equal serial steps."""

from __future__ import annotations

import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] == 7:
    pytest.skip("sm_80 NVFP4 kernels; sm_70 runs test_qwen27_nvfp4_tp_volta", allow_module_level=True)

from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear
from tensorfold.families.qwen3_5.cuda import distributed
from tensorfold.families.qwen3_5.cuda.distributed import output_rows, split_input, split_output, split_weights
from tensorfold.families.qwen3_5.cuda.forward import State, commit, tree_forward
from tensorfold.families.qwen3_5.cuda.nvfp4_load import Plain8
from tensorfold.families.qwen3_5.cuda.weights import GDN, Attention, Config, Layer, Plain, Weights


def _raw4(n: int, k: int, gen: torch.Generator):
    packed = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8).cuda()
    scale = torch.randint(0x20, 0x40, (n, k // 16), generator=gen, dtype=torch.uint8).cuda()
    return packed, scale


def _raw8(n: int, k: int, gen: torch.Generator) -> torch.Tensor:
    w = torch.randint(0, 256, (n, k), generator=gen, dtype=torch.uint8)
    w[(w & 0x7F) >= 0x60] = 0x30                            # finite, magnitudes below 2^5
    return w.cuda().view(torch.float8_e4m3fn)


def _same(a, b) -> None:
    for name in ("words", "bs", "w8", "groups", "weight"):
        x, y = getattr(a, name, None), getattr(b, name, None)
        assert (x is None) == (y is None), name
        if x is not None:
            assert x.shape == y.shape and torch.equal(x, y), name
    for name in ("n", "k", "npad", "scale"):
        if hasattr(a, name):
            assert getattr(a, name) == getattr(b, name), name


@pytest.mark.parametrize("n,k,segments", [(512, 256, None), (384, 256, None), (768, 128, (256, 256, 256))])
def test_fp4_shards_are_the_sliced_checkpoint(n, k, segments):
    """NVFP4 column and row shards equal projections loaded from the sliced checkpoint tensors (codes, e4m3 scales)."""

    gen = torch.Generator().manual_seed(n + k)
    packed, scale = _raw4(n, k, gen)
    full = Fp4Linear.from_checkpoint(packed, scale, 0.05)
    for rank in (0, 1):
        rows = output_rows(n, rank, segments=segments, device="cuda")
        _same(split_output(full, rank, segments=segments), Fp4Linear.from_checkpoint(packed[rows], scale[rows], 0.05))
        h = k // 2
        _same(split_input(full, rank),
              Fp4Linear.from_checkpoint(packed[:, rank * h // 2:(rank + 1) * h // 2].contiguous(),
                                        scale[:, rank * h // 16:(rank + 1) * h // 16].contiguous(), 0.05))


def test_fp8_and_bf16_shards_are_the_sliced_checkpoint():
    """FP8 (one tensor scale) and bf16 shards, with a bf16 gate's e4m3 prompt copy, equal ones built from the slices."""

    gen = torch.Generator().manual_seed(3)
    w8 = _raw8(768, 256, gen)
    full = Fp8Linear.from_checkpoint(w8, 0.004)
    wb = (torch.randn(384, 256, generator=gen) * 0.05).to(torch.bfloat16).cuda()
    gate = Plain8(wb, rows8=Fp8Linear.from_bf16(wb))
    for rank in (0, 1):
        rows = output_rows(768, rank, segments=(256, 256, 256), device="cuda")
        _same(split_output(full, rank, segments=(256, 256, 256)), Fp8Linear.from_checkpoint(w8[rows], 0.004))
        _same(split_input(full, rank), Fp8Linear.from_checkpoint(w8[:, rank * 128:(rank + 1) * 128], 0.004))
        rows = output_rows(384, rank, device="cuda")
        got = split_output(gate, rank)
        _same(got, Plain8(wb[rows], rows8=Fp8Linear.from_bf16(wb[rows].contiguous())))
        _same(got.rows8, Fp8Linear.from_bf16(wb[rows].contiguous()))
        cut = wb[:, rank * 128:(rank + 1) * 128].contiguous()
        got = split_input(gate, rank)
        _same(got, Plain(cut))
        _same(got.rows8, Fp8Linear.from_bf16(cut))
    tiny = Plain((torch.randn(48, 256, generator=gen)).to(torch.bfloat16).cuda())          # the GDN gates b and a
    assert torch.equal(split_output(tiny, 1).weight, tiny.weight[24:])


def test_shards_refuse_partial_tiles():
    gen = torch.Generator().manual_seed(5)
    full = Fp4Linear.from_checkpoint(*_raw4(192, 128, gen), 0.05)
    with pytest.raises(ValueError, match="64-output tiles"):
        split_output(full, 0)                           # 96 outputs a rank
    with pytest.raises(ValueError, match="64-input groups"):
        split_input(Fp8Linear.from_checkpoint(_raw8(128, 64, gen), 0.01), 0)


@pytest.mark.parametrize("kind", ["fp4", "fp8"])
@pytest.mark.parametrize("n,k", [(128, 256), (5120, 8704 * 2)])
def test_row_partials_are_fp32_row_invariant_and_sum_to_the_product(kind, n, k):
    """A rank's fp32 partial rounds to its own bf16 output, never depends on the rows beside it, and the rank sum is the product."""

    gen = torch.Generator().manual_seed(n)
    if kind == "fp4":
        packed, scale = _raw4(n, k, gen)
        full = Fp4Linear.from_checkpoint(packed, scale, 0.01)
        codes = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
        dense = torch.stack([codes[(packed & 15).long()], codes[(packed >> 4).long()]], -1).view(n, k)
        dense = dense * scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, 1) * 0.01
    else:
        w8 = _raw8(n, k, gen)
        full = Fp8Linear.from_checkpoint(w8, 0.004)
        dense = w8.float() * 0.004
    x = (torch.randn(64, k, generator=gen) * 0.5).to(torch.bfloat16).cuda()
    shards = [split_input(full, r) for r in (0, 1)]
    xs = [x[:, r * k // 2:(r + 1) * k // 2] for r in (0, 1)]
    alone = [torch.cat([s.partial(xr[i:i + 1].contiguous()) for i in range(64)]) for s, xr in zip(shards, xs)]
    for rows in (1, 5, 16, 64):
        for s, xr, a in zip(shards, xs, alone):
            p = s.partial(xr[:rows].contiguous())
            assert p.dtype == torch.float32 and torch.equal(p, a[:rows])
            assert torch.equal(p.to(torch.bfloat16), s(xr[:rows].contiguous()))
    total = distributed.sum_rank_partials(alone).float()
    ref = x.float() @ dense.t()
    assert (total - ref).abs().max().item() <= ref.abs().max().item() * 2 ** -7


# ---- the two-rank forward on one GPU (two threads) -----------------------------------------------------------------


class _Ranks:
    """``gather_rank_partials`` for two threads: each posts its partial, both add rank 0's first in fp32, round once."""

    def __init__(self) -> None:
        self.local = threading.local()
        self.barrier = threading.Barrier(2)
        self.posted: dict[int, torch.Tensor] = {}
        self.calls = 0

    def gather(self, local: torch.Tensor, group=None, dtype=torch.bfloat16) -> torch.Tensor:
        rank = self.local.rank
        self.calls += rank == 0
        self.barrier.wait()
        self.posted[rank] = local.contiguous()
        self.barrier.wait()
        out = (self.posted[0].float() + self.posted[1].float()).to(dtype)
        self.barrier.wait()
        return out

    def run(self, fn):
        """``fn(rank)`` on two threads; both results in rank order."""

        got, errors = [None, None], []

        def body(rank):
            self.local.rank = rank
            try:
                got[rank] = fn(rank)
            except BaseException as exc:        # a failing rank would leave the other in the barrier
                errors.append(exc)
                self.barrier.abort()
        threads = [threading.Thread(target=body, args=(r,)) for r in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        torch.cuda.synchronize()
        return got


def _model() -> Weights:
    """Two layers (GDN, then attention) as the RadixArk 27B stores them: NVFP4 MLP, FP8 GDN/attention, bf16 gates and head."""

    gen = torch.Generator().manual_seed(11)

    def fp4(n, k):
        packed, scale = _raw4(n, k, gen)
        return Fp4Linear.from_checkpoint(packed, scale, 0.02)

    def fp8(n, k):
        return Fp8Linear.from_checkpoint(_raw8(n, k, gen), 0.003)

    def bf16(n, k, scale=0.05):
        return Plain((torch.randn(n, k, generator=gen) * scale).to(torch.bfloat16).cuda())

    c = Config(hidden=128, intermediate=256, layers=2, heads=2, kv_heads=2, head_dim=128, vocab=256, k_heads=2,
               v_heads=2, dk=128, dv=128, conv_kernel=4, interval=2, eps=1e-6, rope_dims=32, rope_theta=10000000.0,
               eos=(0,))
    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    gdn = GDN(fp8(768, 128), fp8(256, 128), bf16(2, 128), bf16(2, 128), fp8(128, 256),
              (torch.randn(768, 4, generator=gen) * 0.1).to(torch.bfloat16).cuda(),
              torch.zeros(2, device="cuda"), torch.zeros(2, device="cuda"), norm)
    attn = Attention(fp8(512, 128), fp8(256, 128), fp8(256, 128), fp8(128, 256), norm, norm)
    layers = [Layer(True, norm, norm, gdn, None, fp4(256, 128), fp4(256, 128), fp4(128, 256)),
              Layer(False, norm, norm, None, attn, fp4(256, 128), fp4(256, 128), fp4(128, 256))]
    half = c.rope_dims // 2
    inv = (c.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)).float().cuda()
    w = Weights(c, bf16(256, 128, 0.5), layers, norm, bf16(256, 128), inv, quant="nvfp4")
    w.precision = "full"
    return w


@pytest.fixture(scope="module")
def two():
    from tensorfold.families.qwen3_5.cuda.prefill import prefill_state

    full = _model()
    shards = [split_weights(full, r, tiled=True, split_head=True) for r in (0, 1)]
    for w in [full] + shards:                   # build every kernel once, before two threads would race to
        st = State(w)
        prefill_state(w, [5, 6, 7], st)
        for rows in ([-1], [-1, 0, 0]):
            tree_forward(w, torch.arange(len(rows), device="cuda", dtype=torch.int32) + 3, rows, st)
        for q in (w.layers[0].down, w.layers[0].gdn.out, w.layers[1].attn.o):
            q.partial(torch.zeros((1, q.k), dtype=torch.bfloat16, device="cuda"))
    return full, shards


@pytest.fixture
def ranks(monkeypatch):
    r = _Ranks()
    monkeypatch.setattr(distributed, "gather_rank_partials", r.gather)
    return r


def test_split_weights_keeps_the_checkpoint_math(two):
    full, shards = two
    for s in shards:
        assert s.quant == "nvfp4" and s.precision == "full" and s.config.heads == 1 and s.config.intermediate == 128
        assert isinstance(s.layers[0].down, Fp4Linear) and s.layers[0].down.k == 128
        assert isinstance(s.layers[0].gdn.out, Fp8Linear) and s.layers[0].gdn.qkv.n == 384
        assert s.head.n == 128 and s.embed is full.embed


def test_two_rank_window_rows_equal_two_rank_serial_steps(two, ranks):
    """Two-rank tree window rows and commits equal two-rank serial steps; logits equal one GPU's to bf16 rounding."""

    full, shards = two
    tokens, parents, path = [7, 8, 9, 10, 11], [-1, 0, 0, 1, 3], [0, 1, 3, 4]
    prompt = [3 + (5 * i) % 250 for i in range(9)]

    def rank_run(rank):
        from tensorfold.families.qwen3_5.cuda.prefill import prefill_state

        w = shards[rank]
        st = State(w)
        prefill_state(w, prompt, st, tp=True)
        win_state = State(w)
        prefill_state(w, prompt, win_state, tp=True)
        logits, record = tree_forward(w, torch.tensor(tokens, device="cuda", dtype=torch.int32), parents, win_state,
                                      tp=True)
        serial = {}
        for row in path:
            out, rec = tree_forward(w, torch.tensor([tokens[row]], device="cuda", dtype=torch.int32), [-1], st, tp=True)
            serial[row] = out[0]
            commit(st, rec, [0])
        commit(win_state, record, path)
        return logits, serial, win_state, st

    (l0, s0, w0, t0), (l1, s1, w1, t1) = ranks.run(rank_run)
    assert ranks.calls == 7 * 4             # 2 prompts, 1 window, 4 steps: two row-parallel sums in each layer
    assert l0.shape == l1.shape == (5, 128)
    for row in path:
        assert torch.equal(l0[row], s0[row]) and torch.equal(l1[row], s1[row]), f"row {row} differs from serial"
    for a, b in ((w0, t0), (w1, t1)):
        assert torch.equal(a.rec[0], b.rec[0]) and torch.equal(a.conv[0], b.conv[0])
        assert torch.equal(a.kv[1][0][:a.pos], b.kv[1][0][:b.pos])
    # one GPU: the same bits are not promised (sums split differently), the values are
    from tensorfold.families.qwen3_5.cuda.prefill import prefill_state

    st = State(full)
    prefill_state(full, prompt, st)
    one, _ = tree_forward(full, torch.tensor(tokens, device="cuda", dtype=torch.int32), parents, st)
    both = torch.cat([l0, l1], dim=1).float()
    assert (both - one.float()).abs().max().item() <= 0.05 * one.float().abs().max().item()


def test_two_rank_prompts_ignore_chunking(two, ranks):
    full, shards = two
    prompt = [3 + (7 * i) % 250 for i in range(67)]

    def rank_run(rank):
        from tensorfold.families.qwen3_5.cuda.prefill import prefill_state

        w = shards[rank]
        runs = []
        for size in (67, 16, 5):
            st = State(w)
            runs.append((prefill_state(w, prompt, st, size=size, tp=True), st))
        return runs

    for runs in ranks.run(rank_run):
        for normed, st in runs[1:]:
            assert torch.equal(normed, runs[0][0])
            assert torch.equal(st.rec[0], runs[0][1].rec[0]) and torch.equal(st.conv[0], runs[0][1].conv[0])


def test_two_rank_drafter_halves_cover_the_draft_vocabulary():
    """Each rank's DFlash2 head rows on an NVFP4 checkpoint: rank 0 the first ceil(n/2) draft ids, rank 1 the rest."""

    from tensorfold.families.qwen3_5.cuda.dflash2 import _rank_spans, _sub_parts

    spans = ((0, 98304), (248032, 248320))
    halves = [_rank_spans(spans, r) for r in (0, 1)]
    assert halves == [((0, 49296),), ((49296, 98304), (248032, 248320))]
    ids = lambda ss: [i for a, b in ss for i in range(a, b)]
    assert ids(halves[0]) + ids(halves[1]) == ids(spans)
    gen = torch.Generator().manual_seed(8)
    head = Plain((torch.randn(1024, 128, generator=gen)).to(torch.bfloat16).cuda())
    small = ((0, 384), (672, 1000))
    x = torch.randn(3, 128, generator=gen).to(torch.bfloat16).cuda()
    want = torch.cat([head(x)[:, a:b] for a, b in small], dim=1)
    got = torch.cat([torch.cat([p(x)[:, lo:hi] for p, lo, hi in _sub_parts(head, _rank_spans(small, r))], dim=1)
                     for r in (0, 1)], dim=1)
    assert torch.equal(got, want)
