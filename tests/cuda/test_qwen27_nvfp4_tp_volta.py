"""Qwen3.8-27B NVFP4 on two or four ranks on sm_70: shards are the checkpoint's bytes, partials fp32, windows exact."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
    pytest.skip("Volta only", allow_module_level=True)

from tensorfold.cuda.kernels.qmmf_volta import F16, FP4, FP8, VoltaLinear  # noqa: E402
from tensorfold.families.qwen3_5.cuda import distributed  # noqa: E402
from tensorfold.families.qwen3_5.cuda.distributed import (output_rows, split_input, split_output,  # noqa: E402
                                                          split_weights)
from tensorfold.families.qwen3_5.cuda.forward import State, commit, tree_forward  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import GDN, Attention, Config, Layer, Plain, Weights  # noqa: E402

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _raw4(n: int, k: int, gen: torch.Generator):
    packed = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8).cuda()
    scale = torch.randint(0x20, 0x40, (n, k // 16), generator=gen, dtype=torch.uint8).cuda()
    return packed, scale.view(torch.float8_e4m3fn)


def _raw8(n: int, k: int, gen: torch.Generator) -> torch.Tensor:
    w = torch.randint(0, 256, (n, k), generator=gen, dtype=torch.uint8)
    w[(w & 0x7F) >= 0x60] = 0x30                            # finite, magnitudes below 2^5
    return w.cuda().view(torch.float8_e4m3fn)


def _rawb(n: int, k: int, gen: torch.Generator, scale: float = 0.05) -> torch.Tensor:
    w = torch.randn(n, k, generator=gen) * scale
    w[1::7] *= 1e-3                                         # columns of very different ranges
    return w.to(torch.bfloat16).cuda()


def _same(a: VoltaLinear, b: VoltaLinear) -> None:
    assert (a.fmt, a.n, a.k, a.layout) == (b.fmt, b.n, b.k, b.layout)
    for name in ("words", "alpha") + (("scales",) if a.fmt == FP4 else ()):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y), name


SEGMENTS = [(512, 256, None), (384, 256, None), (768, 128, (256, 256, 256)), (96, 128, None), (48, 256, None),
            (2, 128, None), (5120 + 64, 512, None)]


def _splits(n: int, k: int, segments, world: int) -> tuple[bool, bool]:
    """NVFP4 column and row shards equal projections built from the sliced checkpoint tensors, whatever tiles they touch."""

    lengths = segments or (n,)
    return all(v % world == 0 for v in lengths), k % (64 * world) == 0


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("n,k,segments", SEGMENTS + [(1024, 1024, None), (3072, 512, (1024, 1024, 1024))])
def test_fp4_shards_are_the_sliced_checkpoint(n, k, segments, world):
    """NVFP4 column and row shards equal projections built from the sliced checkpoint tensors, whatever tiles they touch."""

    gen = torch.Generator().manual_seed(n + k)
    packed, scale = _raw4(n, k, gen)
    full = VoltaLinear.from_nvfp4(packed, scale, 0.05)
    outs, ins = _splits(n, k, segments, world)
    if not (outs or ins):
        pytest.skip(f"({n}, {k}) does not split over {world} ranks")
    for rank in range(world):
        if outs:
            rows = output_rows(n, rank, world, segments=segments, device="cuda")
            _same(split_output(full, rank, world, segments=segments),
                  VoltaLinear.from_nvfp4(packed[rows], scale[rows], 0.05))
            _same(full.outputs(rows, chunk=64), VoltaLinear.from_nvfp4(packed[rows], scale[rows], 0.05))
        if ins:
            h = k // world
            _same(split_input(full, rank, world),
                  VoltaLinear.from_nvfp4(packed[:, rank * h // 2:(rank + 1) * h // 2].contiguous(),
                                         scale[:, rank * h // 16:(rank + 1) * h // 16].contiguous(), 0.05))


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("n,k,segments", SEGMENTS + [(1024, 1024, None), (3072, 512, (1024, 1024, 1024))])
def test_fp8_and_bf16_shards_are_the_sliced_checkpoint(n, k, segments, world):
    """FP8 shards and 16-bit column and row shards equal the projections built from the slices."""

    gen = torch.Generator().manual_seed(3 * n + k)
    w8 = _raw8(n, k, gen)
    full = VoltaLinear.from_fp8(w8, 0.004)
    wb = _rawb(n, k, gen)
    fb = VoltaLinear.from_bf16(wb)
    outs, ins = _splits(n, k, segments, world)
    if not (outs or ins):
        pytest.skip(f"({n}, {k}) does not split over {world} ranks")
    for rank in range(world):
        if outs:
            rows = output_rows(n, rank, world, segments=segments, device="cuda")
            _same(split_output(full, rank, world, segments=segments), VoltaLinear.from_fp8(w8[rows], 0.004))
            _same(split_output(fb, rank, world, segments=segments), VoltaLinear.from_bf16(wb[rows].contiguous()))
        if ins:
            h = k // world
            _same(split_input(full, rank, world),
                  VoltaLinear.from_fp8(w8[:, rank * h:(rank + 1) * h].contiguous(), 0.004))
            got = split_input(fb, rank, world)
            values = got.dense().double() * got.alpha[:n, None].double()
            assert torch.equal(values, wb[:, rank * h:(rank + 1) * h].double())


def test_rows_need_not_be_ordered_or_whole_tiles():
    gen = torch.Generator().manual_seed(4)
    packed, scale = _raw4(200, 128, gen)
    full = VoltaLinear.from_nvfp4(packed, scale, 0.25)
    rows = torch.tensor([7, 199, 0, 64, 65, 31, 32, 150], device="cuda")
    _same(full.outputs(rows), VoltaLinear.from_nvfp4(packed[rows], scale[rows], 0.25))


def test_shards_refuse_split_groups():
    gen = torch.Generator().manual_seed(5)
    with pytest.raises(ValueError, match="64-input groups"):
        split_input(VoltaLinear.from_fp8(_raw8(128, 64, gen), 0.01), 0)
    with pytest.raises(ValueError, match="output rows"):
        VoltaLinear.from_fp8(_raw8(128, 64, gen), 0.01).outputs(torch.tensor([128], device="cuda"))


def test_copy_owns_the_tiles():
    gen = torch.Generator().manual_seed(6)
    full = VoltaLinear.from_nvfp4(*_raw4(1000, 256, gen), 0.05)
    view = full.tiles(3, 9)
    own = view.copy()
    _same(own, view)
    assert own.words.data_ptr() != view.words.data_ptr() and own.scales.data_ptr() != view.scales.data_ptr()


@pytest.mark.parametrize("fmt", [FP4, FP8, F16])
@pytest.mark.parametrize("n,k", [(128, 256), (5120, 6144), (5120, 17408)])
def test_row_partials_are_fp32_row_invariant_and_sum_to_the_product(fmt, n, k):
    """A rank's fp32 partial rounds to its bf16 output, ignores the rows beside it, and the rank sum is the product."""

    gen = torch.Generator().manual_seed(n + fmt)
    if fmt == FP4:
        packed, scale = _raw4(n, k, gen)
        full = VoltaLinear.from_nvfp4(packed, scale, 0.01)
        codes = torch.tensor(E2M1, device="cuda")
        dense = torch.stack([codes[(packed & 15).long()], codes[(packed >> 4).long()]], -1).view(n, k)
        dense = dense * scale.float().repeat_interleave(16, 1) * 0.01
    elif fmt == FP8:
        w8 = _raw8(n, k, gen)
        full = VoltaLinear.from_fp8(w8, 0.004)
        dense = w8.float() * 0.004
    else:
        wb = _rawb(n, k, gen)
        full = VoltaLinear.from_bf16(wb)
        dense = wb.float()
    x = (torch.randn(64, k, generator=gen) * 0.5).to(torch.bfloat16).cuda()
    shards = [split_input(full, r) for r in (0, 1)]
    xs = [x[:, r * k // 2:(r + 1) * k // 2] for r in (0, 1)]
    alone = [torch.cat([s.partial(xr[i:i + 1].contiguous()) for i in range(64)]) for s, xr in zip(shards, xs)]
    for rows in (1, 5, 9, 16, 17, 33, 64):
        for s, xr, a in zip(shards, xs, alone):
            p = s.partial(xr[:rows].contiguous())
            assert p.dtype == torch.float32 and torch.equal(p, a[:rows])
            assert torch.equal(p.to(torch.bfloat16), s(xr[:rows].contiguous()))
    total = distributed.sum_rank_partials(alone).float()
    ref = x.float() @ dense.t()
    assert (total - ref).abs().max().item() <= ref.abs().max().item() * 2 ** -7


# ---- the real checkpoint's projections ------------------------------------------------------------------------------


MODEL = os.environ.get("TENSORFOLD_QWEN27_NVFP4", "")


@pytest.mark.parametrize("world", [2, 4])
def test_real_checkpoint_shards_are_its_slices(world):
    """The real checkpoint's projections: each rank's shard equals the one built from its sliced tensors."""

    if not MODEL:
        pytest.skip("set TENSORFOLD_QWEN27_NVFP4 to shard the real checkpoint")
    from safetensors import safe_open

    d = Path(MODEL)
    index = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    root = "model.language_model." if any(n.startswith("model.language_model.") for n in index) else "model."

    def tensor(name):
        with safe_open(str(d / index[name]), framework="pt", device="cuda") as f:
            return f.get_tensor(name)

    def nvfp4(name):
        return tensor(name + ".weight"), tensor(name + ".weight_scale"), float(tensor(name + ".weight_scale_2"))

    w, s, g = nvfp4(root + "layers.3.mlp.down_proj")
    full = VoltaLinear.from_nvfp4(w, s, g)
    h = w.shape[1] * 2 // world
    for rank in range(world):
        _same(split_input(full, rank, world), VoltaLinear.from_nvfp4(w[:, rank * h // 2:(rank + 1) * h // 2].contiguous(),
                                                              s[:, rank * h // 16:(rank + 1) * h // 16].contiguous(), g))
    w, s, g = nvfp4(root + "layers.3.mlp.gate_proj")
    full = VoltaLinear.from_nvfp4(w, s, g)
    for rank in range(world):
        rows = output_rows(w.shape[0], rank, world, device="cuda")
        _same(split_output(full, rank, world), VoltaLinear.from_nvfp4(w[rows], s[rows], g))
    for name, kind in (("layers.3.self_attn.o_proj", "in"), ("layers.3.self_attn.q_proj", "out"),
                       ("layers.0.linear_attn.in_proj_qkv", "out"), ("layers.0.linear_attn.out_proj", "in")):
        w8, sc = tensor(root + name + ".weight"), float(tensor(root + name + ".weight_scale").float().reshape(-1)[0])
        full = VoltaLinear.from_fp8(w8, sc)
        for rank in range(world):
            if kind == "in":
                h = w8.shape[1] // world
                _same(split_input(full, rank, world), VoltaLinear.from_fp8(w8[:, rank * h:(rank + 1) * h].contiguous(), sc))
            else:
                rows = output_rows(w8.shape[0], rank, world, device="cuda")
                _same(split_output(full, rank, world), VoltaLinear.from_fp8(w8[rows], sc))
    wb = tensor(root + "layers.0.linear_attn.in_proj_b.weight").to(torch.bfloat16)
    full = VoltaLinear.from_bf16(wb)
    for rank in range(world):
        rows = output_rows(wb.shape[0], rank, world, device="cuda")
        _same(split_output(full, rank, world), VoltaLinear.from_bf16(wb[rows].contiguous()))
    head = tensor("lm_head.weight").to(torch.bfloat16)[:20000].contiguous()     # a slice of the head: memory
    full = VoltaLinear.from_bf16(head)
    for rank in range(world):
        rows = output_rows(head.shape[0], rank, world, device="cuda")
        _same(split_output(full, rank, world), VoltaLinear.from_bf16(head[rows].contiguous()))


# ---- the multi-rank forward on one GPU (a thread a rank) ------------------------------------------------------------


class _Ranks:
    """``gather_rank_partials`` for a thread a rank: each posts its partial, all add them rank 0 first in fp32, round once."""

    def __init__(self, world: int = 2) -> None:
        self.world = world
        self.local = threading.local()
        self.barrier = threading.Barrier(world)
        self.posted: dict[int, torch.Tensor] = {}
        self.calls = 0

    def gather(self, local: torch.Tensor, group=None, dtype=torch.bfloat16) -> torch.Tensor:
        rank = self.local.rank
        self.calls += rank == 0
        self.barrier.wait()
        self.posted[rank] = local.contiguous()
        self.barrier.wait()
        acc = self.posted[0].float()
        for r in range(1, self.world):
            acc = acc + self.posted[r].float()
        out = acc.to(dtype)
        self.barrier.wait()
        return out

    def run(self, fn):
        got, errors = [None] * self.world, []

        def body(rank):
            self.local.rank = rank
            try:
                got[rank] = fn(rank)
            except BaseException as exc:        # a failing rank would leave the other in the barrier
                errors.append(exc)
                self.barrier.abort()
        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.world)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        torch.cuda.synchronize()
        return got


def _model(n: int = 2) -> Weights:
    """Two layers (GDN, then attention) stored as the RadixArk 27B stores them, on the sm_70 linears; ``n`` heads a kind."""

    gen = torch.Generator().manual_seed(11)

    def fp4(n, k):
        return VoltaLinear.from_nvfp4(*_raw4(n, k, gen), 0.02)

    def fp8(n, k):
        return VoltaLinear.from_fp8(_raw8(n, k, gen), 0.003)

    def bf16(n, k, scale=0.05):
        return VoltaLinear.from_bf16((torch.randn(n, k, generator=gen) * scale).to(torch.bfloat16).cuda())

    c = Config(hidden=128, intermediate=128 * n, layers=2, heads=n, kv_heads=n, head_dim=128, vocab=256, k_heads=n,
               v_heads=n, dk=128, dv=128, conv_kernel=4, interval=2, eps=1e-6, rope_dims=32, rope_theta=10000000.0,
               eos=(0,))
    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    m = 128 * n
    gdn = GDN(fp8(3 * m, 128), fp8(m, 128), bf16(n, 128), bf16(n, 128), fp8(128, m),
              (torch.randn(3 * m, 4, generator=gen) * 0.1).to(torch.bfloat16).cuda(),
              torch.zeros(n, device="cuda"), torch.zeros(n, device="cuda"), norm)
    attn = Attention(fp8(2 * m, 128), fp8(m, 128), fp8(m, 128), fp8(128, m), norm, norm)
    layers = [Layer(True, norm, norm, gdn, None, fp4(m, 128), fp4(m, 128), fp4(128, m)),
              Layer(False, norm, norm, None, attn, fp4(m, 128), fp4(m, 128), fp4(128, m))]
    half = c.rope_dims // 2
    inv = (c.rope_theta ** (-torch.arange(0, half, dtype=torch.float64) / half)).float().cuda()
    embed = Plain((torch.randn(256, 128, generator=gen) * 0.5).to(torch.bfloat16).cuda())
    w = Weights(c, embed, layers, norm, bf16(256, 128), inv, quant="nvfp4")
    w.precision = "full"
    return w


def _sharded(world: int):
    from tensorfold.families.qwen3_5.cuda.prefill import prefill_state

    full = _model(world)
    shards = [split_weights(full, r, world, tiled=True, split_head=True) for r in range(world)]
    for w in [full] + shards:                   # build every kernel once, so two threads never race to build one
        st = State(w)
        prefill_state(w, [5, 6, 7], st)
        for rows in ([-1], [-1, 0, 0]):
            tree_forward(w, torch.arange(len(rows), device="cuda", dtype=torch.int32) + 3, rows, st)
        for q in (w.layers[0].down, w.layers[0].gdn.out, w.layers[1].attn.o):
            q.partial(torch.zeros((1, q.k), dtype=torch.bfloat16, device="cuda"))
    return full, shards


@pytest.fixture(scope="module")
def two():
    return _sharded(2)


@pytest.fixture(scope="module", params=[2, 4], ids=["two", "four"])
def sharded(request):
    return _sharded(request.param)


@pytest.fixture
def ranks(monkeypatch, sharded):
    r = _Ranks(len(sharded[1]))
    monkeypatch.setattr(distributed, "gather_rank_partials", r.gather)
    return r


def test_split_weights_keeps_the_stored_formats(two):
    full, shards = two
    for s in shards:
        assert s.quant == "nvfp4" and s.precision == "full" and s.config.heads == 1 and s.config.intermediate == 128
        assert s.layers[0].down.layout == "volta-nvfp4" and s.layers[0].down.k == 128
        assert s.layers[0].gdn.out.layout == "volta-fp8" and s.layers[0].gdn.qkv.n == 384
        assert s.layers[0].gdn.b.layout == "volta-f16" and s.layers[0].gdn.b.n == 1
        assert s.head.n == 128 and s.head.layout == "volta-f16" and s.embed is full.embed


def test_split_weights_on_four_ranks_take_a_quarter():
    full, shards = _sharded(4)
    for s in shards:
        assert s.config.heads == s.config.kv_heads == 1 and s.config.intermediate == 128
        assert s.layers[0].down.k == 128 and s.layers[0].gdn.qkv.n == 384 and s.layers[0].gdn.b.n == 1
        assert s.layers[1].attn.o.k == 128 and s.head.n == 64 and s.embed is full.embed


def test_rank_window_rows_equal_that_rank_counts_serial_steps(sharded, ranks):
    """A tree window on two or four ranks equals that rank count's serial steps; logits match one GPU's to bf16 rounding."""

    full, shards = sharded
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

    got = ranks.run(rank_run)
    assert ranks.calls == 7 * 4             # 2 prompts, 1 window, 4 steps: two row-parallel sums in each layer
    for logits, serial, win, st in got:
        assert logits.shape == (5, 256 // len(shards))
        for row in path:
            assert torch.equal(logits[row], serial[row]), f"row {row} differs from serial"
        assert torch.equal(win.rec[0], st.rec[0]) and torch.equal(win.conv[0], st.conv[0])
        assert torch.equal(win.kv[1][0][:win.pos], st.kv[1][0][:st.pos])
    from tensorfold.families.qwen3_5.cuda.prefill import prefill_state

    st = State(full)
    prefill_state(full, prompt, st)
    one, _ = tree_forward(full, torch.tensor(tokens, device="cuda", dtype=torch.int32), parents, st)
    every = torch.cat([g[0] for g in got], dim=1).float()
    assert (every - one.float()).abs().max().item() <= 0.05 * one.float().abs().max().item()


def test_rank_runs_repeat_bit_for_bit(sharded, ranks):
    full, shards = sharded
    prompt = [3 + (11 * i) % 250 for i in range(30)]

    def rank_run(rank):
        from tensorfold.families.qwen3_5.cuda.prefill import prefill_state

        w = shards[rank]
        got = []
        for _ in range(2):
            st = State(w)
            prefill_state(w, prompt, st, tp=True)
            got.append(tree_forward(w, torch.tensor([4, 5, 6], device="cuda", dtype=torch.int32), [-1, 0, 1], st,
                                    tp=True)[0])
        return got

    for a, b in ranks.run(rank_run):
        assert torch.equal(a, b)


def test_rank_prompts_ignore_chunking(sharded, ranks):
    full, shards = sharded
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


def test_two_rank_drafter_halves_are_the_head_columns():
    """Each rank's DFlash2 head rows on an sm_70 head (owned copies, not views): together the draft columns."""

    from tensorfold.families.qwen3_5.cuda.dflash2 import _rank_spans, _sub_parts

    gen = torch.Generator().manual_seed(8)
    head = VoltaLinear.from_bf16((torch.randn(1024, 128, generator=gen)).to(torch.bfloat16).cuda())
    small = ((0, 384), (672, 1000))
    x = torch.randn(3, 128, generator=gen).to(torch.bfloat16).cuda()
    want = torch.cat([head(x)[:, a:b] for a, b in small], dim=1)
    got = torch.cat([torch.cat([p.copy()(x)[:, lo:hi] for p, lo, hi in _sub_parts(head, _rank_spans(small, r))], dim=1)
                     for r in (0, 1)], dim=1)
    assert torch.equal(got, want)
