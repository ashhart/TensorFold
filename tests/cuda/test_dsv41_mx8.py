"""DeepSeek's FP8 projections: bf16 is ``Mx8Linear``'s bits, fp32 rounds to them, rows never follow the row count,
and real rank shapes track an fp64 product of the dequantized weights."""

from __future__ import annotations

import math
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from dsv41_layouts import mx8_from_block

from tensorfold.cuda import prompt_precision
from tensorfold.cuda.nvfp4.linear import Mx8Linear
from tensorfold.families.deepseek_v41.cuda import mx8
from tensorfold.families.deepseek_v41.cuda.split import slice_for

MODEL = os.environ.get("TF_DSV41_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set TF_DSV41_MODEL to the checkpoint")
ROWS = range(1, 7)
PROMPTS = (1, 7, 256, 2048)
GROUPS = 4                          # wo_a groups a rank holds
# one rank's projections at world 2: (n, K, fp32 output)
SHAPES = {
    "wqa_kv": (1792, 5120, False), "wq_b": (16384, 1280, False), "wo_a": (1024, 4096, False),
    "wo_b": (5120, 4096, True), "shared_gu": (2304, 5120, False), "shared_d": (5120, 1152, True),
    "index_wq_b": (4096, 1280, False), "engram_wkv": (12800, 6144, False), "main_proj": (2560, 15360, False),
    "head": (64640, 5120, True),
}
# the checkpoint's projections: (module, fp32 output)
REAL = [("layers.2.attn.wq_a", False), ("layers.2.attn.wq_b", False), ("layers.2.attn.wo_a", False),
        ("layers.2.attn.wo_b", True), ("layers.2.ffn.shared_experts.w1", False),
        ("layers.2.ffn.shared_experts.w2", True), ("layers.2.attn.indexer.wq_b", False),
        ("layers.1.engram.wkv", False), ("mtp.0.main_proj", False), ("lm_head", True)]
_LINEARS: dict[str, Mx8Linear] = {}


def _random(n: int, k: int, seed: int) -> Mx8Linear:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randint(0, 256, (n, k), generator=g, dtype=torch.uint8, device="cuda")
    w[(w & 0x7F) >= 0x70] = 0x30                                               # finite, below 2^7
    s = torch.randint(118, 131, (n, k // 32), generator=g, dtype=torch.uint8, device="cuda")   # 2^-9 .. 2^3
    return Mx8Linear.from_checkpoint(w.view(torch.float8_e4m3fn), s)


def _linear(name: str) -> Mx8Linear:
    if name not in _LINEARS:
        n, k, _ = SHAPES[name]
        _LINEARS[name] = _random(n, k, len(_LINEARS) + 1)
    return _LINEARS[name]


def _x(m: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn((m, k), generator=g, device="cuda") * 0.5).to(torch.bfloat16)


@pytest.mark.parametrize("name", SHAPES)
def test_bf16_is_mx8linears_bits_and_fp32_rounds_to_them(name):
    lin = _linear(name)
    x, xp = _x(6, lin.k, 1), _x(2048, lin.k, 2)
    assert torch.equal(mx8.mm(lin, x), lin(x))
    assert torch.equal(mx8.mm(lin, xp, prompt=True), lin.prefill(xp))
    assert lin.fold                                                             # the prompt GEMM, not the fallback
    assert torch.equal(mx8.mm(lin, x, f32=True).to(torch.bfloat16), lin(x))
    assert torch.equal(mx8.mm(lin, xp, f32=True, prompt=True).to(torch.bfloat16), lin.prefill(xp))
    assert torch.equal(mx8.mm(lin, xp), lin(xp))                                # 2048 rows stay decode unless asked


@pytest.mark.parametrize("f32", [False, True])
@pytest.mark.parametrize("name", SHAPES)
def test_rows_do_not_depend_on_the_row_count(name, f32):
    lin = _linear(name)
    x = _x(6, lin.k, 3)
    full = mx8.mm(lin, x, f32=f32)
    for r in ROWS:
        assert torch.equal(mx8.mm(lin, x[:r], f32=f32), full[:r]), r
        assert torch.equal(mx8.mm(lin, x[6 - r:], f32=f32), full[6 - r:]), r
    xp = _x(2048, lin.k, 4)
    full = mx8.mm(lin, xp, f32=f32, prompt=True)
    for p in PROMPTS:
        for a in (0, 2048 - p):
            assert torch.equal(mx8.mm(lin, xp[a:a + p], f32=f32, prompt=True), full[a:a + p]), (p, a)
    assert torch.equal(mx8.mm(lin, xp[1777:1778], f32=f32, prompt=True), full[1777:1778])


def test_prompts_past_the_fold_window_take_the_decode_kernel_and_ignore_fp8_prompts():
    n, k = SHAPES["wo_b"][:2]
    lin = _random(n, k, 40)
    xp = _x(300, k, 5)
    want = mx8.mm(lin, xp, f32=True, prompt=True)
    with prompt_precision.using(True):                                          # fp32 never takes the FP8 GEMM
        assert torch.equal(mx8.mm(lin, xp, f32=True, prompt=True), want)
    lin.bs[0, 0, 0, 0] = 140
    lin.fold = None
    assert torch.equal(mx8.mm(lin, xp, prompt=True), lin.prefill(xp))
    assert lin.fold is False
    assert torch.equal(mx8.mm(lin, xp, f32=True, prompt=True), mx8.mm(lin, xp, f32=True))
    assert torch.equal(mx8.mm(lin, xp, f32=True, prompt=True).to(torch.bfloat16), lin.prefill(xp))


def _wo_a() -> list[Mx8Linear]:
    n, k, _ = SHAPES["wo_a"]
    return [_linear("wo_a")] + [_random(n, k, 50 + g) for g in range(1, GROUPS)]


@pytest.mark.parametrize("prompt", [False, True])
def test_grouped_projects_each_group_from_its_heads(prompt):
    lins = _wo_a()
    k, n = lins[0].k, lins[0].n
    for m in (1, 5, 2048 if prompt else 6):
        o = _x(m, GROUPS * k, 6).view(m, 32, 512).view(m, GROUPS * k)          # the rank's heads, flattened
        out = torch.empty((m, GROUPS * n), dtype=torch.bfloat16, device="cuda")
        mx8.grouped(lins, o, out, prompt=prompt)
        want = [(lin.prefill if prompt else lin)(o[:, g * k:(g + 1) * k].contiguous()) for g, lin in enumerate(lins)]
        assert torch.equal(out, torch.cat(want, dim=1)), m


def test_an_output_buffer_takes_no_allocation():
    lin = _linear("wo_b")
    for m, prompt in ((6, False), (2048, True)):
        x = _x(m, lin.k, 7)
        out = torch.empty((m, lin.n), dtype=torch.float32, device="cuda")
        mx8.mm(lin, x, out, f32=True, prompt=prompt)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        mx8.mm(lin, x, out, f32=True, prompt=prompt)
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() == base, (m, prompt)


def _capture(step) -> torch.cuda.CUDAGraph:
    step()                                                                      # warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph


def test_graph_replays_equal_eager():
    wo_b, wq_b, lins = _linear("wo_b"), _linear("wq_b"), _wo_a()
    k = lins[0].k
    for r in ROWS:
        xb, xq, o = _x(r, wo_b.k, 8), _x(r, wq_b.k, 9), _x(r, GROUPS * k, 10)
        yb = torch.empty((r, wo_b.n), dtype=torch.float32, device="cuda")
        yq = torch.empty((r, wq_b.n), dtype=torch.bfloat16, device="cuda")
        u = torch.empty((r, GROUPS * lins[0].n), dtype=torch.bfloat16, device="cuda")

        def step(xb=xb, yb=yb, xq=xq, yq=yq, o=o, u=u):
            mx8.mm(wo_b, xb, yb, f32=True)
            mx8.mm(wq_b, xq, yq)
            mx8.grouped(lins, o, u)

        graph = _capture(step)
        for seed in (11, 12):
            xb.copy_(_x(r, wo_b.k, seed))
            xq.copy_(_x(r, wq_b.k, seed + 1))
            o.copy_(_x(r, GROUPS * k, seed + 2))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(yb, mx8.mm(wo_b, xb, f32=True)), r
            assert torch.equal(yq, mx8.mm(wq_b, xq)), r
            assert torch.equal(u, mx8.grouped(lins, o, torch.empty_like(u))), r


def _rank_part(pack, module: str, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank's part of a checkpoint projection: (e4m3 weight, scale bytes) on the device, fp32 dequantized on the CPU."""

    key = f"{module}.weight"
    scale = next(f"{module}.{s}" for s in ("scale", "weight_scale") if pack.has(f"{module}.{s}"))
    rows, cols = slice_for(rank, 2, key, pack.shape(key))
    srows, scols = slice_for(rank, 2, scale, pack.shape(scale))
    w = pack.tensor(key, rows)[:, cols].contiguous().cuda()
    s = pack.tensor(scale, srows)[:, scols].contiguous().cuda()
    return w, s, pack.dense_fp32(module, rows)[:, cols].contiguous()


def _mx8(w: torch.Tensor, s: torch.Tensor) -> Mx8Linear:
    """MXFP8 (a scale a row, the head) or FP8 32x32 blocks, as the loader builds them."""

    return Mx8Linear.from_checkpoint(w, s) if s.shape[0] == w.shape[0] else mx8_from_block(w, s)


def _t1(got: torch.Tensor, x: torch.Tensor, w: torch.Tensor, k: int, f32: bool) -> dict:
    """The op-level bound against the fp64 product: the fp32 sum bound over K; bf16 within 1 ulp with <= 0.5% off, or
    (a sum that cancels to far below its terms) within that bound plus the half ulp of the bf16 rounding."""

    ref = x.double() @ w.t()
    bound = 8 * math.sqrt(k) * 2.0 ** -24 * (x.double().abs() @ w.abs().t())
    d = (got.double() - ref).abs()
    if f32:
        ratio = float((d / bound.clamp_min(1e-300)).max())
        assert ratio <= 1, ratio
        return {"bound_used": ratio}
    top = torch.maximum(got.double().abs(), ref.abs())
    ulp = torch.ldexp(torch.ones_like(top), torch.frexp(top).exponent - 8).clamp_min(2.0 ** -133)
    off = float((got != ref.float().to(torch.bfloat16)).double().mean())
    cancels = (d > ulp) & (d <= bound + 0.5 * ulp)
    worst = float((d / ulp)[~cancels].max())
    assert worst <= 1, worst
    assert off <= 0.005, off
    return {"max_ulp": worst, "off": off, "cancelled": int(cancels.sum())}


@needs_model
@pytest.mark.parametrize("module,f32", REAL)
def test_real_rank_parts_track_the_fp64_product(module, f32):
    pytest.importorskip("safetensors")
    from dsv41_ref_pack import RefPack

    pack = RefPack(MODEL)
    for rank in (0, 1):
        w8, s, dense = _rank_part(pack, module, rank)
        w = dense.cuda().double()
        groups = GROUPS if module.endswith("wo_a") else 1                      # wo_a: the rank's groups, as attention
        n = w8.shape[0] // groups
        lins = [_mx8(w8[g * n:(g + 1) * n], s[g * s.shape[0] // groups:(g + 1) * s.shape[0] // groups])
                for g in range(groups)]
        w = torch.block_diag(*[w[g * n:(g + 1) * n] for g in range(groups)])
        for m, prompt in ((6, False), (256, True)):
            x = _x(m, w.shape[1], 20 + rank)
            if groups > 1:
                out = torch.empty((m, w.shape[0]), dtype=torch.bfloat16, device="cuda")
                got = mx8.grouped(lins, x, out, prompt=prompt)
            else:
                got = mx8.mm(lins[0], x, f32=f32, prompt=prompt)
            print(module, "rank", rank, "prompt" if prompt else "decode", _t1(got, x, w, lins[0].k, f32))
        del w8, s, dense, w, lins
        torch.cuda.empty_cache()
