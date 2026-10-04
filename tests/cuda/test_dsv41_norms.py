"""DeepSeek RMSNorm and mHC kernels: the op-level bound against the fp32 reference port in mirror mode, bit-equal rows,
graphs.

The reference runs on the CPU: NVIDIA's container runs fp32 CUDA matmuls in TF32, too coarse for the mixes.
"""

from __future__ import annotations

import math
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_reference as ref

from tensorfold.families.deepseek_v41.cuda import norms

MODEL = os.environ.get("TF_DSV41_MODEL")
EPS, HC_EPS, ITERS = 1e-20, 1e-6, 20
D = 5120
PROMPT = (1, 7, 129, 2048)


def _ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 distance in units in the last place (signed zeros equal)."""

    def ordered(x):
        bits = x.contiguous().view(torch.int16).cpu().to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(a) - ordered(b)).abs()


def _t1(out: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """Every element within 1 bf16 ulp of the reference, at most 0.5% of them differing."""

    u = _ulps(out, want.to(torch.bfloat16))
    assert int(u.max()) <= 1, f"{what}: {int(u.max())} ulps"
    assert float((u > 0).double().mean()) <= 0.005, f"{what}: {float((u > 0).double().mean()):.4%} differ"


def _streams(rows: int, gen: torch.Generator, d: int = D) -> torch.Tensor:
    """bf16 [rows, 4*d]: copies of different magnitudes with a few loud channels, as a residual stream has."""

    x = torch.randn((rows, 4, d), generator=gen) * torch.tensor([1.0, 0.3, 2.5, 0.8])[None, :, None]
    loud = torch.randint(0, d, (16,), generator=gen)
    x[:, :, loud] *= 40
    return x.to(torch.bfloat16).view(rows, 4 * d)


def _mhc(gen: torch.Generator, d: int = D):
    fn = torch.randn((norms.MIX, 4 * d), generator=gen) * 0.02
    base = torch.randn(norms.MIX, generator=gen)
    scale = torch.tensor([0.6, 0.4, 1.3])
    w = (1 + 0.5 * torch.randn(d, generator=gen)).to(torch.bfloat16)
    return fn, base, scale, w


def _coeffs(rows: int):
    """part, pre, post, comb buffers for ``rows``."""

    return (torch.empty(shape, device="cuda", dtype=torch.float32)
            for shape in ((rows, norms.HC_BLOCKS, 32), (rows, 4), (rows, 4), (rows, 4, 4)))


def _mix(x, fn, base, scale):
    part, pre, post, comb = _coeffs(x.shape[0])
    norms.hc_mix(x, fn, base, scale, part, pre, post, comb, EPS, HC_EPS, ITERS)
    return pre, post, comb


def _collapse(x, pre, w, keep: bool = True):
    rows, d = x.shape[0], w.shape[0]
    out = torch.empty((rows, d), device="cuda", dtype=torch.bfloat16)
    raw = torch.empty_like(out) if keep else None
    norms.collapse_norm(x, pre, w, out, EPS, raw)
    return out, raw


def _norm(x, w):
    out = torch.empty(x.shape, device="cuda", dtype=torch.bfloat16)
    return norms.rmsnorm(x, w, EPS, out)


def _cuda(*ts):
    return [t.cuda() for t in ts]


def _check_mix(x, fn, base, scale, got) -> None:
    """pre/post/comb within 8 sqrt(K) 2^-24 of the fp64 scale of the row's mix logits, plus a fp32 floor."""

    rows, k = x.shape
    xf, f64 = x.double().cpu(), fn.double().cpu()
    mixes = ref.hc_mixes(xf.view(rows, 4, -1), f64, EPS)
    want = ref.hc_split_sinkhorn(mixes, scale.double().cpu(), base.double().cpu(), 4, ITERS, HC_EPS)
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    size = (xf.abs() @ f64.abs().T) * r * scale.double().cpu().repeat_interleave(torch.tensor([4, 4, 16]))
    tol = 8 * math.sqrt(k) * 2.0**-24 * size.amax(-1) + 64 * 2.0**-24
    for name, g, w in zip(("pre", "post", "comb"), got, want):
        err = (g.double().cpu() - w).abs().flatten(1).amax(-1)
        assert bool((err <= tol).all()), f"{name}: {float((err / tol).max()):.3g} of the bound"


def test_rmsnorm_matches_the_mirror_reference():
    gen = torch.Generator().manual_seed(0)
    for d, rows in ((5120, 64), (1024, 64), (512, 200), (128, 300)):
        x = (torch.randn((rows, d), generator=gen) * torch.rand((rows, 1), generator=gen) * 8).to(torch.bfloat16)
        w = (1 + 0.5 * torch.randn(d, generator=gen)).to(torch.bfloat16)
        out = _norm(*_cuda(x, w))
        _t1(out, ref.rms_norm(x.float(), w.float(), EPS, ref.MIRROR), f"rmsnorm D={d}")


def test_rmsnorm_reads_strided_rows_and_fp32_weights():
    gen = torch.Generator().manual_seed(1)
    both = (torch.randn((9, 1536), generator=gen) * 3).to(torch.bfloat16).cuda()      # [qa | kva] of one GEMM
    w = (1 + 0.5 * torch.randn(1024, generator=gen)).cuda()
    got = _norm(both[:, :1024], w)
    assert torch.equal(got, _norm(both[:, :1024].contiguous(), w))
    _t1(got, ref.rms_norm(both[:, :1024].float().cpu(), w.cpu(), EPS, ref.MIRROR), "strided")


def test_hc_mix_and_collapse_match_the_reference():
    gen = torch.Generator().manual_seed(2)
    x = _streams(48, gen)
    fn, base, scale, w = _mhc(gen)
    xc, fnc, basec, scalec, wc = _cuda(x, fn, base, scale, w)
    got = _mix(xc, fnc, basec, scalec)
    _check_mix(xc, fnc, basec, scalec, got)
    pre = got[0].cpu()
    out, raw = _collapse(xc, got[0], wc)
    collapsed = ref.hc_pre(x.float().view(48, 4, D), pre, ref.MIRROR)
    _t1(raw, collapsed, "collapse")
    _t1(out, ref.rms_norm(collapsed, w.float(), EPS, ref.MIRROR), "collapse_norm")


def test_one_hot_collapse_is_the_plain_norm_of_copy_zero():
    """Layer 0 collapses with pre = [1, 0, 0, 0]: exactly the first copy, normed by the RMSNorm kernel's own code."""

    gen = torch.Generator().manual_seed(3)
    x = _streams(7, gen).cuda()
    w = (1 + 0.5 * torch.randn(D, generator=gen)).to(torch.bfloat16).cuda()
    pre = torch.zeros((7, 4), device="cuda")
    pre[:, 0] = 1
    out, raw = _collapse(x, pre, w)
    assert torch.equal(raw, x[:, :D])
    assert torch.equal(out, _norm(x[:, :D], w))
    assert torch.equal(_collapse(x, pre, w, keep=False)[0], out)


def _all(x, fn, base, scale, w):
    pre, post, comb = _mix(x, fn, base, scale)
    out, raw = _collapse(x, pre, w)
    return pre, post, comb, out, raw, _norm(x[:, :1024], w[:1024])


def test_rows_alone_equal_rows_in_windows_and_prompt_chunks():
    gen = torch.Generator().manual_seed(4)
    x = _streams(2048, gen).cuda()
    args = _cuda(*_mhc(gen))
    full = _all(x, *args)
    for rows in range(1, 7):
        for start in (0, 5, 1000, 2048 - rows):
            win = _all(x[start:start + rows].clone(), *args)
            for i, (a, b) in enumerate(zip(win, full)):
                assert torch.equal(a, b[start:start + rows]), (rows, start, i)
    for rows in PROMPT:
        for start in (0, 2048 - rows):
            chunk = _all(x[start:start + rows].clone(), *args)
            for i, (a, b) in enumerate(zip(chunk, full)):
                assert torch.equal(a, b[start:start + rows]), (rows, start, i)


@pytest.mark.parametrize("rows", [1, 6])
def test_cuda_graph_replay_equals_eager(rows):
    gen = torch.Generator().manual_seed(5)
    fn, base, scale, w = _cuda(*_mhc(gen))
    x = _streams(rows, gen).cuda()
    part, pre, post, comb = _coeffs(rows)
    out, raw, qn = (torch.empty((rows, d), device="cuda", dtype=torch.bfloat16) for d in (D, D, 1024))

    def step():
        norms.hc_mix(x, fn, base, scale, part, pre, post, comb, EPS, HC_EPS, ITERS)
        norms.collapse_norm(x, pre, w, out, EPS, raw)
        norms.rmsnorm(x[:, :1024], w[:1024], EPS, qn)

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    x.copy_(_streams(rows, gen).cuda())
    graph.replay()
    replayed = [t.clone() for t in (pre, post, comb, out, raw, qn)]
    step()
    for i, (a, b) in enumerate(zip(replayed, (pre, post, comb, out, raw, qn))):
        assert torch.equal(a, b), i


@pytest.mark.skipif(not MODEL, reason="set TF_DSV41_MODEL to the checkpoint")
@pytest.mark.parametrize("layer", [0, 39])
def test_real_hc_attn_weights_pass_t1(layer):
    from dsv41_ref_pack import RefPack

    pack = RefPack(MODEL)
    p = f"layers.{layer}"
    fn, base, scale = (pack.tensor(f"{p}.hc_attn_{n}").float() for n in ("fn", "base", "scale"))
    w = pack.tensor(f"{p}.attn_norm.weight")
    gen = torch.Generator().manual_seed(6)
    if layer == 0:
        x = pack.tensor("embed.weight", slice(1000, 1064)).to(torch.bfloat16).repeat(1, 4)   # the embedding, 4 copies
    else:
        x = _streams(64, gen)
    xc, fnc, basec, scalec, wc = _cuda(x, fn, base, scale, w)
    got = _mix(xc, fnc, basec, scalec)
    _check_mix(xc, fnc, basec, scalec, got)
    out, raw = _collapse(xc, got[0], wc)
    collapsed = ref.hc_pre(x.float().view(64, 4, D), got[0].cpu(), ref.MIRROR)
    _t1(raw, collapsed, f"layer {layer} collapse")
    _t1(out, ref.rms_norm(collapsed, w.float(), EPS, ref.MIRROR), f"layer {layer} collapse_norm")
