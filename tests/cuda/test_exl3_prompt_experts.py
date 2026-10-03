"""The prompt-chunk EXL3 experts kernel (``tensorfold/cuda/exl3/prompt_experts``) against the universal, row-invariant
``experts.routed`` and a float64 reference on synthetic GLM-5.3-shaped layers (one TP=4 rank: D 6144, I 512, top-8):
mixed 2/3/4-bit experts (68 / 184 / 4 of 256, like layer 10), mul1, gate|up in one fused trellis read through
``gu_stride`` as experts_cx lays them out; unrouted slots, bf16 out, the row chunking and experts_cx's switch."""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

D, I, TOPK = 6144, 512, 8


def _scale(n, mag, gen):
    """suh / svh: signs times per-channel scales (ExLlamaV3 folds input / output channel scales into them)."""

    sign = torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1
    return (sign * (torch.rand((n,), generator=gen) + 0.5) * mag).half().cuda()


def make_layer(E=256, k2_counts=((4, 68), (6, 184), (8, 4)), cb=2, seed=0):
    """Experts as experts_cx lays them out: gate|up fused [D/16, 2I/16, 16K] per expert, down [I/16, D/16, 16K]."""

    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(seed)
    widths = [k for k, n in k2_counts for _ in range(n)]
    assert len(widths) == E
    widths = [widths[i] for i in torch.randperm(E, generator=g).tolist()]
    it = I // 16
    gate, up, down = [], [], []
    for e in range(E):
        k2 = widths[e]
        fused = torch.randint(-32768, 32768, (D // 16, 2 * it, 8 * k2), dtype=torch.int32, generator=g)
        fused = fused.to(torch.int16).cuda()
        td = torch.randint(-32768, 32768, (it, D // 16, 8 * k2), dtype=torch.int32, generator=g).to(torch.int16).cuda()
        gate.append((fused[:, :it], _scale(D, 1.0, g), _scale(I, 0.02, g)))
        up.append((fused[:, it:], _scale(D, 1.0, g), _scale(I, 0.02, g)))
        down.append((td, _scale(I, 1.0, g), _scale(D, 0.05, g)))
    return experts.prepare(gate, up, down, cb, gu_stride=2 * it)


def picks(R, E, gen, skew=0.0):
    """top-8 distinct of E per row; ``skew`` > 0 favours low expert ids (a mildly skewed router)."""

    logits = torch.randn((R, E), generator=gen) + skew * torch.linspace(1.0, -1.0, E)
    sel = logits.topk(TOPK, dim=1).indices.to(torch.int32)
    w = torch.softmax(torch.randn((R, TOPK), generator=gen), dim=1)
    return sel.cuda().contiguous(), w.float().cuda().contiguous()


def reference(x, sel, w, ex, chunk=128):
    from tensorfold.cuda.exl3 import experts

    R = x.shape[0]
    s = experts.Scratch(ex, chunk, TOPK)
    out = torch.empty((R, D), dtype=torch.float32, device="cuda")
    for r0 in range(0, R, chunk):
        n = min(chunk, R - r0)
        out[r0:r0 + n] = experts.routed(x[r0:r0 + n], sel[r0:r0 + n].contiguous(), w[r0:r0 + n].contiguous(), ex, s,
                                        None, n)
    return out


def reference64(x, sel, w, ex, rows):
    """float64 out rows from the decoded W_q (Hadamard rotations and SwiGLU exact)."""

    import math

    from tensorfold.cuda.exl3 import experts

    i = torch.arange(128)
    par = torch.tensor([bin(v).count("1") & 1 for v in range(128)])
    H = (torch.where(par[i[:, None] & i[None, :]] == 1, -1.0, 1.0).double() / math.sqrt(128)).cuda()
    rot = lambda v: (v.reshape(*v.shape[:-1], -1, 128) @ H).reshape(v.shape)   # noqa: E731
    E = ex.count
    W = {}

    def mat(j, e):
        if (j, e) not in W:
            W[(j, e)] = experts.dequant(ex.keep[j * E + e].contiguous(), ex.cb).double()
        return W[(j, e)]

    out = torch.zeros((len(rows), D), dtype=torch.float64, device="cuda")
    for n, r in enumerate(rows):
        xr = x[r].double()
        for s in range(sel.shape[1]):
            e = int(sel[r, s])
            if not 0 <= e < E:
                continue
            gg = rot(rot(xr * ex.suh_g[e].double()) @ mat(0, e)) * ex.svh_g[e].double()
            uu = rot(rot(xr * ex.suh_u[e].double()) @ mat(1, e)) * ex.svh_u[e].double()
            a = gg * torch.sigmoid(gg) * uu
            out[n] += float(w[r, s]) * rot(rot(a * ex.suh_d[e].double()) @ mat(2, e)) * ex.svh_d[e].double()
    return out


@pytest.fixture(scope="module")
def layer():
    return make_layer()


@pytest.mark.parametrize("R", [64, 1024, 4096])
def test_prompt_routed_matches_routed(layer, R):
    from tensorfold.cuda.exl3 import prompt_experts as pe

    g = torch.Generator().manual_seed(R)
    x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = picks(R, layer.count, g, skew=0.5)
    ref = reference(x, sel, w, layer)
    got = pe.prompt_routed(x, sel, w, layer)
    torch.cuda.synchronize()
    err = (got - ref).abs()
    rel = (err.norm() / ref.norm()).item()
    print(f"R={R}: max abs {err.max().item():.3e}, |ref| max {ref.abs().max().item():.3e}, rel (fro) {rel:.3e}, "
          f"max rel to row max {(err.amax(1) / ref.abs().amax(1)).max().item():.3e}")
    rows = list(range(0, R, max(1, R // 16)))
    r64 = reference64(x, sel, w, layer, rows)
    e_new = ((got[rows].double() - r64).norm() / r64.norm()).item()
    e_old = ((ref[rows].double() - r64).norm() / r64.norm()).item()
    print(f"  vs float64 ({len(rows)} rows): prompt_routed {e_new:.3e}, routed {e_old:.3e}")
    assert torch.isfinite(got).all()
    assert rel <= 2e-3
    assert e_new <= 2e-3


def test_unrouted_slots_and_bf16_out(layer):
    from tensorfold.cuda.exl3 import prompt_experts as pe

    g = torch.Generator().manual_seed(5)
    R = 200
    x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = picks(R, layer.count, g)
    sel[::3, 2] = layer.count                     # e.g. a shared-expert id: skipped
    sel[1::5, 0] = -1
    keep = (sel >= 0) & (sel < layer.count)
    # routed() leaves unrouted slots' outputs to the caller (stale scratch rows): weigh them 0 there
    ref = reference(x, torch.where(keep, sel, torch.full_like(sel, layer.count)), w * keep, layer)
    out = torch.empty((R, D), dtype=torch.bfloat16, device="cuda")
    pe.prompt_routed(x, sel, w, layer, out=out)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel <= 5e-3, rel


def test_row_chunks_match_one_call(layer):
    """R above CHUNK_ROWS runs in row chunks: the same as separate calls."""

    from tensorfold.cuda.exl3 import prompt_experts as pe

    g = torch.Generator().manual_seed(9)
    R = pe.CHUNK_ROWS + 300
    x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = picks(R, layer.count, g)
    got = pe.prompt_routed(x, sel, w, layer)
    a = pe.prompt_routed(x[:pe.CHUNK_ROWS], sel[:pe.CHUNK_ROWS], w[:pe.CHUNK_ROWS], layer)
    b = pe.prompt_routed(x[pe.CHUNK_ROWS:], sel[pe.CHUNK_ROWS:], w[pe.CHUNK_ROWS:], layer)
    ref = torch.cat([a, b])
    assert ((got - ref).norm() / ref.norm()).item() < 1e-6       # only the atomics' order differs


def test_experts_cx_switch(layer, monkeypatch):
    """experts_cx's prefill takes the TensorFold kernel under TF_EXL3_PROMPT_EXPERTS=1 (bf16 out)."""

    from types import SimpleNamespace

    from tensorfold.families.glm_moe_dsa.cuda import experts_cx

    monkeypatch.setenv("TF_EXL3_PROMPT_EXPERTS", "1")
    fake = SimpleNamespace(ex=layer, dims=D)
    fake._prompt_ok = lambda: experts_cx.SharedExperts._prompt_ok(fake)
    fake._prefill_tf = lambda *a: experts_cx.SharedExperts._prefill_tf(fake, *a)
    g = torch.Generator().manual_seed(11)
    R = 96
    x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = picks(R, layer.count, g)
    out = experts_cx.SharedExperts.prefill(fake, x, sel.long(), w)
    assert out.dtype == torch.bfloat16 and out.shape == (R, D)
    ref = reference(x, sel, w, layer)
    assert ((out.float() - ref).norm() / ref.norm()).item() < 5e-3


def test_f16_accumulation_mode(layer):
    """f16_acc (the slots summed into fp16 with red.add.f16x2, then converted): error vs float64 reported, bounded."""

    from tensorfold.cuda.exl3 import prompt_experts as pe

    g = torch.Generator().manual_seed(4096)
    R = 2048
    x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = picks(R, layer.count, g, skew=0.5)
    ref = reference(x, sel, w, layer)
    f32 = pe.prompt_routed(x, sel, w, layer, f16_acc=False)
    f16 = pe.prompt_routed(x, sel, w, layer, f16_acc=True)
    assert f16.dtype == torch.float32
    rows = list(range(0, R, R // 16))
    r64 = reference64(x, sel, w, layer, rows)
    e32 = ((f32[rows].double() - r64).norm() / r64.norm()).item()
    e16 = ((f16[rows].double() - r64).norm() / r64.norm()).item()
    rel = ((f16 - ref).norm() / ref.norm()).item()
    print(f"f16_acc: vs routed {rel:.3e}; vs float64 fp32-acc {e32:.3e}, f16-acc {e16:.3e}")
    assert rel <= 2e-3 and e16 <= 2e-3


def test_item_order_by_count(layer):
    """Items ordered by expert size (route's by_count) give the same sums as expert-id order."""

    from tensorfold.cuda.exl3 import prompt_experts as pe

    g = torch.Generator().manual_seed(77)
    R = 1024
    x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = picks(R, layer.count, g, skew=1.0)
    old = pe.ORDER_BY_COUNT
    try:
        pe.ORDER_BY_COUNT = 0
        a = pe.prompt_routed(x, sel, w, layer)
        pe.ORDER_BY_COUNT = 1
        b = pe.prompt_routed(x, sel, w, layer)
    finally:
        pe.ORDER_BY_COUNT = old
    assert ((a - b).norm() / a.norm()).item() < 1e-6



@pytest.mark.parametrize("mode", [3, 2])
@pytest.mark.parametrize("R", [1024, 4096])
def test_deterministic_slot_sums(layer, R, mode, monkeypatch):
    """Modes 3 (fixed point, default) and 2 (slot rows): the same call twice gives the same bits (fp32 red.add sums a
    row's slots in arrival order); each matches the fp32 atomic sums and the float64 reference as closely."""
    from tensorfold.cuda.exl3 import prompt_experts as pe

    g = torch.Generator().manual_seed(7 + R)
    x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = picks(R, layer.count, g, skew=0.5)
    sel[::5, 3] = -1                                      # unrouted slots are skipped by the sum
    monkeypatch.setattr(pe, "DETERMINISTIC", True)
    monkeypatch.setattr(pe, "DET_MODE", mode)
    a = pe.prompt_routed(x, sel, w, layer)
    b = pe.prompt_routed(x, sel, w, layer)
    monkeypatch.setattr(pe, "DETERMINISTIC", False)
    atom = pe.prompt_routed(x, sel, w, layer)
    torch.cuda.synchronize()
    assert torch.equal(a, b), "deterministic mode differs between two identical calls"
    rel = ((a - atom).norm() / atom.norm()).item()
    print(f"R={R} mode {mode}: deterministic vs atomic rel {rel:.3e}")
    assert rel <= 1e-5
    rows = list(range(0, R, max(1, R // 16)))
    r64 = reference64(x, sel, w, layer, rows)
    e_det = ((a[rows].double() - r64).norm() / r64.norm()).item()
    e_atom = ((atom[rows].double() - r64).norm() / r64.norm()).item()
    print(f"  vs float64: deterministic {e_det:.3e}, atomic {e_atom:.3e}")
    assert e_det <= max(2e-3, 1.1 * e_atom)
