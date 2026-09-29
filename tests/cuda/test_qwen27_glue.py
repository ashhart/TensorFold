"""Glue kernels: rows alone equal rows in a window, and values match plain torch."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda import glue  # noqa: E402

dev = "cuda"


def _rows_alone(fn, *args):
    full = fn(*args)
    W = args[0].shape[0]
    for r in range(W):
        one = fn(*[a[r:r + 1] if torch.is_tensor(a) and a.dim() > 0 and a.shape[0] == W else a for a in args])
        for f, o in zip(full, one):
            assert torch.equal(f[r:r + 1], o), f"row {r} differs"
    return full


def test_add_rmsnorm():
    x = torch.randn(9, 5120, device=dev).bfloat16()
    r = torch.randn(9, 5120, device=dev).bfloat16()
    w = (torch.rand(5120, device=dev) + 0.5).bfloat16()
    h, y, xs = _rows_alone(lambda x, r, w: glue.add_rmsnorm(x, r, w, 1e-6), x, r, w)
    hf = (x.float() + r.float()).bfloat16().float()
    ref = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    assert (y.float() - ref).abs().max() < 0.05
    assert torch.allclose(xs, y.float().reshape(9, 80, 64).sum(-1), atol=1e-3)


def test_swiglu_and_gate_mul():
    g = torch.randn(5, 17408, device=dev).bfloat16()
    u = torch.randn(5, 17408, device=dev).bfloat16()
    a, xs = _rows_alone(glue.swiglu, g, u)
    ref = torch.nn.functional.silu(g.float()) * u.float()
    assert (a.float() - ref).abs().max() < 0.05
    o = torch.randn(5, 24, 256, device=dev).bfloat16()
    qg = torch.randn(5, 24 * 512, device=dev).bfloat16()
    out, _ = _rows_alone(lambda o, qg: glue.gate_mul(o, qg, heads=24, head_dim=256), o, qg)
    gate = qg.reshape(5, 24, 512)[..., 256:].float()
    assert (out.float().reshape(5, 24, 256) - o.float() * torch.sigmoid(gate)).abs().max() < 0.05


def test_gdn_pre_matches_torch():
    W, C = 6, 10240
    qkv = torch.randn(W, C, device=dev).bfloat16()
    cs = torch.randn(3, C, device=dev).bfloat16()
    cw = torch.randn(C, 4, device=dev).bfloat16()
    win = torch.tensor([[0, 1, 2, 3 + i] if i == 0 else [1, 2, 3, 3 + i] for i in range(W)], device=dev, dtype=torch.int32)
    a = torch.randn(W, 48, device=dev).bfloat16()
    b = torch.randn(W, 48, device=dev).bfloat16()
    alog = torch.randn(48, device=dev)
    dtb = torch.randn(48, device=dev)
    q, k, v, g, beta = glue.gdn_pre(qkv, cs, cw, win, a, b, alog, dtb, kh=16, vh=48, dk=128)
    src = torch.cat([cs, qkv]).float()
    conv = sum(src[win[:, j].long()] * cw.float()[:, j] for j in range(4))
    c = torch.nn.functional.silu(conv).bfloat16().float()
    qr = c[:, :2048].reshape(W, 16, 128)
    qr = qr * torch.rsqrt(qr.pow(2).mean(-1, keepdim=True) + 1e-6) / 128
    assert (q.float() - qr).abs().max() < 2e-3
    # plain torch adds the 4 taps unfused; the kernel may fuse multiply-adds, so allow one bf16 step
    assert (v.float() - c[:, 4096:].reshape(W, 48, 128)).abs().max() <= c.abs().max() * 2 ** -7
    gr = torch.exp(-torch.exp(alog) * torch.nn.functional.softplus(a.float() + dtb))
    assert torch.allclose(g, gr, rtol=1e-5, atol=1e-6)
    assert torch.allclose(beta, torch.sigmoid(b.float()), atol=1e-6)


def test_strided_projection_slices_equal_their_copies():
    """Slices of the fused [z | b | a] and [k | v] projections, read in place, give the bits of contiguous copies."""

    W, C, vh, dv, kvh, hd = 6, 10240, 48, 128, 4, 256
    zba = torch.randn(W, vh * dv + 2 * vh, device=dev).bfloat16()
    z, b, a = zba[:, :vh * dv].reshape(W, vh, dv), zba[:, vh * dv:vh * dv + vh], zba[:, vh * dv + vh:]
    assert not (z.is_contiguous() or b.is_contiguous() or a.is_contiguous())
    qkv = torch.randn(W, C, device=dev).bfloat16()
    cs = torch.randn(3, C, device=dev).bfloat16()
    cw = torch.randn(C, 4, device=dev).bfloat16()
    win = torch.tensor([[0, 1, 2, 3] if i == 0 else [1, 2, 3, 3 + i] for i in range(W)], device=dev, dtype=torch.int32)
    alog, dtb = torch.randn(vh, device=dev), torch.randn(vh, device=dev)

    def pre(a, b):
        return glue.gdn_pre(qkv, cs, cw, win, a, b, alog, dtb, kh=16, vh=vh, dk=128)

    assert all(torch.equal(x, y) for x, y in zip(pre(a, b), pre(a.contiguous(), b.contiguous())))
    yr = torch.randn(W, vh, dv, device=dev).bfloat16()
    norm = (torch.rand(dv, device=dev) + 0.5).bfloat16()
    assert all(torch.equal(x, y) for x, y in zip(glue.gated_norm(yr, z, norm, 1e-6),
                                                 glue.gated_norm(yr, z.contiguous(), norm, 1e-6)))
    kv = torch.randn(W, 2 * kvh * hd, device=dev).bfloat16()
    qg = torch.randn(W, 24 * 2 * hd, device=dev).bfloat16()
    qn, kn = (torch.rand(hd, device=dev) + 0.5).bfloat16(), (torch.rand(hd, device=dev) + 0.5).bfloat16()
    pos, inv = torch.arange(W, device=dev, dtype=torch.int32), torch.rand(hd // 8, device=dev)

    def prep(k):
        return glue.attn_prep(qg, k, qn, kn, pos, inv, 1e-6, heads=24, kv_heads=kvh, head_dim=hd)

    key = kv[:, :kvh * hd]
    assert not key.is_contiguous()
    assert all(torch.equal(x, y) for x, y in zip(prep(key), prep(key.contiguous())))
