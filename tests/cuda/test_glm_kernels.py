"""GLM-5.3-Flash CUDA kernels on synthetic weights: row invariance (a row's bits alone equal its bits inside
a window) and agreement with the torch definitions.

The references are computed in float64: NVIDIA's PyTorch container sets TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which
runs every fp32 matmul in TF32, too coarse for the hyper-connection mix's 1e-5 check.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _q4(n: int, k: int, gen: torch.Generator):
    from tensorfold.families.glm5_next.cuda.qmm import make_q4

    words = torch.randint(-2**31, 2**31 - 1, (n, k // 8), generator=gen, dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((n, k // 64), generator=gen) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // 64), generator=gen) * 0.05).to(torch.bfloat16)
    return make_q4(words.cuda(), scales.cuda(), biases.cuda()), (words, scales, biases)


@cuda
def test_matmul_rows_invariant_and_close():
    from tensorfold.families.glm5_next.cuda import qmm

    gen = torch.Generator().manual_seed(0)
    for n, k in ((12576, 4096), (4096, 4096), (2048, 1536), (77440 // 20 * 1, 512)):
        q, (w, s, b) = _q4(n, k, gen)
        x = (torch.randn((40, k), generator=gen) * 0.5).to(torch.bfloat16).cuda()
        full = qmm.matmul(x, q)
        for rows in (1, 3, 16, 17):
            part = qmm.matmul(x[:rows].contiguous(), q)
            assert torch.equal(part, full[:rows]), (n, k, rows)
        ref = x.double() @ qmm.dequantize(w, s, b).cuda().double().t()
        assert torch.allclose(full.double(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item())
        f32 = qmm.matmul(x, q, f32=True)
        assert torch.equal(f32.to(torch.bfloat16), full)


@cuda
def test_moe_kernels_rows_invariant():
    from tensorfold.families.glm5_next.cuda import glue, qmm

    gen = torch.Generator().manual_seed(1)
    E, NI, D, K = 9, 128, 256, 3
    gate = [_q4(NI, D, gen)[1] for _ in range(E)]
    up = [_q4(NI, D, gen)[1] for _ in range(E)]
    down = [_q4(D, NI, gen)[1] for _ in range(E)]
    st = lambda parts: tuple(torch.stack([p[i] for p in parts]).cuda() for i in range(3))  # noqa: E731
    ex = qmm.make_experts(st(gate), st(up), st(down))
    router = (torch.randn((E - 1, D), generator=gen) * 0.1).to(torch.bfloat16).cuda()
    bias = (torch.randn((E - 1,), generator=gen) * 0.01).cuda()

    def run(x):
        R = x.shape[0]
        xs = qmm.group_sums(x)
        logits = glue.router(x, router, torch.empty((R, E - 1), device="cuda"))
        pick = torch.empty((R, K + 1), dtype=torch.int32, device="cuda")
        wts = torch.empty((R, K + 1), device="cuda")
        maxu = min(R * K, E - 1) + 1
        grp = qmm.Group(torch.zeros((maxu,), dtype=torch.int32, device="cuda"),
                        torch.zeros((1,), dtype=torch.int32, device="cuda"),
                        torch.full((maxu, R), -1, dtype=torch.int32, device="cuda"))
        glue.select(logits, bias, pick, wts, grp.ids, grp.count, grp.members, K, E - 1, 2.5, True)
        act = torch.empty((R, K + 1, NI), dtype=torch.bfloat16, device="cuda")
        axs = torch.empty((R, K + 1, NI // 64), device="cuda")
        y = torch.empty((R, K + 1, D), device="cuda")
        qmm.moe_gateup(x, xs, ex, grp, act, axs, 10.0)
        qmm.moe_down(act, axs, ex, grp, y)
        out = torch.empty((R, D), device="cuda")
        glue.combine(y, wts, out)
        return out, pick, wts

    x = (torch.randn((12, D), generator=gen)).to(torch.bfloat16).cuda()
    full, pick, wts = run(x)
    for r in range(12):
        one, p1, w1 = run(x[r:r + 1].contiguous())
        assert torch.equal(one[0], full[r]) and torch.equal(p1[0], pick[r]) and torch.equal(w1[0], wts[r])
    assert (pick[:, K] == E - 1).all()
    # the definition (float64)
    scores = torch.sigmoid(x.double() @ router.double().t())
    idx = torch.topk(scores + bias.double(), K, dim=-1).indices
    assert torch.equal(torch.sort(idx, dim=-1).values, torch.sort(pick[:, :K].long(), dim=-1).values)


@cuda
def test_hc_pre_post_against_fp32():
    from tensorfold.families.glm5_next.cuda import glue

    gen = torch.Generator().manual_seed(2)
    R, S, D = 5, 4, 4096
    x = (torch.randn((R, S * D), generator=gen)).to(torch.bfloat16).cuda()
    fn = (torch.randn((24, S * D), generator=gen) * 0.01).to(torch.bfloat16).cuda()
    base = (torch.randn((24,), generator=gen) * 0.1).cuda()
    scale = torch.tensor([0.5, 0.7, 1.1]).cuda()
    nw = (1 + 0.1 * torch.randn((D,), generator=gen)).to(torch.bfloat16).cuda()

    def run(xx):
        n = xx.shape[0]
        out = torch.empty((n, D), dtype=torch.bfloat16, device="cuda")
        xs = torch.empty((n, D // 64), device="cuda")
        post = torch.empty((n, S), device="cuda")
        comb = torch.empty((n, S * S), device="cuda")
        part = torch.empty((n, glue.HC_BLOCKS, 32), device="cuda")
        glue.hc_pre(xx, fn, base, scale, nw, out, xs, post, comb, part, 1e-5, 1e-6, 20)
        return out, post, comb

    out, post, comb = run(x)
    for r in range(R):
        o1, p1, c1 = run(x[r:r + 1].contiguous())
        assert torch.equal(o1[0], out[r]) and torch.equal(p1[0], post[r]) and torch.equal(c1[0], comb[r])
    # the definition (float64): mix = rms(X) . fn, pre = sigmoid + eps, post = 2 sigmoid, comb = Sinkhorn(softmax)
    eps, hc_eps = 1e-5, 1e-6
    X = x.double().view(R, S, D)
    flat = X.reshape(R, S * D)
    flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + eps)
    mix = flat @ fn.double().t()
    b64, s64 = base.double(), scale.double()
    rpre = torch.sigmoid(mix[:, :S] * s64[0] + b64[:S]) + hc_eps
    rpost = 2 * torch.sigmoid(mix[:, S:2 * S] * s64[1] + b64[S:2 * S])
    rcomb = torch.softmax(mix[:, 2 * S:].view(R, S, S) * s64[2] + b64[2 * S:].view(S, S), dim=-1) + hc_eps
    rcomb = rcomb / (rcomb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(19):
        rcomb = rcomb / (rcomb.sum(dim=-1, keepdim=True) + hc_eps)
        rcomb = rcomb / (rcomb.sum(dim=-2, keepdim=True) + hc_eps)
    col = (rpre.unsqueeze(-1) * X).sum(dim=1)
    rout = col * torch.rsqrt(col.pow(2).mean(-1, keepdim=True) + eps) * nw.double()
    assert torch.allclose(post.double(), rpost, atol=1e-5) and torch.allclose(comb.view(R, S, S).double(), rcomb, atol=1e-5)
    assert (out.double() - rout).abs().max() < 0.05
    # post: X_s = post_s * branch + sum_j comb[j, s] X_j
    branch = torch.randn((1, R, D), generator=gen).cuda()
    xo = torch.empty_like(x)
    glue.hc_post(x, xo, branch, post, comb)
    ref = post.double().unsqueeze(-1) * branch[0].to(torch.bfloat16).double().unsqueeze(1) + torch.matmul(
        comb.view(R, S, S).double().transpose(-1, -2), X)
    assert (xo.double().view(R, S, D) - ref).abs().max() < 0.05


@cuda
def test_kda_chain_serial_window_and_reference():
    from tensorfold.families.glm5_next.cuda import kda

    gen = torch.Generator().manual_seed(3)
    H, R = 2, 6
    C = 3 * H * 128
    width = C + 256 + H
    P = (torch.randn((R, width), generator=gen) * 0.5).to(torch.bfloat16).cuda()
    A = (torch.randn((R, H * 128), generator=gen)).to(torch.bfloat16).cuda()
    G = (torch.randn((R, H * 128), generator=gen)).to(torch.bfloat16).cuda()
    cw = (torch.randn((C, 4), generator=gen) * 0.3).to(torch.bfloat16).cuda()
    cs0 = (torch.randn((3, C), generator=gen) * 0.5).to(torch.bfloat16).cuda()
    st0 = (torch.randn((H, 128, 128), generator=gen) * 0.05).cuda()
    a_log = torch.randn((H,), generator=gen).cuda()
    dt = torch.randn((H * 128,), generator=gen).cuda()
    nw = (1 + 0.1 * torch.randn((128,), generator=gen)).to(torch.bfloat16).cuda()
    b_off = C + 256

    sc = kda.KDAScratch(R, H, "cuda")
    s_out = torch.empty_like(st0)
    win = kda.chain(P, b_off, A, G, cs0, cw, st0, a_log, dt, nw, 1e-5, -5.0, R, sc, s_out).clone()
    # serial: one row at a time from the evolving state and conv window
    state = st0.clone()
    cs = cs0.clone()
    for r in range(R):
        sc1 = kda.KDAScratch(1, H, "cuda")
        nxt = torch.empty_like(state)
        one = kda.chain(P[r:r + 1], b_off, A[r:r + 1], G[r:r + 1], cs, cw, state, a_log, dt, nw, 1e-5, -5.0, 1,
                        sc1, nxt)
        assert torch.equal(one[0], win[r]), r
        state = nxt
        cs = torch.cat([cs, P[r:r + 1, :C]])[1:].contiguous()
    assert torch.equal(state, s_out)
    # replay of a kept prefix gives the serial state after that many rows
    for keep in (1, 3, 5):
        rep = torch.empty_like(st0)
        kda.replay(st0, sc, keep, rep)
        state = st0.clone()
        cs = cs0.clone()
        for r in range(keep):
            nxt = torch.empty_like(state)
            kda.chain(P[r:r + 1], b_off, A[r:r + 1], G[r:r + 1], cs, cw, state, a_log, dt, nw, 1e-5, -5.0, 1,
                      kda.KDAScratch(1, H, "cuda"), nxt)
            state = nxt
            cs = torch.cat([cs, P[r:r + 1, :C]])[1:].contiguous()
        assert torch.equal(rep, state), keep


@cuda
def test_chain_attention_serial_window_and_reference():
    from tensorfold.families.glm5_next.cuda.attention import AttnScratch, attention, kv_write

    gen = torch.Generator().manual_seed(4)
    H, D, cap = 4, 256, 1200
    P, R = 700, 6
    kc = torch.zeros((cap, H, D), dtype=torch.bfloat16, device="cuda")
    vc = torch.zeros_like(kc)
    kc[:P] = torch.randn((P, H, D), generator=gen).to(torch.bfloat16).cuda()
    vc[:P] = torch.randn((P, H, D), generator=gen).to(torch.bfloat16).cuda()
    q = torch.randn((R, H, D), generator=gen).to(torch.bfloat16).cuda()
    kn = torch.randn((R, H, D), generator=gen).to(torch.bfloat16).cuda()
    vn = torch.randn((R, H, D), generator=gen).to(torch.bfloat16).cuda()
    scr = AttnScratch(8, H, D, cap, "cuda")
    pos = torch.tensor([P], dtype=torch.int32, device="cuda")
    kv_write(kn, vn, kc, vc, pos)
    win = attention(q, kc, vc, pos, scr, scale=D ** -0.5).clone()
    full = attention(q, kc, vc, pos, scr, scale=D ** -0.5, nch=None).clone()
    assert torch.equal(win, full)
    for r in range(R):
        one = attention(q[r:r + 1].contiguous(), kc, vc, torch.tensor([P + r], dtype=torch.int32, device="cuda"), scr,
                        scale=D ** -0.5, nch=-(-(P + r + 1) // 512))
        assert torch.equal(one[0], win[r]), r
        keys = kc[:P + r + 1].double()
        s = torch.einsum("hd,thd->ht", q[r].double(), keys) * D ** -0.5
        ref = torch.einsum("ht,thd->hd", torch.softmax(s, dim=-1), vc[:P + r + 1].double())
        assert (one[0].double() - ref).abs().max() < 0.02


@cuda
def test_commit_kernels_match_per_layer_updates():
    from tensorfold.families.glm5_next.cuda import kda

    gen = torch.Generator().manual_seed(5)
    L, H, R = 3, 2, 5
    sc = kda.KDAScratchSet(L, R, H, "cuda")
    sc.k.copy_(torch.randn(sc.k.shape, generator=gen).cuda() * 0.1)
    sc.v.copy_(torch.randn(sc.v.shape, generator=gen).to(torch.bfloat16).cuda())
    sc.g.copy_(torch.rand(sc.g.shape, generator=gen).cuda() * 0.5 + 0.5)
    sc.b.copy_(torch.rand(sc.b.shape, generator=gen).cuda())
    st = torch.randn((L, H, 128, 128), generator=gen).cuda() * 0.05
    out = torch.empty_like(st)
    kda.replay_layers(st, sc, 3, out)
    for layer in range(L):
        one = torch.empty_like(st[layer])
        kda.replay(st[layer].contiguous(), sc.views[layer], 3, one)
        assert torch.equal(one, out[layer]), layer


@cuda
def test_sparse_selection_and_attention_rows_invariant():
    from tensorfold.families.glm5_next.cuda import sparse

    gen = torch.Generator().manual_seed(6)
    cap, pos, R, H, D = 2600, 2200, 5, 4, 256
    npmax = cap // 4
    pk = torch.randn((npmax + 2, 128), generator=gen).to(torch.bfloat16).cuda()
    qi = torch.randn((R, 32 * 128), generator=gen).to(torch.bfloat16).cuda()
    wts = torch.randn((R, 160), generator=gen).to(torch.bfloat16).cuda()[:, 128:]
    kc = torch.randn((cap, H, D), generator=gen).to(torch.bfloat16).cuda()
    vc = torch.randn((cap, H, D), generator=gen).to(torch.bfloat16).cuda()
    q = torch.randn((R, H, D), generator=gen).to(torch.bfloat16).cuda()
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    tokens, counts = sparse.select_tokens(qi, wts, pk, pos, R, npmax, pos_dev)
    out = torch.zeros((R, H, D), dtype=torch.bfloat16, device="cuda")
    sparse.sparse_attention(q, kc, vc, tokens, counts, out, D ** -0.5)
    for r in range(R):
        p1 = torch.tensor([pos + r], dtype=torch.int32, device="cuda")
        t1, c1 = sparse.select_tokens(qi[r:r + 1].contiguous(), wts[r:r + 1], pk, pos + r, 1, npmax, p1)
        assert torch.equal(t1[0], tokens[r]) and int(c1[0]) == int(counts[r]) == 2048 + (pos + r + 1) % 4
        o1 = torch.zeros((1, H, D), dtype=torch.bfloat16, device="cuda")
        sparse.sparse_attention(q[r:r + 1].contiguous(), kc, vc, t1, c1, o1, D ** -0.5)
        assert torch.equal(o1[0], out[r]), r
        sel = t1[0, :int(c1[0])].long()
        assert bool((sel[1:] > sel[:-1]).all()) and int(sel[-1]) == pos + r
        s = torch.einsum("hd,thd->ht", q[r].double(), kc[sel].double()) * D ** -0.5
        ref = torch.einsum("ht,thd->hd", torch.softmax(s, dim=-1), vc[sel].double())
        assert (o1[0].double() - ref).abs().max() < 0.02


@cuda
def test_dflash2_attention_and_conv_against_torch():
    """The DFlash2 drafter's attention kernel (keys [0, s + N), sliding window, bidirectional or causal block)
    against SDPA with the explicit mask, and its two-tap dynamic convolution against the torch definition."""

    import torch.nn.functional as F

    from tensorfold.families.glm5_next.cuda.dflash2 import _dattn_kernel, _dconv

    gen = torch.Generator(device="cuda").manual_seed(0)
    H, KV, N, HD, CAP, window = 16, 4, 8, 128, 2568, 2047
    for s in (0, 1, 60, 2100):
        for causal in (False, True):
            q = torch.randn((H, N, HD), generator=gen, device="cuda").bfloat16()
            kc = torch.randn((KV, CAP, HD), generator=gen, device="cuda").bfloat16()
            vc = torch.randn((KV, CAP, HD), generator=gen, device="cuda").bfloat16()
            out = torch.empty((N, H * HD), dtype=torch.bfloat16, device="cuda")
            _dattn_kernel[(KV,)](q, kc, vc, out, torch.tensor([s], device="cuda"), window, HD ** -0.5, N=N,
                                 G=H // KV, NH=H, HD=HD, CAP=CAP, BK=64, CAUSAL=causal, num_warps=4)
            kidx = torch.arange(s + N, device="cuda")[None, :]
            qidx = torch.arange(N, device="cuda")[:, None]
            block = kidx >= s
            if causal:
                block = block & (kidx <= s + qidx)
            mask = ((kidx < s) & (s + qidx - kidx <= window)) | block
            ref = F.scaled_dot_product_attention(q.double()[None], kc[:, :s + N].double()[None],
                                                 vc[:, :s + N].double()[None], attn_mask=mask[None, None],
                                                 scale=HD ** -0.5, enable_gqa=True)[0].transpose(0, 1).reshape(N, -1)
            assert (out.double() - ref).abs().max().item() < 0.02, (s, causal)
    D, gs = 4096, 16
    x = torch.randn((N, D), generator=gen, device="cuda").bfloat16()
    dyn = (torch.randn((N, 2 * 2 * D // gs), generator=gen, device="cuda") * 0.1).bfloat16()
    base = torch.randn((2, 2, D), generator=gen, device="cuda").bfloat16()
    res = torch.randn((N, D), generator=gen, device="cuda").bfloat16()
    for branch in (0, 1):
        got = _dconv(x, dyn, base, branch, gs, res)
        d = dyn.view(N, 2, 2, D // gs)
        k0 = (base[branch, 0].float() + d[:, branch, 0].float().repeat_interleave(gs, -1)).bfloat16().float()
        k1 = (base[branch, 1].float() + d[:, branch, 1].float().repeat_interleave(gs, -1)).bfloat16().float()
        prev = torch.cat((torch.zeros_like(x[:1]), x[:-1])).float()
        want = (res.float() + (x.float() * k0 + prev * k1).bfloat16().float()).bfloat16()
        assert torch.equal(got, want), branch
