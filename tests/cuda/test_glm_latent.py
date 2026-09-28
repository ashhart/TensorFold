"""GLM's latent-cache attention (families/glm5_next/cuda/latent.py): quality against a float64 definition of the
expanded MLA attention, and the engine's exactness contract (a window row gets the bits of the serial step)."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")

H, D, L = 32, 256, 512


def _weights(gen):
    wk = torch.randn((H, D, L), generator=gen) * L ** -0.5
    wv = torch.randn((H, D, L), generator=gen) * L ** -0.5
    return wk, wv


def _reference(q, lat, wk, wv, pos, scale):
    """Expanded MLA in float64: keys Wk_h c_j, values Wv_h c_j, causal softmax. q [R, H, D], lat [n, L]."""
    q, lat, wk, wv = (x.double() for x in (q, lat, wk, wv))
    keys = torch.einsum("hdl,nl->nhd", wk, lat)
    vals = torch.einsum("hdl,nl->nhd", wv, lat)
    out = []
    for r in range(q.shape[0]):
        n = pos + r + 1
        s = torch.einsum("hd,nhd->hn", q[r], keys[:n]) * scale
        out.append(torch.einsum("hn,nhd->hd", torch.softmax(s, dim=1), vals[:n]))
    return torch.stack(out)


def _latent_forward(latent, q, cache, pos, a, scratch, R, sparse=None):
    dev = q.device
    pos_dev = torch.tensor([pos], dtype=torch.int32, device=dev)
    qa = latent.absorb_q(q, a, scratch.qa[:R])
    ol = scratch.ol[:R]
    latent.attention(qa, cache, pos_dev, scratch, scale=D ** -0.5, nch=scratch.nch, out=ol)
    if sparse is not None:
        tokens, counts = sparse
        latent.sparse_attention(qa, cache, tokens, counts, ol, D ** -0.5)
    out = torch.empty((R, H, D), dtype=torch.bfloat16, device=dev)
    return latent.expand_v(ol, a, out).clone()


@cuda
def test_dequant_mlx4_matches_manual_unpack():
    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator().manual_seed(0)
    q = torch.randint(0, 16, (8, 128), generator=gen)
    words = torch.zeros((8, 16), dtype=torch.int64)
    for j in range(8):
        words |= q[:, j::8] << (4 * j)
    words = words.to(torch.int32)                       # wraps like MLX's uint32 read as int32
    s = torch.rand((8, 2), generator=gen).to(torch.bfloat16)
    b = torch.rand((8, 2), generator=gen).to(torch.bfloat16)
    got = latent.dequant_mlx4(words.cuda(), s.cuda(), b.cuda()).cpu()
    want = q.float() * s.float().repeat_interleave(64, 1) + b.float().repeat_interleave(64, 1)
    assert torch.equal(got, want)


@cuda
def test_latent_attention_matches_expanded_definition():
    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator().manual_seed(1)
    wk, wv = _weights(gen)
    a = latent.AbsorbW(wk.cuda(), wv.cuda())
    P, R = 700, 8
    lat = torch.randn((P + R, L), generator=gen).to(torch.bfloat16)
    q = torch.randn((R, H, D), generator=gen).to(torch.bfloat16)
    cache = torch.zeros((2048, L), dtype=torch.bfloat16, device="cuda")
    cache[:P + R] = lat.cuda()
    s = latent.LatentScratch(16, H, latent.chunks_for(2048), "cuda")
    got = _latent_forward(latent, q.cuda(), cache, P, a, s, R).double().cpu()
    want = _reference(q, lat, wk.to(torch.bfloat16).float(), wv.to(torch.bfloat16).float(), P, D ** -0.5)
    err = (got - want).abs().max().item()
    assert err < 2e-2 * want.abs().max().item(), err


@cuda
@pytest.mark.parametrize("R", [2, 5, 8, 16])
def test_window_rows_equal_serial_steps(R):
    """Row r of a window at pos P equals a one-row step at P + r, bit for bit (dense path)."""
    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator().manual_seed(2)
    wk, wv = _weights(gen)
    a = latent.AbsorbW(wk.cuda(), wv.cuda())
    P = 1500
    lat = torch.randn((P + R, L), generator=gen).to(torch.bfloat16).cuda()
    q = torch.randn((R, H, D), generator=gen).to(torch.bfloat16).cuda()
    cache = torch.zeros((2560, L), dtype=torch.bfloat16, device="cuda")
    cache[:P + R] = lat
    s = latent.LatentScratch(64, H, latent.chunks_for(2560 + 64), "cuda")
    window = _latent_forward(latent, q, cache, P, a, s, R)
    for r in range(R):
        one = _latent_forward(latent, q[r:r + 1].contiguous(), cache, P + r, a, s, 1)
        assert torch.equal(window[r], one[0]), f"row {r} differs from its serial step"


@cuda
def test_sparse_rows_equal_serial_steps_and_reference():
    """Sparse rows (selected tokens) are row-invariant and match the float64 definition over the same tokens."""
    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator().manual_seed(3)
    wk, wv = _weights(gen)
    a = latent.AbsorbW(wk.cuda(), wv.cuda())
    n, R, W = 6000, 4, 2051
    lat = torch.randn((n, L), generator=gen).to(torch.bfloat16)
    q = torch.randn((R, H, D), generator=gen).to(torch.bfloat16)
    cache = lat.cuda()
    tokens = torch.full((R, W), -1, dtype=torch.int32)
    picks = []
    for r in range(R):
        sel = torch.sort(torch.randperm(n - 10, generator=gen)[:2048 - r]).values
        picks.append(sel)
        tokens[r, :sel.numel()] = sel.to(torch.int32)
    counts = torch.tensor([p.numel() for p in picks], dtype=torch.int32)
    tokens, counts = tokens.cuda(), counts.cuda()
    qa = latent.absorb_q(q.cuda(), a, torch.empty((R, H, L), dtype=torch.bfloat16, device="cuda"))
    ol = torch.zeros((R, H, L), dtype=torch.bfloat16, device="cuda")
    latent.sparse_attention(qa, cache, tokens, counts, ol, D ** -0.5)
    for r in range(R):
        one = torch.zeros((1, H, L), dtype=torch.bfloat16, device="cuda")
        latent.sparse_attention(qa[r:r + 1].contiguous(), cache, tokens[r:r + 1].contiguous(),
                                counts[r:r + 1].contiguous(), one, D ** -0.5)
        assert torch.equal(ol[r], one[0]), f"sparse row {r} differs from its serial step"
    out = latent.expand_v(ol, a, torch.empty((R, H, D), dtype=torch.bfloat16, device="cuda")).double().cpu()
    wk16, wv16 = wk.to(torch.bfloat16).double(), wv.to(torch.bfloat16).double()
    for r in range(R):
        c = lat[picks[r]].double()
        keys = torch.einsum("hdl,nl->nhd", wk16, c)
        vals = torch.einsum("hdl,nl->nhd", wv16, c)
        sc = torch.einsum("hd,nhd->hn", q[r].double(), keys) * D ** -0.5
        want = torch.einsum("hn,nhd->hd", torch.softmax(sc, dim=1), vals)
        assert (out[r] - want).abs().max().item() < 2e-2 * want.abs().max().item()


def _select_tokens_loop(pools, pos, R, POOL=4, TOPK=512):
    """TensorFold 0.3.4's per-row loop, kept as the reference for the vectorized select_tokens."""
    width = TOPK * POOL + POOL - 1
    tokens = torch.full((R, width), -1, dtype=torch.int32)
    counts = []
    for r in range(R):
        q = pos + r
        npool = (q + 1) // POOL
        if npool <= TOPK:
            counts.append(0)
            continue
        body = (pools[r, :, None] * POOL + torch.arange(POOL)).reshape(-1)
        tail = torch.arange(npool * POOL, q + 1)
        row = torch.cat([body, tail])
        tokens[r, :row.numel()] = row.to(torch.int32)
        counts.append(row.numel())
    return tokens, torch.tensor(counts, dtype=torch.int32)


@cuda
@pytest.mark.parametrize("pos,R", [(2040, 16), (2047, 64), (9000, 5), (130000, 257)])
def test_select_tokens_vectorized_equals_loop(pos, R):
    from tensorfold.families.glm5_next.cuda import sparse

    gen = torch.Generator().manual_seed(pos + R)
    H, D = 32, 128
    npool_max = (pos + R) // 4 + 2
    qi = torch.randn((R, H * D), generator=gen).to(torch.bfloat16).cuda()
    wts = torch.randn((R, H), generator=gen).to(torch.bfloat16).cuda()
    pk = torch.randn((npool_max, D), generator=gen).to(torch.bfloat16).cuda()
    pos_dev = torch.tensor([pos], dtype=torch.int32, device="cuda")
    tokens, counts = sparse.select_tokens(qi, wts, pk, pos, R, npool_max - 2, pos_dev)
    # the same pool choice the function made, fed to the reference loop
    scores = torch.empty((R, npool_max - 2), dtype=torch.float32, device="cuda")
    sparse._scores[(R, -(-(npool_max - 2) // 64))](qi, wts, wts.stride(0), pk, scores, pos_dev, R, npool_max - 2,
                                                    128 ** -0.5, 1.0 / 5.656854249492381, H=32, HP=32, D=128, BP=64,
                                                    RB=1, num_warps=4)
    blocked = torch.empty_like(scores)
    sparse._scores[(-(-R // 16), -(-(npool_max - 2) // 64))](qi, wts, wts.stride(0), pk, blocked, pos_dev, R,
                                                              npool_max - 2, 128 ** -0.5, 1.0 / 5.656854249492381, H=32,
                                                              HP=32, D=128, BP=64, RB=16,
                                                              num_warps=4)
    assert torch.equal(scores, blocked), "row-blocked scores differ from one row a program"
    order = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :512]
    pools = torch.sort(order, dim=1).values.cpu()
    want_t, want_c = _select_tokens_loop(pools, pos, R)
    got_t, got_c = tokens.cpu(), counts.cpu()
    assert torch.equal(got_c, want_c)
    for r in range(R):
        n = int(want_c[r])
        assert torch.equal(got_t[r, :n], want_t[r, :n]), f"row {r}"


def _q4_rows(n, k, gen):
    """Random MLX affine 4-bit rows: words [n, k / 8] int32, scales and biases [n, k / 64] bf16."""
    w = torch.randint(0, 2 ** 31 - 1, (n, k // 8), generator=gen, dtype=torch.int64).to(torch.int32)
    s = (torch.rand((n, k // 64), generator=gen) * 0.01 + 0.002).to(torch.bfloat16)
    b = (torch.randn((n, k // 64), generator=gen) * 0.01).to(torch.bfloat16)
    return w, s, b


def _absorb_q4(gen):
    from tensorfold.families.glm5_next.cuda import latent

    k = _q4_rows(H * D, L, gen)
    v = _q4_rows(H * D, L, gen)
    a = latent.AbsorbQ4(tuple(t.cuda() for t in k), tuple(t.cuda() for t in v), H)
    wk = latent.dequant_mlx4(*(t.cuda() for t in k)).reshape(H, D, L).cpu()
    wv = latent.dequant_mlx4(*(t.cuda() for t in v)).reshape(H, D, L).cpu()
    return a, wk, wv


@cuda
def test_q4_latent_attention_matches_expanded_definition():
    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator().manual_seed(11)
    a, wk, wv = _absorb_q4(gen)
    P, R = 900, 8
    lat = torch.randn((P + R, L), generator=gen).to(torch.bfloat16)
    q = torch.randn((R, H, D), generator=gen).to(torch.bfloat16)
    cache = torch.zeros((2048, L), dtype=torch.bfloat16, device="cuda")
    cache[:P + R] = lat.cuda()
    s = latent.LatentScratch(16, H, latent.chunks_for(2048), "cuda")
    got = _latent_forward(latent, q.cuda(), cache, P, a, s, R).double().cpu()
    want = _reference(q, lat, wk, wv, P, D ** -0.5)
    err = (got - want).abs().max().item()
    assert err < 2e-2 * want.abs().max().item(), err


@cuda
@pytest.mark.parametrize("R", [2, 8, 64, 300])
def test_q4_window_rows_equal_serial_steps(R):
    from tensorfold.families.glm5_next.cuda import latent

    gen = torch.Generator().manual_seed(12 + R)
    a, _, _ = _absorb_q4(gen)
    P = 1200
    lat = torch.randn((P + R, L), generator=gen).to(torch.bfloat16).cuda()
    q = torch.randn((R, H, D), generator=gen).to(torch.bfloat16).cuda()
    cache = torch.zeros((2560, L), dtype=torch.bfloat16, device="cuda")
    cache[:P + R] = lat
    s = latent.LatentScratch(512, H, latent.chunks_for(2560 + 512), "cuda")
    window = _latent_forward(latent, q, cache, P, a, s, R)
    for r in sorted({0, 1, R // 2, R - 1}):
        one = _latent_forward(latent, q[r:r + 1].contiguous(), cache, P + r, a, s, 1)
        assert torch.equal(window[r], one[0]), f"row {r} of {R} differs from its serial step"


@cuda
def test_top_pools_equals_stable_sort_with_ties():
    from tensorfold.families.glm5_next.cuda import sparse

    gen = torch.Generator().manual_seed(5)
    scores = torch.randint(-3, 4, (64, 3000), generator=gen).float()          # many exact ties
    scores[:, 2000:] = float("-inf")
    scores[0, :10] = -0.0
    scores = scores.cuda()
    want = torch.sort(torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :512], dim=1).values
    assert torch.equal(sparse._top_pools(scores, 512), want)


@cuda
def test_prefill_head_tiles_give_the_decode_tiles_bits():
    """Prefill chunks (64 rows or more) run a rank's heads in one 32-head tile, decode windows in 16-head tiles:
    each row gets the same bits either way, dense and sparse."""
    from tensorfold.families.glm5_next.cuda import latent

    assert latent.head_block(64) == latent.HB_WIDE and latent.head_block(8) == latent.HB
    gen = torch.Generator().manual_seed(7)
    heads = 32
    n, R, W = 6000, 72, 2051
    cache = torch.randn((n, L), generator=gen).to(torch.bfloat16).cuda()
    qa = (torch.randn((R, heads, L), generator=gen) * 0.05).to(torch.bfloat16).cuda()
    # dense: rows at P .. P + R - 1 against their serial steps
    P = 1500
    s = latent.LatentScratch(R, heads, latent.chunks_for(P + R), "cuda")
    pos = torch.tensor([P], dtype=torch.int32, device="cuda")
    wide = latent.attention(qa, cache, pos, s, scale=0.07, nch=latent.chunks_for(P + R),
                            out=torch.empty((R, heads, L), dtype=torch.bfloat16, device="cuda")).clone()
    for r in (0, 1, 37, R - 1):
        one = latent.attention(qa[r:r + 1].contiguous(), cache, torch.tensor([P + r], dtype=torch.int32, device="cuda"),
                               s, scale=0.07, nch=latent.chunks_for(P + R),
                               out=torch.empty((1, heads, L), dtype=torch.bfloat16, device="cuda"))
        assert torch.equal(wide[r], one[0]), f"dense row {r}"
    # sparse: each row's own selection
    tokens = torch.full((R, W), -1, dtype=torch.int32)
    counts = []
    for r in range(R):
        sel = torch.sort(torch.randperm(n, generator=gen)[:2048 - r % 5]).values
        tokens[r, :sel.numel()] = sel.to(torch.int32)
        counts.append(sel.numel())
    tokens, counts = tokens.cuda(), torch.tensor(counts, dtype=torch.int32, device="cuda")
    wide = torch.zeros((R, heads, L), dtype=torch.bfloat16, device="cuda")
    latent.sparse_attention(qa, cache, tokens, counts, wide, 0.07)
    for r in (0, 3, 41, R - 1):
        one = torch.zeros((1, heads, L), dtype=torch.bfloat16, device="cuda")
        latent.sparse_attention(qa[r:r + 1].contiguous(), cache, tokens[r:r + 1].contiguous(),
                                counts[r:r + 1].contiguous(), one, 0.07)
        assert torch.equal(wide[r], one[0]), f"sparse row {r}"
