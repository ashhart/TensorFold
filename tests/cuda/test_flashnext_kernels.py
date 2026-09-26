"""Flash Next CUDA kernels: a row's bits never depend on the other rows of a window, and each kernel
matches an fp32 reference within bf16 tolerance."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import gdn, moe, qmm  # noqa: E402

DEV = "cuda"


def _mlx_weights(n: int, k: int, seed: int, lead: tuple = ()):
    g = torch.Generator(device=DEV).manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (*lead, n, k // 8), generator=g, device=DEV,
                          dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((*lead, n, k // 32), generator=g, device=DEV) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((*lead, n, k // 32), generator=g, device=DEV) * 0.02).to(torch.bfloat16)
    return words, scales, biases


SHAPES = [(324, 10240), (10240, 320), (16480, 2560), (2560, 6144), (2560, 2560), (13952, 2560)]
ROWS = [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33, 64]


@pytest.mark.parametrize("n,k", SHAPES)
def test_qmm_rows_do_not_depend_on_the_window(n, k):
    q = qmm.make_q4(*_mlx_weights(n, k, n + k))
    x = torch.randn((64, k), device=DEV).to(torch.bfloat16)
    alone = torch.cat([qmm.matmul(x[r:r + 1], q) for r in range(64)])
    for m in ROWS:
        assert torch.equal(qmm.matmul(x[:m], q), alone[:m]), (n, k, m)
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(3)).to(DEV)
    assert torch.equal(qmm.matmul(x[perm], q), alone[perm])


@pytest.mark.parametrize("n,k", SHAPES)
def test_qmm_launch_settings_keep_the_bits(n, k):
    """Program tile width, unrolling, warps and stages change the schedule, never a row's sums."""

    q = qmm.make_q4(*_mlx_weights(n, k, 3 * n + k))
    x = torch.randn((5, k), device=DEV).to(torch.bfloat16)
    ref = qmm.matmul(x, q)
    for block_n, gpi, warps, stages in ((32, 1, 4, 2), (32, 4, 8, 4), (64, 2, 8, 2), (64, 1, 4, 4)):
        got = qmm.matmul(x, q, gpi=gpi, num_warps=warps, num_stages=stages, block_n=block_n)
        assert torch.equal(got, ref), (n, k, block_n, gpi, warps, stages)


@pytest.mark.parametrize("n,k", SHAPES[:3])
def test_qmm_matches_fp32(n, k):
    w = _mlx_weights(n, k, 7 * n + k)
    q = qmm.make_q4(*w)
    x = torch.randn((5, k), device=DEV).to(torch.bfloat16)
    y = qmm.matmul(x, q).float()
    ref = x.float() @ qmm.dequantize(*w).T
    err = (y - ref).abs().max().item()
    assert err <= ref.abs().max().item() * 2 ** -7, err
    back = qmm.to_mlx(q)
    assert all(torch.equal(a, b) for a, b in zip(back, w))


def _experts(e: int, width: int, dims: int):
    gate = _mlx_weights(width, dims, 11, (e,))
    up = _mlx_weights(width, dims, 12, (e,))
    down = _mlx_weights(dims, width, 13, (e,))
    sg = _mlx_weights(width, dims, 14)
    su = _mlx_weights(width, dims, 15)
    sd = _mlx_weights(dims, width, 16)
    return qmm.make_experts(gate, up, down, (sg, su, sd)), (gate, up, down, sg, su, sd)


class _Cfg:
    num_experts_per_tok = 10
    num_experts = 64
    moe_intermediate_size = 640
    hidden_size = 2560


def _moe_rows(x, router_rows, ex, rows_max):
    buf = moe.MoEBuffers(rows_max, _Cfg, DEV)
    xs = qmm.group_sums(x)
    moe.moe(x, xs, router_rows, ex, buf, _Cfg)
    return buf


def test_moe_rows_do_not_depend_on_the_window():
    ex, _ = _experts(64, 640, 2560)
    torch.manual_seed(5)
    router_rows = (torch.randn((65, 2560), device=DEV) * 0.02).to(torch.bfloat16)
    x = torch.randn((8, 2560), device=DEV).to(torch.bfloat16)
    singles = []
    for r in range(8):
        b = _moe_rows(x[r:r + 1], router_rows, ex, 1)
        singles.append((b.pick[0].clone(), b.wts[0].clone(), b.y[0].clone()))
    for m in (2, 3, 4, 8):
        b = _moe_rows(x[:m], router_rows, ex, m)
        for r in range(m):
            assert torch.equal(b.pick[r], singles[r][0])
            assert torch.equal(b.wts[r], singles[r][1])
            assert torch.equal(b.y[r], singles[r][2]), (m, r)
        count = int(b.group.count.item())
        ids = b.group.ids[:count].tolist()
        assert ids == sorted(set(b.pick[:m].reshape(-1).tolist()))


def test_moe_launch_settings_keep_the_bits():
    ex, _ = _experts(64, 640, 2560)
    torch.manual_seed(7)
    router_rows = (torch.randn((65, 2560), device=DEV) * 0.02).to(torch.bfloat16)
    x = torch.randn((4, 2560), device=DEV).to(torch.bfloat16)
    ref = _moe_rows(x, router_rows, ex, 4)
    y0 = ref.y.clone()
    for block_n, gpi, warps, stages in ((32, 1, 4, 2), (32, 4, 8, 3), (64, 2, 8, 4)):
        xs = qmm.group_sums(x)
        qmm.moe_gateup(x, xs, ex, ref.group, ref.act, ref.axs, gpi=gpi, num_warps=warps, num_stages=stages,
                       block_n=block_n)
        qmm.moe_down(ref.act, ref.axs, ex, ref.group, ref.y, gpi=gpi, num_warps=warps, num_stages=stages,
                     block_n=block_n)
        assert torch.equal(ref.y[:4], y0[:4]), (block_n, gpi, warps, stages)


def test_moe_matches_fp32():
    ex, (gate, up, down, sg, su, sd) = _experts(64, 640, 2560)
    torch.manual_seed(6)
    router_rows = (torch.randn((65, 2560), device=DEV) * 0.02).to(torch.bfloat16)
    x = torch.randn((3, 2560), device=DEV).to(torch.bfloat16)
    b = _moe_rows(x, router_rows, ex, 3)
    xf = x.float()
    logits = xf @ router_rows.float().T
    for r in range(3):
        top = sorted(range(64), key=lambda e: (-logits[r, e].item(), e))[:10]
        assert b.pick[r, :10].tolist() == top
        for k, e in enumerate(top + [64]):
            if e < 64:
                gw, uw, dw = (qmm.dequantize(t[0][e], t[1][e], t[2][e]) for t in (gate, up, down))
            else:
                gw, uw, dw = (qmm.dequantize(*t) for t in (sg, su, sd))
            act = (torch.nn.functional.silu(xf[r] @ gw.T) * (xf[r] @ uw.T))
            y = act @ dw.T
            err = (b.y[r, k] - y).abs().max().item()
            assert err <= y.abs().max().item() * 2 ** -5 + 1e-3, (r, k, err)


def _gdn_inputs(rows: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    p = (torch.randn((rows, gdn.PW), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    cs = (torch.randn((3, gdn.CONV), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    cw = (torch.randn((gdn.CONV, 4), generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    state = torch.randn((gdn.NV, gdn.DV, gdn.DK), generator=g, device=DEV) * 0.05
    a_log = torch.randn((gdn.NV,), generator=g, device=DEV) * 0.5
    dt = torch.randn((gdn.NV,), generator=g, device=DEV) * 0.5
    nw = (1 + 0.1 * torch.randn((gdn.DV,), generator=g, device=DEV)).to(torch.bfloat16)
    return p, cs, cw, state, a_log, dt, nw


def test_gdn_window_rows_and_replay_match_serial_steps():
    rows = 6
    p, cs, cw, state, a_log, dt, nw = _gdn_inputs(rows, 21)
    win = gdn.GDNScratch(rows, DEV)
    out_state = torch.empty_like(state)
    gdn.chain(p, cs, cw, state, a_log, dt, nw, 1e-6, rows, win, out_state)
    # serial: one row at a time, conv window and state carried forward
    st = state.clone()
    conv = cs.clone()
    one = gdn.GDNScratch(1, DEV)
    states = []
    for r in range(rows):
        nxt = torch.empty_like(st)
        gdn.chain(p[r:r + 1].contiguous(), conv, cw, st, a_log, dt, nw, 1e-6, 1, one, nxt)
        assert torch.equal(one.out[0], win.out[r]), r
        assert torch.equal(one.xs[0], win.xs[r]), r
        conv = torch.cat([conv, p[r:r + 1, :gdn.CONV]])[1:].contiguous()
        st = nxt
        states.append(st.clone())
    assert torch.equal(out_state, states[-1])
    for keep in range(1, rows + 1):
        rep = torch.empty_like(state)
        gdn.replay(state, win, keep, rep)
        assert torch.equal(rep, states[keep - 1]), keep


def test_gdn_matches_fp32():
    rows = 3
    p, cs, cw, state, a_log, dt, nw = _gdn_inputs(rows, 22)
    win = gdn.GDNScratch(rows, DEV)
    out_state = torch.empty_like(state)
    gdn.chain(p, cs, cw, state, a_log, dt, nw, 1e-6, rows, win, out_state)
    F = torch.nn.functional
    C, NV, DV, NK, DK = gdn.CONV, gdn.NV, gdn.DV, gdn.NK, gdn.DK
    inp = torch.cat([cs, p[:, :C]]).float()
    conv = sum(inp[j:j + rows] * cw.float()[:, j] for j in range(4))
    conv = F.silu(conv)
    q = conv[:, :NK * DK].reshape(rows, NK, DK)
    k = conv[:, NK * DK:2 * NK * DK].reshape(rows, NK, DK)
    v = conv[:, 2 * NK * DK:].reshape(rows, NV, DV)
    q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * DK ** -0.5
    k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    z = p[:, C:C + NV * DV].float().reshape(rows, NV, DV)
    b = p[:, C + NV * DV:C + NV * DV + NV].float()
    a = p[:, C + NV * DV + NV:].float()
    beta = torch.sigmoid(b)
    g = torch.exp(-torch.exp(a_log) * F.softplus(a + dt))
    S = state.clone()
    outs = []
    for t in range(rows):
        qt, kt = q[t].repeat_interleave(3, 0), k[t].repeat_interleave(3, 0)
        S = S * g[t][:, None, None]
        kv = (S * kt[:, None, :]).sum(-1)
        S = S + kt[:, None, :] * ((v[t] - kv) * beta[t][:, None])[:, :, None]
        y = (S * qt[:, None, :]).sum(-1)
        yn = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6) * nw.float()
        outs.append((yn * torch.sigmoid(z[t])).reshape(-1))
    ref = torch.stack(outs)
    err = (win.out.float() - ref).abs().max().item()
    assert err < 0.05 * ref.abs().max().item(), err
    assert (out_state - S).abs().max().item() < 1e-2


def test_qsa_selection_matches_a_plain_reference():
    """Sparse rows: pooled block keys, fp32 block scores, the 512 best blocks (lower ids among equal scores) in
    block order with 4 keys each, then the unfinished tail; dense rows keep their causal length."""

    from tensorfold.families.qwen4_exp.cuda import attention as att

    torch.manual_seed(11)
    cap, rows, di, hi = 4096, 6, 128, 4
    p0 = 2600
    scratch = att.AttnScratch(rows, 24, 256, cap, DEV)
    ikc = torch.randn((cap, di), device=DEV).to(torch.bfloat16)
    pooled = torch.zeros((cap // 4, di), dtype=torch.bfloat16, device=DEV)
    w_ = (1 + 0.1 * torch.randn((di,), device=DEV)).float()
    inv = (1e7 ** (-torch.arange(0, 32, dtype=torch.float64) / 32)).float().to(DEV)
    iq = torch.randn((rows, hi, di), device=DEV).to(torch.bfloat16)
    pos = torch.tensor([p0], dtype=torch.int32, device=DEV)
    # every block before the window is pooled once, as earlier forwards would have
    for start in range(0, p0, 64):
        att._pool[(64 // 4 + 2,)](ikc, pooled, torch.tensor([start], dtype=torch.int32, device=DEV), w_, inv, 1e-6,
                                  64, DI=di, HALF=32, RATIO=4, num_warps=1)
    att.qsa_select(iq, ikc, pooled, pos, w_, inv, 1e-6, scratch, rows)
    # force ties: rescore with a copy where pairs of blocks share a score
    sc = scratch.scores.clone()
    for r in range(rows):
        end = p0 + r + 1
        c = end // 4
        # reference pooled keys and scores
        blocks = ikc[:c * 4].float().reshape(c, 4, di).sum(1) / 4
        blocks = blocks.to(torch.bfloat16).float()
        xn = (blocks * torch.rsqrt(blocks.pow(2).mean(-1, keepdim=True) + 1e-6) * w_).to(torch.bfloat16).float()
        ang = (torch.arange(c, device=DEV) * 4).float()[:, None] * inv[None, :]
        cos, sin = torch.cos(ang), torch.sin(ang)
        rot = xn.clone()
        rot[:, :32] = xn[:, :32] * cos - xn[:, 32:64] * sin
        rot[:, 32:64] = xn[:, 32:64] * cos + xn[:, :32] * sin
        assert (pooled[:c].float() - rot.to(torch.bfloat16).float()).abs().max() < 0.05
        s_ref = torch.relu(iq[r].float() @ pooled[:c].float().T).sum(0) / di ** 0.5
        assert torch.allclose(sc[r, :c], s_ref, rtol=1e-4, atol=1e-4)
        order = sorted(range(c), key=lambda b: (-sc[r, b].item(), b))[:512]
        want = [4 * b + k for b in sorted(order) for k in range(4)] + list(range(4 * c, end))
        n = int(scratch.nk[r])
        assert int(scratch.sparse[r]) == 1 and n == len(want)
        assert scratch.ids[r, :n].tolist() == want, r
    # ties: equal scores at the cut go to the lower block ids
    scratch.scores.copy_(torch.floor(sc * 4) / 4)
    att._select[(rows,)](scratch.scores, pos, scratch.ids, scratch.nk, scratch.sparse, scratch.nb, RATIO=4, TOP=512,
                         IDW=scratch.idw, BLOCK=1024, num_warps=16)
    for r in range(rows):
        c = (p0 + r + 1) // 4
        s = scratch.scores[r, :c]
        order = sorted(range(c), key=lambda b: (-s[b].item(), b))[:512]
        assert scratch.ids[r, :2048:4].tolist() == [4 * b for b in sorted(order)], r


def test_gdn_at_a_tensor_parallel_ranks_head_counts():
    """8 key and 24 value heads (one of two ranks): a window's rows and a replayed prefix give serial bits."""

    nk, nv, rows = 8, 24, 5
    conv, pw = gdn.widths(nk, nv)
    g = torch.Generator(device=DEV).manual_seed(31)
    p = (torch.randn((rows, pw), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    cs = (torch.randn((3, conv), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    cw = (torch.randn((conv, 4), generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    state = torch.randn((nv, 128, 128), generator=g, device=DEV) * 0.05
    a_log = torch.randn((nv,), generator=g, device=DEV) * 0.5
    dt = torch.randn((nv,), generator=g, device=DEV) * 0.5
    nw = torch.ones((128,), device=DEV).to(torch.bfloat16)
    win = gdn.GDNScratch(rows, DEV, nk, nv)
    out_state = torch.empty_like(state)
    gdn.chain(p, cs, cw, state, a_log, dt, nw, 1e-6, rows, win, out_state)
    st, cv = state.clone(), cs.clone()
    one = gdn.GDNScratch(1, DEV, nk, nv)
    for r in range(rows):
        nxt = torch.empty_like(st)
        gdn.chain(p[r:r + 1].contiguous(), cv, cw, st, a_log, dt, nw, 1e-6, 1, one, nxt)
        assert torch.equal(one.out[0], win.out[r]), r
        cv = torch.cat([cv, p[r:r + 1, :conv]])[1:].contiguous()
        st = nxt
        if r == 2:
            rep = torch.empty_like(state)
            gdn.replay(state, win, 3, rep)
            assert torch.equal(rep, st)
    assert torch.equal(out_state, st)


def test_rank_ordered_partials_write_back_like_their_rounded_sum():
    """Tensor parallel: streams take bf16(p0 + p1) (rank 0 first) exactly as a plain branch would."""

    from tensorfold.families.qwen4_exp.cuda import glue

    torch.manual_seed(12)
    rows, d, s = 3, 2560, 4
    h = torch.randn((rows, s * d), device=DEV).to(torch.bfloat16)
    parts = torch.randn((2, rows, d), device=DEV)
    inj = (torch.rand((rows, s), device=DEV) * 2).to(torch.bfloat16)
    pss_a = torch.empty((rows, d // 256, s), device=DEV)
    pss_b = torch.empty_like(pss_a)
    ha, hb = h.clone(), h.clone()
    glue.hc_writeback(ha, ha, pss_a, s, 3, branch=parts.contiguous(), inject=inj)
    branch = (parts[0] + parts[1]).to(torch.bfloat16)
    glue.hc_writeback(hb, hb, pss_b, s, 1, branch=branch, inject=inj)
    assert torch.equal(ha, hb) and torch.equal(pss_a, pss_b)


def test_host_table_gathers_the_rows_across_shards_and_files(tmp_path):
    """The n-gram rows come out of the memory-mapped checkpoint files byte for byte, whatever shard and file."""

    import json
    import struct

    import numpy as np

    from tensorfold.families.qwen4_exp.cuda.weights import HostTable, _header

    rng = np.random.default_rng(0)
    files, words, scales, biases = [], [], [], []
    for fi, counts in enumerate(((7, 5), (3, 9, 4))):
        tensors = {}
        for si, rows in enumerate(counts):
            key = f"shard_{fi}_{si}"
            tensors[key + ".weight"] = rng.integers(0, 2**32, (rows, 20), dtype=np.uint64).astype(np.uint32)
            tensors[key + ".scales"] = rng.integers(0, 2**16, (rows, 5), dtype=np.uint64).astype(np.uint16)
            tensors[key + ".biases"] = rng.integers(0, 2**16, (rows, 5), dtype=np.uint64).astype(np.uint16)
        header, blobs, at = {"__metadata__": {}}, [], 0
        for name, arr in tensors.items():
            raw = arr.tobytes()
            header[name] = {"dtype": "U32" if arr.dtype == np.uint32 else "U16", "shape": list(arr.shape),
                            "data_offsets": [at, at + len(raw)]}
            blobs.append(raw)
            at += len(raw)
        head = json.dumps(header).encode()
        head += b" " * (-len(head) % 8)
        path = tmp_path / f"part{fi}.safetensors"
        path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))
        h = _header(path)
        for si in range(len(counts)):
            key = f"shard_{fi}_{si}"
            files.append((path, h[key + ".weight"], h[key + ".scales"], h[key + ".biases"]))
            words.append(tensors[key + ".weight"])
            scales.append(tensors[key + ".scales"])
            biases.append(tensors[key + ".biases"])
    table = HostTable(files)
    W, S, B = np.concatenate(words), np.concatenate(scales), np.concatenate(biases)
    ids = rng.integers(0, table.rows, (13, 16))
    w, s, b = table.gather(ids)
    flat = ids.reshape(-1)
    assert table.rows == len(W)
    assert np.array_equal(w, W[flat]) and np.array_equal(s, S[flat]) and np.array_equal(b, B[flat])


@pytest.mark.parametrize("rows", [1, 3, 7, 16])
def test_hc_upmix_gives_the_bits_of_the_matmul_then_the_mix(rows):
    """The fused up projection and stream mix give the bits of the matmul then glue.hc_mix."""

    from tensorfold.families.qwen4_exp.cuda import glue

    S, D, LOW = 4, 2560, 320
    q = qmm.make_q4(*_mlx_weights(S * D, LOW, 11))
    g = torch.Generator(device=DEV).manual_seed(rows)
    act = torch.randn((rows, LOW), generator=g, device=DEV).to(torch.bfloat16)
    xs = qmm.group_sums(act)
    normed = torch.randn((rows, S * D), generator=g, device=DEV).to(torch.bfloat16)
    up = qmm.matmul(act, q, xs)
    ref_mixed = torch.empty((rows, D), dtype=torch.bfloat16, device=DEV)
    ref_xs = torch.empty((rows, D // 32), dtype=torch.float32, device=DEV)
    glue.hc_mix(up, normed, ref_mixed, ref_xs, S)
    mixed = torch.empty_like(ref_mixed)
    xsm = torch.empty_like(ref_xs)
    qmm.hc_upmix(act, xs, q, normed, mixed, xsm, S)
    assert torch.equal(mixed, ref_mixed) and torch.equal(xsm, ref_xs)


@pytest.mark.parametrize("rows", [1, 7, 16])
@pytest.mark.parametrize("inject", [True, False])
def test_hc_readout_fused_gives_the_separate_kernels_bits(rows, inject):
    """A hyper-connection read-out in 3 kernels (decode windows) and in 5 (prefill chunks): the same normed
    streams, mix, group sums and inject gates."""

    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.cuda import forward as fwd, glue
    from tensorfold.families.qwen4_exp.cuda.weights import HC

    S, D, LOW = 4, 2560, 320
    down = qmm.stack_q4([_mlx_weights(LOW, S * D, 5)] + ([_mlx_weights(S, S * D, 6)] if inject else []))
    hc = HC(down, qmm.make_q4(*_mlx_weights(S * D, LOW, 7)),
            1 + 0.05 * torch.randn((S * D,), generator=torch.Generator(device=DEV).manual_seed(8), device=DEV), inject)
    g = torch.Generator(device=DEV).manual_seed(rows + 100 * inject)
    h = (torch.randn((rows, S * D), generator=g, device=DEV) * 3).to(torch.bfloat16)

    def buffers():
        f32, bf = torch.float32, torch.bfloat16
        b = SimpleNamespace(pss=torch.empty((rows, D // 256, S), dtype=f32, device=DEV),
                            normed=torch.empty((rows, S * D), dtype=bf, device=DEV),
                            xs_normed=torch.empty((rows, S * D // 32), dtype=f32, device=DEV),
                            dn=torch.empty((rows, LOW + S), dtype=bf, device=DEV),
                            dn_mix=torch.empty((rows, LOW), dtype=bf, device=DEV),
                            act=torch.empty((rows, LOW), dtype=bf, device=DEV),
                            xs_act=torch.empty((rows, LOW // 32), dtype=f32, device=DEV),
                            up=torch.empty((rows, S * D), dtype=bf, device=DEV),
                            mixed=torch.empty((rows, D), dtype=bf, device=DEV),
                            xs_mixed=torch.empty((rows, D // 32), dtype=f32, device=DEV),
                            part=torch.empty((32 * 16 * 2560,), dtype=f32, device=DEV),
                            inj=torch.empty((rows, S), dtype=bf, device=DEV))
        glue.hc_writeback(h, h, b.pss, S, 0)
        return b

    fused, plain = buffers(), buffers()
    fwd._readout_fused(hc, fused, h, rows, 1e-6, S, LOW, fused.inj if inject else None)
    fwd._readout_plain(hc, plain, h, rows, 1e-6, S, LOW, plain.inj if inject else None)
    assert torch.equal(fused.normed, plain.normed)
    assert torch.equal(fused.mixed, plain.mixed) and torch.equal(fused.xs_mixed, plain.xs_mixed)
    if inject:
        assert torch.equal(fused.inj, plain.inj)


def test_the_packaged_draft_vocabulary():
    """The MTP drafts' token ids: sorted, distinct, inside the vocabulary, every id below 65,536 and the
    tokenizer's added tokens among them."""

    import numpy as np

    from tensorfold.families.qwen4_exp.cuda.weights import draft_token_ids

    ids = draft_token_ids("default")
    assert np.all(np.diff(ids) > 0) and ids[0] == 0 and ids[-1] < 248_320 and 65_536 < len(ids) < 100_000
    assert np.array_equal(ids[:65_536], np.arange(65_536))
    assert {248_044, 248_045, 248_046, 248_068, 248_069} <= set(ids.tolist())     # end of text, im, think tags
