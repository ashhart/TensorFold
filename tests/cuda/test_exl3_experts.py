"""The universal EXL3 routed-expert kernels (``tensorfold/cuda/exl3/experts``) on synthetic layers.

- The kernels' lane decode against a bit-by-bit decoder written from the format description, for every codebook
  and every bit width (and against ExLlamaV3's own ``reconstruct`` when ExLlamaV3 is importable).
- On 4-bit mcg experts with the GLM tile settings and bf16 SwiGLU, the per-slot outputs are bit-identical to the
  GLM family's EXL3 path (``families/glm5_next/cuda/exl3_mm.py``), GLM-shaped (4096 -> 1024 and 4096 -> 2048).
- Mixed-K layers (a bit width per expert matrix, codebooks 3inst / mcg / mul1): each row gives identical bits alone
  and inside windows of 1, 2, 3, 16, 17, 64 and 128 rows, and agrees with a float64 reference.
- A layer captured in CUDA graphs at 1, 2, 4 and 8 rows replays to exactly the eager call's output.
"""

from __future__ import annotations

import math

import random

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

HAD = 128


def _end(p: int, k2: int) -> int:
    return (p >> 1) * k2 + (k2 if p & 1 else k2 >> 1)


def _cb_value(s: int, cb: int) -> np.float16:
    if cb == 2:
        x = (s * 0x83DCD12D) & 0xFFFFFFFF
        h = 1024.0 + float(sum((x >> (8 * i)) & 0xFF for i in range(4)))
        kinv = float(np.array([0x1EEE], dtype=np.uint16).view(np.float16)[0])
        bias = float(np.array([0xC931], dtype=np.uint16).view(np.float16)[0])
        return np.float16(h * kinv + bias)                      # exact in float64, then one fp16 rounding (fma)
    x = (s * 0xCBAC1FED) & 0xFFFFFFFF if cb == 1 else (s * 89226354 + 64248484) & 0xFFFFFFFF
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo = np.array([x & 0xFFFF], dtype=np.uint16).view(np.float16)[0]
    hi = np.array([x >> 16], dtype=np.uint16).view(np.float16)[0]
    return np.float16(np.float64(lo) + np.float64(hi))


def _slow_tile(words16: np.ndarray, k2: int, cb: int) -> np.ndarray:
    """One tile, bit by bit: int16 words -> [16, 16] fp16 W_q (row = k, column = n)."""

    stream = []
    for j in range(0, len(words16), 2):
        word = (int(words16[j]) & 0xFFFF) | ((int(words16[j + 1]) & 0xFFFF) << 16)
        stream += [(word >> (31 - b)) & 1 for b in range(32)]
    n = len(stream)
    out = np.zeros((16, 16), dtype=np.float16)
    for p in range(256):
        end = _end(p, k2)
        s = 0
        for b in range(end - 16, end):
            s = (s << 1) | stream[b % n]
        lane, j = p // 8, p % 8
        row = 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1)
        col = lane // 4 + 8 * (j >> 2)
        out[row, col] = _cb_value(s, cb)
    return out


def _trellis(k: int, n: int, k2: int, gen: torch.Generator, device="cuda") -> torch.Tensor:
    v = torch.randint(-32768, 32768, (k // 16, n // 16, 8 * k2), dtype=torch.int32, generator=gen)
    return v.to(torch.int16).to(device).contiguous()


def _scale(n: int, mag: float, gen: torch.Generator, device="cuda") -> torch.Tensor:
    sign = torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1
    return (sign * (torch.rand((n,), generator=gen) + 0.5) * mag).half().to(device)


CASES = [(cb, k2) for cb in (0, 1, 2) for k2 in range(2, 17)]


@pytest.mark.parametrize("cb,k2", CASES)
def test_decode_matches_the_bitstream(cb, k2):
    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(100 * cb + k2)
    t = _trellis(32, 48, k2, g)
    w = experts.dequant(t, cb).cpu().numpy()
    tc = t.cpu().numpy()
    for kt, nt in ((0, 0), (1, 2), (0, 1)):
        ref = _slow_tile(tc[kt, nt], k2, cb)
        got = w[kt * 16:(kt + 1) * 16, nt * 16:(nt + 1) * 16]
        assert np.array_equal(ref.view(np.uint16), got.view(np.uint16)), (cb, k2, kt, nt)


@pytest.mark.parametrize("cb,k2", CASES)
def test_decode_matches_exllamav3_reconstruct(cb, k2):
    if k2 % 2 and cb != 2:
        pytest.skip("ExLlamaV3 only allows the half-integer rates with the mul1 codebook")
    if k2 in (9, 11, 13, 15):
        pytest.skip("this ExLlamaV3 fork's reconstruct instantiates only 1.5/2.5/3.5 (the kernel also covers 4.5..7.5)")
    ext = pytest.importorskip("exllamav3.ext").exllamav3_ext
    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(7 + 100 * cb + k2)
    t = _trellis(256, 512, k2, g)
    ours = experts.dequant(t, cb)
    ref = torch.empty_like(ours)
    ext.reconstruct(ref, t, k2 / 2 if k2 % 2 else k2 // 2, cb == 1, cb == 2)
    torch.cuda.synchronize()
    assert torch.equal(ours.view(torch.int16), ref.view(torch.int16))


def _layer(E, D, I, k2s, cb, seed):
    """k2s: list of (gate, up, down) half-bit widths a expert."""

    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(seed)
    gate, up, down = [], [], []
    for e in range(E):
        kg, ku, kd = k2s[e]
        gate.append((_trellis(D, I, kg, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        up.append((_trellis(D, I, ku, g), _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        down.append((_trellis(I, D, kd, g), _scale(I, 1 / math.sqrt(I), g), _scale(D, 0.25, g)))
    return experts.prepare(gate, up, down, cb), (gate, up, down)


def _hadamard(dev):
    i = torch.arange(HAD)
    par = torch.tensor([bin(v).count("1") & 1 for v in range(HAD)])
    return (torch.where(par[i[:, None] & i[None, :]] == 1, -1.0, 1.0).double() / math.sqrt(HAD)).to(dev)


def _rot(v, H):
    s = v.shape
    return (v.reshape(*s[:-1], s[-1] // HAD, HAD) @ H).reshape(s)


def _reference(x, sel, wts, mats, cb):
    """float64 routed output from the decoded W_q (the kernels' decode is checked bit-exact above)."""

    from tensorfold.cuda.exl3 import experts

    H = _hadamard(x.device)
    gate, up, down = mats
    R, k = sel.shape
    out = torch.zeros((R, x.shape[1]), dtype=torch.float64, device=x.device)
    xd = x.double()
    for r in range(R):
        for j in range(k):
            e = int(sel[r, j])
            if e >= len(gate):
                continue

            def lin(v, m):
                t, suh, svh = m
                return _rot(_rot(v * suh.double(), H) @ experts.dequant(t, cb).double(), H) * svh.double()

            gg = lin(xd[r], gate[e])
            uu = lin(xd[r], up[e])
            a = gg * torch.sigmoid(gg) * uu
            out[r] += float(wts[r, j]) * lin(a, down[e])
    return out


def _picks(E, R, k, gen, shared=False):
    sel = torch.stack([torch.randperm(E, generator=gen)[:k] for _ in range(R)]).to(torch.int32)
    w = torch.rand((R, k), generator=gen) * 0.2 + 0.05
    if shared:
        sel = torch.cat([sel, torch.full((R, 1), E, dtype=torch.int32)], 1)
        w = torch.cat([w, torch.ones((R, 1))], 1)
    return sel.cuda().contiguous(), w.float().cuda().contiguous()


MIXED = [
    ("mul1", 2, lambda e: (4 + 2 * (e % 3), 4 + 2 * (e % 3), 4 + 2 * ((e + 1) % 3))),     # MiMo-like 2/3/4 bits
    ("mul1", 2, lambda e: ((2, 3, 5, 6, 7, 8, 10, 12, 14, 16)[e % 10],) * 2 + ((3, 5, 7, 8, 16)[e % 5],)),
    ("mcg", 1, lambda e: ((2, 4, 6, 8, 10, 12, 14, 16)[e % 8],) * 2 + ((4, 6, 8)[e % 3],)),
    ("3inst", 0, lambda e: ((2, 4, 6, 8, 10, 12, 14, 16)[(e + 3) % 8],) * 2 + ((8, 6, 16)[e % 3],)),
]


@pytest.mark.parametrize("name,cb,kfun", MIXED, ids=[m[0] + str(i) for i, m in enumerate(MIXED)])
def test_mixed_k_rows_are_independent_and_match_the_reference(name, cb, kfun):
    from tensorfold.cuda.exl3 import experts

    E, D, I, TOPK = 24, 512, 256, 6
    ex, mats = _layer(E, D, I, [kfun(e) for e in range(E)], cb, seed=11 + cb)
    g = torch.Generator().manual_seed(3)
    ROWS = 128
    x = torch.randn((ROWS, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = _picks(E, ROWS, TOPK, g, shared=True)
    scratch = experts.Scratch(ex, ROWS, TOPK + 1)

    def run(rows):
        out = experts.routed(x[rows], sel[rows].contiguous(), w[rows].contiguous(), ex, scratch, None, len(rows))
        return out.clone()

    full = run(list(range(ROWS)))
    assert torch.isfinite(full).all()
    for n in (1, 2, 3, 16, 17, 64, 128):
        for start in (0, ROWS - n):
            rows = list(range(start, start + n))
            assert torch.equal(run(rows), full[rows]), (name, n, start)
    ref = _reference(x[:6].float(), sel[:6], w[:6], mats, cb)
    err = (full[:6].double() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-2, err


def _glm_case(E, D, NI, rows_list, seed):
    from tensorfold.cuda.exl3 import experts
    from tensorfold.families.glm5_next.cuda import exl3_mm, glue, qmm

    import triton

    g = torch.Generator().manual_seed(seed)

    def trel(k, n):
        return torch.stack([_trellis(k, n, 8, g) for _ in range(E)])

    gt, ut, dt = trel(D, NI), trel(D, NI), trel(NI, D)
    def sc(n, m):
        return torch.stack([_scale(n, m, g) for _ in range(E)]).contiguous()

    suh_g, suh_u, svh_g, svh_u = sc(D, 0.02), sc(D, 0.02), sc(NI, 0.5), sc(NI, 0.5)
    suh_d, svh_d = sc(NI, 0.05), sc(D, 0.2)
    glm = exl3_mm.Exl3Experts(exl3_mm.words(gt), exl3_mm.words(ut), exl3_mm.words(dt), suh_g, suh_u, svh_g, svh_u,
                              suh_d, svh_d, E, NI, D)
    ours = experts.prepare_stacked(gt, ut, dt, suh_g, suh_u, svh_g, svh_u, suh_d, svh_d, "mcg")
    TOPK, SLOTS, LIMIT = 8, 9, 10.0
    maxr = max(rows_list)
    x = torch.randn((maxr, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = _picks(E, maxr, TOPK, g, shared=True)
    s_ours = experts.Scratch(ours, maxr, SLOTS, experts.GLM_GATEUP, experts.GLM_DOWN)
    for R in rows_list:
        pick = sel[:R].contiguous()
        maxu = min(R * TOPK, E) + 1
        grp = qmm.Group(torch.zeros((maxu,), dtype=torch.int32, device="cuda"),
                        torch.zeros((1,), dtype=torch.int32, device="cuda"),
                        torch.full((maxu, R), -1, dtype=torch.int32, device="cuda"))
        glue._group[(1,)](pick, grp.ids, grp.count, grp.members, R, SLOTS=SLOTS, MAXU=maxu, MAXM=R,
                          BLOCK=triton.next_power_of_2(E + 1), num_warps=8)
        s_glm = exl3_mm.Scratch(R, SLOTS, D, NI, "cuda")
        y_glm = torch.full((R * SLOTS, D), 7.0, dtype=torch.float32, device="cuda")
        exl3_mm.routed(x[:R], pick, grp, glm, s_glm, y_glm, R, LIMIT)
        s_ours.y.fill_(7.0)
        y = experts.routed(x[:R], pick, None, ours, s_ours, None, R, limit=LIMIT, act_mode=experts.ACT_BF16)
        torch.cuda.synchronize()
        assert torch.equal(y.view(torch.int32), y_glm.view(torch.int32)), (E, D, NI, R)
        out_glm = torch.empty((R, D), dtype=torch.float32, device="cuda")
        glue.combine(y_glm, w[:R].contiguous(), out_glm)
        out = experts.routed(x[:R], pick, w[:R].contiguous(), ours, s_ours, None, R, limit=LIMIT,
                             act_mode=experts.ACT_BF16, group=False)
        assert torch.equal(out.view(torch.int32), out_glm.view(torch.int32)), ("combine", E, D, NI, R)


def test_glm_small_bit_identical():
    _glm_case(12, 512, 256, (1, 2, 3, 5, 8), seed=1)


@pytest.mark.parametrize("NI", (1024, 2048))
def test_glm_shaped_bit_identical(NI):
    _glm_case(288, 4096, NI, (1, 2, 4, 8, 16), seed=2)


# ------------------------------------------------------------------------------------------------ CUDA graphs

def test_graph_replay_equals_eager_at_1_2_4_8_rows():
    """One graph per row count, all sharing one layer and one scratch (as an engine captures them), replayed after
    the allocator has been churned and after new inputs and picks are copied into the captured buffers: every
    replay is torch.equal to the eager call on the same inputs. A graph keeps device addresses, so this also pins
    that routed() allocates nothing and reads its inputs, tables and scratch where the caller keeps them."""

    from tensorfold.cuda.exl3 import experts

    E, D, I, TOPK, SLOTS = 32, 512, 256, 6, 7
    _, cb, kfun = MIXED[1]                                       # mul1, 1..8 bits and half-integer widths
    ex, _ = _layer(E, D, I, [kfun(e) for e in range(E)], cb, seed=21)
    ROWS = (1, 2, 4, 8)
    scratch = experts.Scratch(ex, max(ROWS), SLOTS)
    g = torch.Generator().manual_seed(5)

    def inputs(R, gen):
        x = torch.randn((R, D), generator=gen).to(torch.bfloat16).cuda()
        sel, w = _picks(E, R, TOPK, gen, shared=True)
        return x, sel, w

    bufs, graphs = {}, {}
    for R in ROWS:
        x, sel, w = inputs(R, g)
        out = torch.empty((R, D), dtype=torch.float32, device="cuda")
        bufs[R] = (x, sel, w, out)
        experts.routed(x, sel, w, ex, scratch, out, R)          # warm up outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            experts.routed(x, sel, w, ex, scratch, out, R)
        graphs[R] = graph
    churn = [torch.full((1 << 18,), -1, dtype=torch.int64, device="cuda") for _ in range(16)]
    for trial in range(2):
        for R in ROWS:
            x, sel, w, out = bufs[R]
            if trial:
                nx, nsel, nw = inputs(R, g)
                x.copy_(nx)
                sel.copy_(nsel)
                w.copy_(nw)
            eager = experts.routed(x, sel, w, ex, scratch, None, R).clone()
            out.fill_(float("nan"))
            graphs[R].replay()
            torch.cuda.synchronize()
            assert torch.isfinite(eager).all(), (R, trial)
            assert torch.equal(out, eager), (R, trial)
    del churn


@pytest.mark.parametrize("slots", (1, 7, 9, 32))
def test_fused_down_and_combine_equal_the_separate_launches(slots):
    """routed() finishes with one launch (down_combine) where it used to run down_epilogue then combine: the
    per-slot outputs and the combined rows are bit-identical to the two launches, with a non-routed slot whose
    output the caller left in y, 1 to 16 rows and 1 to 32 slots."""

    from tensorfold.cuda.exl3 import experts

    E, D, I = 40, 512, 256
    _, cb, kfun = MIXED[1]
    ex, _ = _layer(E, D, I, [kfun(e) for e in range(E)], cb, seed=31)
    ext = experts._ext()
    g = torch.Generator().manual_seed(slots)
    scratch = experts.Scratch(ex, 16, slots)
    for R in (1, 2, 3, 8, 16):
        x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
        # distinct experts a row; with more than one slot the last is a shared-expert slot (y is the caller's)
        sel, w = _picks(E, R, max(slots - 1, 1), g, shared=slots > 1)
        shared = torch.randn((R * slots, D), generator=g).float().cuda()
        P, sk = R * slots, scratch.cfg_d[2]

        scratch.y[:P].copy_(shared)
        fused = experts.routed(x, sel, w, ex, scratch, None, R).clone()
        y_fused = scratch.y[:P].clone()
        # the same call's down projection output is still in scratch.z: finish it with the two separate launches
        scratch.y[:P].copy_(shared)
        ext.down_epilogue(scratch.z, sel, ex.svh_d, scratch.y, R, P, D, sk, slots, E)
        apart = torch.empty((R, D), dtype=torch.float32, device="cuda")
        ext.combine(scratch.y, w, apart, R, D, slots)
        torch.cuda.synchronize()
        assert torch.equal(y_fused.view(torch.int32), scratch.y[:P].view(torch.int32)), (slots, R)
        assert torch.equal(fused.view(torch.int32), apart.view(torch.int32)), (slots, R)


# -------------------------------------------------------------------------------------------------- the lane map

def test_lane_map_extracts_every_window():
    """Mirror ``LaneMap``/``Fmt<K2>`` and check, on a random tile, that every lane's eight windows come out right.

    The kernel merges two 32-bit words and extracts the run's windows by fixed shifts, so a run of ``GV``
    windows must fit in the 64-bit merge. This is the check that pins the ``GV`` table: it fails for
    K2 = 9, 13, 15 with the wrong ``GV`` (the half-bit widths 4.5, 6.5 and 7.5).
    """
    def end(p, k2):
        return (p >> 1) * k2 + (k2 if (p & 1) else (k2 >> 1))

    def gv_of(k2):
        return 2 if k2 >= 13 else (4 if (k2 == 7 or 9 <= k2 <= 12 or k2 == 16) else 8)

    for k2 in range(1, 17):
        tw, gv = 4 * k2, gv_of(k2)
        assert 8 % gv == 0
        words = [random.getrandbits(32) for _ in range(tw)]
        for _ in range(4):
            words = [random.getrandbits(32) for _ in range(tw)]

            def absbit(b):
                i, r = divmod(b, 32)
                return (words[i % tw] >> (31 - r)) & 1

            def state(p):
                e = end(p, k2)
                return sum(absbit(e - 16 + i) << (15 - i) for i in range(16))

            for lane in range(32):
                for g in range(8 // gv):
                    p_last = 8 * lane + g * gv + gv - 1
                    last_end = end(p_last, k2) + 128 * k2
                    hr = (last_end - 1) >> 5
                    sh = (hr + 1) * 32 - last_end
                    assert 0 <= sh <= 32, (k2, lane, g, sh)
                    hi, lo = hr % tw, (hr + tw - 1) % tw
                    mm = ((words[lo] << 32) | words[hi]) >> sh
                    for j in range(gv):
                        p = 8 * lane + g * gv + j
                        off = end(p_last, k2) - end(p, k2)
                        assert off + 16 <= 64, (k2, lane, g, j, off)
                        assert (mm >> off) & 0xFFFF == state(p), (k2, lane, g, j)
