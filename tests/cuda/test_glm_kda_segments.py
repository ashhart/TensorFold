"""GLM-5.3-Flash's segmented KDA kernels (kda.chain_segments, kda.replay_layers_segments, forward.conv_shift_segments):
one verify window holding several streams' rows back to back, each stream with its own state slot, parity, conv slot
and keep. Every segment's outputs, saved rows and states must equal (torch.equal) a solo run of its rows through the
single-stream path (``chain`` as a decode window runs it, below WIDE_ROWS on the one-kernel chain, and its three-kernel
twin, ``replay_layers``, ``_shift_conv``), and nothing outside its slots may change. Real per-rank shapes: 64 KDA heads
over 2 ranks (32 a rank), 34 KDA layers, 4-tap conv. Builds no engine; needs < 2 GiB of GPU memory."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

H = 32                     # KDA heads a rank (linear_attn_config num_heads 64 / world 2)
LAYERS = 34                # KDA layers
C = 3 * H * 128            # conv channels q | k | v
B_OFF = C + 256            # [q | k | v | f_a | g_a | b]
WIDTH = B_OFF + H
LOWER, EPS = -5.0, 1e-5
SLOTS, CONV_SLOTS = 6, 5


def _kda():
    from tensorfold.families.glm5_next.cuda import kda

    return kda


def _layer_weights(g: torch.Generator) -> dict:
    return dict(conv_w=(torch.randn((C, 4), generator=g) * 0.5).to(torch.bfloat16).cuda(),
                a_log=(torch.rand(H, generator=g) * 2 - 1).cuda(),
                dt_bias=(torch.randn(H * 128, generator=g) * 0.5).cuda(),
                norm_w=(torch.rand(128, generator=g) + 0.5).to(torch.bfloat16).cuda())


def _window(rows: int, g: torch.Generator) -> dict:
    return dict(p=(torch.randn((rows, WIDTH), generator=g) * 0.5).to(torch.bfloat16).cuda(),
                a=(torch.randn((rows, H * 128), generator=g) * 3.0).to(torch.bfloat16).cuda(),
                g=torch.randn((rows, H * 128), generator=g).to(torch.bfloat16).cuda())


def _layout(rng: np.random.Generator, n: int | None = None, sizes=None) -> list[tuple]:
    """1-4 segments of 1..16 rows on distinct state and conv slots, random parities and keeps (keep == rows and
    keep == 1 frequent)."""

    n = n or int(rng.integers(1, 5))
    sizes = sizes or [int(rng.choice([1, 16])) if rng.random() < 0.3 else int(rng.integers(1, 17)) for _ in range(n)]
    slots = rng.permutation(SLOTS)[:n]
    cslots = rng.permutation(CONV_SLOTS)[:n]
    out = []
    for rows, slot, cslot in zip(sizes, slots, cslots):
        u = rng.random()
        keep = rows if u < 0.3 else 1 if u < 0.5 else int(rng.integers(1, rows + 1))
        out.append((rows, int(slot), int(rng.integers(0, 2)), int(cslot), keep))
    return out


def _starts(layout) -> list[int]:
    return list(np.cumsum([0] + [s[0] for s in layout])[:-1])


def _eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


# -- chain ------------------------------------------------------------------------------------------------------
def _check_chain(layout, seed: int, table_rows: int | None = None) -> None:
    kda = _kda()
    g = torch.Generator().manual_seed(seed)
    wt = _layer_weights(g)
    R = sum(s[0] for s in layout)
    x = _window(R, g)
    conv = (torch.randn((CONV_SLOTS, 3, C), generator=g) * 0.5).to(torch.bfloat16).cuda()
    rec0 = (torch.randn((SLOTS, 2, H, 128, 128), generator=g) * 0.1).cuda()
    rec = rec0.clone()
    conv0 = conv.clone()
    out_tab = None if table_rows is None else torch.empty((table_rows, kda.SEG_COLS), dtype=torch.int32, device="cuda")
    seg = kda.segment_table(layout, "cuda", out=out_tab)
    sc = kda.KDAScratch(R, H, "cuda")
    out = kda.chain_segments(seg, x["p"], B_OFF, x["a"], x["g"], conv, wt["conv_w"], rec, wt["a_log"],
                             wt["dt_bias"], wt["norm_w"], EPS, LOWER, R, sc).clone()
    assert _eq(conv, conv0), "conv states are read only"
    touched = set()
    for (rows, slot, parity, cslot, _), r0 in zip(layout, _starts(layout)):
        touched.add((slot, 1 - parity))
        solo = kda.KDAScratch(16, H, "cuda")
        so = torch.empty((H, 128, 128), dtype=torch.float32, device="cuda")
        ref = kda.chain(x["p"][r0:r0 + rows], B_OFF, x["a"][r0:r0 + rows], x["g"][r0:r0 + rows], conv0[cslot],
                        wt["conv_w"], rec0[slot, parity], wt["a_log"], wt["dt_bias"], wt["norm_w"], EPS, LOWER, rows,
                        solo, so, wide=True).clone()
        tag = (layout, seed, slot, parity, cslot)
        assert _eq(out[r0:r0 + rows], ref), (tag, "out")
        assert _eq(rec[slot, 1 - parity], so), (tag, "state")
        # the single-stream decode window's own path (one kernel below WIDE_ROWS): the same bits
        narrow = kda.KDAScratch(16, H, "cuda")
        sn = torch.empty((H, 128, 128), dtype=torch.float32, device="cuda")
        ref_n = kda.chain(x["p"][r0:r0 + rows], B_OFF, x["a"][r0:r0 + rows], x["g"][r0:r0 + rows], conv0[cslot],
                          wt["conv_w"], rec0[slot, parity], wt["a_log"], wt["dt_bias"], wt["norm_w"], EPS, LOWER,
                          rows, narrow, sn, wide=False)
        assert _eq(ref_n, ref) and _eq(sn, so), (tag, "the one-kernel chain")
        for name in ("k", "v", "g", "b"):
            assert _eq(getattr(sc, name)[r0:r0 + rows], getattr(solo, name)[:rows]), (tag, name)
    for s in range(SLOTS):
        for par in (0, 1):
            if (s, par) not in touched:
                assert _eq(rec[s, par], rec0[s, par]), ("an untouched state changed", s, par)


def test_chain_segments_equal_solo_runs():
    rng = np.random.default_rng(7)
    layouts = [[(1, 0, 0, 0, 1)], [(16, 3, 1, 2, 16)], [(1, 1, 1, 4, 1)] * 1 + [(16, 0, 0, 1, 3)],
               [(16, 5, 0, 0, 1), (16, 4, 1, 1, 16), (16, 3, 0, 2, 8), (16, 2, 1, 3, 1)],
               [(1, 0, 1, 4, 1), (1, 1, 0, 3, 1), (1, 2, 1, 2, 1), (1, 3, 0, 1, 1)]]
    layouts += [_layout(rng) for _ in range(40)]
    for i, layout in enumerate(layouts):
        _check_chain(layout, 100 + i)


def test_chain_segments_skip_empty_table_rows():
    """A table sized for 4 streams serving 2 (the rest 0-row segments): same bits, nothing else touched."""

    rng = np.random.default_rng(11)
    for i in range(6):
        _check_chain(_layout(rng, n=int(rng.integers(1, 4))), 300 + i, table_rows=4)


# -- replay -----------------------------------------------------------------------------------------------------
def _saved_rows(L: int, R: int, g: torch.Generator):
    """Realistic saved rows: unit k, bf16 v, decays exp(-5 sigmoid(x)), beta in (0, 1)."""

    k = torch.randn((L, R, H, 128), generator=g)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn((L, R, H, 128), generator=g).to(torch.bfloat16)
    gd = torch.exp(LOWER * torch.sigmoid(torch.randn((L, R, H, 128), generator=g) * 2))
    b = torch.sigmoid(torch.randn((L, R, H), generator=g)).to(torch.bfloat16).float()
    return k.cuda(), v.cuda(), gd.cuda(), b.cuda()


def _check_replay(layout, seed: int, L: int = LAYERS) -> None:
    kda = _kda()
    g = torch.Generator().manual_seed(seed)
    R = sum(s[0] for s in layout)
    sset = kda.KDAScratchSet(L, R, H, "cuda")
    k, v, gd, b = _saved_rows(L, R, g)
    sset.k.copy_(k), sset.v.copy_(v), sset.g.copy_(gd), sset.b.copy_(b)
    gc = torch.Generator(device="cuda").manual_seed(seed)
    rec0 = torch.randn((SLOTS, 2, L, H, 128, 128), generator=gc, device="cuda") * 0.1
    rec = rec0.clone()
    kda.replay_layers_segments(kda.segment_table(layout, "cuda"), rec, sset)
    touched = set()
    for (rows, slot, parity, _, keep), r0 in zip(layout, _starts(layout)):
        if keep == rows:
            continue                                       # skipped: the chain's state stays
        touched.add((slot, 1 - parity))
        solo = kda.KDAScratchSet(L, 16, H, "cuda")         # a solo stream's scratch: its window from row 0
        for name in ("k", "v", "g", "b"):
            getattr(solo, name)[:, :rows].copy_(getattr(sset, name)[:, r0:r0 + rows])
        so = torch.empty((L, H, 128, 128), dtype=torch.float32, device="cuda")
        kda.replay_layers(rec0[slot, parity], solo, keep, so)
        assert _eq(rec[slot, 1 - parity], so), (layout, seed, slot, parity, keep)
    for s in range(SLOTS):
        for par in (0, 1):
            if (s, par) not in touched:
                assert _eq(rec[s, par], rec0[s, par]), ("an untouched state changed", s, par)


def test_replay_layers_segments_equal_solo_runs():
    rng = np.random.default_rng(13)
    layouts = [[(16, 0, 0, 0, 16)], [(16, 1, 1, 0, 1)], [(1, 2, 0, 0, 1), (16, 3, 1, 1, 15)],
               [(16, 5, 0, 0, 1), (16, 4, 1, 1, 16), (16, 3, 0, 2, 8), (16, 2, 1, 3, 1)],
               [(3, 0, 1, 0, 3), (5, 1, 0, 1, 2), (2, 2, 1, 2, 1), (7, 3, 0, 3, 7)]]
    layouts += [_layout(rng) for _ in range(16)]
    for i, layout in enumerate(layouts):
        _check_replay(layout, 500 + i)


# -- conv shift ---------------------------------------------------------------------------------------------------
def test_conv_shift_segments_equal_solo_runs():
    from tensorfold.families.glm5_next.cuda.forward import _shift_conv, conv_shift_segments

    kda = _kda()
    rng = np.random.default_rng(17)
    L = LAYERS
    for i in range(20):
        layout = _layout(rng)
        R = sum(s[0] for s in layout)
        g = torch.Generator().manual_seed(700 + i)
        conv0 = torch.randn((CONV_SLOTS, L, 3, C), generator=g).to(torch.bfloat16).cuda()
        proj = torch.randn((L, 16 * 4, WIDTH), generator=g).to(torch.bfloat16).cuda()   # st.proj-like: spare rows
        conv = conv0.clone()
        table = kda.segment_table(layout, "cuda", out=torch.empty((4, kda.SEG_COLS), dtype=torch.int32,
                                                                  device="cuda"))
        conv_shift_segments(conv, proj[:, :R], table)
        used = set()
        for (rows, _, _, cslot, keep), r0 in zip(layout, _starts(layout)):
            used.add(cslot)
            ref = conv0[cslot].clone()
            _shift_conv(ref, proj[:, r0:r0 + rows], keep)
            assert _eq(conv[cslot], ref), (layout, cslot, keep)
        for s in set(range(CONV_SLOTS)) - used:
            assert _eq(conv[s], conv0[s])


# -- a whole round: every layer's chain, then the commit ------------------------------------------------------------
def test_a_multi_stream_round_equals_solo_rounds():
    """4 layers: chain_segments per layer, replay_layers_segments, conv_shift_segments, against each stream's own
    chain(wide) per layer + commit (replay_layers when keep < rows, _shift_conv) on its own state and scratch."""

    from tensorfold.families.glm5_next.cuda.forward import _shift_conv, conv_shift_segments

    kda = _kda()
    L = 4
    rng = np.random.default_rng(19)
    for trial in range(6):
        layout = _layout(rng, n=4) if trial else [(16, 0, 0, 0, 1), (1, 1, 1, 1, 1), (9, 2, 1, 2, 9), (5, 3, 0, 3, 2)]
        R = sum(s[0] for s in layout)
        g = torch.Generator().manual_seed(900 + trial)
        wts = [_layer_weights(g) for _ in range(L)]
        wins = [_window(R, g) for _ in range(L)]
        rec0 = (torch.randn((SLOTS, 2, L, H, 128, 128), generator=g) * 0.1).cuda()
        conv0 = (torch.randn((CONV_SLOTS, L, 3, C), generator=g) * 0.5).to(torch.bfloat16).cuda()
        proj = torch.stack([w["p"] for w in wins])                   # [L, R, W]: st.proj's role
        rec, conv = rec0.clone(), conv0.clone()
        seg = kda.segment_table(layout, "cuda")
        sset = kda.KDAScratchSet(L, R, H, "cuda")
        outs = []
        for li in range(L):
            x, wt = wins[li], wts[li]
            outs.append(kda.chain_segments(seg, proj[li], B_OFF, x["a"], x["g"], conv[:, li], wt["conv_w"],
                                           rec[:, :, li], wt["a_log"], wt["dt_bias"], wt["norm_w"], EPS, LOWER, R,
                                           sset.views[li]).clone())
        kda.replay_layers_segments(seg, rec, sset)
        conv_shift_segments(conv, proj, seg)
        for (rows, slot, parity, cslot, keep), r0 in zip(layout, _starts(layout)):
            srec = rec0[slot].clone()                                # [2, L, H, 128, 128]: the solo stream's rec
            sconv = conv0[cslot].clone()
            sproj = proj[:, r0:r0 + rows]
            solo = kda.KDAScratchSet(L, 16, H, "cuda")
            for li in range(L):
                x, wt = wins[li], wts[li]
                ref = kda.chain(sproj[li], B_OFF, x["a"][r0:r0 + rows], x["g"][r0:r0 + rows], sconv[li],
                                wt["conv_w"], srec[parity, li], wt["a_log"], wt["dt_bias"], wt["norm_w"], EPS, LOWER,
                                rows, solo.views[li], srec[1 - parity, li], wide=True)
                assert _eq(outs[li][r0:r0 + rows], ref), (layout, slot, li, "out")
            if keep < rows:
                kda.replay_layers(srec[parity], solo, keep, srec[1 - parity])
            _shift_conv(sconv, sproj, keep)
            assert _eq(rec[slot], srec), (layout, slot, keep, "states")
            assert _eq(conv[cslot], sconv), (layout, cslot, keep, "conv")


# -- timing -------------------------------------------------------------------------------------------------------
def _time(fn, iters: int = 200) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters


def test_timing_one_launch_vs_solo_launches():
    """One 4-segment launch against 4 solo launches (one layer's chain; all 34 layers' replay; the conv shift)."""

    from tensorfold.families.glm5_next.cuda.forward import _shift_conv, conv_shift_segments

    kda = _kda()
    g = torch.Generator().manual_seed(1)
    wt = _layer_weights(g)
    rec = torch.randn((4, 2, LAYERS, H, 128, 128), device="cuda") * 0.1
    conv = (torch.randn((4, LAYERS, 3, C), generator=g) * 0.5).to(torch.bfloat16).cuda()
    lines = []
    for n in (1, 2, 4, 8, 16):
        layout = [(n, s, s % 2, s, max(1, n // 2)) for s in range(4)]
        R = 4 * n
        x = _window(R, g)
        seg = kda.segment_table(layout, "cuda")
        sc = kda.KDAScratch(R, H, "cuda")
        solos = [kda.KDAScratch(16, H, "cuda") for _ in range(4)]
        kda.reserve(64, H, "cuda")
        starts = _starts(layout)

        def seg_chain():
            kda.chain_segments(seg, x["p"], B_OFF, x["a"], x["g"], conv[:, 0], wt["conv_w"], rec[:, :, 0],
                               wt["a_log"], wt["dt_bias"], wt["norm_w"], EPS, LOWER, R, sc)

        def solo_chain():
            for s, r0 in enumerate(starts):
                kda.chain(x["p"][r0:r0 + n], B_OFF, x["a"][r0:r0 + n], x["g"][r0:r0 + n], conv[s, 0], wt["conv_w"],
                          rec[s, s % 2, 0], wt["a_log"], wt["dt_bias"], wt["norm_w"], EPS, LOWER, n, solos[s],
                          rec[s, 1 - s % 2, 0], wide=True)

        sset = kda.KDAScratchSet(LAYERS, R, H, "cuda")
        ssets = [kda.KDAScratchSet(LAYERS, 16, H, "cuda") for _ in range(4)]
        for t in [sset] + ssets:
            t.k.normal_(), t.v.normal_(), t.g.uniform_(0.5, 1.0), t.b.uniform_(0, 1)
        keep = max(1, n // 2)
        tab = seg if keep < n else kda.segment_table([(n, s, s % 2, s, n - 1 if n > 1 else 0) for s in range(4)],
                                                     "cuda")
        rkeep = keep if keep < n else max(n - 1, 0)

        def seg_replay():
            kda.replay_layers_segments(tab, rec, sset)

        def solo_replay():
            for s in range(4):
                kda.replay_layers(rec[s, s % 2], ssets[s], rkeep, rec[s, 1 - s % 2])

        proj = torch.randn((LAYERS, R, WIDTH), generator=g).to(torch.bfloat16).cuda()

        def seg_shift():
            conv_shift_segments(conv, proj, seg)

        def solo_shift():
            for s, r0 in enumerate(starts):
                _shift_conv(conv[s], proj[:, r0:r0 + n], keep)

        c1, c4 = _time(seg_chain), _time(solo_chain)
        r1, r4 = _time(seg_replay), _time(solo_replay)
        s1, s4 = _time(seg_shift), _time(solo_shift)
        lines.append(f"4 x {n:2d} rows | chain (1 layer) {c1:7.1f} vs {c4:7.1f} us | replay (34 layers, keep "
                     f"{rkeep}) {r1:7.1f} vs {r4:7.1f} us | conv shift {s1:6.1f} vs {s4:6.1f} us")
    print("\nsegmented (one launch) vs 4 solo launches:\n" + "\n".join(lines))
