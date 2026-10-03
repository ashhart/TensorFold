"""Flash Next's kernels give each row the bits a one-row call gives it: the multi-stream kernels against the
single-stream ones, the hyper-connection at any row count (GPU)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():
    pytest.skip("needs a Metal GPU", allow_module_level=True)

from tensorfold.kernels.qwen.flash_next.v1 import attention, base, gdn, hc  # noqa: E402

NK, NV, DK, DV, TAPS = 16, 48, 128, 128, 4
C = 2 * NK * DK + NV * DV
PW = C + NV * DV + 2 * NV


def _gdn_inputs(rng, rows):
    projected = mx.array(rng.normal(size=(rows, PW)).astype(np.float32)).astype(mx.bfloat16)
    conv = mx.array(rng.normal(size=(TAPS - 1, C)).astype(np.float32)).astype(mx.bfloat16)
    ssm = mx.array((0.1 * rng.normal(size=(NV, DV, DK))).astype(np.float32))
    return projected, conv, ssm


@pytest.mark.parametrize("rows", [[1, 1], [2, 1, 3], [1, 4, 2, 1], [3], [1, 1, 1, 1], [2] * 8, [1, 2, 1, 1, 3, 1, 2, 1]])
def test_gdn_step_multi_matches_each_stream(rows):
    rng = np.random.default_rng(len(rows) * 10 + sum(rows))
    conv_w = mx.array(rng.normal(size=(C, TAPS)).astype(np.float32)).astype(mx.bfloat16)
    a_log = mx.array(rng.normal(size=(NV,)).astype(np.float32)).astype(mx.bfloat16)
    dt = mx.array(rng.normal(size=(NV,)).astype(np.float32)).astype(mx.bfloat16)
    norm = mx.array(rng.normal(size=(DV,)).astype(np.float32)).astype(mx.bfloat16)
    eps = mx.array([1e-6], dtype=mx.float32)
    parts = [_gdn_inputs(rng, n) for n in rows]
    kw = dict(nk=NK, nv=NV, dk=DK, dv=DV)
    single = [gdn.gdn_step(p, c, s, conv_w, a_log, dt, norm, eps, **kw) for p, c, s in parts]
    multi = gdn.gdn_step_multi(mx.concatenate([p for p, _, _ in parts]), [c for _, c, _ in parts],
                             [s for _, _, s in parts], rows, conv_w, a_log, dt, norm, eps, **kw)
    for k in range(3):
        # the per-row state buffers are allocated in multiples of 8 rows: compare the rows written
        joined = mx.concatenate([out[k][:n] for out, n in zip(single, rows)])
        assert bool(mx.array_equal(joined, multi[k][:sum(rows)]).item()), k


def test_attention_and_selection_multi_match_each_stream():
    rng = np.random.default_rng(7)
    heads, kvh, dims = 24, 2, 256
    caps, lengths, rows = [256, 3072, 512], [200, 2600, 300], [2, 1, 3]
    keys = [mx.array(rng.normal(size=(1, kvh, cap, dims)).astype(np.float32)).astype(mx.bfloat16) for cap in caps]
    values = [mx.array(rng.normal(size=(1, kvh, cap, dims)).astype(np.float32)).astype(mx.bfloat16) for cap in caps]
    qs = [mx.array(rng.normal(size=(n, heads, dims)).astype(np.float32)).astype(mx.bfloat16) for n in rows]
    # stream 1 is past the dense limit: its row reads 512 selected blocks' keys and its tail
    top, ratio, idim, iheads = 512, 4, 128, 4
    iq = [mx.array(rng.normal(size=(n, iheads, idim)).astype(np.float32)).astype(mx.bfloat16) for n in rows]
    pooled = [mx.array(rng.normal(size=(max(1, (length + n) // ratio), idim)).astype(np.float32)).astype(mx.bfloat16)
              for length, n in zip(lengths, rows)]
    counts_all, sparse_all, ids_all, srow, single = [], [], [], [], []
    ends_all, complete_all = [], []
    for b, (n, length) in enumerate(zip(rows, lengths)):
        ends = [length + r + 1 for r in range(n)]
        complete = [e // ratio for e in ends]
        sparse = [c > top for c in complete]
        counts = [ratio * top + e - ratio * c if sp else e for e, c, sp in zip(ends, complete, sparse)]
        ids = attention.index_select(iq[b], pooled[b], complete, ends, top=top) if complete[-1] > top else None
        single.append(attention.attention_rows(qs[b], keys[b], values[b], counts, ids, sparse, 0.0625))
        counts_all += counts
        sparse_all += sparse
        srow += [b] * n
        ends_all += ends
        complete_all += complete
        ids_all.append(ids)
    ids_multi = attention.index_select_multi(mx.concatenate(iq), pooled, srow, complete_all, ends_all, top=top)
    at = 0
    for b, n in enumerate(rows):
        for r in range(n):
            if ids_all[b] is not None and sparse_all[at + r]:
                used = counts_all[at + r]              # a row reads the first counts[r] ids; the rest are unset
                assert bool(mx.array_equal(ids_multi[at + r, :used], ids_all[b][r, :used]).item())
        at += n
    out = attention.attention_rows_multi(mx.concatenate(qs), keys, values, srow, counts_all, ids_multi, sparse_all, 0.0625)
    assert bool(mx.array_equal(out, mx.concatenate(single)).item())
    # the output gate inside the merge: bf16(bf16(merged) * sigmoid(gate)), each row as its stream's own call
    width = heads * 2 * dims + 1024
    proj = [mx.array(rng.normal(size=(n, width)).astype(np.float32)).astype(mx.bfloat16) for n in rows]
    gated = attention.attention_rows_multi(mx.concatenate(qs), keys, values, srow, counts_all, ids_multi, sparse_all,
                                           0.0625, gate=mx.concatenate(proj))
    at = 0
    for b, n in enumerate(rows):
        counts = counts_all[at:at + n]
        one = attention.attention_rows(qs[b], keys[b], values[b], counts, ids_all[b], sparse_all[at:at + n], 0.0625,
                                       gate=proj[b])
        assert bool(mx.array_equal(gated[at:at + n], one).item()), b
        g = proj[b][:, :heads * 2 * dims].reshape(n, heads, 2, dims)[:, :, 1].astype(mx.float32)
        want = (single[b].astype(mx.float32) * mx.sigmoid(g)).reshape(n, heads * dims)
        assert float(mx.abs(one.astype(mx.float32) - want).max().item()) <= 0.02 * float(mx.abs(want).max().item())
        at += n



def _qweights(rng, rows, cols):
    w = mx.array(rng.integers(0, 2**32, size=(rows, cols // 8), dtype=np.uint32))
    sc = mx.array((0.02 * rng.random((rows, cols // 32))).astype(np.float32)).astype(mx.bfloat16)
    bi = mx.array((0.01 * rng.normal(size=(rows, cols // 32))).astype(np.float32)).astype(mx.bfloat16)
    return base.QWeights(w, sc, bi)


@pytest.mark.parametrize("inject", [True, False])
def test_hyper_connection_rows_equal_one_row_calls(inject):
    S, D, LOW = 4, 2560, 320
    rng = np.random.default_rng(7 + inject)
    down, up = _qweights(rng, LOW + (S if inject else 0), S * D), _qweights(rng, S * D, LOW)
    scale = mx.array((1.0 + 0.1 * rng.normal(size=(S * D,))).astype(np.float32))
    eps = mx.array([1e-6], dtype=mx.float32)
    h = mx.array((0.3 * rng.normal(size=(19, S * D))).astype(np.float32)).astype(mx.bfloat16)
    hn, ssp = hc.hc_norm(h, streams=S)
    ones = [hc.hc_project(hn[r:r + 1], ssp[r:r + 1], down, up, scale, eps=eps, streams=S, low=LOW) for r in range(19)]
    for rows in (2, 3, 5, 9, 17, 19):
        mixed, gates = hc.hc_project(hn[:rows], ssp[:rows], down, up, scale, eps=eps, streams=S, low=LOW)
        for r in range(rows):
            assert mx.array_equal(mixed[r], ones[r][0][0]).item(), (rows, r)
            if inject:
                assert mx.array_equal(gates[r], ones[r][1][0]).item(), (rows, r)


@pytest.mark.parametrize("inject", [True, False])
def test_hyper_connection_scalar_rows_equal_the_mma_path(inject, monkeypatch):
    """One- and two-row calls take the scalar kernels: the same bits as the 8-row-tile MMA kernels' rows."""

    S, D, LOW = 4, 2560, 320
    rng = np.random.default_rng(17 + inject)
    down, up = _qweights(rng, LOW + (S if inject else 0), S * D), _qweights(rng, S * D, LOW)
    scale = mx.array((1.0 + 0.1 * rng.normal(size=(S * D,))).astype(np.float32))
    eps = mx.array([1e-6], dtype=mx.float32)
    h = mx.array((0.3 * rng.normal(size=(2, S * D))).astype(np.float32)).astype(mx.bfloat16)
    hn, ssp = hc.hc_norm(h, streams=S)
    for rows in (1, 2):
        scalar = hc.hc_project(hn[:rows], ssp[:rows], down, up, scale, eps=eps, streams=S, low=LOW)
        monkeypatch.setattr(hc, "SCALAR_ROWS", 0)
        mma = hc.hc_project(hn[:rows], ssp[:rows], down, up, scale, eps=eps, streams=S, low=LOW)
        monkeypatch.undo()
        assert mx.array_equal(scalar[0], mma[0]).item(), rows
        if inject:
            assert mx.array_equal(scalar[1][:rows], mma[1][:rows]).item(), rows


@pytest.mark.parametrize("nib, half", [(0, False), (2, False), (2, True)])
def test_per_row_projections_do_not_depend_on_the_row_count(nib, half, monkeypatch):
    """rows.qmv_rows and rows.hc_project: every row of a call equals that row's one-row call, bit for bit, with the
    multi-row calls' dots on the convert (nib 0), or from two rows on nib or on half nibbles."""

    from tensorfold.kernels.qwen.flash_next.v1 import rows

    monkeypatch.setattr(base, "nib_rows", lambda: nib)
    monkeypatch.setattr(base, "half_nibs", lambda: half)

    S, D, LOW = 4, 2560, 320
    rng = np.random.default_rng(51)
    lin = _qweights(rng, 1024, 2560)
    x = mx.array((0.5 * rng.normal(size=(40, 2560))).astype(np.float32)).astype(mx.bfloat16)
    full = rows.qmv_rows(x, lin)
    mx.eval(full)
    for r in (0, 7, 31, 32, 39):
        mx.clear_cache()
        assert mx.array_equal(rows.qmv_rows(x[r:r + 1], lin), full[r:r + 1]).item(), r
    down, up = _qweights(rng, LOW + S, S * D), _qweights(rng, S * D, LOW)
    scale = mx.array((1.0 + 0.1 * rng.normal(size=(S * D,))).astype(np.float32))
    eps = mx.array([1e-6], dtype=mx.float32)
    h = mx.array((0.3 * rng.normal(size=(9, S * D))).astype(np.float32)).astype(mx.bfloat16)
    hn, ssp = hc.hc_norm(h, streams=S)
    mixed, gates = rows.hc_project(hn, ssp, down, up, scale, eps=eps, streams=S, low=LOW)
    mx.eval(mixed, gates)
    for r in range(9):
        mx.clear_cache()
        one = rows.hc_project(hn[r:r + 1], ssp[r:r + 1], down, up, scale, eps=eps, streams=S, low=LOW)
        assert mx.array_equal(one[0], mixed[r:r + 1]).item(), r
        assert mx.array_equal(one[1][:1], gates[r:r + 1]).item(), r


@pytest.mark.parametrize("inject", [True, False])
def test_hyper_connection_tiles_keep_the_per_row_bits(inject):
    """Before M5 rows.hc_project takes 8-row tiles from HC_MMA_FROM rows: every row keeps the per-row kernels' bits."""

    from tensorfold.kernels.qwen.flash_next.v1 import row_tiles, rows

    S, D, LOW = 4, 2560, 320
    rng = np.random.default_rng(71 + inject)
    down, up = _qweights(rng, LOW + (S if inject else 0), S * D), _qweights(rng, S * D, LOW)
    scale = mx.array((1.0 + 0.1 * rng.normal(size=(S * D,))).astype(np.float32))
    eps = mx.array([1e-6], dtype=mx.float32)
    h = mx.array((0.3 * rng.normal(size=(40, S * D))).astype(np.float32)).astype(mx.bfloat16)
    hn, ssp = hc.hc_norm(h, streams=S)
    rows._hc_mma_ok.clear()                           # another test's check (patched kernels) never answers this one
    exact = rows._hc_mma_exact(down, up, scale, eps, S, LOW)
    if base.nib_rows():                               # before M5 the dispatch must take the tiles
        assert exact
    for n in (1, 2, 3, 5, 8, 9, 16, 17, 32, 40):
        tiles = row_tiles.hc_tiles(hn[:n], ssp[:n], down, up, scale, eps=eps, streams=S, low=LOW)
        per_row = rows._hc_rows(hn[:n], ssp[:n], down, up, scale, eps=eps, streams=S, low=LOW)
        if exact:
            assert mx.array_equal(tiles[0], per_row[0]).item(), n
            if inject:
                assert mx.array_equal(tiles[1][:n], per_row[1][:n]).item(), n
        called = rows.hc_project(hn[:n], ssp[:n], down, up, scale, eps=eps, streams=S, low=LOW)
        assert mx.array_equal(called[0], per_row[0]).item(), n


def test_windows_are_checked_with_the_tiles_and_timed_without(monkeypatch):
    """check_windows checks exactness on the tiles and times the per-row kernels, whose costs the allocator prices."""

    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.runtime import FlashNext
    from tensorfold.kernels.qwen.flash_next.v1 import row_tiles, rows

    seen = []

    def hidden(tokens, cache):
        seen.append(rows.hc_tiles_on)
        return mx.array(np.asarray(tokens, dtype=np.float32)[..., None])      # [1, R, 1]: row r is its token

    fake = SimpleNamespace(model=SimpleNamespace(make_cache=list, hidden=hidden), head=lambda h: h, fused_rows=4,
                           _check_streams=lambda base, window: True)
    assert FlashNext.check_windows(fake)[0] == 4
    assert seen == [True] * 8 + [False] * 12 and rows.hc_tiles_on    # prompt, 4 serial steps, widths 2-4; then timing
    monkeypatch.setattr(row_tiles, "hc_tiles", lambda *a, **k: pytest.fail("tiles while they are off"))
    monkeypatch.setattr(rows, "hc_tiles_on", False)
    S, LOW = 4, 320
    rng = np.random.default_rng(5)
    down, up = _qweights(rng, LOW + S, S * 2560), _qweights(rng, S * 2560, LOW)
    h = mx.array((0.3 * rng.normal(size=(16, S * 2560))).astype(np.float32)).astype(mx.bfloat16)
    hn, ssp = hc.hc_norm(h, streams=S)
    scale, eps = mx.ones((S * 2560,), dtype=mx.float32), mx.array([1e-6], dtype=mx.float32)
    mx.eval(rows.hc_project(hn, ssp, down, up, scale, eps=eps, streams=S, low=LOW))


@pytest.mark.parametrize("has_state", [True, False])
def test_gdn_pipelined_rows_equal_the_row_by_row_kernel(has_state, monkeypatch):
    """The three-phase GDN step gives the row-by-row kernel's outputs and states bit for bit."""

    rng = np.random.default_rng(41 + has_state)
    conv_w = mx.array(rng.normal(size=(C, TAPS)).astype(np.float32)).astype(mx.bfloat16)
    a_log = mx.array(rng.normal(size=(NV,)).astype(np.float32)).astype(mx.bfloat16)
    dt = mx.array(rng.normal(size=(NV,)).astype(np.float32)).astype(mx.bfloat16)
    norm = mx.array(rng.normal(size=(DV,)).astype(np.float32)).astype(mx.bfloat16)
    eps = mx.array([1e-6], dtype=mx.float32)
    kw = dict(nk=NK, nv=NV, dk=DK, dv=DV)
    for rows in (1, 2, 3, 4):
        p, c, s = _gdn_inputs(rng, rows)
        s = s if has_state else None
        monkeypatch.setattr(gdn, "PIPE_ROWS", 0)
        ref = gdn.gdn_step(p, c, s, conv_w, a_log, dt, norm, eps, **kw)
        mx.eval(ref)
        mx.clear_cache()
        monkeypatch.setattr(gdn, "PIPE_ROWS", 4)
        got = gdn.gdn_step(p, c, s, conv_w, a_log, dt, norm, eps, **kw)
        for k in range(3):
            assert bool(mx.array_equal(got[k][:rows], ref[k][:rows]).item()), (rows, k)


def test_ple_kernels_are_the_math_and_row_invariant():
    """embed.ple_gate and ple_conv against float64 math, and each row of a 6-row call against its one-row call."""

    from tensorfold.kernels.qwen.flash_next.v1 import embed

    rng = np.random.default_rng(29)
    S, D, R, TAPS, DIL = 4, 2560, 6, 4, 3
    W = S * D

    def bf(a):
        return mx.array(np.asarray(a, dtype=np.float32)).astype(mx.bfloat16)

    kv, h = bf(rng.normal(size=(R, W + D))), bf(0.5 * rng.normal(size=(R, W)))
    ks, qs, cs = (mx.array((1.0 + 0.1 * rng.normal(size=(W,))).astype(np.float32)) for _ in range(3))
    eps = mx.array([1e-6], dtype=mx.float32)
    gated, normed = embed.ple_gate(kv, h, ks, qs, cs, eps, streams=S)
    for r in range(R):
        g1, n1 = embed.ple_gate(kv[r:r + 1], h[r:r + 1], ks, qs, cs, eps, streams=S)
        assert mx.array_equal(g1[0], gated[r]).item() and mx.array_equal(n1[0], normed[r]).item(), r
    f = lambda a: np.asarray(a.astype(mx.float32)).astype(np.float64)      # noqa: E731
    k, q, v = f(kv)[:, :W].reshape(R, S, D), f(h).reshape(R, S, D), f(kv)[:, W:]
    norm = lambda x, w: x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6) * f(w).reshape(S, D)  # noqa: E731
    g = (norm(k, ks) * norm(q, qs)).sum(-1, keepdims=True) / np.sqrt(D)
    g = np.sign(g) * np.sqrt(np.maximum(np.abs(g), 1e-6))
    want = (1.0 / (1.0 + np.exp(-g)) * v[:, None, :])
    assert np.abs(f(gated).reshape(R, S, D) - want).max() <= 0.02 * np.abs(want).max()
    tail = bf(rng.normal(size=(1, (TAPS - 1) * DIL, W)))
    conv_in = mx.concatenate([tail, normed[None]], axis=1)
    w = mx.array((0.3 * rng.normal(size=(W, TAPS))).astype(np.float32))
    out = embed.ple_conv(conv_in[0], w, gated, h, streams=S, dilation=DIL)
    x = f(conv_in[0])
    y = sum(f(w)[:, j] * x[j * DIL: j * DIL + R] for j in range(TAPS))
    ref = f(h) + f(gated) + y / (1.0 + np.exp(-y))
    assert np.abs(f(out) - ref).max() <= 0.02 * np.abs(ref).max()
    for r in range(R):
        one = embed.ple_conv(conv_in[0, r:r + 1 + (TAPS - 1) * DIL], w, gated[r:r + 1], h[r:r + 1], streams=S,
                             dilation=DIL)
        assert mx.array_equal(one[0], out[r]).item(), r


def test_split_router_rows_equal_one_row_calls_and_the_math():
    """experts.router(split=True): every row of 1-16-row calls as its one-row call, and float64 math."""

    from tensorfold.kernels.qwen.flash_next.v1 import experts

    rng = np.random.default_rng(37)
    rows_w = mx.array((0.05 * rng.normal(size=(513, 2560))).astype(np.float32)).astype(mx.bfloat16)
    x = mx.array(rng.normal(size=(16, 2560)).astype(np.float32)).astype(mx.bfloat16)
    ones = [experts.router(x[r:r + 1], rows_w, split=True) for r in range(16)]
    for count in (2, 3, 4, 9, 16):
        got = experts.router(x[:count], rows_w, split=True)
        for r in range(count):
            assert mx.array_equal(got[r], ones[r][0]).item(), (count, r)
    f64 = lambda a: np.asarray(a.astype(mx.float32)).astype(np.float64)      # noqa: E731
    want = f64(x) @ f64(rows_w).T
    got = np.asarray(mx.concatenate(ones))
    assert np.abs(got - want).max() <= 1e-4 * np.abs(want).max()
