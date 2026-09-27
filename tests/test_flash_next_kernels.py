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
