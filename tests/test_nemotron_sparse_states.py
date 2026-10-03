"""A lone wide window stores its Mamba states every 8th row and at its last; the state of any row between comes back by
re-scanning from the nearest stored state with the same kernel, so every kept state is the every-row path's bit for bit
(tiny random shapes, any Mac)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.nemotron.lightning.v1 import kernels as K  # noqa: E402

H, DH, NG, DS, KC = 4, 8, 2, 32, 4
XD = H * DH
CD = XD + 2 * NG * DS
PROJ = XD + CD + H
SHAPE = {"heads": H, "head_dim": DH, "groups": NG, "state_dim": DS}


def _params(seed: int = 0):
    rng = np.random.default_rng(seed)
    conv_w = mx.array(rng.normal(size=(KC, CD)).astype(np.float32) * 0.5)
    conv_b = mx.array(rng.normal(size=(CD,)).astype(np.float32) * 0.1)
    a_log = mx.array(rng.normal(size=(H,)).astype(np.float32) * 0.3)
    d_skip = mx.array(rng.normal(size=(H,)).astype(np.float32))
    dt_bias = mx.array(rng.normal(size=(H,)).astype(np.float32) * 0.1)
    limits = mx.array([0.0, 1e4], dtype=mx.float32)
    return (conv_w, conv_b, a_log, d_skip, dt_bias), limits


def _stream(seed: int, rows: int):
    rng = np.random.default_rng(100 + seed)
    proj = mx.array(rng.normal(size=(rows, PROJ)).astype(np.float32)).astype(mx.bfloat16)
    conv = mx.array(rng.normal(size=(1, KC - 1, CD)).astype(np.float32)).astype(mx.bfloat16)
    ssm = mx.array(rng.normal(size=(1, H, DH, DS)).astype(np.float32) * 0.1)
    return proj, conv, ssm


def _equal(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


def test_sparse_store_keeps_every_eighth_row_and_the_last():
    assert K.sparse_store(16) is None
    assert K.sparse_store(17) == (-1,) * 7 + (0,) + (-1,) * 7 + (1, 2)
    store = K.sparse_store(64)
    assert [r for r, s in enumerate(store) if s >= 0] == [7, 15, 23, 31, 39, 47, 55, 63]
    assert [s for s in store if s >= 0] == list(range(8))


@pytest.mark.parametrize("rows", [17, 24, 40, 64])
def test_every_kept_state_equals_the_every_row_path(rows):
    params, limits = _params()
    proj, conv, ssm = _stream(rows, rows)
    y_all, conv_all, ssm_all = K.mamba_step(proj, conv, ssm, *params, limits, **SHAPE)
    store = K.sparse_store(rows)
    y, conv_rows, ssm_rows = K.mamba_step(proj, conv, ssm, *params, limits, store=store, **SHAPE)
    assert _equal(y, y_all) and ssm_rows.shape[0] == max(store) + 1
    for row in range(rows):
        conv_at, ssm_at, slot = K.kept_state(proj, conv_rows, ssm_rows, store, conv, ssm, row, params, limits, **SHAPE)
        assert _equal(conv_at[slot:slot + 1], conv_all[row:row + 1]), f"conv state after row {row}"
        assert _equal(ssm_at[slot:slot + 1], ssm_all[row:row + 1]), f"SSM state after row {row}"
        if store[row] >= 0:
            assert conv_at is conv_rows and ssm_at is ssm_rows and slot == store[row]


def test_a_narrow_window_still_stores_every_row():
    params, limits = _params()
    proj, conv, ssm = _stream(3, 16)
    _, conv_rows, ssm_rows = K.mamba_step(proj, conv, ssm, *params, limits, store=K.sparse_store(16), **SHAPE)
    assert conv_rows.shape[0] == 16 and ssm_rows.shape[0] == 16
    conv_at, ssm_at, slot = K.kept_state(proj, conv_rows, ssm_rows, None, conv, ssm, 5, params, limits, **SHAPE)
    assert conv_at is conv_rows and ssm_at is ssm_rows and slot == 5
