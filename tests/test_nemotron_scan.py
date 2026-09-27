"""Nemotron's Mamba-2 scan kernel: a row's bits do not depend on the other rows of a call, the block size, or the other
streams' segments beside it (tiny random shapes)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.nemotron.lightning.v1 import kernels as K  # noqa: E402

H, DH, NG, DS, KC = 4, 8, 2, 32, 4
XD = H * DH
CD = XD + 2 * NG * DS
PROJ = XD + CD + H


def _params(seed: int = 0):
    rng = np.random.default_rng(seed)
    conv_w = mx.array(rng.normal(size=(KC, CD)).astype(np.float32) * 0.5)
    conv_b = mx.array(rng.normal(size=(CD,)).astype(np.float32) * 0.1)
    a_log = mx.array(rng.normal(size=(H,)).astype(np.float32) * 0.3)
    d_skip = mx.array(rng.normal(size=(H,)).astype(np.float32))
    dt_bias = mx.array(rng.normal(size=(H,)).astype(np.float32) * 0.1)
    limits = mx.array([0.0, 1e4], dtype=mx.float32)
    return conv_w, conv_b, a_log, d_skip, dt_bias, limits


def _stream(seed: int, rows: int):
    rng = np.random.default_rng(100 + seed)
    proj = mx.array(rng.normal(size=(rows, PROJ)).astype(np.float32)).astype(mx.bfloat16)
    conv = mx.array(rng.normal(size=(1, KC - 1, CD)).astype(np.float32)).astype(mx.bfloat16)
    ssm = mx.array(rng.normal(size=(1, H, DH, DS)).astype(np.float32) * 0.1)
    return proj, conv, ssm


def _one_by_one(proj, conv, ssm, params):
    ys, convs, ssms = [], [], []
    for r in range(int(proj.shape[0])):
        y, c, s = K.mamba_step(proj[r:r + 1], conv, ssm, *params, heads=H, head_dim=DH, groups=NG, state_dim=DS)
        conv, ssm = c[-1:], s[-1:]
        ys.append(y)
        convs.append(c)
        ssms.append(s)
    return mx.concatenate(ys), mx.concatenate(convs), mx.concatenate(ssms)


def _equal(a, b) -> bool:
    return bool(mx.array_equal(a, b).item())


@pytest.mark.parametrize("rows", [2, 5, 16, 17, 40])
def test_a_window_equals_one_row_steps(rows):
    params = _params()
    proj, conv, ssm = _stream(0, rows)
    window = K.mamba_step(proj, conv, ssm, *params, heads=H, head_dim=DH, groups=NG, state_dim=DS)
    serial = _one_by_one(proj, conv, ssm, params)
    for got, want in zip(window, serial):
        assert _equal(got, want)


@pytest.mark.parametrize("lengths", [(1, 1), (3, 5), (16, 2, 7), (4, 13, 1, 9)])
def test_segments_of_several_streams_equal_each_stream_alone(lengths):
    params = _params(1)
    streams = [_stream(i, n) for i, n in enumerate(lengths)]
    proj = mx.concatenate([p for p, _, _ in streams])
    convs = mx.concatenate([c for _, c, _ in streams])
    ssms = mx.concatenate([s for _, _, s in streams])
    y, conv_rows, ssm_rows = K.mamba_scan(proj, convs, ssms, lengths, *params, heads=H, head_dim=DH, groups=NG,
                                          state_dim=DS)
    at = 0
    for (p, c, s), n in zip(streams, lengths):
        alone = K.mamba_step(p, c, s, *params, heads=H, head_dim=DH, groups=NG, state_dim=DS)
        assert _equal(y[at:at + n], alone[0])
        assert _equal(conv_rows[at:at + n], alone[1])
        assert _equal(ssm_rows[at:at + n], alone[2])
        at += n


def test_states_read_through_slots_equal_compact_states():
    params = _params(2)
    lengths = (2, 3, 1)
    streams = [_stream(10 + i, n) for i, n in enumerate(lengths)]
    proj = mx.concatenate([p for p, _, _ in streams])
    convs = mx.concatenate([c for _, c, _ in streams])
    ssms = mx.concatenate([s for _, _, s in streams])
    compact = K.mamba_scan(proj, convs, ssms, lengths, *params, heads=H, head_dim=DH, groups=NG, state_dim=DS)
    # the same states among others, in another order: rows 5, 0, 3 of a 7-row pool
    slots = (5, 0, 3)
    pool = [_stream(40 + i, 1) for i in range(7)]
    for stream, slot in zip(streams, slots):
        pool[slot] = stream
    pool_conv = mx.concatenate([c for _, c, _ in pool])
    pool_ssm = mx.concatenate([s for _, _, s in pool])
    pooled = K.mamba_scan(proj, pool_conv, pool_ssm, lengths, *params, heads=H, head_dim=DH, groups=NG,
                          state_dim=DS, slots=slots)
    for got, want in zip(pooled, compact):
        assert _equal(got, want)
