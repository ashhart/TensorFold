"""Flash Next's DeltaNet input and output kernels around the shared recurrence give gdn.cu's chain kernel bits."""

import importlib.util

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import gdn, gdn_io  # noqa: E402

DEV = "cuda"


def _inputs(rows: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    p = (torch.randn((rows, gdn.PW), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    cs = (torch.randn((3, gdn.CONV), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    cw = (torch.randn((gdn.CONV, 4), generator=g, device=DEV) * 0.3).to(torch.bfloat16)
    state = torch.randn((gdn.NV, gdn.DV, gdn.DK), generator=g, device=DEV) * 0.05
    a_log = torch.randn((gdn.NV,), generator=g, device=DEV) * 0.5
    dt = torch.randn((gdn.NV,), generator=g, device=DEV) * 0.5
    nw = (1 + 0.1 * torch.randn((gdn.DV,), generator=g, device=DEV)).to(torch.bfloat16)
    return p, cs, cw, state, a_log, dt, nw


def _chain(p, cs, cw, state, a_log, dt, nw, rows):
    sc = gdn.GDNScratch(rows, DEV)
    out = torch.empty((rows, gdn.NV * gdn.DV), dtype=torch.bfloat16, device=DEV)
    xs = torch.empty((rows, gdn.NV * gdn.DV // 32), dtype=torch.float32, device=DEV)
    state_out = torch.empty_like(state)
    gdn.chain(p, cs, cw, state, a_log, dt, nw, 1e-6, rows, sc, state_out, out, xs)
    return sc, out, xs, state_out


def _chain_windows(rows: int, base: int = 0) -> list[list[int]]:
    """A chain's taps: row j reads [conv state | rows] positions j .. j + 3 (window rows start at ``base``)."""

    return [[j + t if j + t < 3 else base + j + t for t in range(4)] for j in range(rows)]


def _front(p, conv_states, windows, sid, cw, a_log, dt):
    ptrs = torch.tensor([c.data_ptr() for c in conv_states], dtype=torch.int64, device=DEV)
    win = torch.tensor(windows, dtype=torch.int32, device=DEV)
    ids = torch.tensor(sid, dtype=torch.int32, device=DEV)
    return gdn_io.front(p, ptrs, ids, win, cw, a_log, dt, gdn.NK)


def test_front_gives_the_chain_kernels_replay_inputs():
    rows = 7
    p, cs, cw, state, a_log, dt, nw = _inputs(rows, 3)
    sc, _, _, _ = _chain(p, cs, cw, state, a_log, dt, nw, rows)
    _, k, v, g, beta = _front(p, [cs], _chain_windows(rows), [0] * rows, cw, a_log, dt)
    assert torch.equal(k, sc.k[:rows]) and torch.equal(v, sc.v[:rows])
    assert torch.equal(g, sc.g[:rows]) and torch.equal(beta, sc.b[:rows])


def test_front_takes_each_rows_own_stream():
    p1, cs1, cw, state, a_log, dt, nw = _inputs(5, 4)
    p2, cs2, _, _, _, _, _ = _inputs(3, 5)
    alone = [_front(p1, [cs1], _chain_windows(5), [0] * 5, cw, a_log, dt),
             _front(p2, [cs2], _chain_windows(3), [0] * 3, cw, a_log, dt)]
    p = torch.cat([p1, p2])
    together = _front(p, [cs1, cs2], _chain_windows(5) + _chain_windows(3, base=5), [0] * 5 + [1] * 3, cw, a_log,
                      dt)
    for x, (a, b) in zip(together, zip(*alone)):
        assert torch.equal(x, torch.cat([a, b]))


@pytest.mark.skipif(importlib.util.find_spec("tensorfold.cuda.kernels") is None,
                    reason="needs the shared recurrence (tensorfold.cuda.kernels.gdn)")
def test_front_recurrence_back_give_the_chain_kernels_layer():
    from tensorfold.cuda.kernels import gdn as shared

    rows = 6
    p, cs, cw, state, a_log, dt, nw = _inputs(rows, 6)
    _, out, xs, state_out = _chain(p, cs, cw, state, a_log, dt, nw, rows)
    q, k, v, g, beta = _front(p, [cs], _chain_windows(rows), [0] * rows, cw, a_log, dt)
    plan = shared.plan([list(range(-1, rows - 1))], DEV)
    y = shared.tree(q, k, v, g, beta, plan, state=state)
    got, got_xs = torch.empty_like(out), torch.empty_like(xs)
    gdn_io.back(y, p, nw, 1e-6, got, got_xs)
    assert torch.equal(got, out) and torch.equal(got_xs, xs)


@pytest.mark.skipif(importlib.util.find_spec("tensorfold.cuda.kernels") is None,
                    reason="needs the shared recurrence (tensorfold.cuda.kernels.gdn)")
def test_shared_replay_gives_the_chain_kernels_states():
    from tensorfold.cuda.kernels import gdn as shared

    rows, keep = 6, 4
    p, cs, cw, state, a_log, dt, nw = _inputs(rows, 7)
    sc, _, _, state_all = _chain(p, cs, cw, state, a_log, dt, nw, rows)
    state_keep = torch.empty_like(state)
    gdn.replay(state, sc, keep, state_keep)
    q, k, v, g, beta = _front(p, [cs], _chain_windows(rows), [0] * rows, cw, a_log, dt)
    two = [state, state.clone()]                     # two streams from the same state: all rows, then a prefix
    table = shared.to_device(shared.replay_table([k], [v], [g], [beta], [[t] for t in two]), torch.int64, DEV)
    paths = torch.tensor([list(range(rows)), list(range(keep)) + [0] * (rows - keep)], dtype=torch.int32, device=DEV)
    counts = torch.tensor([rows, keep], dtype=torch.int32, device=DEV)
    got = shared.replay(table, 1, 2, paths, counts, k, v)
    assert torch.equal(got[0, 0], state_all) and torch.equal(got[1, 0], state_keep)
