"""Flash launch bounds and graph reuse follow live context, not allocated cache slots."""

import importlib
from types import SimpleNamespace

import pytest

from tests.test_cuda_geometry import Allocation, allocations


class Kernel:
    def __init__(self):
        self.calls = []
    def __getitem__(self, grid):
        def launch(*args, **kw):
            self.calls.append((grid, args, kw))
        return launch


@pytest.mark.parametrize("slots", [8192, 262151])
def test_actual_attention_callsite_bounds_short_request_launches(monkeypatch, allocations, slots):
    mod = importlib.import_module("tensorfold.families.qwen4_exp.cuda.forward")
    attention = mod.attn_mod
    kernels = {name: Kernel() for name in ("_pool", "_scores", "_select", "_chunks", "_merge")}
    for name, kernel in kernels.items():
        monkeypatch.setattr(attention, name, kernel)
    tensor = Allocation((8, 8, 64), "bf16", "cpu")
    cfg = SimpleNamespace(heads=8, kv_heads=2, head_dim=64, index_heads=4, index_dim=128, eps=1e-6)
    weights = SimpleNamespace(cfg=cfg, inv_freq=SimpleNamespace(numel=lambda: 16))
    layer = SimpleNamespace(attn=SimpleNamespace(proj=None, q_scale=None, k_scale=None, iq_scale=None,
                                                ik_scale=None, o=None))
    scratch = SimpleNamespace(qsa=True, nch=(slots+511)//512, nb=(slots+3)//4, budget=2048, ratio=4,
                              idw=2052, ids=None, nk=None, sparse=None, scores=None, po=None, pm=None, pl=None, out=tensor)
    buffers = SimpleNamespace(attn=scratch, mixed=tensor, xs_mixed=tensor, pa=tensor, q=tensor, iq=tensor,
                              gated=tensor, xs_gated=tensor)
    monkeypatch.setattr(mod, "_mm", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "_out_proj", lambda *a, **kw: None)
    monkeypatch.setattr(mod.glue, "attn_prep", lambda *a, **kw: None)
    monkeypatch.setattr(mod.glue, "attn_gate", lambda *a, **kw: None)
    ikc = Allocation((slots, 128), "bf16", "cpu")
    cache = Allocation((slots, 2, 64), "bf16", "cpu")
    mod.attn_block(layer, weights, cache, cache, ikc, None, None, buffers, 8, 8192)
    assert kernels["_chunks"].calls[0][0] == (8, 2, 5)
    assert kernels["_scores"].calls[0][0] == (8, 32)
    assert kernels["_select"].calls[0][2]["BLOCK"] == 2048
    # Row strides stay tied to allocated storage even when launch grids shrink.
    assert kernels["_chunks"].calls[0][2]["NCH"] == scratch.nch
    assert kernels["_scores"].calls[0][1][4] == scratch.nb


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("rows", [1, 8])
def test_actual_graph_calls_recapture_when_live_context_crosses_bucket(monkeypatch, allocations, mtp, rows):
    mod = importlib.import_module("tensorfold.families.qwen4_exp.cuda.graphs")
    captures, computes = [], []
    monkeypatch.setattr(mod, "stage", lambda w, st, b, tokens: len(tokens))
    monkeypatch.setattr(mod, "mtp_stage", lambda w, st, b, tokens, streams: len(tokens))
    def compute(*args, **kw):
        computes.append(kw.get("context"))
        return None
    monkeypatch.setattr(mod, "compute", compute)
    monkeypatch.setattr(mod, "mtp_compute", compute)
    obj = mod.Graphs.__new__(mod.Graphs)
    obj.max_rows, obj.main, obj.mtp, obj.mtp_out = 8, {}, {}, {}
    state = SimpleNamespace(pos=2000, mtp_len=2000, cur=[0], capacity=262151)
    obj.e = SimpleNamespace(w=None, st=state, buf=SimpleNamespace(logits=[0] * rows), mbuf=None, capacity=262151)
    def capture(fn):
        captures.append(fn)
        fn()
        return SimpleNamespace(replay=lambda: None)
    obj._capture = capture
    for context in (2000, 8192 - rows, 8193 - rows, 9000, 10000, 0):
        state.pos = state.mtp_len = context
        if mtp:
            obj.mtp_forward([1] * rows, None)
        else:
            obj.forward([1] * rows)
    assert len(captures) == 2
    assert set(computes) == {8192, 16384}
