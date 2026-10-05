"""Bounded actual-weight qualification of dynamic total-row graph reuse."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import time
import weakref
from types import SimpleNamespace
from typing import Any

LIMIT_BYTES = 4 * 1024**3


def _bits(left: Any, right: Any) -> bool:
    """Compare raw bytes, including signed zeros and NaN payloads."""

    import torch

    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def _equal(left: Any, right: Any) -> None:
    """Compare whole State tensors/caches and committed host state exactly."""

    import numpy as np
    import torch

    assert left.pos == right.pos and left.cur == right.cur
    for name in ("rec", "conv", "ple_tail"):
        assert _bits(getattr(left, name), getattr(right, name)), name
    assert (left.ple_history is None) == (right.ple_history is None)
    if left.ple_history is not None:
        assert np.array_equal(left.ple_history, right.ple_history)
    for a, b in zip(left.kc, right.kc):
        for name in ("k", "v", "ks", "vs"):
            assert _bits(getattr(a, name), getattr(b, name)), name
    for name in ("ikc", "pooled"):
        for a, b in zip(getattr(left, name), getattr(right, name)):
            assert _bits(a, b), name


def _step(w: Any, pairs: list, helper: Any, pending: list, widths: tuple, keep: tuple,
          seed: int) -> tuple[list, dict]:
    """Eager reference uses original per-segment PLE; candidate uses real helper."""

    import torch
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda import attn_multi, gdn_multi, mixed_row_target_graphs
    from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, stage

    staged = []
    for i, pair in enumerate(pairs):
        tokens = [[(seed + s * 31 + j * 7) % (w.cfg.vocab - 1) + 1 for j in range(n)]
                  for s, n in enumerate(widths)]
        segs = stage(w, pair.buf, list(zip(pair.states, tokens)))
        streams = [Stream([1], 64, st=st, sid=s) for s, st in enumerate(pair.states)]
        admitted = i == 0 and mixed_row_target_graphs.eligible(
            w, pair.buf, segs, pending[i], streams, depth=6, filling=False)
        tables = gdn_multi.Tables(w, pair.gdn, segs, pending[i], pending_width=7 if admitted else None)
        pair.buf.gdn_tables = tables
        pair.buf.attn_step = attn_multi.Step(w, segs, mtp=False)
        pair.buf.mixed_ple = None
        staged.append((pair, segs, tables, admitted))
    ref, rsegs, rt, _ = staged[1]
    want = compute(w, rsegs, ref.buf).clone()
    expected = {name: getattr(ref.buf, name)[:sum(widths)].clone()
                for name in ("streams", "ple_nrow")}
    graph, segs, tables, admitted = staged[0]
    before = helper.captures, helper.replays
    got = helper.run(w, graph.buf, segs, tables, graph.buf.attn_step, admitted=admitted)
    replay = got is not None
    if got is None:
        got = compute(w, segs, graph.buf)
    assert _bits(got, want), "logits"
    for name, value in expected.items():
        assert _bits(getattr(graph.buf, name)[:sum(widths)], value), name
    for name in ("q", "k", "v", "g", "beta"):
        assert _bits(getattr(graph.gdn, name), getattr(ref.gdn, name)), name
    for a, b in zip(graph.states, ref.states):
        _equal(a, b)
    next_pending = []
    for pair, current, tables_, _ in staged:
        pair.buf.gdn_tables = pair.buf.attn_step = pair.buf.mixed_ple = None
        next_pending.append(gdn_multi.keep(tables_, keep))
        for (st, a0, a1), n in zip(current, keep):
            commit(w, st, pair.buf, a1 - a0, n, at=a0, states=False)
    assert next_pending[0] == next_pending[1]
    for a, b in zip(graph.states, ref.states):
        _equal(a, b)
    return next_pending, {"widths": list(widths), "pending_counts": list(map(len, pending[0])),
                          "parity": tables.cur, "admitted": admitted, "replayed": replay,
                          "captures_delta": helper.captures - before[0],
                          "replays_delta": helper.replays - before[1], "exact": True}


def qualify(decoder: Any) -> dict:
    """Use loaded actual weights; release all qualification allocations before HTTP."""

    import torch
    from tensorfold.families.qwen4_exp.cuda import gdn_multi
    from tensorfold.families.qwen4_exp.cuda.mixed_row_target_graphs import MixedRowTargetGraphs
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    w = decoder.w
    if decoder.streams or decoder.filling or w.comm is not None:
        raise ValueError("qualification requires idle single-rank decoder")
    if not any(layer.ple is not None for layer in w.layers):
        raise ValueError("qualification must exercise actual PLE")
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    pairs, helper, cases = [], MixedRowTargetGraphs(), []
    try:
        for _ in range(2):
            b = Buffers(w, 28, 1024, moe_prefill=True)
            sc = gdn_multi.Scratch(w, 28)
            for name in ("q", "k", "v", "g", "beta"):
                getattr(sc, name).zero_()
            states = [State(w, 256, 7, "int8", limit=1024) for _ in range(4)]
            pairs.append(SimpleNamespace(buf=b, gdn=sc, states=states))
        allocation_bytes = torch.cuda.memory_allocated() - before
        if allocation_bytes > LIMIT_BYTES:
            raise ValueError("qualification allocations exceed4GiB")
        pending = [[[] for _ in range(4)] for _ in range(2)]
        sequence = [((7, 7, 7, 7), (1, 2, 3, 7)),
                    ((1, 3, 2, 7), (1, 3, 2, 7)),
                    ((7, 1, 4, 1), (7, 1, 4, 1)),
                    ((7, 7, 2, 2), (7, 7, 2, 1)),
                    ((3, 4, 4, 7), (3, 4, 4, 7)),
                    ((2, 7, 1, 3), (1, 7, 1, 2)),
                    ((7, 2, 7, 2), (7, 1, 6, 2))]
        for i, (widths, keep) in enumerate(sequence):
            pending, case = _step(w, pairs, helper, pending, widths, keep, 101 + i * 17)
            cases.append(case)
        assert helper.entries == helper.captures == 4
        # Entries retain only fixed tensors, not the replaced original States.
        refs = [weakref.ref(st) for pair in pairs for st in pair.states]
        states = st = None
        for pair in pairs:
            pair.states = [original.clone() for original in pair.states]
        gc.collect()
        assert all(ref() is None for ref in refs), "graph pinned old State"
        # Same graphs, new slot order, changed tails, reset (zero pending) and resized cache pointers.
        order = (2, 0, 3, 1)
        for side, pair in enumerate(pairs):
            pair.states = [pair.states[i] for i in order]
            pending[side] = [pending[side][i] for i in order]
            for i, st in enumerate(pair.states):
                st.resize(512)
                if i < 3:
                    st.reset(w)
                    pending[side][i] = []
                st.ple_tail.fill_(0.01 * (i + 1))
        pending, case = _step(w, pairs, helper, pending, (1, 7, 3, 2), (1, 7, 2, 2), 299)
        cases.append(case)
        assert case["replayed"] and helper.captures == 4
        # Actual attention ends512, then513 must eagerly fall back. Old KV is equal on both sides.
        for pair in pairs:
            for st, n in zip(pair.states, (7, 7, 2, 2)):
                st.resize(1024)
                st.set_pos(512 - n)
        pending, case = _step(w, pairs, helper, pending, (7, 7, 2, 2), (1, 1, 1, 1), 317)
        cases.append(case)
        assert case["replayed"] and helper.captures == 4
        pending, case = _step(w, pairs, helper, pending, (7, 7, 2, 2), (1, 1, 1, 1), 337)
        cases.append(case)
        assert not case["admitted"] and not case["replayed"]
        # Signature replacement invalidates, falls back once, then safely captures fresh roots.
        for side, pair in enumerate(pairs):
            for st in pair.states:
                st.reset(w)
            pending[side] = [[0], [], [], []]
        pairs[0].gdn.q = pairs[0].gdn.q.clone()
        pending, case = _step(w, pairs, helper, pending, (1, 3, 2, 7), (1, 3, 2, 7), 357)
        cases.append(case)
        assert case["admitted"] and not case["replayed"] and helper.entries == 0
        pending, case = _step(w, pairs, helper, pending, (7, 1, 4, 1), (7, 1, 4, 1), 377)
        cases.append(case)
        assert case["replayed"] and helper.captures == 5
        torch.cuda.synchronize()
        peak_extra = torch.cuda.max_memory_allocated() - before
        if peak_extra > LIMIT_BYTES or time.perf_counter() - start > 180:
            raise ValueError("qualification exceeded4GiB/180s bound")
        result = {"schema": "mixed_row_qualification_v1", "status": "passed", "exact": True,
                "cases": cases, "case_count": len(cases), "captures": helper.captures,
                "replays": helper.replays, "allocated_bytes": allocation_bytes,
                "elapsed_seconds": time.perf_counter() - start, "peak_additional_bytes": peak_extra,
                "qualification_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "old_states_released": True, "private_buffer_capacity": 1024,
                "scope": "actual loaded weights; eager PLE versus dynamic layout replay; no speed claim"}
    finally:
        pairs.clear()
        helper = None
        # Locals in the loop can retain their last allocations; explicitly release them.
        b = sc = states = pair = st = None
        gc.collect()
        torch.cuda.synchronize()
    result.update(memory_allocated_after=torch.cuda.memory_allocated(),
                  memory_reserved_after=torch.cuda.memory_reserved())
    return result
