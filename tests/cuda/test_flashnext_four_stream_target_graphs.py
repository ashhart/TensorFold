"""Four-stream target graphs keep eager target bits across real decoder geometry changes."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("Flash Next requires sm_12x", allow_module_level=True)

from test_flashnext_forward import _Rand, _bf16_table, _cfg, _model, _ple  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attn_multi, gdn_multi, multi_target_graphs  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, stage  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402


def _same_state(left, right) -> None:
    """Compare committed state and all attention cache bytes after a target step."""

    assert left.pos == right.pos and left.cur == right.cur
    for name in ("rec", "conv", "ple_tail"):
        assert torch.equal(getattr(left, name), getattr(right, name)), name
    if left.ple_history is not None:
        assert np.array_equal(left.ple_history, right.ple_history)
    for a, b in zip(left.kc, right.kc):
        for name in ("k", "v", "ks", "vs"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name
    for name in ("ikc", "pooled"):
        for a, b in zip(getattr(left, name), getattr(right, name)):
            assert torch.equal(a, b), name


def _step(w, graph, eager, states, pending, width, keep, seed):
    """Run the actual graph helper against original unpadded eager Tables."""

    graph_states, eager_states = states
    graph_pending, eager_pending = pending
    staged = []
    for decoder, current, held, candidate in ((graph, graph_states, graph_pending, True),
                                                (eager, eager_states, eager_pending, False)):
        tokens = [[(seed + i * 31 + j * 7) % 4095 + 1 for j in range(width)] for i in range(4)]
        segs = stage(w, decoder.buf, list(zip(current, tokens)))
        streams = [Stream([1], 100, st=st, sid=i) for i, st in enumerate(current)]
        admitted = candidate and multi_target_graphs.eligible(
            w, decoder.buf, segs, held, streams, depth=6, filling=False)
        tables = gdn_multi.Tables(w, decoder.gdn, segs, held, pending_width=7 if admitted else None)
        decoder.buf.gdn_tables = tables
        decoder.buf.attn_step = attn_multi.Step(w, segs, mtp=False)
        if candidate and admitted:
            assert tables.held.shape == (4, 7)
        if not candidate:
            assert tables.held.shape[1] == max(max(map(len, held)), 1)
        staged.append((decoder, current, segs, tables, admitted))
    gd, gs, gsegs, gt, admitted = staged[0]
    ed, es, esegs, et, _ = staged[1]
    want = compute(w, esegs, ed.buf).clone()
    want_streams = ed.buf.streams[:4 * width].clone()
    before_replays = graph.four_stream_target_graphs.replays
    got = graph.four_stream_target_graphs.run(
        w, gd.buf, gsegs, gt, gd.buf.attn_step, admitted=admitted)
    if got is None:
        got = compute(w, gsegs, gd.buf)
    got = got.clone()
    assert torch.equal(got, want)
    assert torch.equal(gd.buf.streams[:4 * width], want_streams)
    assert graph.four_stream_target_graphs.replays == before_replays + int(admitted)
    for a, b in zip(gs, es):
        _same_state(a, b)
    gd.buf.gdn_tables = gd.buf.attn_step = None
    ed.buf.gdn_tables = ed.buf.attn_step = None
    next_graph = gdn_multi.keep(gt, keep)
    next_eager = gdn_multi.keep(et, keep)
    assert next_graph == next_eager
    for (st, a0, a1), (ref, e0, e1), count in zip(gsegs, esegs, keep):
        assert (a0, a1) == (e0, e1)
        commit(w, st, gd.buf, width, count, at=a0, states=False)
        commit(w, ref, ed.buf, width, count, at=e0, states=False)
        _same_state(st, ref)
    return next_graph, next_eager, admitted


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_four_keys_replay_pending_slots_and_context_fallback(monkeypatch, kv_dtype):
    """Capture 2/7×both parities, then replay changed rows and fall back beyond 512."""

    w = _model()
    monkeypatch.setenv("TF_FOUR_STREAM_TARGET_GRAPH", "1")
    graph = MultiDecoder(w, slots=4, capacity=1024, depth=6, kv_dtype=kv_dtype, graphs=True)
    monkeypatch.setenv("TF_FOUR_STREAM_TARGET_GRAPH", "0")
    eager = MultiDecoder(w, slots=4, capacity=1024, depth=6, kv_dtype=kv_dtype, graphs=True)
    assert graph.four_stream_target_graphs is not None and eager.four_stream_target_graphs is None
    states = (list(graph.free), list(eager.free))
    pending = ([[] for _ in range(4)], [[] for _ in range(4)])
    sequence = [(7, [1, 2, 3, 4]), (2, [2] * 4), (7, [1, 3, 5, 7]),
                (7, [7] * 4), (2, [1] * 4), (2, [2] * 4)]
    for i, (width, keep) in enumerate(sequence):
        left, right, admitted = _step(w, graph, eager, states, pending, width, keep, 100 + i * 17)
        pending = (left, right)
        assert admitted is (i > 0)
    helper = graph.four_stream_target_graphs
    assert helper.captures == helper.entries == 4 and helper.replays == 5
    # The next replay reads fresh table pointers after slot reorder, reset and growth.
    order = [2, 0, 3, 1]
    states = ([states[0][i] for i in order], [states[1][i] for i in order])
    pending = ([pending[0][i] for i in order], [pending[1][i] for i in order])
    for side in states:
        side[0].resize(1024)
        side[1].reset(w)
    pending[0][1].clear()
    pending[1][1].clear()
    left, right, admitted = _step(w, graph, eager, states, pending, 2, [2] * 4, 299)
    assert admitted and helper.captures == 4
    pending = (left, right)
    # Synthetic near-boundary positions exercise the same host eligibility and
    # dense attention path. Both arms have identical zero-initialized older KV.
    for side in states:
        for st in side:
            st.resize(1024)
            st.set_pos(510)
    left, right, admitted = _step(w, graph, eager, states, pending, 2, [1] * 4, 317)
    assert admitted and all(st.pos == 511 for st in states[0])
    replay_before, eager_before = helper.replays, helper.eager_calls
    _, _, admitted = _step(w, graph, eager, states, (left, right), 2, [1] * 4, 337)
    assert not admitted and helper.replays == replay_before and helper.eager_calls == eager_before + 1


def test_ple_tail_changes_replay_exactly(monkeypatch):
    """A retained graph reads each current State's changed PLE tail."""

    c = _cfg(ple=True)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "shard_0.safetensors"
        table = _bf16_table(path, c.ngram(0).rows, c.ngram(0).dims)
        w = _model(ple=_ple(c, table, _Rand(19)))
        monkeypatch.setenv("TF_FOUR_STREAM_TARGET_GRAPH", "1")
        graph = MultiDecoder(w, slots=4, capacity=1024, depth=6, kv_dtype="int8", graphs=True)
        monkeypatch.setenv("TF_FOUR_STREAM_TARGET_GRAPH", "0")
        eager = MultiDecoder(w, slots=4, capacity=1024, depth=6, kv_dtype="int8", graphs=True)
        states = (list(graph.free), list(eager.free))
        pending = ([[] for _ in range(4)], [[] for _ in range(4)])
        left, right, _ = _step(w, graph, eager, states, pending, 7, [2] * 4, 71)
        left, right, admitted = _step(w, graph, eager, states, (left, right), 2, [2] * 4, 89)
        assert admitted and graph.four_stream_target_graphs.captures == 1
        left, right, admitted = _step(w, graph, eager, states, (left, right), 2, [2] * 4, 98)
        assert admitted and graph.four_stream_target_graphs.captures == 2
        for side in states:
            for i, st in enumerate(side):
                st.ple_tail.fill_(0.02 * (i + 1))
        _, _, admitted = _step(w, graph, eager, states, (left, right), 2, [1] * 4, 107)
        assert admitted and graph.four_stream_target_graphs.captures == 2


def test_real_round_hook_and_disabled_filling_guards(monkeypatch):
    """The actual MultiDecoder.round path captures, verifies and commits equal tokens."""

    w = _model()
    monkeypatch.setenv("TF_FOUR_STREAM_TARGET_GRAPH", "1")
    graph = MultiDecoder(w, slots=4, capacity=1024, depth=6, kv_dtype="int8", graphs=True)
    monkeypatch.setenv("TF_FOUR_STREAM_TARGET_GRAPH", "0")
    eager = MultiDecoder(w, slots=4, capacity=1024, depth=6, kv_dtype="int8", graphs=True)
    monkeypatch.setenv("TF_FOUR_STREAM_TARGET_GRAPH", "1")
    off = MultiDecoder(w, slots=4, capacity=1024, depth=6, kv_dtype="int8", graphs=False)
    assert eager.four_stream_target_graphs is None and off.four_stream_target_graphs is None
    prompts = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13, 21, 31], [8, 8, 9, 2000]]
    streams = []
    for decoder in (graph, eager):
        decoder._draft_all = lambda _: None  # fix geometry; round still samples, accepts and commits normally
        current = [Stream(list(p), 64, draft=True, stop_eos=False) for p in prompts]
        for stream in current:
            decoder.admit(stream)
        streams.append(current)
    for _ in range(8):
        if all(s.out for s in streams[0]) and all(s.out for s in streams[1]):
            break
        for decoder in (graph, eager):
            decoder.finish(decoder.round())
    else:
        pytest.fail("four prompts did not reach a common target round")
    assert all(not s.done for side in streams for s in side)
    assert graph.held and eager.held
    for side in streams:
        for stream in side:
            stream.drafts = [101, 102, 103, 104, 105, 106]
    for decoder in (graph, eager):
        decoder.finish(decoder.round())
    assert graph.four_stream_target_graphs.captures >= 1
    for left, right in zip(*streams):
        assert left.out == right.out and left.committed == right.committed
        _same_state(left.st, right.st)
    live = streams[0]
    segs = stage(w, graph.buf, [(s.st, [11, 12]) for s in live])
    pending = [[0] for _ in live]
    assert multi_target_graphs.eligible(w, graph.buf, segs, pending, live, depth=6, filling=False)
    assert not multi_target_graphs.eligible(w, graph.buf, segs, pending, live, depth=6, filling=True)
