"""Host admission, capture restoration and lifetime checks for bounded target graphs."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import weakref

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'src/tensorfold/families/qwen4_exp/cuda/multi_target_graphs.py'
SPEC = importlib.util.spec_from_file_location('four_target_host', PATH)
g = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = g
SPEC.loader.exec_module(g)


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values).copy()
        self.shape, self.dtype = self.values.shape, self.values.dtype

    def data_ptr(self):
        return self.values.ctypes.data

    def clone(self):
        return Tensor(self.values)

    def copy_(self, other):
        self.values[:] = other.values

    def __getitem__(self, key):
        other = Tensor(self.values[key])
        other.values = self.values[key]
        return other


class State(NS):
    pass


def fixture(width=2, counts=(0, 1, 2, 7), parity=0, end=512):
    states = [State(pos=end-width, capacity=1024, kv_dtype='int8', image_positions=None,
                    lin_index={0: 0}, cur=[0], rec=Tensor(np.zeros((2, 1, 3))), ratio=4,
                    ple_tail=Tensor(np.full((2, 3), i)),
                    kc=[NS(k=Tensor(np.zeros((1024, 3))), v=Tensor(np.zeros((1024, 3))),
                           ks=Tensor(np.zeros((1024, 1))), vs=Tensor(np.zeros((1024, 1))))],
                    ikc=[Tensor(np.zeros((1024, 3)))], pooled=[Tensor(np.zeros((256, 3)))])
              for i in range(4)]
    segs = [(st, i * width, (i+1) * width) for i, st in enumerate(states)]
    streams = [NS(st=st, vision=None, constraint=None) for st in states]
    pending = [list(range(count)) for count in counts]
    scratch = NS(parity=parity, q=Tensor(np.zeros(4)))
    tables = NS(scratch=scratch, segs=segs, cur=parity, held=Tensor(np.zeros((4, 7))),
                ints=Tensor(np.arange(20)), ptrs=Tensor(np.arange(8)), folds=True,
                state_arrays=[st.rec[0, 0] for st in states])
    step = NS(segs=segs, ints=Tensor(np.arange(12)), ptrs=Tensor(np.arange(24)))
    b = NS(prefill=False, attn=NS(budget=2048), moe=NS(plan=NS(tile=64)),
           gdn_tables=tables, attn_step=step, ids=Tensor(np.zeros(width*4)))
    w = NS(comm=None, x3=None, head=object())
    return w, b, segs, pending, streams, tables, step


@pytest.mark.parametrize('width,end,ok', [(2, 512, True), (7, 512, True), (2, 513, False),
                                         (7, 513, False), (3, 512, False), (2, 1, True)])
def test_admission_boundaries(width, end, ok):
    args = fixture(width=width, end=end)
    assert g.eligible(*args[:5], depth=6, filling=False) is ok


@pytest.mark.parametrize('kind', ['counts8', 'empty', 'mixed', 'duplicate', 'vision', 'image',
                                  'grammar', 'tp', 'prefill', 'depth', 'filling', 'budget', 'dtype', 'missing'])
def test_noneligible_shapes_fall_back(kind):
    w, b, segs, pending, streams, _, _ = fixture()
    depth, filling = 6, False
    if kind == 'counts8':
        pending[0] = list(range(8))
    elif kind == 'empty':
        pending = [[] for _ in range(4)]
    elif kind == 'mixed':
        segs[1] = (segs[1][0], 2, 5)
    elif kind == 'duplicate':
        streams[1].st = streams[0].st
    elif kind == 'vision':
        streams[0].vision = object()
    elif kind == 'image':
        streams[0].st.image_positions = object()
    elif kind == 'grammar':
        streams[0].constraint = object()
    elif kind == 'tp':
        w.comm = object()
    elif kind == 'prefill':
        b.prefill = True
    elif kind == 'depth':
        depth = 5
    elif kind == 'filling':
        filling = True
    elif kind == 'budget':
        b.attn.budget = 511
    elif kind == 'dtype':
        streams[0].st.kv_dtype = 'bf16'
    else:
        streams[0] = None
    assert not g.eligible(w, b, segs, pending, streams, depth=depth, filling=filling)


@pytest.mark.parametrize('value,result', [('0', False), ('1', True), ('2', None), ('true', None)])
def test_flag(monkeypatch, value, result):
    monkeypatch.setenv(g.FLAG, value)
    if result is None:
        with pytest.raises(ValueError):
            g.requested()
    else:
        assert g.requested() is result


def fake_cuda(monkeypatch, w, b, compute):
    module = ModuleType('fake_target.forward')
    module.compute = compute
    torch = ModuleType('torch')
    torch.empty_like = lambda t: Tensor(np.zeros(t.shape))
    torch.cuda = NS(synchronize=lambda: None, graph=lambda *a, **kw: nullcontext(),
                    CUDAGraph=lambda: NS(replay=lambda: compute(w, [], b)))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'fake_target.forward', module)
    monkeypatch.setattr(g, '__package__', 'fake_target')


@pytest.mark.parametrize('fails', [False, True])
def test_non_idempotent_fold_restored_before_replay_and_on_failure(monkeypatch, fails):
    w, b, segs, _, _, tables, step = fixture()
    calls = []
    def compute(w, proxies, b):
        calls.append(1)
        for rec in b.gdn_tables.state_arrays:
            rec.values[:] += 1
        if fails and len(calls) == 2:
            raise RuntimeError('capture failure')
        return 'logits'
    fake_cuda(monkeypatch, w, b, compute)
    helper = g.FourStreamTargetGraphs()
    if fails:
        with pytest.raises(RuntimeError, match='capture failure'):
            helper.run(w, b, segs, tables, step, admitted=True)
        assert helper.entries == helper.captures == helper.replays == 0
        assert b.gdn_tables is tables and b.attn_step is step
    else:
        assert helper.run(w, b, segs, tables, step, admitted=True) == 'logits'
        assert helper.entries == helper.captures == helper.replays == 1
        entry = helper._entries[(2, 0)]
        assert entry.tables.segs == entry.step.segs == []
        assert tables.segs is segs and step.segs is segs
    for st, _, _ in segs:
        np.testing.assert_array_equal(st.rec.values[0, 0], np.full(3, 0 if fails else 1))
        assert not st.rec.values[1].any()


def test_entry_does_not_keep_old_states_alive(monkeypatch):
    w, b, segs, _, _, tables, step = fixture()
    refs = [weakref.ref(st) for st, _, _ in segs]
    fake_cuda(monkeypatch, w, b, lambda *a: 'logits')
    helper = g.FourStreamTargetGraphs()
    helper.run(w, b, segs, tables, step, admitted=True)
    del segs, tables, step, _
    import gc
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert helper.entries == 1


def test_failed_restore_still_detaches_capture_tables_and_rebinds_buffers(monkeypatch):
    w, b, segs, _, _, tables, step = fixture()
    captured = []
    def compute(*args):
        captured.append((b.gdn_tables, b.attn_step))
        return 'logits'
    fake_cuda(monkeypatch, w, b, compute)
    original_restore = g._restore
    calls = []
    def restore(saved):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('restore failed')
        original_restore(saved)
    monkeypatch.setattr(g, '_restore', restore)
    helper = g.FourStreamTargetGraphs()
    with pytest.raises(RuntimeError, match='restore failed'):
        helper.run(w, b, segs, tables, step, admitted=True)
    assert b.gdn_tables is tables and b.attn_step is step
    assert tables.segs is segs and step.segs is segs
    assert all(t.segs == s.segs == [] for t, s in captured)
    assert helper.entries == helper.captures == helper.replays == 0


def test_replay_copies_fresh_tables_and_tails(monkeypatch):
    w, b, segs, _, _, tables, step = fixture()
    fake_cuda(monkeypatch, w, b, lambda *a: 'logits')
    helper = g.FourStreamTargetGraphs()
    helper.run(w, b, segs, tables, step, admitted=True)
    entry = helper._entries[(2, 0)]
    _, _, fresh, _, _, new_tables, new_step = fixture()
    new_tables.scratch = tables.scratch
    new_tables.ints.values += 100
    new_tables.ptrs.values += 1000
    new_step.ints.values += 10
    new_step.ptrs.values += 10000
    fresh.reverse()
    assert helper.run(w, b, fresh, new_tables, new_step, admitted=True) == 'logits'
    for original, changed in [(entry.tables.ints, new_tables.ints), (entry.tables.ptrs, new_tables.ptrs),
                              (entry.step.ints, new_step.ints), (entry.step.ptrs, new_step.ptrs)]:
        np.testing.assert_array_equal(original.values, changed.values)
    for tail, (st, _, _) in zip(helper.tails, fresh):
        np.testing.assert_array_equal(tail.values, st.ple_tail.values)
    assert helper.captures == 1 and helper.replays == 2


def test_four_keys_and_static_scratch_invalidation(monkeypatch):
    w, b, segs, _, _, tables, step = fixture()
    fake_cuda(monkeypatch, w, b, lambda *a: 'logits')
    helper = g.FourStreamTargetGraphs()
    for width in (2, 7):
        windows = [(st, i * width, (i+1) * width) for i, (st, _, _) in enumerate(segs)]
        for parity in (0, 1):
            tables.cur = parity
            helper.run(w, b, windows, tables, step, admitted=True)
    assert helper.entries == helper.captures == helper.replays == 4
    tables.scratch.q = tables.scratch.q.clone()
    assert helper.run(w, b, segs, tables, step, admitted=True) is None
    assert helper.entries == 0 and helper.eager_calls == 1


def test_noneligible_does_not_mutate_graph_family():
    w, b, segs, _, _, tables, step = fixture()
    helper = g.FourStreamTargetGraphs()
    before = b.gdn_tables, b.attn_step
    assert helper.run(w, b, segs, tables, step, admitted=False) is None
    assert (b.gdn_tables, b.attn_step) == before
    assert helper.entries == 0 and helper.tails == [] and helper.eager_calls == 1


def test_actual_tables_padding_preserves_plan_rows_counts_and_pointer_values(monkeypatch):
    """Exercise the real constructor with its real host planner and fake device transfers."""
    torch = ModuleType('torch')
    torch.int32, torch.int64 = np.int32, np.int64
    monkeypatch.setitem(sys.modules, 'torch', torch)
    shared_path = ROOT / 'src/tensorfold/cuda/kernels/gdn.py'
    spec = importlib.util.spec_from_file_location('target_shared_host', shared_path)
    shared = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, shared)
    spec.loader.exec_module(shared)
    shared.to_device = lambda values, dtype, dev: Tensor(np.asarray(values, dtype=dtype))
    kernels = ModuleType('tensorfold.cuda.kernels')
    kernels.gdn = shared
    monkeypatch.setitem(sys.modules, 'tensorfold.cuda.kernels', kernels)
    package = ModuleType('target_table_host')
    package.__path__ = []
    monkeypatch.setitem(sys.modules, 'target_table_host', package)
    monkeypatch.setitem(sys.modules, 'target_table_host.gdn_io', ModuleType('target_table_host.gdn_io'))
    constants = ModuleType('target_table_host.gdn')
    constants.DK = constants.DV = 128
    monkeypatch.setitem(sys.modules, 'target_table_host.gdn', constants)
    path = ROOT / 'src/tensorfold/families/qwen4_exp/cuda/gdn_multi.py'
    spec = importlib.util.spec_from_file_location('target_table_host.gdn_multi', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    # Match torch.Tensor.view while retaining the original allocation root.
    def view(self, *shape):
        other = Tensor(self.values.reshape(*shape))
        other.values = self.values.reshape(*shape)
        return other
    monkeypatch.setattr(Tensor, 'view', view, raising=False)
    spec.loader.exec_module(module)
    for width in (2, 7):
        w, _, segs, _, _, _, _ = fixture(width=width)
        w.layers, w.device = [], 'fake'
        for st, _, _ in segs:
            st.conv = Tensor(np.zeros((1, 3)))
        scratch = NS(lin=1, parity=1)
        for pending in ([[], [3], [4, 5], []], [[], [], [], list(range(7))]):
            ordinary = module.Tables(w, scratch, segs, pending)
            padded = module.Tables(w, scratch, segs, pending, pending_width=7)
            assert padded.held.shape == (4, 7)
            assert padded.plan.slots == ordinary.plan.slots
            assert padded.plan.max_rows == ordinary.plan.max_rows == width
            for name in ('sid', 'win', 'held_counts', 'ptrs'):
                np.testing.assert_array_equal(getattr(padded, name).values, getattr(ordinary, name).values)
            for name in ('entries', 'starts'):
                np.testing.assert_array_equal(getattr(padded.plan, name).values, getattr(ordinary.plan, name).values)
            for i, rows in enumerate(pending):
                np.testing.assert_array_equal(padded.held.values[i, :len(rows)], rows)
                assert not padded.held.values[i, len(rows):].any()
            # Mutating the held view must update the retained integer root.
            padded.held.values[0, 0] = 99
            assert 99 in padded.ints.values
        with pytest.raises(ValueError, match='hold every pending row'):
            module.Tables(w, scratch, segs, pending, pending_width=1)
