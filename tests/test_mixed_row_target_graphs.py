"""Host checks of bounded admission, dynamic metadata and exception cleanup."""

from contextlib import nullcontext
import copy
import gc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import weakref

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REL = 'src/tensorfold/families/qwen4_exp/cuda/'
BASE = ROOT


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = ModuleType('mixed_host')
package.__path__ = []
sys.modules[package.__name__] = package
original = load('mixed_host.multi_target_graphs', BASE / REL / 'multi_target_graphs.py')
g = load('mixed_host.mixed_row_target_graphs', ROOT / REL / 'mixed_row_target_graphs.py')


class Tensor:
    def __init__(self, values, dtype=None, device='cuda'):
        self.values = np.array(values, dtype=dtype)
        self.shape, self.dtype, self.device = self.values.shape, self.values.dtype, device

    def data_ptr(self):
        return self.values.ctypes.data

    def clone(self):
        return Tensor(self.values)

    def copy_(self, other):
        self.values[:] = other.values

    def numel(self):
        return self.values.size

    def element_size(self):
        return self.values.itemsize

    def __getitem__(self, index):
        other = Tensor(self.values[index])
        other.values = self.values[index]
        return other


class State(NS):
    pass


def fixture(widths=(1, 3, 2, 7), counts=(0, 1, 2, 7), parity=0, end=512):
    states = [State(pos=end-n, capacity=1024, kv_dtype='int8', image_positions=None,
                    lin_index={0: 0}, cur=[0], rec=Tensor(np.zeros((2, 1, 3))), ratio=4,
                    ple_tail=Tensor(np.full((2, 3), i)),
                    kc=[NS(k=Tensor(np.zeros((1024, 3))), v=Tensor(np.zeros((1024, 3))),
                           ks=Tensor(np.zeros((1024, 1))), vs=Tensor(np.zeros((1024, 1))))],
                    ikc=[Tensor(np.zeros((1024, 3)))], pooled=[Tensor(np.zeros((256, 3)))])
              for i, n in enumerate(widths)]
    at, segs = 0, []
    for st, n in zip(states, widths):
        segs.append((st, at, at+n))
        at += n
    streams = [NS(st=st, vision=None, constraint=None) for st in states]
    pending = [list(range(n)) for n in counts]
    scratch = NS(parity=parity, q=Tensor(np.zeros(4)))
    tables = NS(scratch=scratch, segs=segs, cur=parity, held=Tensor(np.zeros((4, 7))),
                ints=Tensor(np.arange(8*at+37)), ptrs=Tensor(np.arange(8)), folds=True,
                plan=NS(slots=0, max_rows=max(widths), starts=Tensor(np.array([s[1] for s in segs]+[at]))),
                sid=Tensor(np.repeat(np.arange(4), widths)))
    step = NS(segs=segs, ints=Tensor(np.arange(2*at+8)), ptrs=Tensor(np.arange(24)),
              ends=[end]*4, most=max(widths))
    b = NS(prefill=False, attn=NS(budget=2048), moe=NS(plan=NS(tile=64)),
           gdn_tables=tables, attn_step=step, mixed_ple=None, ids=Tensor(np.zeros(at)))
    w = NS(comm=None, x3=None, head=object())
    return w, b, segs, pending, streams, tables, step


@pytest.mark.parametrize('widths,end,ok', [((1,3,2,7),512,True), ((7,7,2,2),512,True),
    ((1,3,2,7),513,False), ((2,2,2,2),512,False), ((7,7,7,7),512,False),
    ((8,1,2,2),512,False), ((0,6,2,5),512,False)])
def test_admission(widths, end, ok):
    args = fixture(widths=widths, end=end)
    assert g.eligible(*args[:5], depth=6, filling=False) is ok


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'vision', 'image', 'grammar', 'tp',
    'exl3', 'budget', 'prefill', 'depth', 'filling', 'pending8', 'empty', 'dtype', 'gap'])
def test_fallback(kind):
    w,b,segs,pending,streams,*_ = fixture()
    depth, filling = 6, False
    if kind == 'missing': streams[0] = None
    elif kind == 'duplicate': streams[0].st = streams[1].st
    elif kind == 'vision': streams[0].vision = object()
    elif kind == 'image': segs[0][0].image_positions = object()
    elif kind == 'grammar': streams[0].constraint = object()
    elif kind == 'tp': w.comm = object()
    elif kind == 'exl3': w.x3 = object()
    elif kind == 'budget': b.attn.budget = 256
    elif kind == 'prefill': b.prefill = True
    elif kind == 'depth': depth = 5
    elif kind == 'filling': filling = True
    elif kind == 'pending8': pending[1] = list(range(8))
    elif kind == 'empty': pending[:] = [[] for _ in pending]
    elif kind == 'dtype': segs[0][0].kv_dtype = 'bf16'
    elif kind == 'gap': segs[1] = (segs[1][0], 2, 5)
    assert not g.eligible(w,b,segs,pending,streams,depth=depth,filling=filling)


def install_fake(monkeypatch, failure=None):
    forward = ModuleType('mixed_host.forward')
    forward.compute = lambda w,segs,b: Tensor(np.ones((segs[-1][2], 4)))
    if failure == 'compute':
        def compute(*args): raise RuntimeError('compute failed')
        forward.compute = compute
    monkeypatch.setitem(sys.modules, 'mixed_host.forward', forward)
    class Graph:
        def replay(self): pass
    torch = NS(empty=lambda shape,dtype,device: Tensor(np.zeros(shape),dtype),
               cuda=NS(CUDAGraph=Graph, graph=lambda *a,**k: nullcontext(), synchronize=lambda:None))
    monkeypatch.setitem(sys.modules,'torch',torch)
    return torch


def test_capture_refresh_and_no_state_retention(monkeypatch):
    install_fake(monkeypatch)
    w,b,segs,pending,streams,tables,step = fixture()
    helper = g.MixedRowTargetGraphs()
    helper.run(w,b,segs,tables,step,admitted=True)
    entry = helper._entries[(13,0)]
    assert entry.tables.plan.max_rows == 7 and entry.tables.plan.slots == 0
    assert entry.step.ends == [512]*4 and entry.step.most == 7
    assert entry.tables.segs == entry.step.segs == []
    assert helper.entries == helper.captures == helper.replays == 1
    references = [weakref.ref(st) for st,_,_ in segs]
    # Fresh same-total metadata/tails/pointers must overwrite every recorded table.
    _,_,new_segs,_,_,new_tables,new_step = fixture(widths=(7,1,4,1))
    new_tables.scratch = tables.scratch
    new_tables.ints.values += 11
    new_tables.ptrs.values += 17
    new_step.ints.values += 23
    new_step.ptrs.values += 29
    helper.run(w,b,new_segs,new_tables,new_step,admitted=True)
    for a,c in ((entry.tables.ints,new_tables.ints),(entry.tables.ptrs,new_tables.ptrs),
                (entry.step.ints,new_step.ints),(entry.step.ptrs,new_step.ptrs)):
        assert np.array_equal(a.values,c.values)
    assert helper.captures == 1 and helper.replays == 2
    segs.clear(); streams.clear(); tables.segs=[]; step.segs=[]
    gc.collect()
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize('failure', ['compute','restore','sync'])
def test_exception_rebinds_and_detaches(monkeypatch, failure):
    fake = install_fake(monkeypatch, failure)
    w,b,segs,_,_,tables,step = fixture()
    observed = []
    original_inputs = g.capture_inputs
    def capture_inputs(*args):
        result = original_inputs(*args); observed.append(result); return result
    monkeypatch.setattr(g,'capture_inputs',capture_inputs)
    if failure == 'restore':
        monkeypatch.setattr(g,'_restore',lambda _: (_ for _ in ()).throw(RuntimeError('restore failed')))
    if failure == 'sync':
        fake.cuda.synchronize=lambda: (_ for _ in ()).throw(RuntimeError('sync failed'))
    helper = g.MixedRowTargetGraphs()
    with pytest.raises(RuntimeError): helper.run(w,b,segs,tables,step,admitted=True)
    assert b.gdn_tables is tables and b.attn_step is step and b.mixed_ple is None
    assert observed[0][0].segs == observed[0][1].segs == []
    assert helper.entries == helper.captures == helper.replays == 0


def test_signature_invalidation_falls_back_without_replay(monkeypatch):
    install_fake(monkeypatch)
    w,b,segs,_,_,tables,step = fixture()
    helper = g.MixedRowTargetGraphs()
    helper.run(w,b,segs,tables,step,admitted=True)
    b.moe.plan = NS(tile=64)
    assert helper.run(w,b,segs,tables,step,admitted=True) is None
    assert helper.entries == 0 and helper.replays == 1 and helper.eager_calls == 1


@pytest.mark.parametrize('bad', ['slots','folds','held','parity'])
def test_bad_chain_geometry_rejects(monkeypatch,bad):
    install_fake(monkeypatch)
    w,b,segs,_,_,tables,step=fixture()
    if bad=='slots': tables.plan.slots=1
    elif bad=='folds': tables.folds=False
    elif bad=='held': tables.held=Tensor(np.zeros((4,2)))
    elif bad=='parity': tables.cur=2
    with pytest.raises(ValueError): g.MixedRowTargetGraphs().run(w,b,segs,tables,step,admitted=True)


def test_default_off_and_typo(monkeypatch):
    monkeypatch.delenv(g.FLAG,raising=False)
    assert not g.requested()
    monkeypatch.setenv(g.FLAG,'1'); assert g.requested()
    monkeypatch.setenv(g.FLAG,'yes')
    with pytest.raises(ValueError): g.requested()




def test_shared_hook_clears_ple_and_does_not_change_solo():
    text=(ROOT/REL/'multi.py').read_text()
    assert text.count('self.buf.mixed_ple = None')==1
    assert 'if logits is None and mixed is not None:' in text
    assert 'mixed_row_target_graphs' not in (ROOT/REL/'multi_solo.py').read_text()
