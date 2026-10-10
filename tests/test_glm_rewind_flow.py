"""Exercise GLM request cache lifecycle with the GPU arithmetic replaced by small CPU states."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

from tensorfold.families.glm5_next.cuda.engine import GlmEngine


class _Tensor:
    def __init__(self, values=()):
        self.values = list(values)

    def clone(self):
        return _Tensor(self.values)

    def copy_(self, other):
        self.values = list(other.values)

    def __getitem__(self, item):
        return _Tensor(self.values[item])

    @property
    def shape(self):
        return (len(self.values),)


class _State:
    def __init__(self):
        self.rec = [_Tensor()]
        self.conv = _Tensor()
        self.cur = [0]
        self.pos = self.mtp_len = self.mtp_drafted = 0

    def set_pos(self, pos):
        self.pos = pos

    def set_mtp_len(self, length):
        self.mtp_len = length


def _mod(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _fake_decode(monkeypatch):
    base = "tensorfold.families.glm5_next.cuda"
    torch = _mod(monkeypatch, "torch", no_grad=lambda: lambda fn: fn, empty_like=lambda x: _Tensor())
    torch.cuda = SimpleNamespace(empty_cache=lambda: None)
    _mod(monkeypatch, "tensorfold.cuda.sampling", comm_gather=None, nucleus_rows=None, one_rank=None)
    _mod(monkeypatch, "tensorfold.engine.exact_sampling", MARGIN=0, Sampling=object, choose_rows=None)
    for name in ("glue", "qmm", "sparse", "weights"):
        _mod(monkeypatch, f"{base}.{name}", **({"pool_bucket": None} if name == "sparse" else
                                              {"Weights": object} if name == "weights" else {}))
    prof = _mod(monkeypatch, f"{base}.prof", active=False, timed=lambda label: nullcontext(),
                report=lambda count: None)

    def stage(w, st, buf, tokens):
        buf.tokens = list(tokens)
        return len(tokens)

    def compute(w, st, buf, rows, **kwargs):
        before = st.rec[0].values
        cut = kwargs.get("cut")
        if cut is not None:
            cut.rec.copy_(_Tensor(before + buf.tokens[:cut.point]))
            cut.conv.copy_(_Tensor(before + buf.tokens[:cut.point]))
        st.rec[0].values = before + buf.tokens
        st.conv.values = list(st.rec[0].values)
        buf.fnormed = _Tensor(buf.tokens)
        return _Tensor([len(st.rec[0].values)])

    def commit(w, st, buf, rows, keep):
        st.pos += rows

    _mod(monkeypatch, f"{base}.forward", Buffers=object, State=_State,
         Cut=lambda point, rec, conv: SimpleNamespace(point=point, rec=rec, conv=conv),
         chunks_for=lambda st, rows: 1, stage=stage, compute=compute, commit=commit)
    _mod(monkeypatch, f"{base}.mtp", mtp_compute=None, mtp_stage=None,
         mtp_forward=lambda w, st, buf, tokens, hidden: None)
    _mod(monkeypatch, f"{base}.drafter_choice", DrafterChoice=object, auto_decode=None)
    path = Path(__file__).parents[1] / "src/tensorfold/families/glm5_next/cuda/decode.py"
    spec = importlib.util.spec_from_file_location(f"{base}.decode", path)
    decode = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, decode)
    spec.loader.exec_module(decode)
    decode.row_bytes = lambda e, snap: 1
    decode.snapshot_bytes = lambda snap: 1 + snap.nbytes
    decode.save_rows = lambda e, snap: (setattr(snap, "rows", [snap.ids]), setattr(snap, "nbytes", 1))
    decode.load_rows = lambda e, snap: None
    return decode


def _engine():
    engine = GlmEngine.__new__(GlmEngine)
    st = _State()
    e = SimpleNamespace(w=SimpleNamespace(mtp=object()), st=st, pbuf=SimpleNamespace(fnormed=_Tensor()),
                        prefill_rows=4, constraint=None, window=None, reset=st.__init__,
                        sample=lambda logits, positions, sampling: [777], follow=lambda tokens: None)
    engine.e = e
    engine.cache, engine.live = [], []
    engine.cache_entries, engine.cache_bytes = 8, 100
    engine.limit, engine.eos = 100, ()
    engine.rank, engine.drafter = 0, None
    engine.user_token, engine.policy, engine.serial_only = 900, "0", False
    engine.request = SimpleNamespace(policy=None, stop_eos=True)
    engine._effective = lambda code: code
    engine._ring = lambda: None
    engine._share = lambda values: values
    return engine


def test_request_flow_keeps_rewind_and_normal_continuation(monkeypatch):
    _fake_decode(monkeypatch)
    engine = _engine()
    first = [1, 2, 900, 10, 901, 20]
    warm = engine.generate(first, 1, None, lambda tokens: None)
    assert warm["cached"] == 0
    assert [(len(s.ids), s.rewind) for s in engine.cache] == [(2, True), (5, False)]
    assert [s.rec.values for s in engine.cache] == [first[:2], first[:5]]

    replacement = [1, 2, 900, 11, 901, 20]
    assert engine.generate(replacement, 1, None, lambda tokens: None)["cached"] == 2
    assert engine.generate(replacement + [777, 900, 30], 1, None,
                           lambda tokens: None)["cached"] == len(replacement) - 1


def test_request_flow_without_user_marker_keeps_original_prefix_behavior(monkeypatch):
    _fake_decode(monkeypatch)
    engine = _engine()
    engine.user_token = None
    first = [1, 2, 900, 10, 901, 20]
    engine.generate(first, 1, None, lambda tokens: None)
    assert [(len(s.ids), s.rewind) for s in engine.cache] == [(len(first) - 1, False)]
    assert engine.generate(first + [777, 900, 30], 1, None,
                           lambda tokens: None)["cached"] == len(first) - 1
