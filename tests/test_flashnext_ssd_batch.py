"""Check batched staging against the pre-batching stage with CPU bytes and event guards."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'src/tensorfold/families/qwen4_exp/cuda'
CANDIDATE = BASE / 'forward.py'


def _unbatched_stage(w: Weights, b: Buffers, windows: Sequence[tuple[State, Sequence[int]]]) -> list[Seg]:
    """Host work before a forward (token ids, n-gram rows into static buffers); returns each stream's segment."""

    segs: list[Seg] = []
    for st, tokens in windows:
        a0 = segs[-1][2] if segs else 0
        if st.pos + len(tokens) > st.capacity:
            raise ValueError("context past the cache capacity")
        segs.append((st, a0, a0 + len(tokens)))
    R = segs[-1][2]
    if R > b.rows:
        raise ValueError(f"window of {R} rows, buffers hold {b.rows}")
    lookups = []                         # (layer's PLE, row ids [rows, heads], first staging row)
    for layer in w.layers:
        if layer.ple is not None:
            p = layer.ple
            for (st, tokens), (_, a0, _) in zip(windows, segs):
                toks = np.asarray(tokens, dtype=np.int64)
                ids = p.ngram.ids(st.ple_history, toks)
                st.ple_last = (st.ple_history, toks)
                lookups.append((p, ids, a0 * (ids.size // len(toks))))
    # the table reads before the wait, which ends once the GPU has run the previous step: their faults overlap it
    ahead = [p.table.gather(ids) for p, ids, _ in lookups] if w.x3 is None and stage_ahead() else None
    b.staged.synchronize()               # the previous step's copies out of the pinned buffers are done
    b.ids_host[:R].numpy()[:] = np.asarray([t for _, tokens in windows for t in tokens], dtype=np.int32)
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    for i, (p, ids, at) in enumerate(lookups):
        if w.x3 is not None:
            # Test recorder supplies the same EXL3 call boundary without CUDA import.
            stage_ple(p.table, w.x3, ids, at=at)
        else:
            stage_ple_rows(p, b, ids, at=at, got=None if ahead is None else ahead[i])
    b.staged.record()
    return segs

def bodies(path: Path | None) -> dict:
    """Execute actual helpers plus either candidate stage or the original stage oracle."""
    tree = ast.parse(CANDIDATE.read_text())
    names = {'stage_ahead', 'stage', 'stage_ple_rows', 'commit'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
        # Supply EXL3's actual call boundary as a CPU recorder rather than import CUDA.
        for child in ast.walk(node):
            for field, value in ast.iter_fields(child):
                if isinstance(value, list):
                    setattr(child, field, [n for n in value if not isinstance(n, ast.ImportFrom)])
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    ns = {'np': np, 'os': os, 'STAGE_AHEAD': 'TF_FLASH_STAGE_AHEAD',
          'torch': SimpleNamespace(int16=np.int16, bfloat16=np.uint16)}
    exec(compile(ast.fix_missing_locations(module), str(CANDIDATE), 'exec'), ns)
    if path is None:
        ns['stage'] = FunctionType(_unbatched_stage.__code__, ns)
    return ns


spec = importlib.util.spec_from_file_location('batch_ngram', BASE / 'ngram.py')
ngram = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ngram)


class Tensor:
    """CPU storage with guarded pinned writes and recorded copy boundaries."""
    def __init__(self, array, log, event, name):
        self.array, self.log, self.event, self.name = array, log, event, name

    def __getitem__(self, key):
        return Tensor(self.array[key], self.log, self.event, self.name)

    def view(self, dtype):
        return Tensor(self.array.view(dtype), self.log, self.event, self.name)

    def numpy(self):
        assert self.event.waited, 'pinned storage touched before previous copy completed'
        self.log.append(('write', self.name))
        return self.array

    def copy_(self, other, non_blocking=False):
        assert self.event.waited and non_blocking
        self.array[:] = other.array
        self.log.append(('copy', self.name, self.array.tobytes()))


class Event:
    def __init__(self, log):
        self.log, self.waited = log, False

    def synchronize(self):
        self.log.append(('wait',))
        self.waited = True

    def record(self):
        self.log.append(('record',))
        self.waited = False


class Table:
    """Decode face test double: bit-preserving indexed rows across a shard split."""
    def __init__(self, face, log, layer):
        self.bits, self.log, self.layer = (4 if face == 'affine' else 16), log, layer
        self.calls = []
        ids = np.arange(512, dtype=np.uint32)[:, None]
        if self.bits == 16:
            self.values = (ids * 17 + np.arange(3) + layer * 500).astype(np.uint16)
        else:
            self.values = ((ids * 1234567 + np.arange(2)).astype(np.uint32),
                           (ids + np.arange(2) + 16000).astype(np.uint16),
                           (ids + np.arange(2) + 33000).astype(np.uint16))

    def gather(self, ids):
        self.calls.append(ids.copy())
        self.log.append(('gather', self.layer))
        flat = ids.reshape(-1)
        arrays = (self.values,) if self.bits == 16 else self.values
        # Exercise boundaries and duplicate occurrences, never sort returned rows.
        assert np.any(flat < 100) and np.any(flat >= 100)
        out = tuple(a[flat].copy() for a in arrays)
        return out[0] if self.bits == 16 else out


def fixture(face='nvfp4', one=False, exl3=False, layers=2):
    log, states, windows = [], [], []
    histories = [[0, 0], [4, 5], [0, 0], [7, 8]]
    tokens = [[9, 9], [5, 0, 11], [9, 9], [6]]
    for i, (history, toks) in enumerate(zip(histories, tokens)):
        st = SimpleNamespace(pos=8+i, capacity=64, ple_history=np.array(history, dtype=np.int64),
                             ple_last=None, cur=[], ple_tail=np.zeros((2, 3), np.uint16))
        st.set_pos = lambda pos, st=st: setattr(st, 'pos', pos)
        states.append(st)
        windows.append((st, toks))
    if one:
        states, windows = states[:1], windows[:1]
    event = Event(log)
    b = SimpleNamespace(rows=32, staged=event, prefill=False, ple_nrow=np.zeros((32, 3), np.uint16))
    sizes = {'ids_host': ((32,), np.int32), 'ids': ((32,), np.int32),
             'ple_hv': ((128, 3), np.uint16), 'ple_v': ((128, 3), np.uint16),
             'ple_hw': ((128, 2), np.int32), 'ple_w': ((128, 2), np.int32),
             'ple_hs': ((128, 2), np.uint16), 'ple_s': ((128, 2), np.uint16),
             'ple_hb': ((128, 2), np.uint16), 'ple_b': ((128, 2), np.uint16)}
    for name, (shape, dtype) in sizes.items():
        setattr(b, name, Tensor(np.full(shape, 123, dtype), log, event, name))
    ple = []
    for layer in range(layers):
        gram = ngram.NGram(vocab=32, ngram_size=3, heads_per_ngram=2, vocab_base=31,
                          divisor=1, shards=2, seed=42, eos=0, embed_dim=12, ple_index=layer)
        ple.append(SimpleNamespace(ngram=gram, table=Table(face, log, layer)))
    w = SimpleNamespace(layers=[SimpleNamespace(ple=p) for p in ple], x3=object() if exl3 else None,
                        cfg=SimpleNamespace(ngram_size=3))
    return w, b, states, windows, log


class NoConcatenate:
    """Reject batched ID allocation on the original single-window/EXL3 paths."""

    def __getattr__(self, name):
        return getattr(np, name)

    def concatenate(self, *args, **kwargs):
        raise AssertionError('single-window and EXL3 staging must not concatenate IDs')


def execute(path, face, ahead, one=False, exl3=False):
    ns = bodies(path)
    w, b, states, windows, log = fixture(face, one, exl3)
    shifts = []
    ns['shift_windows'] = lambda old, new, keep, channels: shifts.append((new.copy(), keep, channels))
    ns['stage_ple'] = lambda table, x3, ids, at: log.append(('exl3', ids.copy(), at))
    before = [s.ple_history.copy() for s in states]
    if one or exl3:
        ns['np'] = NoConcatenate()
    segs = ns['stage'](w, b, windows)
    ns['np'] = np  # Commit legitimately concatenates the accepted token history.
    assert [(a, z) for _, a, z in segs] == ([(0, 2)] if one else [(0, 2), (2, 5), (5, 7), (7, 8)])
    for st, history, (_, toks) in zip(states, before, windows):
        np.testing.assert_array_equal(st.ple_history, history)
        np.testing.assert_array_equal(st.ple_last[0], history)
        np.testing.assert_array_equal(st.ple_last[1], toks)
    # Commit an accepted prefix: rejected suffixes must not enter subsequent hashes.
    for i, (st, a0, a1) in enumerate(segs):
        keep = 1 if i == 1 else a1-a0
        b.ple_nrow[:] = np.arange(32)[:, None]
        ns['commit'](w, st, b, a1-a0, keep, at=a0)
        np.testing.assert_array_equal(st.ple_history, np.concatenate([before[i], windows[i][1][:keep]])[-2:])
        assert st.ple_last is None and st.pos == 8+i+keep
        assert shifts[-1][1] == keep
        np.testing.assert_array_equal(shifts[-1][0][0, :, 0], np.arange(a0, a1))
    first_log = list(log)
    resumed = [(s, [12, 0, 9]) for s in states]
    if one or exl3:
        ns['np'] = NoConcatenate()
    ns['stage'](w, b, resumed)
    return w, b, states, first_log, log


@pytest.mark.parametrize('face', ['affine', 'bf16', 'nvfp4'])
@pytest.mark.parametrize('ahead', [False, True])
@pytest.mark.parametrize('one', [False, True])
def test_actual_stage_and_commit_bytes_equal(face, ahead, one, monkeypatch):
    monkeypatch.setenv('TF_FLASH_STAGE_AHEAD', '1' if ahead else '0')
    left = execute(None, face, ahead, one)
    right = execute(CANDIDATE, face, ahead, one)
    for name in ('ids', 'ple_v', 'ple_w', 'ple_s', 'ple_b'):
        np.testing.assert_array_equal(getattr(left[1], name).array, getattr(right[1], name).array)
    for ls, rs in zip(left[2], right[2]):
        np.testing.assert_array_equal(ls.ple_history, rs.ple_history)
        np.testing.assert_array_equal(ls.ple_last[1], rs.ple_last[1])
    for lp, rp in zip(left[0].layers, right[0].layers):
        n = 1 if one else 4
        assert len(lp.ple.table.calls) == 2*n and len(rp.ple.table.calls) == 2
        for start in (0, n):
            np.testing.assert_array_equal(np.concatenate(lp.ple.table.calls[start:start+n]), rp.ple.table.calls[start//n])
    for result in (left, right):
        log = result[3]
        assert log[-1] == ('record',)
        wait = next(i for i, row in enumerate(log) if row[0] == 'wait')
        gathers = [i for i, row in enumerate(log) if row[0] == 'gather']
        assert all(i < wait if ahead else i > wait for i in gathers)
        assert all(i > wait for i, row in enumerate(log) if row[0] in ('write', 'copy'))
    if one:
        assert left[3] == right[3], 'single-window calls and copy ordering changed'
    # Compare each complete layer copy, not only the final layer's reused buffer.
    names = ('ple_v',) if face != 'affine' else ('ple_w', 'ple_s', 'ple_b')
    for name in names:
        copies = lambda result: [row[2] for row in result[3] if row[:2] == ('copy', name)]
        original, batched = copies(left), copies(right)
        n = 1 if one else 4
        assert len(original) == 2*n and len(batched) == 2
        for i in range(2):
            assert b''.join(original[i*n:(i+1)*n]) == batched[i]


@pytest.mark.parametrize('ahead', [False, True])
def test_exl3_calls_offsets_unchanged(ahead, monkeypatch):
    monkeypatch.setenv('TF_FLASH_STAGE_AHEAD', '1' if ahead else '0')
    left = execute(None, 'bf16', ahead, exl3=True)
    right = execute(CANDIDATE, 'bf16', ahead, exl3=True)
    rows = lambda result: [(row[1].tobytes(), row[2]) for row in result[4] if row[0] == 'exl3']
    assert rows(left) == rows(right)
    assert [at for _, at in rows(right)[:4]] == [0, 8, 20, 28]
    assert all(not p.ple.table.calls for p in right[0].layers)


def test_per_layer_batch_preserves_head_and_segment_order(monkeypatch):
    """The gather batch is the ordered concatenation, including repeated rows."""
    monkeypatch.setenv('TF_FLASH_STAGE_AHEAD', '1')
    reference = execute(None, 'nvfp4', True)
    candidate = execute(CANDIDATE, 'nvfp4', True)
    for old_layer, new_layer in zip(reference[0].layers, candidate[0].layers):
        expected = np.concatenate(old_layer.ple.table.calls[:4], axis=0)
        np.testing.assert_array_equal(new_layer.ple.table.calls[0], expected)
        assert expected.shape[0] == 8 and expected.shape[1] == old_layer.ple.ngram.heads


@pytest.mark.parametrize('bad', ['capacity', 'rows'])
@pytest.mark.parametrize('filename', ['baseline', 'candidate'])
def test_bounds_fail_before_hash_reads_or_event_wait(bad, filename):
    path = None if filename == 'baseline' else CANDIDATE
    ns = bodies(path)
    w, b, states, windows, log = fixture()
    if bad == 'capacity':
        states[-1].capacity = states[-1].pos
    else:
        b.rows = 7
    with pytest.raises(ValueError):
        ns['stage'](w, b, windows)
    assert log == [] and all(s.ple_last is None for s in states)


def test_without_ple_only_stages_tokens_and_records_event():
    results = []
    for path in (None, CANDIDATE):
        ns = bodies(path)
        w, b, states, windows, log = fixture()
        w.layers = [SimpleNamespace(ple=None)]
        ns['stage'](w, b, windows)
        assert all(s.ple_last is None for s in states)
        results.append((b.ids.array.tobytes(), log))
    assert results[0] == results[1]
