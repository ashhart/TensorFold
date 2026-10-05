"""GlmEngine._take_over on CPU: snapshots that will be dropped are never cloned, and the survivors are unchanged."""

from __future__ import annotations

import random
import sys
import types
from dataclasses import dataclass, field

from tensorfold.families.glm5_next.cuda.engine import GlmEngine

DECODE = "tensorfold.families.glm5_next.cuda.decode"
GIB = 1 << 30


class Rows:
    """A tensor stand-in with a byte size."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes

    def numel(self):
        return self.nbytes

    def element_size(self):
        return 1


def rows_bytes(rows) -> int:
    return sum(t.numel() * t.element_size() for t in rows or [])


@dataclass
class Snap:
    ids: list[int]
    need: int                         # what save_rows would copy
    base: int = 1000                  # KDA state and conv window
    rows: list | None = None
    nbytes: int = 0
    drafter_rows: list | None = None
    drafter_end: int = -1
    cloned: bool = field(default=False)


class Recorder:
    def __init__(self):
        self.saved: list[Snap] = []
        self.empty_cache = 0

    def install(self, monkeypatch):
        def snapshot_bytes(snap):
            return snap.base + rows_bytes(snap.drafter_rows) + (snap.nbytes if snap.rows is not None else 0)

        def save_rows(e, snap):
            snap.rows, snap.nbytes, snap.cloned = [object()], snap.need, True
            snap.drafter_end, snap.drafter_rows = -1, None         # decode.save_rows: DFlash2 caches are not kept
            self.saved.append(snap)

        decode = types.ModuleType(DECODE)
        decode.row_bytes, decode.save_rows, decode.snapshot_bytes = lambda e, snap: snap.need, save_rows, snapshot_bytes
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(empty_cache=lambda: setattr(self, "empty_cache", self.empty_cache + 1))
        monkeypatch.setitem(sys.modules, DECODE, decode)
        monkeypatch.setitem(sys.modules, "torch", torch)
        return snapshot_bytes


class Engine:
    e = None

    def __init__(self, cache, live, cache_bytes, snapshot_bytes):
        self.cache, self.live, self.cache_bytes = list(cache), live, cache_bytes
        self._sizeof = snapshot_bytes

    def _drop(self, snap):
        snap.rows, snap.nbytes, snap.drafter_rows = None, 0, None
        self.cache.remove(snap)

    def _held_bytes(self):
        return sum(self._sizeof(c) for c in self.cache)


def reference(engine, keep, recorder):
    """The save-then-drop walk as it was before the fix, run on a copy of the engine's state."""
    live, dropped = engine.live, False

    def resumes(c):
        return len(c.ids) <= len(keep) and keep[:len(c.ids)] == c.ids

    for snap in list(engine.cache):
        n = len(snap.ids)
        if snap not in engine.cache or snap.rows is not None or resumes(snap):
            continue
        if live[:n] != snap.ids:
            engine._drop(snap)
            continue
        need = snap.need
        while engine._held_bytes() + need > engine.cache_bytes:
            old = next((c for c in engine.cache if c is not snap and not resumes(c)), None)
            if old is None:
                break
            engine._drop(old)
            dropped = True
        if engine._held_bytes() + need > engine.cache_bytes:
            engine._drop(snap)
            dropped = True
            continue
        snap.rows, snap.nbytes = [object()], snap.need
    return dropped


def conversation(turns: int, per_token: int = 6200, first: int = 14_000, last: int = 257_000):
    lengths = [first + round(i * (last - first) / (turns - 1)) for i in range(turns)]
    conv = list(range(last + 1))
    return [Snap(conv[:n], n * per_token) for n in lengths], conv


def run(monkeypatch, snaps, live, keep, cache_bytes):
    rec = Recorder()
    sizeof = rec.install(monkeypatch)
    engine = Engine(snaps, live, cache_bytes, sizeof)
    GlmEngine._take_over(engine, keep)
    return engine, rec


def test_a_new_conversation_clones_only_the_snapshots_that_stay(monkeypatch):
    snaps, conv = conversation(25)
    engine, rec = run(monkeypatch, snaps, conv, [], 3 * GIB)       # a fresh prompt shares no prefix
    stayed = [s for s in snaps if s in engine.cache]
    assert stayed == snaps[-2:]                                    # the two newest fit 3 GiB
    assert rec.saved == stayed                                     # nothing else was ever cloned
    assert sum(s.need for s in rec.saved) <= 3 * GIB
    assert rec.empty_cache == 1                                    # dropped for the budget


def test_the_snapshots_that_stay_are_those_of_the_save_then_drop_walk(monkeypatch):
    rng = random.Random(7)
    for case in range(200):
        convs = [[rng.randrange(1000) for _ in range(rng.randint(1, 60))] for _ in range(rng.randint(1, 4))]
        live = rng.choice(convs)

        def build(case=case, convs=convs):
            r = random.Random(case)
            out = []
            for _ in range(r.randint(0, 40)):
                ids = list(r.choice(convs))
                ids = ids[:r.randint(1, len(ids))]
                if r.random() < 0.15:
                    ids[-1] += 1                                   # rows already overwritten
                out.append(Snap(ids, r.randint(1, 5000), r.choice([0, 1, 50, 300]),
                                rows=[object()] if r.random() < 0.15 else None))
                out[-1].nbytes = out[-1].need if out[-1].rows is not None else 0
            return out

        keep = [] if rng.random() < 0.5 else list(rng.choice(convs))
        budget = rng.randint(0, 20_000)
        old, new = build(), build()
        ref = Engine(old, live, budget, lambda s: s.base + (s.nbytes if s.rows is not None else 0))
        budget_drop = reference(ref, keep, None)
        engine, rec = run(monkeypatch, new, live, keep, budget)
        pos = {id(s): i for i, s in enumerate(new)}
        assert [pos[id(s)] for s in engine.cache] == [{id(s): i for i, s in enumerate(old)}[id(s)] for s in ref.cache]
        assert [s.rows is not None for s in engine.cache] == [s.rows is not None for s in ref.cache], case
        assert rec.empty_cache == (1 if budget_drop else 0), case
        assert all(s in engine.cache for s in rec.saved), case      # no clone was made for a dropped snapshot


def test_a_snapshot_whose_rows_are_gone_is_dropped_without_a_clone(monkeypatch):
    conv = list(range(100))
    snaps = [Snap(conv[:10], 100), Snap([7, 7, 7, 7], 100), Snap(conv[:30], 100)]
    engine, rec = run(monkeypatch, snaps, conv[:30], [], 10**6)
    assert engine.cache == [snaps[0], snaps[2]]
    assert rec.saved == [snaps[0], snaps[2]]
    assert rec.empty_cache == 0                                    # not a budget drop


def test_a_prefix_the_next_prompt_resumes_is_left_alone(monkeypatch):
    conv = list(range(100))
    snaps = [Snap(conv[:10], 100), Snap(conv[:20], 100)]
    engine, rec = run(monkeypatch, snaps, conv[:20], conv[:25], 10**6)
    assert engine.cache == snaps and rec.saved == []


def window_snaps(conv):
    """Snapshots holding a ring drafter's window rows, whether or not drafter_end is the snapshot's own length."""
    return [Snap(conv[:20], 700, 50, drafter_rows=[Rows(30), Rows(12)], drafter_end=20),
            Snap(conv[:40], 900, 50, drafter_rows=[Rows(30), Rows(12)], drafter_end=17),
            Snap(conv[:60], 500, 50, drafter_rows=[Rows(8)], drafter_end=60),
            Snap(conv[:80], 400, 50)]


def test_save_rows_sheds_the_window_rows_and_the_prediction_holds(monkeypatch):
    """save_rows here takes no drafter: it always clears drafter_rows, so the size after a save is
    snapshot_bytes(before) - the window rows' bytes + need, for drafter_end == len(ids) and for any other drafter_end."""
    conv = list(range(200))
    rec = Recorder()
    snapshot_bytes = rec.install(monkeypatch)
    from tensorfold.families.glm5_next.cuda.decode import save_rows as fake_save_rows
    for snap in window_snaps(conv):
        predicted = snapshot_bytes(snap) - rows_bytes(snap.drafter_rows) + snap.need
        fake_save_rows(None, snap)
        assert snap.drafter_rows is None and snap.drafter_end == -1
        assert snapshot_bytes(snap) == predicted


def test_the_budget_decision_uses_the_exact_size_after_a_save(monkeypatch):
    conv = list(range(200))
    probe = window_snaps(conv)
    rec = Recorder()
    snapshot_bytes = rec.install(monkeypatch)
    from tensorfold.families.glm5_next.cuda.decode import save_rows as fake_save_rows
    for snap in probe:
        fake_save_rows(None, snap)
    total = sum(snapshot_bytes(s) for s in probe)                  # the true size once all four are saved

    snaps = window_snaps(conv)
    engine, rec = run(monkeypatch, snaps, conv[:80], [], total)    # exactly the budget: all four stay
    assert engine.cache == snaps and rec.saved == snaps and rec.empty_cache == 0
    assert [s.drafter_rows for s in snaps] == [None] * 4

    snaps = window_snaps(conv)
    engine, rec = run(monkeypatch, snaps, conv[:80], [], total - 1)  # one byte less: the oldest goes
    assert engine.cache == snaps[1:] and rec.saved == snaps[1:] and rec.empty_cache == 1
    assert sum(engine._sizeof(s) for s in engine.cache) <= total - 1
