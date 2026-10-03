"""A fake LaneForward whose logits hash each lane's tokens as read back through the page pool, and two ranks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tensorfold.cuda.drafting import Proposals
from tensorfold.cuda.kvpool import PagePool, Plane
from tensorfold.cuda.lanes.follow import Lanes
from tensorfold.cuda.lanes.link import AGREED
from tensorfold.cuda.lanes.forward import Candidates
from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream

VOCAB = 64
M64 = (1 << 64) - 1
PLANES = (Plane("kv", 8), Plane("pair", 8, 2))


def mix(x: int) -> int:
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & M64
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & M64
    return x ^ (x >> 31)


def roll(h: int, tokens) -> int:
    for t in tokens:
        h = mix(h ^ (int(t) + 0x9E3779B97F4A7C15) & M64)
    return h


def values(h: int) -> np.ndarray:
    """The logits after a context hashing to ``h``: coarse steps, so ties between ids happen."""

    return np.array([(mix(h ^ (j * 0xD1B54A32D192ED03 & M64)) >> 56) / 8.0 for j in range(VOCAB)], dtype=np.float32)


def greedy(context) -> int:
    v = values(roll(0, context))
    return int(np.lexsort((np.arange(VOCAB), -v))[0])


@dataclass
class LaneState:
    pos: int = 0
    ring: int = 0  # the bounded state a snapshot carries: a hash of every token before ``pos``
    tail: int = 0  # what ``finish_prompt`` rebuilt from the prompt's tail (replay)
    tokens: list[int] = field(default_factory=list)  # the lane's tokens when there is no pool
    last: tuple | None = None


class FakeForward:
    """Writes each row's token into the pool's ``kv`` plane and pair sums into ``pair``; reads both back every row."""

    def __init__(self, pool: PagePool | None, lanes: int, *, grid: int = 1, replay_tail: int = 0) -> None:
        self.vocab, self.pool, self.grid, self.replay_tail = VOCAB, pool, grid, replay_tail
        self.st = [LaneState() for _ in range(lanes)]
        self.tables = None
        self.calls: list[tuple] = []

    def bind(self, tables) -> None:
        self.tables = tables

    def _row(self, lane: int, plane: str, pos: int) -> int:
        n = self.pool.page_tokens
        per = self.pool.planes[plane].per_tokens
        return self.tables[lane].pages[pos // n] * self.pool.rows_per_page(plane) + (pos % n) // per

    def _write(self, lane: int, pos: int, token: int, before: int | None) -> None:
        if self.pool is None:
            del self.st[lane].tokens[pos:]
            self.st[lane].tokens.append(int(token))
            return
        self.pool.view("kv")[self._row(lane, "kv", pos)] = np.frombuffer(np.int64(token).tobytes(), np.uint8)
        if pos % 2 == 1:
            pair = np.int64(int(before) + int(token))
            self.pool.view("pair")[self._row(lane, "pair", pos)] = np.frombuffer(pair.tobytes(), np.uint8)

    def _held(self, lane: int, end: int) -> list[int]:
        """The lane's tokens below ``end`` as its caches hold them; a pair row that disagrees spoils the hash."""

        if self.pool is None:
            return list(self.st[lane].tokens[:end])
        kv, pair = self.pool.view("kv"), self.pool.view("pair")
        out = [int(np.frombuffer(bytes(kv[self._row(lane, "kv", p)]), np.int64)[0]) for p in range(end)]
        for p in range(1, end, 2):
            if int(np.frombuffer(bytes(pair[self._row(lane, "pair", p)]), np.int64)[0]) != out[p - 1] + out[p]:
                out[p] ^= 0x5A5A
        return out

    def reset(self, lane: int) -> None:
        self.st[lane] = LaneState()

    def prefill(self, pieces) -> None:
        self.calls.append(("prefill", tuple((p.lane, p.start, p.end) for p in pieces)))
        for p in pieces:
            st = self.st[p.lane]
            if p.start != st.pos:
                raise AssertionError(f"lane {p.lane}: a piece at {p.start}, the lane is at {st.pos}")
            before = self._held(p.lane, p.start)[-1:] or [0]
            for i, t in enumerate(p.ids):
                self._write(p.lane, p.start + i, t, before[0] if i == 0 else p.ids[i - 1])
            st.ring, st.pos = roll(st.ring, p.ids), p.end

    def finish_prompt(self, lane: int, end: int, tail) -> None:
        self.calls.append(("finish", lane, end, tuple(tail)))
        self.st[lane].tail = roll(7, tail)

    def window(self, rows, count, masks=None) -> Candidates:
        self.calls.append(("window", tuple((r.lane, r.start, len(r.tokens)) for r in rows)))
        ids, vals = [], []
        for k, r in enumerate(rows):
            st = self.st[r.lane]
            if r.start != st.pos:
                raise AssertionError(f"lane {r.lane}: a window at {r.start}, the lane is at {st.pos}")
            held = self._held(r.lane, r.start)
            spoil = roll(0, held) ^ st.ring  # zero when the snapshot's ring matches the pages
            prev = held[-1] if held else 0
            for i, t in enumerate(r.tokens):
                self._write(r.lane, r.start + i, t, prev if i == 0 else r.tokens[i - 1])
            for i in range(len(r.tokens)):
                v = values(roll(0, [*held, *r.tokens[: i + 1]]) ^ spoil ^ st.tail)
                w = masks[k] if masks is not None else None
                if w is not None and i in w.rows:
                    v = np.where(w.allowed(w.rows.index(i)), v, -np.inf).astype(np.float32)
                order = np.lexsort((np.arange(VOCAB), -v))[:count]
                ids.append(order)
                vals.append(v[order])
            st.last = (r.start, tuple(r.tokens))
        return Candidates(np.array(ids, dtype=np.int64), np.array(vals, dtype=np.float32))

    def commit(self, lane: int, kept: int) -> None:
        st = self.st[lane]
        start, tokens = st.last
        st.ring, st.pos, st.last = roll(st.ring, tokens[: kept + 1]), start + kept + 1, None

    def snapshot(self, lane: int):
        st = self.st[lane]
        return {"pos": st.pos, "ring": st.ring, "tail": st.tail, "tokens": list(st.tokens[: st.pos])}

    def restore(self, lane: int, snap) -> None:
        self.st[lane] = LaneState(snap["pos"], snap["ring"], snap["tail"], list(snap["tokens"]))

    def state(self) -> tuple:
        return tuple((s.pos, s.ring, s.tail, tuple(s.tokens), s.last) for s in self.st)


class Codec:
    """The fake's snapshot as host arrays."""

    def to_host(self, snap) -> dict:
        return {
            "head": np.array([snap["pos"], snap["ring"], snap["tail"]], dtype=np.uint64),
            "tokens": np.array(snap["tokens"], dtype=np.int64),
        }

    def from_host(self, a) -> dict:
        pos, ring, tail = (int(v) for v in a["head"])
        return {"pos": pos, "ring": ring, "tail": tail, "tokens": [int(t) for t in a["tokens"]]}


class Pattern:
    """Drafts the greedy continuation for ``good[i]`` tokens of proposal i, then wrong ones; confidences to match."""

    def __init__(self, block: int = 6, good=(3, 0, 5, 1, 6, 2)) -> None:
        self.block, self.good = block, list(good)
        self.context: dict[int, list[int]] = {}
        self.calls = 0

    def reset(self, lane, prompt) -> None:
        self.context[lane] = list(prompt)

    def observe(self, lane, tokens) -> None:
        self.context[lane].extend(tokens)

    def propose(self, lanes, pendings, starts, depths, samplings) -> Proposals:
        tokens, conf = [], []
        for lane, pending, most in zip(lanes, pendings, depths):
            assert self.context[lane][-1] == pending
            good = self.good[self.calls % len(self.good)]
            self.calls += 1
            ctx, out = list(self.context[lane]), []
            for j in range(most):
                t = greedy(ctx)
                t = t if j < good else (t + 1) % VOCAB
                out.append(t)
                ctx.append(t)
            tokens.append(out)
            conf.append([0.9 if j < good else 0.2 for j in range(most)])
        return Proposals(tokens, conf)


class FailingTier:
    """A tier whose every write fails, as a full disk's would."""

    used, limit = 0, 1 << 20

    def find(self, prompt):
        return None

    def put(self, key, ids, arrays, *, owned=False):
        raise OSError("no space left on device")

    def get(self, key):
        raise KeyError(key)

    def drop(self, key):
        return None


class Mirror:
    """Rank 0's link to a second rank in the same thread: each message runs there as it is sent, and votes."""

    def __init__(self, follower: Lanes) -> None:
        self.follower, self.sent, self.last = follower, [], None

    def send(self, kind, ints) -> None:
        self.sent.append((kind, list(ints)))
        if kind not in AGREED:
            self.follower.apply(kind, list(ints))
            return
        try:
            self.follower.apply(kind, list(ints))
            self.last = (kind, list(ints), True)
        except Exception:  # noqa: BLE001  (its vote says so)
            self.last = (kind, list(ints), False)

    def agree(self, ok: bool) -> bool:
        kind, ints, theirs = self.last
        both = bool(ok) and theirs
        if not both:
            self.follower.disagreed(kind, ints, theirs)
        return both

    def recv(self):
        raise RuntimeError("rank 0 does not receive")

    def idle(self) -> None:
        return None


def drive(decoder, streams: list[Stream], *, lanes: int, check=None, arrive=None) -> int:
    """The ``Scheduler``'s loop without its thread: admit while lanes are free, round, finish; ``check`` after each."""

    waiting, rounds, arrive = list(streams), 0, dict(arrive or {})
    while waiting or decoder.live():
        while waiting and decoder.live() < lanes and arrive.get(id(waiting[0]), 0) <= rounds:
            try:
                decoder.admit(waiting[0])
            except NoRoom:
                if not decoder.live():
                    raise
                break
            waiting.pop(0)
        done = decoder.round()
        decoder.finish(done)
        rounds += 1
        if check is not None:
            check()
        assert rounds < 10_000
    return rounds


def same_state(a: Lanes, b: Lanes) -> None:
    assert a.state() == b.state()
    assert a.forward.state() == b.forward.state()
    if a.pool is not None:
        for name in a.pool.buffers:
            assert np.array_equal(a.pool.buffers[name], b.pool.buffers[name]), name
