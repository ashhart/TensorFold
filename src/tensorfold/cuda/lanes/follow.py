"""Every rank's lanes: rank 0's messages applied in order to the forward, pages, cache, drafter, depth and grammars."""
# Rank 0 runs this same code on the messages it sends, so the ranks' state differs only if a message does.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..drafting import Proposals
from ..sessions import TieredCache
from .forward import Candidates, Piece, Rows, chain, check
from .link import ADMIT, AGREED, DONE, EVICT, ROUND, SAMPLING_WORDS, STOP, Link, unpack_sampling


def _takes(fn: Any, name: str) -> bool:
    import inspect

    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


@dataclass
class Held:
    """A cache entry's state: the forward's snapshot and the pool pages holding its rows (None without a pool)."""

    snap: Any
    pages: list[int] | None = None
    data: dict | None = None  # a lower tier's page rows, before they are written into a lane
    tokens: int = 0  # positions it keeps: complete rows below them are its, the rest of its last page is not


class HeldCodec:
    """A ``Held`` as host arrays: the family codec's arrays under ``state/``, each plane's page rows under ``page/``."""

    def __init__(self, codec: Any, pool: Any = None) -> None:
        self.codec, self.pool = codec, pool

    def to_host(self, held: Held) -> dict:
        out = {f"state/{k}": v for k, v in self.codec.to_host(held.snap).items()}
        if self.pool is not None:
            for name, rows in self.pool.read_pages(held.pages).items():
                keep = held.tokens // self.pool.planes[name].per_tokens if held.tokens else len(rows)
                rows = rows.copy()
                rows[keep:] = 0  # stale rows past the kept positions never leave the device
                out[f"page/{name}"] = rows
        return out

    def from_host(self, arrays: dict) -> Held:
        snap = self.codec.from_host({k[6:]: v for k, v in arrays.items() if k.startswith("state/")})
        data = {k[5:]: v for k, v in arrays.items() if k.startswith("page/")}
        return Held(snap, None, data if self.pool is not None else None)


@dataclass
class Lane:
    prompt: list[int]
    sampling: Any = None
    constraint: Any = None
    history: list[int] = field(default_factory=list)  # the tokens whose state the lane holds
    last: Rows | None = None  # the window waiting for its commit


@dataclass
class Result:
    rows: list[Rows] = field(default_factory=list)  # the windows run, after grammar cuts
    cand: Candidates | None = None
    errors: dict[int, Exception] = field(default_factory=dict)  # lane -> why its request ends


class Lanes:
    """One rank's lanes. ``pool``: a ``kvpool.PagePool``; ``cache``: a ``TieredCache`` whose entries hold ``Held``."""

    def __init__(
        self,
        forward: Any,
        lanes: int,
        capacity: int,
        *,
        pool: Any = None,
        cache: TieredCache | None = None,
        drafter: Any = None,
        depth: Any = None,
        grammars: Any = None,
    ) -> None:
        check(forward)
        self.forward, self.capacity = forward, int(capacity)
        self._counts = _takes(forward.window, "counts")
        self.pool, self.cache, self.drafter, self.depth, self.grammars = pool, cache, drafter, depth, grammars
        self.lanes: list[Lane | None] = [None] * int(lanes)
        self._kept: tuple | None = None
        self.tables = [pool.table(capacity) for _ in range(lanes)] if pool is not None else None
        if self.tables is not None and callable(getattr(forward, "bind", None)):
            forward.bind(self.tables)
        if cache is not None:
            if pool is not None and cache.release is None:
                cache.release = self._release
            if cache.codec is not None and not isinstance(cache.codec, HeldCodec):
                cache.codec = HeldCodec(cache.codec, pool)

    def apply(
        self, kind: int, ints: Sequence[int], *, constraint: Any = None, held: Held | None = None
    ) -> Result | None:
        """Run one message; rank 0 passes an admission its own grammar object and the lower-tier state it read."""

        if kind == ADMIT:
            self._admit(ints, constraint, held)
        elif kind == ROUND:
            return self._round(ints)
        elif kind == DONE:
            for lane in ints:
                self._done(int(lane))
        elif kind == EVICT:
            self.cache.drop_at(int(ints[0]))
        elif kind != STOP:
            raise ValueError(f"lane message kind {kind}")
        return None

    def follow(self, link: Link) -> None:
        """Ranks past 0: apply rank 0's messages until it sends STOP."""

        while True:
            kind, ints = link.recv()
            if kind == STOP:
                return
            if kind not in AGREED:
                self.apply(kind, ints)
                continue
            try:
                self.apply(kind, ints)
                ok = True
            except Exception:  # noqa: BLE001  (rank 0 learns it from the vote and decides for every rank)
                ok = False
            if not link.agree(ok):
                self.disagreed(kind, ints, ok)

    def disagreed(self, kind: int, ints: Sequence[int], ok: bool) -> None:
        """Another rank failed a message this rank applied: an admission is undone; anything else ends the engine."""

        if kind == ADMIT and ok:
            self._undo(int(ints[0]), self._kept)

    def _undo(self, lane: int, kept: tuple | None = None) -> None:
        """An admission taken back: the pages only it held cleared, the cache's order restored, the lane's parts reset."""

        if self.tables is not None:
            table = self.tables[lane]
            self.pool.zero_pages([p for p in table.pages if self.pool.holders(p) == 1])
        self._done(lane)
        if kept is not None:
            self.cache.entries, self.cache.hit = kept
        for reset in (lambda: self.forward.reset(lane), lambda: self.drafter and self.drafter.reset(lane, []),
                      lambda: self.depth and self.depth.reset(lane)):
            try:  # best effort: a part that cannot reset leaves only a free lane's scratch behind
                reset()
            except Exception:  # noqa: BLE001
                pass

    def _release(self, held: Held) -> None:
        if held.pages:
            self.pool.drop(held.pages)

    # -- admission -----------------------------------------------------------------------------------------------
    def _admit(self, ints: Sequence[int], constraint: Any, held: Held | None) -> None:
        lane, quota, cached, tier = (int(v) for v in ints[:4])
        sampling = unpack_sampling(ints[4 : 4 + SAMPLING_WORDS])
        at = 4 + SAMPLING_WORDS
        n = int(ints[at])
        prompt = [int(t) for t in ints[at + 1 : at + 1 + n]]
        packed = list(ints[at + 1 + n :])
        if self.lanes[lane] is not None:
            raise RuntimeError(f"lane {lane} admitted while busy")
        if constraint is None and packed:
            if self.grammars is None:
                raise RuntimeError("a request with a grammar needs the engine's grammars on every rank")
            constraint = self.grammars.follow(packed)
        state = Lane(prompt, sampling, constraint)
        self.lanes[lane] = state
        table = self.tables[lane] if self.tables is not None else None
        # a resume touches the device cache (its order addresses EVICT): an undone admission puts it back
        kept = (list(self.cache.entries), set(self.cache.hit)) if self.cache is not None else None
        self._kept = kept  # for ``disagreed``, when another rank fails this admission
        try:  # a failure anywhere past here leaves the lane as it was: free, no quota, no pages
            if table is not None:
                table.reserve(quota)
            self.forward.reset(lane)
            if cached:
                self._resume(lane, prompt, cached, tier, table, held)
                state.history = prompt[:cached]
            if self.drafter is not None:
                self.drafter.reset(lane, prompt)
            if self.depth is not None:
                self.depth.reset(lane)
        except Exception:
            self._undo(lane, kept)
            raise

    def _resume(self, lane: int, prompt: list[int], cached: int, tier: int, table: Any, held: Held | None) -> None:
        if tier < 0:
            entry = self.cache.named(prompt, cached)
            if entry is None:
                raise RuntimeError(f"no kept state for the {cached} tokens rank 0 resumes lane {lane} from")
            held = entry[1]
            if table is not None:
                table.adopt(held.pages, cached)
        else:
            held = held or self.cache.load(prompt, cached, tier)
            if table is not None:
                table.ensure(cached)
                self.pool.write_pages(table.pages[: self.pool.need(cached)], held.data)
        self.forward.restore(lane, held.snap)

    def _done(self, lane: int) -> None:
        self.lanes[lane] = None
        if self.tables is not None:
            self.tables[lane].release()

    # -- rounds --------------------------------------------------------------------------------------------------
    def _round(self, ints: Sequence[int]) -> Result:
        """Commits, pieces, saves and finals, the candidates a row, then windows: each group led by its length."""

        it = iter(int(v) for v in ints)

        def group(width: int) -> list[tuple[int, ...]]:
            return [tuple(next(it) for _ in range(width)) for _ in range(next(it))]

        commits, pieces, saves, finals = group(3), group(3), group(1), group(1)
        count = next(it)
        windows = group(5)
        res = Result()
        for lane, kept, bonus in commits:
            self._commit(lane, kept, bonus, res)
        if pieces:
            run = []
            for lane, start, end in pieces:
                if self.tables is not None:
                    self.tables[lane].ensure(end)
                ids = tuple(self.lanes[lane].prompt[start:end])
                self.lanes[lane].history.extend(ids)
                run.append(Piece(lane, start, ids))
            self.forward.prefill(run)
        for (lane,) in saves:
            self._save(lane)
        tail = int(getattr(self.forward, "replay_tail", 0) or 0)
        for (lane,) in finals:
            if tail:
                p = self.lanes[lane].prompt
                end = len(p) - 1
                self.forward.finish_prompt(lane, end, tuple(p[max(0, end - tail) : end]))
        if windows:
            self._windows(windows, count, res)
        return res

    def _commit(self, lane: int, kept: int, bonus: int, res: Result) -> None:
        s = self.lanes[lane]
        rows, s.last = s.last, None
        if rows is None:
            raise RuntimeError(f"a commit for lane {lane} without a window")
        self.forward.commit(lane, kept)
        s.history.extend(rows.tokens[: kept + 1])
        emitted = list(rows.tokens[1 : kept + 1]) + [bonus]
        if self.drafter is not None:
            self.drafter.observe(lane, emitted)
        if self.depth is not None:
            self.depth.record(lane, len(rows.tokens) - 1, kept)
        if s.constraint is not None:
            try:
                s.constraint.advance(emitted)
            except Exception as exc:  # noqa: BLE001  (GrammarError: this request ends)
                res.errors[lane] = exc

    def _save(self, lane: int) -> None:
        """The lane's state at its prompt's save point joins the cache (its pages shared, not copied)."""

        s = self.lanes[lane]
        pages = self.tables[lane].share(len(s.history)) if self.tables is not None else None
        self.cache.add(list(s.history), Held(self.forward.snapshot(lane), pages, tokens=len(s.history)), None)

    def _windows(self, specs: Sequence[tuple[int, ...]], count: int, res: Result) -> None:
        drafts: dict[int, list[int]] = {}
        deep = [w for w in specs if w[3] > 0 and w[0] not in res.errors]
        if deep and self.drafter is not None:
            got: Proposals = self.drafter.propose(
                [w[0] for w in deep],
                [w[2] for w in deep],
                [w[1] for w in deep],
                [w[3] for w in deep],
                [self.lanes[w[0]].sampling for w in deep],
            )
            for i, (lane, _, _, most, _) in enumerate(deep):
                tokens = [int(t) for t in got.tokens[i]][:most]
                conf = got.confidence[i] if got.confidence is not None else None
                k = self.depth.depth(lane, conf, len(tokens)) if self.depth is not None else len(tokens)
                drafts[lane] = tokens[:k]
        rows, masks, needs = [], [], []
        for lane, start, pending, _, need in specs:
            if lane in res.errors:
                continue
            s = self.lanes[lane]
            tokens, mask = [pending, *drafts.get(lane, [])], None
            if s.constraint is not None:
                try:
                    w = s.constraint.window(tokens, chain(len(tokens)))
                except Exception as exc:  # noqa: BLE001  (GrammarError: this request ends)
                    res.errors[lane] = exc
                    continue
                tokens, mask = list(w.tokens), (w if w.rows else None)
            if self.tables is not None:
                self.tables[lane].ensure(start + len(tokens))
            r = Rows(lane, start, tuple(int(t) for t in tokens))
            s.last = r
            rows.append(r)
            masks.append(mask)
            needs.append(need)
        res.rows = rows
        if rows:
            masks = masks if any(m is not None for m in masks) else None
            if self._counts:  # each window's own need: one nucleus request does not widen every lane's rows
                res.cand = self.forward.window(rows, count, masks, counts=needs)
            else:
                res.cand = self.forward.window(rows, count, masks)

    # -- views ---------------------------------------------------------------------------------------------------
    def state(self) -> tuple:
        """Everything the ranks must agree on, comparable across ranks (tests)."""

        lanes = tuple(None if s is None else (tuple(s.prompt), tuple(s.history), s.last) for s in self.lanes)
        pages = tuple((tuple(t.pages), t.quota) for t in self.tables) if self.tables is not None else ()
        pool = (sorted(self.pool.heap), sorted(self.pool.refs.items())) if self.pool is not None else ()
        cache = (
            (
                tuple((tuple(e[0]), tuple(e[1].pages or ())) for e in self.cache.entries),
                tuple(sorted(map(tuple, self.cache.hit))),
            )
            if self.cache is not None
            else ()
        )
        depth = self.depth.state() if callable(getattr(self.depth, "state", None)) else ()
        return lanes, pages, pool, cache, depth


__all__ = ["Held", "HeldCodec", "Lane", "Lanes", "Result"]
