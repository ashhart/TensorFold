"""The ``Scheduler``'s decoder for any ``LaneForward``: rank 0 plans a round, sends and runs it, then samples."""
# Sampling is exact_sampling's and acceptance is streams.accept's, so replies equal "draft": false token for token.

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tensorfold.engine.exact_sampling import MARGIN, choose_rows
from tensorfold.engine.grammar import pack

from ..memory_gate import NoRoom
from ..sessions import TieredCache
from ..streams import Stream, accept
from .follow import Lanes
from .forward import Candidates, chain, split
from .link import ADMIT, DONE, EVICT, ROUND, STOP, Link, LocalLink, pack_sampling


@dataclass
class Plan:
    """Rank 0's view of one admitted stream."""

    stream: Stream
    lane: int
    done: int  # prompt tokens the lane's state holds
    save_at: int | None  # the prompt position whose state the cache keeps, until it is reached
    pos: int = 0  # decoding: the pending token's position
    pending: int = 0


def piece_end(done: int, target: int, budget: int, grid: int) -> int:
    """Where a prompt piece from ``done`` toward ``target`` ends: ``target``, else the last grid point in budget."""

    if target - done <= budget:
        return target
    end = (done + budget) // grid * grid
    return end if end > done else min(target, (done // grid + 1) * grid)


def save_point(n: int, grid: int) -> int:
    """The last grid point before a prompt's last token: an identical prompt resumes there."""

    return (n - 1) // grid * grid


class LaneDecoder:
    """``live``, ``admit``, ``round``, ``finish`` and ``drop`` for the ``Scheduler`` (rank 0); others ``follow``."""

    def __init__(
        self,
        forward: Any,
        *,
        capacity: int,
        lanes: int = 4,
        link: Link | None = None,
        pool: Any = None,
        cache: TieredCache | None = None,
        drafter: Any = None,
        depth: Any = None,
        admission: Any = None,
        grammars: Any = None,
        eos: Sequence[int] = (),
        prefill_rows: int = 2048,
    ) -> None:
        self.local = Lanes(
            forward, lanes, capacity, pool=pool, cache=cache, drafter=drafter, depth=depth, grammars=grammars
        )
        self.forward, self.capacity = forward, int(capacity)
        self.link = link or LocalLink()
        self.pool, self.cache, self.drafter, self.admission = pool, cache, drafter, admission
        self.eos = tuple(int(t) for t in eos)
        self.prefill_rows = max(1, int(prefill_rows))
        self.grid = max(1, int(getattr(forward, "grid", 1) or 1))
        self.slack = 1 + (int(drafter.block) if drafter is not None else 0)  # rows a window writes past the reply
        self.streams: dict[int, Stream] = {}  # decoding
        self.filling: list[Stream] = []  # admitted, prompts still prefilling, oldest first
        self.plans: dict[int, Plan] = {}
        self.free = list(range(int(lanes)))
        self.commits: list[tuple[int, int, int]] = []
        self.next_id = 0
        self.rounds = 0
        self.broken: Exception | None = None

    def _send(self, kind: int, ints: list[int], **kw):
        self.link.send(kind, ints)
        return self.local.apply(kind, ints, **kw)

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def follow(self) -> None:
        """Ranks past 0: mirror rank 0 until it stops."""

        self.local.follow(self.link)

    def stop(self) -> None:
        """Rank 0: the other ranks leave ``follow``."""

        self._send(STOP, [])

    # -- admission -----------------------------------------------------------------------------------------------
    def admit(self, s: Stream) -> None:
        """A lane for the request, resumed from the longest kept prefix of its prompt; NoRoom while memory is short."""

        if self.broken is not None:
            raise RuntimeError("an earlier round failed on two ranks; restart the engine") from self.broken
        n = len(s.prompt)
        room = self.capacity - n - self.slack
        if n < 1 or room < 1:
            raise ValueError(f"a prompt of {n} tokens leaves no room in a {self.capacity}-token lane")
        s.count = min(s.count, room)
        if not self.free:
            raise NoRoom("every lane is busy")
        # "draft": false never resumes: the serial reference is computed from the prompt alone, not from kept state
        hit = self.cache.find(s.prompt) if self.cache is not None and s.draft else None
        cached, tier = hit if hit is not None else (0, -1)
        quota = self._quota(n + s.count + self.slack)
        need = self._need(n, s.count)
        # with nothing live it starts: no stream would finish to free memory, so waiting could only fail it
        if self.admission is not None and self.live() and not self.admission.fits(need):
            raise NoRoom(self.admission.why(need))
        held = None
        if tier >= 0:  # read and checked here, before any rank is told to resume
            try:
                held = self.cache.load(s.prompt, cached, tier)
            except ValueError:
                cached, tier = 0, -1
        self._room(quota, s.prompt[:cached] if tier < 0 else [])
        lane = self.free.pop(0)
        s.sid, s.cached = self.next_id, cached
        self.next_id += 1
        msg = [lane, quota, cached, tier, *pack_sampling(s.sampling), n, *s.prompt, *pack(s.constraint)]
        try:
            self._send(ADMIT, msg, constraint=s.constraint, held=held)
        except Exception:
            self.free.insert(0, lane)
            raise
        point = save_point(n, self.grid)
        save = point if self.cache is not None and point > cached else None
        self.plans[s.sid] = Plan(s, lane, cached, save)
        self.filling.append(s)

    def _quota(self, tokens: int) -> int:
        """The pages a lane of ``tokens`` positions reserves; 0 without a pool."""

        if self.pool is None:
            return 0
        quota = self.pool.need(tokens)
        if quota > self.pool.pages:
            raise ValueError(f"a request needs {quota} pages; the page pool has {self.pool.pages}")
        return quota

    def _need(self, prompt_len: int, max_new: int) -> int:
        """Bytes the request allocates outside the page pool (its pages exist already): ``request_bytes``, else 0."""

        fn = getattr(self.forward, "request_bytes", None)
        return int(fn(prompt_len, max_new)) if callable(fn) else 0

    def _room(self, quota: int, shared: Sequence[int]) -> None:
        """Kept entries are evicted until the pool can promise the quota (never the one the lane resumes)."""

        if self.pool is None:
            return
        own = len(shared) // self.pool.page_tokens
        while self.pool.available() < quota - own:
            index = self.cache.victim(keep=(shared,) if shared else ()) if self.cache is not None else None
            if index is None:
                raise NoRoom("the page pool is held by live streams; the request waits for one to finish")
            self._send(EVICT, [index])

    # -- rounds --------------------------------------------------------------------------------------------------
    def round(self) -> list[Stream]:
        """Prompt pieces within the budget and a window for each decoding lane, in one message."""

        if self.broken is not None:
            raise RuntimeError("an earlier round failed on two ranks") from self.broken
        pieces, saves, finals = self._pieces()
        windows, count = self._windows()
        if not (self.commits or pieces or saves or finals or windows):
            return []
        commits, self.commits = self.commits, []
        msg = [
            len(commits),
            *[v for c in commits for v in c],
            len(pieces),
            *[v for p in pieces for v in p],
            len(saves),
            *saves,
            len(finals),
            *finals,
            count,
            len(windows),
            *[v for w in windows for v in w],
        ]
        t0 = time.perf_counter()
        res = self._send(ROUND, msg)
        took = time.perf_counter() - t0
        self.rounds += 1
        by_lane = {p.lane: p for p in self.plans.values()}
        for lane, _, _ in pieces:
            by_lane[lane].stream.prefill_s += took
        done = []
        for lane, exc in res.errors.items():
            s = by_lane[lane].stream
            s.error, s.done = exc, True
            done.append(s)
        if res.cand is not None:
            for rows, cand in zip(res.rows, split(res.cand, res.rows)):
                s = self._sample(by_lane[rows.lane], rows, cand)
                if s.done:
                    done.append(s)
        return done

    def _pieces(self) -> tuple[list, list, list]:
        """Foreground prompts first, then oldest, within ``prefill_rows``; a piece stops at its prompt's save point."""

        pieces, saves, finals, budget = [], [], [], self.prefill_rows
        for s in sorted(self.filling, key=lambda x: x.background):
            p = self.plans[s.sid]
            last = len(s.prompt) - 1
            target = p.save_at if p.save_at is not None else last
            if p.done < target and budget > 0:
                end = piece_end(p.done, target, budget, self.grid)
                pieces.append((p.lane, p.done, end))
                budget -= end - p.done
                p.done = end
            if p.save_at is not None and p.done == p.save_at:
                saves.append(p.lane)
                p.save_at = None
            if p.done == last and p.save_at is None:
                finals.append(p.lane)
        for lane in finals:
            p = next(x for x in self.plans.values() if x.lane == lane)
            s = p.stream
            self.filling.remove(s)
            self.streams[s.sid] = s
            p.pos, p.pending = len(s.prompt) - 1, int(s.prompt[-1])
            s.context, s.started = list(s.prompt), time.perf_counter()
        return pieces, saves, finals

    def _windows(self) -> tuple[list, int]:
        windows, count = [], 1
        vocab = int(self.forward.vocab)
        for s in self.streams.values():
            if s.done:
                continue
            p = self.plans[s.sid]
            most = min(int(self.drafter.block), s.count - len(s.out) - 1) if s.draft and self.drafter else 0
            windows.append((p.lane, p.pos, p.pending, max(0, most)))
            smp = s.sampling
            if smp is not None and smp.temperature > 0:
                count = max(count, min(vocab, int(smp.top_k) + MARGIN) if smp.top_k else vocab)
        return windows, count

    def _sample(self, p: Plan, rows, cand: Candidates) -> Stream:
        """Choose each row's token, keep the drafts up to the first miss, and queue the commit for the next message."""

        s = p.stream
        positions = [rows.start + 1 + i for i in range(len(rows.tokens))]
        chosen = choose(cand, positions, s.sampling)
        ends = self.eos if s.stop_eos else ()
        path, terminal = accept(rows.tokens, chain(len(rows.tokens)), chosen, s.count - len(s.out), ends)
        new = [rows.tokens[r] for r in path[1:]] + [terminal]
        self.commits.append((p.lane, len(path) - 1, terminal))
        s.committed.extend(rows.tokens[r] for r in path)
        s.counted(len(rows.tokens))
        p.pos, p.pending = rows.start + len(path), terminal
        s.take(new, ends)
        return s

    # -- endings -------------------------------------------------------------------------------------------------
    def finish(self, done: list[Stream]) -> None:
        """Free the lanes of finished (or yielding) streams on every rank; their last windows are never committed."""

        lanes = []
        for s in done:
            p = self.plans.pop(s.sid, None)
            if p is None:
                continue
            self.streams.pop(s.sid, None)
            if s in self.filling:
                self.filling.remove(s)
            lanes.append(p.lane)
        if lanes:
            self.commits = [c for c in self.commits if c[0] not in lanes]
            self._send(DONE, lanes)
            self.free = sorted(self.free + lanes)
        if not self.live():
            self.link.idle()

    def drop(self) -> list[Stream]:
        """After a failed round: every admitted stream; one rank starts clean, two can no longer be trusted to agree."""

        gone = list(self.streams.values()) + list(self.filling)  # none has had its reply yet
        lanes = [p.lane for p in self.plans.values()]
        self.streams, self.filling, self.plans, self.commits = {}, [], {}, []
        if isinstance(self.link, LocalLink):
            for lane in lanes:
                self.local.apply(DONE, [lane])
            self.free = sorted(self.free + lanes)
        else:
            self.broken = RuntimeError("a round failed")
        return gone


def choose(cand: Candidates, positions: Sequence[int], sampling: Any) -> list[int]:
    """Greedy: the largest logit, ties to the lower id; else ``exact_sampling.choose_rows`` at the rows' positions."""

    values, ids = np.asarray(cand.values, dtype=np.float32), np.asarray(cand.ids, dtype=np.int64)
    if sampling is None or sampling.temperature <= 0:
        return [int(ids[r, np.lexsort((ids[r], -values[r]))[0]]) for r in range(len(ids))]
    return choose_rows(values, ids, positions, sampling)


__all__ = ["LaneDecoder", "Plan", "choose", "piece_end", "save_point"]
