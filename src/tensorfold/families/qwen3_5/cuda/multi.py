"""The 27B's concurrent rounds: every stream commits exactly its own path, so it equals its serial decoding."""

from __future__ import annotations

import time

import torch

from tensorfold.cuda.markers import MIN_GAP
from tensorfold.cuda.sampling import sample_streams
from tensorfold.cuda.streams import PrefixCache, Stream, accept

from .decode import CopyIndex, clone_state
from .decode_tp import _sample_split, _share, first_token, pack_sampling, unpack_sampling
from .draft_tree import allocate
from .forward import State, _paths, commit_streams, multi_tree_forward, path_indices, reserve
from .prefill import prefill_state
from .weights import Weights

ADMIT, ROUND, DONE, FILL = 1, 2, 3, 4   # rank 0's messages
COPY, TREE, ONE = 0, 1, 2               # a stream's window this round
STEP = 1024                             # prompt rows a prefill step takes while other streams decode


def private(st: State, rows: int) -> State:
    """A copy of a committed state with its own attention caches of ``rows`` rows (rows below ``pos`` copied in)."""

    other = clone_state(st)
    reserve(other, rows)                              # new buffers now: prefill never grows or reallocates them
    return other


def own(snap):
    """A drafter snapshot with its own per-layer lists (``add_taps_streams`` replaces their entries in place)."""

    return None if snap is None else (list(snap[0]), list(snap[1]), snap[2], snap[3])


def kept(st: State) -> State:
    """A cached state: attention rows below ``pos`` viewed in place (commits only write past a stream's ``pos``), DeltaNet states copied (decoding replays them in place)."""

    other = clone_state(st)
    other.kv = [None if kv is None else (kv[0][:st.pos], kv[1][:st.pos]) for kv in st.kv]
    other.rec = [None if r is None else r.clone() for r in st.rec]
    other.conv = [None if c is None else c.clone() for c in st.conv]
    return other


def _unflatten(flat: list[int], pairs: bool) -> list:
    """Length-prefixed lists (``pairs``: each a window's tokens then its parents)."""

    out, i = [], 0
    while i < len(flat):
        n = flat[i]
        if pairs:
            out.append((flat[i + 1:i + 1 + n], flat[i + 1 + n:i + 1 + 2 * n]))
            i += 1 + 2 * n
        else:
            out.append(flat[i + 1:i + 1 + n])
            i += 1 + n
    return out


class MultiDecoder:
    """The ``Scheduler``'s decoder on one GPU or as ``rank`` of two; a stream's window holds at most 16 rows."""

    def __init__(self, w: Weights, draft=None, *, max_rows: int = 16, allow_copy: bool = True, stop_eos: bool = True,
                 keep: int = 8, rank: int = 0, world: int = 1, context: int = 0, points=None) -> None:
        if not 1 <= max_rows <= 16:
            raise ValueError("a stream's window is 1 to 16 rows (the multi-stream GDN tree kernel's limit)")
        self.w, self.draft, self.max_rows, self.allow_copy = w, draft, max_rows, allow_copy
        self.context = context                                # prompt plus reply tokens a stream holds (0: no bound)
        self.eos = tuple(w.config.eos) if stop_eos else ()
        self.rank, self.world, self.device = rank, world, w.norm.device
        self.split = world == 2 and 2 * w.head.n == w.config.vocab       # each rank holds half the head
        self.drafts = draft is not None and (rank == 0 or getattr(draft, "world", 1) == 2)
        self.streams: dict[int, Stream] = {}                  # decoding
        self.filling: list[Stream] = []                        # admitted, prompts still prefilling (oldest first)
        self.points = points                                  # a prompt's message starts to keep states at, or None
        self.cache = PrefixCache(keep)
        self.next_id = 0
        self.broken: Exception | None = None
        self.costs: list[tuple[int, float]] | None = None     # (rows, ms) of the forward: tree widths by the curve
        self.overhead = (8.0, 1.5)                            # a round's other ms: fixed, and per stream

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def _send(self, values: list[int]) -> None:
        if self.world == 2 and self.broken is None:
            _share(values, 0, self.device)

    def _check(self) -> None:
        if self.broken is not None:
            raise RuntimeError("the two ranks are out of step after an error; restart both") from self.broken

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Queue a request on the longest cached prefix of its prompt; rounds prefill the rest a step at a time."""

        self._check()
        if self.context:
            room = self.context - len(s.prompt) - 1
            if room < 1:
                raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.context}-token "
                                 "context (--context)")
            s.count = min(s.count, room)
        hit = self.cache.longest(s.prompt) if s.draft else None
        s.sid, s.cached = self.next_id, len(hit[0]) if hit else 0
        self.next_id += 1
        self._send([ADMIT, s.sid, s.count, int(s.draft), s.cached, *pack_sampling(s.sampling)])
        self._send(list(s.prompt))
        self._queue(s, hit)

    def _queue(self, s: Stream, hit) -> None:
        drafter = self.draft if s.draft and self.drafts else None
        need = len(s.prompt) + s.count                       # the most a stream's attention caches ever hold
        state = private(hit[1] if hit else State(self.w), min(self.context, need) if self.context else need)
        s.st = state
        s.snap = None if drafter is None else own(hit[2]) if hit and hit[2] is not None else \
            ([None] * drafter.layers, [None] * drafter.layers, 0, 0)
        s.stops = ([p for p in self.points(s.prompt) if p >= state.pos + MIN_GAP]
                   if self.points is not None and s.draft else [])
        self.filling.append(s)

    def _fill(self) -> list[Stream]:
        """One prefill step for the oldest queued prompt: to its next kept state, or STEP rows while others decode."""

        s = self.filling[0]
        pos, n = s.st.pos, len(s.prompt)
        stop = next((p for p in s.stops if p > pos), n)
        if any(not x.done for x in self.streams.values()):
            stop = min(stop, pos + STEP)
        self._send([FILL, s.sid, stop])
        try:
            first = self._step(s, stop)
        except Exception as exc:                 # noqa: BLE001  (one GPU: this request fails, the others go on)
            if self.world == 2:
                raise
            self.filling = [x for x in self.filling if x is not s]
            s.error, s.done = exc, True
            return [s]
        if first is None:
            return []
        s.take([first], self._ends(s))
        return [s] if s.done else []

    def _step(self, s: Stream, stop: int) -> int | None:
        """Prefill prompt[pos:stop] (the same bits for any stops); at the end, sample the first token and start decoding."""

        t0 = time.perf_counter()
        drafter = self.draft if s.draft and self.drafts else None
        try:
            if drafter is not None:
                drafter.restore(s.snap)
            normed = prefill_state(self.w, s.prompt[:stop], s.st, tp=self.world == 2, draft=drafter)
            if drafter is not None:
                s.snap = drafter.snapshot()
            if stop in s.stops:
                self.cache.add(list(s.prompt[:stop]), kept(s.st), own(s.snap))
            first = None if stop < len(s.prompt) else \
                first_token(self.w, normed, len(s.prompt), s.sampling, self.rank, self.world)
            if first is not None and s.draft and not (s.stops and len(s.prompt) - s.stops[-1] < MIN_GAP):
                self.cache.add(list(s.prompt), kept(s.st), own(s.snap))   # a message start just before the end covers it
        except Exception as exc:
            if self.world == 2:
                self.broken = exc
            raise
        finally:
            s.prefill_s += time.perf_counter() - t0
        if first is None:
            return None
        s.copies = CopyIndex() if self.allow_copy and s.draft and self.rank == 0 else None
        s.context = list(s.prompt)
        s.started = time.perf_counter()
        self.filling = [x for x in self.filling if x is not s]
        self.streams[s.sid] = s                       # after every step that can fail: a failure leaves it queued
        return first

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """A prefill step for the oldest queued prompt, then one round over the decoding streams; returns the finished."""

        self._check()
        done = self._fill() if self.filling else []
        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return done
        copied: dict[int, list[int]] = {}
        plan = [(s.sid, self._mode(s, copied), s.out[-1], len(s.context)) for s in live]
        self._send([ROUND, len(plan), *[x for item in plan for x in item]])
        wins, record, taps, starts, sampled = self._verify(plan, copied)
        paths, ends = [], []
        for s, (tokens, parents), rows in zip(live, wins, sampled):
            path, end = accept(tokens, parents, rows, s.count - len(s.out), self._ends(s))
            paths.append(path)
            ends.append(end)
        self._send([x for path in paths for x in (len(path), *path)])
        self._commit(plan, wins, record, taps, starts, paths)
        for s, (tokens, _), path, end in zip(live, wins, paths, ends):
            s.take([tokens[r] for r in path[1:]] + [end], self._ends(s))
        return done + [s for s in live if s.done]

    def _ends(self, s: Stream) -> tuple[int, ...]:
        """The end tokens that end this stream: none when its request ignores them (rank 1 follows rank 0's paths)."""

        return self.eos if s.stop_eos else ()

    def _mode(self, s: Stream, copied: dict[int, list[int]]) -> int:
        if not s.draft:
            return ONE
        copied[s.sid] = s.copies.propose(s.context, self.max_rows - 1) if s.copies is not None else []
        return COPY if copied[s.sid] else (TREE if self.draft is not None else ONE)

    def _trees(self, plan, blocks) -> dict[int, tuple[list[int], list[int], list[float]]]:
        """Rank 0: each tree stream's nodes, parents and path scores, in the policy's pop order."""

        return {sid: self.draft.finish_tree(blocks[sid], length, self.max_rows - 1, self.streams[sid].sampling)
                for sid, mode, _, length in plan if mode == TREE and blocks.get(sid) is not None}

    def _cost(self, rows: int) -> float:
        pts = self.costs
        for (r0, t0), (r1, t1) in zip(pts, pts[1:]):
            if rows <= r1 or (r1, t1) == pts[-1]:
                return t0 + (t1 - t0) * (rows - r0) / (r1 - r0)
        return pts[0][1]

    def _windows(self, plan, copied, blocks) -> list[tuple[list[int], list[int]]]:
        """Rank 0: every stream's window; with a cost curve, the trees' widths by expected tokens a millisecond."""

        trees = self._trees(plan, blocks)
        keep = {sid: len(t[0]) for sid, t in trees.items()}
        if self.costs is not None and trees:
            sids = list(trees)
            fixed = sum(1 + (len(copied[sid]) if mode == COPY else 0) for sid, mode, _, _ in plan)
            counts = allocate([trees[sid][2] for sid in sids], fixed, float(len(plan)), self._cost,
                              self.overhead[0] + self.overhead[1] * len(plan))
            keep = dict(zip(sids, counts))
        wins = []
        for sid, mode, pending, _ in plan:
            guesses, parents = [], []
            if mode == COPY:
                guesses, parents = copied[sid], list(range(-1, len(copied[sid]) - 1))
            elif sid in trees:
                guesses, parents = trees[sid][0][:keep[sid]], trees[sid][1][:keep[sid]]
            wins.append(([pending] + list(guesses), [-1] + [0 if p < 0 else p + 1 for p in parents]))
        return wins

    @torch.no_grad()
    def calibrate(self, streams: int, reps: int = 3) -> None:
        """Time the forward at the row counts ``streams`` windows bring; every rank runs the same forwards."""

        st, points = State(self.w), []
        rows = sorted({r for r in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
                       if r <= 16 * streams} | {16 * streams})
        for r in rows:
            n = -(-r // 16)
            sizes = [r // n + (i < r % n) for i in range(n)]
            wins = [([0] * k, list(range(-1, k - 1)), st) for k in sizes]
            times = []
            for i in range(reps + 1):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                multi_tree_forward(self.w, wins, full_logits=self.split or self.rank == 0, tp=self.world == 2)
                torch.cuda.synchronize()
                if i:
                    times.append(1e3 * (time.perf_counter() - t0))
            points.append((r, sorted(times)[len(times) // 2]))
        self.costs = points
        torch.cuda.empty_cache()                      # the widest windows' scratch goes back before any request

    def _verify(self, plan, copied=None):
        """Both ranks: drafter blocks, rank 0's windows, the forward and each stream's samples."""

        blocks = {}
        tree = [(sid, pending) for sid, mode, pending, _ in plan if mode == TREE] if self.drafts else []
        if tree:                                      # every stream's block in one drafter pass
            launched = self.draft.launch_blocks([self.streams[sid].snap for sid, _ in tree],
                                                [pending for _, pending in tree], self.max_rows - 1)
            blocks = {sid: block for (sid, _), block in zip(tree, launched)}
        if self.rank == 0:
            wins = self._windows(plan, copied, blocks)
            self._send([x for tokens, parents in wins for x in (len(tokens), *tokens, *parents)])
        else:
            wins = _unflatten(_share(None, 1, self.device), pairs=True)
        states = [self.streams[item[0]].st for item in plan]
        taps_wanted = self.drafts and any(self.streams[item[0]].draft for item in plan)
        logits, record, taps, starts = multi_tree_forward(
            self.w, [(t, p, st) for (t, p), st in zip(wins, states)],
            full_logits=self.split or self.rank == 0, tp=self.world == 2, capture_taps=taps_wanted)
        positions = [[st.pos + d + 1 for d in _paths(parents)[0]] for (_, parents), st in zip(wins, states)]
        samplings = [self.streams[item[0]].sampling for item in plan]
        if self.split:                                # both ranks gather their halves' candidates
            sampled = [_sample_split(logits[starts[k]:starts[k + 1]], positions[k], samplings[k], self.rank)
                       for k in range(len(plan))]
        else:
            sampled = sample_streams(logits, starts, positions, samplings) if self.rank == 0 else [None] * len(plan)
        return wins, record, taps, starts, sampled

    def _commit(self, plan, wins, record, taps, starts, paths) -> None:
        rows = [[starts[k] + r for r in path] for k, path in enumerate(paths)]
        indices = path_indices(record, rows)
        streams = [self.streams[item[0]] for item in plan]
        commit_streams([s.st for s in streams], record, rows, indices, in_place=True)
        drafting = []
        for s, (tokens, _), path, (_, _, take) in zip(streams, wins, paths, indices):
            s.committed.extend(tokens[r] for r in path)
            if s.draft and self.drafts:
                drafting.append((s, taps.index_select(0, take)))
            s.counted(len(tokens))
        if drafting:                                  # every stream's kept rows into its drafter context, one pass
            snaps = self.draft.add_taps_streams([s.snap for s, _ in drafting], [t for _, t in drafting])
            for (s, _), snap in zip(drafting, snaps):
                s.snap = snap

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams on every rank (their prompt-end states joined the prefix cache at admission)."""

        if done:
            self._send([DONE, len(done), *[s.sid for s in done]])
            for s in done:
                self._finish(s.sid)

    def _finish(self, sid: int) -> None:
        self.streams.pop(sid, None)

    def drop(self) -> list[Stream]:
        """After an error in a round: forget the live streams (two ranks can no longer be trusted to agree)."""

        live = [s for s in self.streams.values() if not s.done]
        for s in live:
            del self.streams[s.sid]
        live += self.filling
        self.filling = []
        if self.world == 2 and self.broken is None:
            self.broken = RuntimeError("a round failed")
        return live

    @torch.no_grad()
    def follow(self) -> None:
        """Rank 1: mirror rank 0's admissions, rounds and completions until rank 0 sends an empty message."""

        while True:
            msg = _share(None, 1, self.device)
            if not msg:
                return
            if msg[0] == ADMIT:
                sid, count, draft, cached = msg[1:5]
                s = Stream(_share(None, 1, self.device), count, unpack_sampling(msg[5:19]), draft=bool(draft), sid=sid)
                hit = self.cache.named(s.prompt, cached) if cached else None
                if cached and hit is None:
                    raise RuntimeError(f"rank 1 has no cached state for the {cached} tokens rank 0 resumes from")
                s.cached = cached
                self._queue(s, hit)
            elif msg[0] == FILL:
                self._step(next(s for s in self.filling if s.sid == msg[1]), msg[2])
            elif msg[0] == ROUND:
                plan = [tuple(msg[2 + 4 * i:6 + 4 * i]) for i in range(msg[1])]
                wins, record, taps, starts, _ = self._verify(plan)
                paths = _unflatten(_share(None, 1, self.device), pairs=False)
                self._commit(plan, wins, record, taps, starts, paths)
            elif msg[0] == DONE:
                for sid in msg[2:2 + msg[1]]:
                    self._finish(sid)
