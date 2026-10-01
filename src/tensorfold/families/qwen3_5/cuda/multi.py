"""The 27B's concurrent rounds: every stream commits exactly its own path, so it equals its serial decoding."""

from __future__ import annotations

import time

import torch

from tensorfold.cuda.capacity import available_bytes
from tensorfold.cuda.markers import MIN_GAP
from tensorfold.cuda.memory_gate import MemoryGate, NoRoom, torch_live
from tensorfold.cuda.sampling import sample_streams
from tensorfold.cuda.streams import PrefixCache, Stream, accept, next_fill
from tensorfold.engine.grammar import GrammarError, pack

from .decode import CopyIndex, clone_state
from .decode_tp import SAMPLING_WORDS as W, _sample_split, _share, first_token, pack_sampling, unpack_sampling
from .draft_tree import allocate
from .engine import entry_end
from .forward import State, _paths, commit_streams, multi_tree_forward, path_indices, reserve
from .prefill import CHUNK, Piece, prefill_batch, prefill_state
from .weights import Weights

ADMIT, ROUND, DONE, FILL, FILLS = 1, 2, 3, 4, 5   # rank 0's messages
COPY, TREE, ONE = 0, 1, 2               # a stream's window this round
STEP = 1024                             # prompt rows a prefill step takes while other streams decode
GROW = 8192                             # rows a stream's attention caches grow by at a time (one GPU)
GIB = 1024**3
TIMED = 16                              # the last rounds whose time beside the forward sets a stream count's overhead
DEPTH_CHIPS = ((12, 0),)                # tuned planning (measured overhead, curve steps, the block); else 0.6.0's
BATCH = True                            # queued foreground prompts' steps share one prefill forward


def calibration_rows(streams: int, steps: bool = True) -> list[int]:
    """Row counts the startup curve times: ``streams`` full windows and (``steps``) a point past each tile step."""

    grid = [r for r in (1, 2, 4, 8, 12, 16, 17, 24, 32, 33, 48, 64, 65, 96, 128, 129, 192, 256, 257, 384, 512)
            if steps or r not in (17, 33, 65, 129, 257)]
    return sorted({r for r in grid if r <= 16 * streams} | {16 * streams})


def private(st: State, rows: int) -> State:
    """A copy of a committed state with its own attention caches of ``rows`` rows (rows below ``pos`` copied in)."""

    other = clone_state(st)
    other.kv = list(st.kv)                            # its own list: the buffers reserved next are this stream's
    reserve(other, rows)                              # new buffers now: prefill never grows or reallocates them
    return other


def own(snap):
    """A drafter snapshot with its own per-layer lists (``add_taps_streams`` replaces their entries in place)."""

    return None if snap is None else (list(snap[0]), list(snap[1]), snap[2], snap[3])


def viewed(st: State) -> State:
    """A kept state whose DeltaNet states are its own already: attention rows below ``pos`` viewed in place."""

    st.kv = [None if kv is None else (kv[0][:st.pos], kv[1][:st.pos]) for kv in st.kv]
    return st


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

    memory_gate: MemoryGate | None = None     # one GPU: streams' caches grow by use (two ranks reserve up front)
    block: int = 16                           # rows of the drafter's next block with several streams (pending, masks)
    depth: bool = True                        # whether the block follows the trees here (DEPTH_CHIPS)
    spent: dict | None = None                 # streams -> the last rounds' ms beside the forward
    last: tuple | None = None                 # (start, streams, rows) of the round before

    def __init__(self, w: Weights, draft=None, *, max_rows: int = 16, allow_copy: bool = True, stop_eos: bool = True,
                 keep: int = 8, rank: int = 0, world: int = 1, context: int = 0, points=None, vision=None) -> None:
        if not 1 <= max_rows <= 16:
            raise ValueError("a stream's window is 1 to 16 rows (the multi-stream GDN tree kernel's limit)")
        self.w, self.draft, self.max_rows, self.allow_copy = w, draft, max_rows, allow_copy
        self.vision = vision
        self.context = context                                # prompt plus reply tokens a stream holds (0: no bound)
        self.eos = tuple(w.config.eos) if stop_eos else ()
        self.rank, self.world, self.device = rank, world, w.norm.device
        self.depth = torch.cuda.is_available() and tuple(torch.cuda.get_device_capability(self.device)) in DEPTH_CHIPS
        self.split = world == 2 and 2 * w.head.n == w.config.vocab       # each rank holds half the head
        self.drafts = draft is not None and (rank == 0 or getattr(draft, "world", 1) == 2)
        self.streams: dict[int, Stream] = {}                  # decoding
        self.filling: list[Stream] = []                        # admitted, prompts still prefilling (oldest first)
        self.points = points                                  # a prompt's message starts to keep states at, or None
        self.cache = PrefixCache(keep)
        self.next_id = 0
        self.broken: Exception | None = None
        self.costs: list[tuple[int, float]] | None = None     # (rows, ms) of the forward: tree widths by the curve
        self.overhead = (8.0, 1.5)                            # a round's other ms until measured: fixed, per stream
        self.block = max_rows
        # one GPU: a stream's caches hold its prompt, then grow a step at a time while the gate has room
        c, att = w.config, sum(1 for layer in getattr(w, "layers", ()) if not layer.linear)
        self.layer_bytes = 2 * getattr(c, "kv_heads", 0) * getattr(c, "head_dim", 0) * 2     # a row of one layer
        self.row_bytes = att * self.layer_bytes
        self.memory_gate = (MemoryGate(1 << 62, reserve=2 * GIB, live=torch_live(torch, available_bytes))
                     if world == 1 and torch.cuda.is_available() else None)

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
        if self.memory_gate is not None:
            self._room(s)
        prepared = getattr(s, "vision", None)
        if prepared is not None and self.vision is None:
            raise ValueError("image inputs require starting this engine with --vision")
        encoded = self.vision.encode(prepared, s.prompt) if prepared is not None else None
        hit = self.cache.longest(s.prompt) if s.draft and encoded is None else None
        s.sid, s.cached = self.next_id, len(hit[0]) if hit else 0
        self.next_id += 1
        # the request's grammar rides after the fields (rank 1 compiles the same): a plain ADMIT is unchanged
        self._send([ADMIT, s.sid, s.count, int(s.draft), s.cached, *pack_sampling(s.sampling),
                    int(encoded is not None), *pack(s.constraint)])
        self._send(list(s.prompt))
        if self.world == 2 and encoded is not None:     # rank 1 takes the image rows as rank 0 encoded them
            from tensorfold.vision.qwen_cuda import broadcast_encoded

            encoded = broadcast_encoded(encoded, 0, self.device, hidden=self.w.config.hidden,
                                        prompt_length=len(s.prompt))
        s.vision = encoded
        self._queue(s, hit)

    def _queue(self, s: Stream, hit) -> None:
        drafter = self.draft if s.draft and self.drafts else None
        rows = self._most(s) if self.memory_gate is None else self._first(s)
        state = private(hit[1] if hit else State(self.w), rows)
        s.st = state
        s.snap = None if drafter is None else own(hit[2]) if hit and hit[2] is not None else \
            ([None] * drafter.layers, [None] * drafter.layers, 0, 0)
        s.stops = ([p for p in self.points(s.prompt) if p >= state.pos + MIN_GAP]
                   if self.points is not None and s.draft and s.vision is None else [])    # image prompts keep none
        self.filling.append(s)

    def _most(self, s: Stream) -> int:
        """The most a stream's attention caches ever hold: its prompt and reply, within the context."""

        need = len(s.prompt) + s.count
        return min(self.context, need) if self.context else need

    def _first(self, s: Stream) -> int:
        """The rows a stream's caches start with (one GPU): its prompt and a window, a step at a time."""

        return min(self._most(s), -(-(len(s.prompt) + self.max_rows + 2) // GROW) * GROW)

    def _room(self, s: Stream) -> None:
        """A new prompt's rows fit beside the live streams, cached prompt ends going first; else it waits (NoRoom)."""

        if any(x.waiting for x in self.streams.values()):
            raise NoRoom("streams already wait for memory; a new request waits until one finishes")
        while not self.memory_gate.fits(self._first(s) * self.row_bytes):
            if not self.cache.evict():
                if not self.live():
                    return                          # alone: startup fitted one stream's whole window
                raise NoRoom(f"a {len(s.prompt)}-token prompt waits for memory until a live stream finishes")
            torch.cuda.empty_cache()

    def _grow(self, st: State, have: int, size: int, alone: bool) -> bool:
        """Grow ``st``'s caches from ``have`` to ``size`` rows while the gate has room (a layer's copy at a time)."""

        while not self.memory_gate.fits((size - have) * self.row_bytes + size * self.layer_bytes):
            if not self.cache.evict():
                if alone:
                    break                           # startup fitted one stream's whole window
                return False
            torch.cuda.empty_cache()
        reserve(st, size)
        torch.cuda.empty_cache()                    # the old buffers back to the system: MemAvailable stays true
        return True

    def _make_room(self, live: list[Stream]) -> list[Stream]:
        """Before a round: grow window caches oldest-first; no-growth streams run; the newest may end."""

        live = sorted(live, key=lambda x: x.sid)
        blocked = False
        for s in live:
            rows = min(s.st.pos + self.max_rows + 2, self._most(s))   # a stream never commits past its prompt and reply
            have = next((kv[0].shape[0] for kv in s.st.kv if kv is not None), rows)
            if rows <= have:
                s.waiting = False
                continue
            size = max(rows, min(self._most(s), -(-rows // GROW) * GROW))
            s.waiting = blocked or not self._grow(s.st, have, size, alone=len(live) == 1 and not self.filling)
            blocked = blocked or s.waiting
        if len(live) > 1 and live[0].waiting:          # even the oldest can't grow: the newest ends
            newest = live[-1]
            newest.error = RuntimeError(
                f"This server ran out of memory with {len(live)} streams decoding, so the newest (this request, after "
                f"{len(newest.out)} tokens) was stopped for the older ones to finish. Retry it, shorten the prompt or "
                "max_tokens, or start the server with a smaller --parallel.")
            newest.done, newest.waiting = True, False
            self.memory_gate.ends += 1
            self.streams.pop(newest.sid, None)
            newest.st = None                         # its caches go now (a cached prompt end may still view them)
            torch.cuda.empty_cache()
            return [newest, *self._make_room(live[:-1])]
        self.memory_gate.waits += any(s.waiting for s in live)
        return []

    def _fill(self) -> list[Stream]:
        """Prefill queued prompts a step: several foreground ones in one forward (``_batch``), else the oldest to its
        next kept state, or STEP rows while others decode."""

        batch = self._batch() if BATCH and sum(not x.background for x in self.filling) > 1 else []
        if len(batch) > 1:
            return self._fill_batch(batch)
        s = next_fill(self.filling)
        pos, n = s.st.pos, len(s.prompt)
        stop = next((p for p in s.stops if p > pos), n)
        if s.background or any(not x.done for x in self.streams.values()):   # a later foreground prompt waits one step
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

    def _batch(self) -> list[tuple[Stream, int]]:
        """Foreground prompts for one prefill forward, oldest first, each to its next kept state or its end: STEP rows
        in all while streams decode, a forward's prompt rows otherwise; the last one in takes the rows left."""

        room = STEP if any(not x.done for x in self.streams.values()) else getattr(self.w, "prompt_rows", CHUNK)
        out = []
        for s in self.filling:
            if s.background:
                continue
            if s.vision is not None or room <= 0:          # an image prompt goes alone, in its turn
                break
            pos = s.st.pos
            stop = min(next((p for p in s.stops if p > pos), len(s.prompt)), pos + room)
            out.append((s, stop))
            room -= stop - pos
        return out

    def _fill_batch(self, batch: list[tuple[Stream, int]]) -> list[Stream]:
        """``_fill`` for several prompts at once; on one GPU an error ends each of them alone, as one prompt's would."""

        self._send([FILLS, len(batch), *[x for s, stop in batch for x in (s.sid, stop)]])
        try:
            firsts = self._steps(batch)
        except Exception as exc:                 # noqa: BLE001  (one GPU: these requests fail, the others go on)
            if self.world == 2:
                raise
            failed = {id(s) for s, _ in batch}
            self.filling = [x for x in self.filling if id(x) not in failed]
            for s, _ in batch:
                s.error, s.done = exc, True
            return [s for s, _ in batch]
        done = []
        for (s, _), first in zip(batch, firsts):
            if first is not None:
                s.take([first], self._ends(s))
                if s.done:
                    done.append(s)
        return done

    def _steps(self, batch: list[tuple[Stream, int]]) -> list[int | None]:
        """``_step`` for several streams in one forward (``prefill_batch``): each stream's states, kept entries and
        first token have the bits ``_step`` gives it alone."""

        t0 = time.perf_counter()
        drafter = self.draft if self.drafts else None
        ends, pieces = [], []
        for s, stop in batch:
            n = len(s.prompt)
            end = entry_end(s.prompt) if (stop == n and s.draft and s.vision is None
                                          and not (s.stops and n - s.stops[-1] < MIN_GAP)) else None
            ends.append(end)
            pieces.append(Piece(s.prompt[:stop], s.st, end, s.snap if s.draft and drafter is not None else None))
        firsts = []
        try:
            outs = prefill_batch(self.w, pieces, tp=self.world == 2, draft=drafter)
            for (s, stop), end, (normed, at, snap) in zip(batch, ends, outs):
                if snap is not None:
                    s.snap = snap
                if stop in s.stops:
                    self.cache.add(list(s.prompt[:stop]), kept(s.st), own(s.snap))
                n = len(s.prompt)
                firsts.append(None if stop < n else first_token(self.w, normed, n, s.sampling, self.rank, self.world,
                                                                s.constraint))
                if end is not None:
                    self.cache.add(list(s.prompt[:end]), viewed(at[0]) if end < n else kept(at[0]), own(at[1]))
        except Exception as exc:
            if self.world == 2:
                self.broken = exc
            raise
        finally:
            spent = time.perf_counter() - t0
            for s, _ in batch:
                s.prefill_s += spent
        for (s, _), first in zip(batch, firsts):
            if first is not None:
                s.copies = CopyIndex() if self.allow_copy and s.draft and self.rank == 0 else None
                s.context = list(s.prompt)
                s.started = time.perf_counter()
                self.filling = [x for x in self.filling if x is not s]
                self.streams[s.sid] = s
        return firsts

    def _step(self, s: Stream, stop: int) -> int | None:
        """Prefill prompt[pos:stop] (the same bits for any stops); at the end, sample the first token and start decoding."""

        t0 = time.perf_counter()
        drafter = self.draft if s.draft and self.drafts else None
        try:
            if drafter is not None:
                drafter.restore(s.snap)
            n = len(s.prompt)             # the prompt end is kept one token early, unless a message start covers it
            end = entry_end(s.prompt) if (stop == n and s.draft and s.vision is None
                                          and not (s.stops and n - s.stops[-1] < MIN_GAP)) else None
            out = prefill_state(self.w, s.prompt[:stop], s.st, tp=self.world == 2, draft=drafter, keep_at=end,
                                vision=s.vision)
            normed = out if end is None else out[0]
            if drafter is not None:
                s.snap = drafter.snapshot()
                drafter.skip(0)                  # no reference of its own: rounds read and replace s.snap's context
            if stop in s.stops:
                self.cache.add(list(s.prompt[:stop]), kept(s.st), own(s.snap))
            first = None if stop < n else first_token(self.w, normed, n, s.sampling, self.rank, self.world,
                                                     s.constraint)
            if end is not None:
                at, snap = out[1]
                self.cache.add(list(s.prompt[:end]), viewed(at) if end < n else kept(at), own(snap))
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
        """A prefill step for the next queued prompt, then one round over the decoding streams; returns the finished."""

        self._check()
        self._timed(time.perf_counter())
        done = self._fill() if self.filling else []
        start = time.perf_counter()
        live = [s for s in self.streams.values() if not s.done]
        if self.memory_gate is not None:
            done += self._make_room(live)
            live = [s for s in live if not s.done and not s.waiting]
        if not live:
            self.last = None
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
        self.last = (start, len(plan), sum(len(t) for t, _ in wins)) if self.costs is not None else None
        for s, (tokens, _), path, end in zip(live, wins, paths, ends):
            new = [tokens[r] for r in path[1:]] + [end]
            if s.constraint is not None and s.error is None:
                try:
                    s.constraint.advance(new)
                except GrammarError as exc:
                    s.error = exc
            if s.error is not None:                   # its grammar failed: this request ends alone, with the error
                s.done, s.finished = True, time.perf_counter()
                continue
            s.take(new, self._ends(s))
        if all(s.done for s in live):
            self.last = None                          # the next round waits for requests: not this round's time
        return done + [s for s in live if s.done]

    def _timed(self, now: float) -> None:
        """The round before: its time beside the forward (its start to this round's, less the curve's forward)."""

        if self.last is None:
            return
        start, n, rows = self.last
        self.last = None
        if self.spent is None:
            self.spent = {}
        seen = self.spent.setdefault(n, [])
        seen.append(max(0.0, 1e3 * (now - start) - self._cost(rows)))
        del seen[:-TIMED]

    def _overhead(self, n: int) -> float:
        """A round's ms beside the forward at ``n`` streams: the median of the last rounds' (one stream: the prior)."""

        seen = (self.spent or {}).get(n) if n > 1 and self.depth else None
        if not seen or len(seen) < 4:
            return self.overhead[0] + self.overhead[1] * n
        return sorted(seen)[len(seen) // 2]

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
                              self._overhead(len(plan)))
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
        for r in calibration_rows(streams, self.depth):
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
        block = self.block if self.depth else self.max_rows    # one stream too: DFlash2 drafts best near 8 rows
        if tree:                                      # every stream's block in one drafter pass
            launched = self.draft.launch_blocks([self.streams[sid].snap for sid, _ in tree],
                                                [pending for _, pending in tree], self.max_rows - 1, block)
            blocks = {sid: block for (sid, _), block in zip(tree, launched)}
        grammars = {}
        if self.rank == 0:
            wins = self._windows(plan, copied, blocks)
            grammars = self._constrain(plan, wins)
            self._send([x for tokens, parents in wins for x in (len(tokens), *tokens, *parents)])
        else:
            wins = _unflatten(_share(None, 1, self.device), pairs=True)
            grammars = self._masks(plan, wins) if self.split else {}
        self.block = self._deepest(plan, wins, block)
        states = [self.streams[item[0]].st for item in plan]
        taps_wanted = self.drafts and any(self.streams[item[0]].draft for item in plan)
        logits, record, taps, starts = multi_tree_forward(
            self.w, [(t, p, st) for (t, p), st in zip(wins, states)],
            full_logits=self.split or self.rank == 0, tp=self.world == 2, capture_taps=taps_wanted)
        for k, window in grammars.items():          # a constrained stream's rows, each masked by its path
            self.streams[plan[k][0]].constraint.mask(logits[starts[k]:starts[k + 1]], window,
                                                     self.rank * self.w.head.n if self.split else 0)
        positions = [[st.pos + d + 1 for d in _paths(parents)[0]] for (_, parents), st in zip(wins, states)]
        samplings = [self.streams[item[0]].sampling for item in plan]
        if self.split:                                # both ranks gather their halves' candidates
            sampled = [_sample_split(logits[starts[k]:starts[k + 1]], positions[k], samplings[k], self.rank)
                       for k in range(len(plan))]
        else:
            sampled = sample_streams(logits, starts, positions, samplings) if self.rank == 0 else [None] * len(plan)
        return wins, record, taps, starts, sampled

    def _deepest(self, plan, wins, block: int) -> int:
        """The drafter's next block: past the deepest kept tree node, between its trained block and ``max_rows``."""

        deepest = max((max(_paths(parents)[0]) for (_, mode, _, _), (_, parents) in zip(plan, wins) if mode == TREE),
                      default=-1)
        if deepest < 0:
            return self.block
        floor = max(4, getattr(self.draft, "trained", 4))      # a shorter block cuts runs the drafter would have kept
        return min(self.max_rows, max(floor, deepest + (5 if deepest >= block - 1 else 2)))

    def _constrain(self, plan, wins) -> dict:
        """Rank 0: each constrained stream's window without the drafts its grammar rules out, and its rows' masks."""

        grammars = {}
        for k, (sid, *_) in enumerate(plan):
            s = self.streams[sid]
            if s.constraint is None or s.error is not None:
                continue
            try:
                window = s.constraint.window(*wins[k])
            except GrammarError as exc:              # this request ends after the round; the others go on
                s.error = exc
                continue
            wins[k] = (window.tokens, window.parents)
            grammars[k] = window
        return grammars

    def _masks(self, plan, wins) -> dict:
        """Rank 1 of a split head: each constrained stream's rows masked as rank 0 masks them (it sent the windows)."""

        grammars = {}
        for k, (sid, *_) in enumerate(plan):
            s = self.streams[sid]
            if s.constraint is not None:
                if getattr(s, "behind", False):         # the last round's final token is this window's first
                    s.constraint.advance(wins[k][0][:1])
                grammars[k] = s.constraint.window(*wins[k])
        return grammars

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
                s = Stream(_share(None, 1, self.device), count, unpack_sampling(msg[5:5 + W]), draft=bool(draft),
                           sid=sid)
                vision = None
                if len(msg) > 5 + W and msg[5 + W]:
                    from tensorfold.vision.qwen_cuda import broadcast_encoded

                    vision = broadcast_encoded(None, 1, self.device, hidden=self.w.config.hidden,
                                               prompt_length=len(s.prompt))
                packed = msg[6 + W:]
                if packed:                              # compiled here as on rank 0
                    from tensorfold.engine import grammar

                    s.constraint = grammar.compiler(self, self.model_dir, self.eos).follow(packed)
                hit = self.cache.named(s.prompt, cached) if cached else None
                if cached and hit is None:
                    raise RuntimeError(f"rank 1 has no cached state for the {cached} tokens rank 0 resumes from")
                s.cached = cached
                s.vision = vision
                self._queue(s, hit)
            elif msg[0] == FILL:
                self._step(next(s for s in self.filling if s.sid == msg[1]), msg[2])
            elif msg[0] == FILLS:
                pairs = msg[2:2 + 2 * msg[1]]
                self._steps([(next(s for s in self.filling if s.sid == sid), stop)
                             for sid, stop in zip(pairs[::2], pairs[1::2])])
            elif msg[0] == ROUND:
                plan = [tuple(msg[2 + 4 * i:6 + 4 * i]) for i in range(msg[1])]
                wins, record, taps, starts, _ = self._verify(plan)
                paths = _unflatten(_share(None, 1, self.device), pairs=False)
                self._commit(plan, wins, record, taps, starts, paths)
                for (sid, *_), (tokens, _), path in zip(plan, wins, paths):
                    s = self.streams[sid]
                    if s.constraint is not None and self.split:     # the kept drafts now, the last token next round
                        s.constraint.advance([tokens[r] for r in path[1:]])
                        s.behind = True
            elif msg[0] == DONE:
                for sid in msg[2:2 + msg[1]]:
                    self._finish(sid)
