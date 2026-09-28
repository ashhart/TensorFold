"""Qwen3.6 MoE's concurrent rounds on one GPU: every stream commits exactly its own path, so it equals its serial decoding."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from tensorfold.cuda.markers import MIN_GAP
from tensorfold.cuda.sampling import sample_rows, sample_streams
from tensorfold.cuda.streams import PrefixCache, Stream, accept
from tensorfold.families.qwen3_5.cuda.decode import CopyIndex
from tensorfold.families.qwen3_5.cuda.forward import State, _mm, commit_streams, multi_tree_forward
from tensorfold.families.qwen3_5.cuda.multi import kept, private

from .decode import COPY_ROWS, Carry, extend, mtp_round, picks
from .mtp import Cache, Head

STEP = 1024          # prompt rows a prefill step takes while other streams decode


@dataclass
class Drafts:
    """A drafting stream's MTP head: its attention cache, and the committed rows it has not absorbed (``carry``)."""

    cache: Cache
    carry: Carry


def own(mc: Cache, rows: int) -> Cache:
    """A head cache of ``rows`` slots of its own, the absorbed rows of ``mc`` (a kept one's) copied in."""

    other = object.__new__(Cache)
    other.k, other.v = mc.k.new_empty((rows, *mc.k.shape[1:])), mc.v.new_empty((rows, *mc.v.shape[1:]))
    other.k[:mc.pos], other.v[:mc.pos] = mc.k[:mc.pos], mc.v[:mc.pos]
    other.pos = mc.pos
    return other


class MultiDecoder:
    """The ``Scheduler``'s decoder: each round a prefill step for the oldest queued prompt, then every stream's MTP
    chain (or copied continuation) verified in one forward; a stream's window holds at most 16 rows. With
    ``graphs`` (a ``graphs.Graphs``), a stream decoding alone replays the one-stream engine's CUDA graphs."""

    def __init__(self, w, head: Head | None, *, depth: int, confidence: float, context: int = 0, keep: int = 3,
                 points=None, stop_eos: bool = True, graphs=None) -> None:
        if head is not None and depth > 0 and not 1 <= depth <= 15:
            raise ValueError(f"MTP drafts a round with --parallel: 1 to 15 (a window holds 16 rows), not {depth}")
        self.w, self.head = w, head if depth > 0 else None
        self.depth, self.confidence = int(depth), float(confidence)
        self.context = context                       # prompt, reply and draft slots a stream holds (0: no bound)
        self.eos = tuple(w.config.eos) if stop_eos else ()
        self.ids = head.ids.cpu().numpy() if self.head is not None and head.ids is not None else None
        self.streams: dict[int, Stream] = {}         # decoding
        self.filling: list[Stream] = []              # admitted, prompts still prefilling (oldest first)
        self.points = points                         # a prompt's message starts to keep states at, or None
        self.cache = PrefixCache(keep)               # (ids, state, (head cache, held row)) at message starts and ends
        self.next_id = 0
        self.graphs = graphs if self.head is not None else None
        self.resident: Stream | None = None          # the stream whose state is in the graphs' buffers

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Queue a request on the longest kept prefix of its prompt, its caches sized once; rounds prefill the rest."""

        if self.context:
            room = self.context - len(s.prompt) - self.depth - 1
            if room < 1:
                raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.context}-token "
                                 "context (--context)")
            s.count = min(s.count, room)
        drafting = s.draft and self.head is not None
        hit = self.cache.longest(s.prompt) if drafting else None
        need = len(s.prompt) + s.count                # the most the stream's attention caches ever hold
        s.st = private(hit[1] if hit else State(self.w), need)
        s.snap = None
        if drafting:                                  # the head writes up to depth - 1 draft slots past the committed
            cache = own(hit[2][0], need + self.depth) if hit else Cache(self.w, need + self.depth)
            s.snap = Drafts(cache, Carry(hit[2][1] if hit else None, []))
        s.sid, s.cached = self.next_id, len(hit[0]) if hit else 0
        self.next_id += 1
        s.stops = ([p for p in self.points(s.prompt) if p >= s.st.pos + MIN_GAP]
                   if self.points is not None and drafting else [])
        self.filling.append(s)

    def _fill(self) -> list[Stream]:
        """One prefill step for the oldest queued prompt: to its next kept state, or STEP rows while others decode."""

        s = self.filling[0]
        pos, n = s.st.pos, len(s.prompt)
        stop = next((p for p in s.stops if p > pos), n)
        if any(not x.done for x in self.streams.values()):
            stop = min(stop, pos + STEP)
        try:
            first = self._step(s, stop)
        except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
            self.filling = [x for x in self.filling if x is not s]
            s.error, s.done = exc, True
            return [s]
        if first is None:
            return []
        s.take([first], self.eos)
        return [s] if s.done else []

    def _keep(self, ids: list[int], s: Stream) -> None:
        d = s.snap
        self.cache.add(ids, kept(s.st), (d.cache.view(), d.carry.states.clone()))

    def _step(self, s: Stream, stop: int) -> int | None:
        """Prefill prompt[pos:stop] (the same bits for any stops); at the end, sample the first token and start decoding."""

        t0 = time.perf_counter()
        d: Drafts | None = s.snap
        try:
            normed, held = extend(self.w, self.head if d is not None else None, s.prompt, s.st,
                                  d.cache if d is not None else None, d.carry.states if d is not None else None, stop)
            if d is not None:
                d.carry = Carry(held, [])
            if stop in s.stops:
                self._keep(list(s.prompt[:stop]), s)
            first = None if stop < len(s.prompt) else \
                sample_rows(_mm(normed[-1:], self.w.head), [len(s.prompt)], s.sampling)[0]
            if first is not None and d is not None and not (s.stops and len(s.prompt) - s.stops[-1] < MIN_GAP):
                self._keep(list(s.prompt), s)         # a message start just before the end covers it
        finally:
            s.prefill_s += time.perf_counter() - t0
        if first is None:
            return None
        if d is not None:
            d.carry.tokens = [first]
        s.copies = CopyIndex() if d is not None else None
        s.context = list(s.prompt)
        s.started = time.perf_counter()
        self.filling = [x for x in self.filling if x is not s]
        self.streams[s.sid] = s                       # after every step that can fail: a failure leaves it queued
        return first

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """A prefill step for the oldest queued prompt, then one round over the decoding streams; returns the finished."""

        done = self._fill() if self.filling else []
        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return done
        if len(live) == 1 and self._fits(live[0]):
            return done + self._alone(live[0])
        self._propose(live)
        wins = [[s.out[-1]] + s.drafts for s in live]
        chains = [list(range(-1, len(t) - 1)) for t in wins]
        logits, record, hidden, starts = multi_tree_forward(
            self.w, [(t, p, s.st) for t, p, s in zip(wins, chains, live)], hidden=True)
        positions = [[s.st.pos + 1 + i for i in range(len(t))] for s, t in zip(live, wins)]
        sampled = sample_streams(logits, starts, positions, [s.sampling for s in live])
        kept_rows = [accept(t, p, rows, s.count - len(s.out), self.eos)
                     for s, t, p, rows in zip(live, wins, chains, sampled)]
        commit_streams([s.st for s in live], record, [[starts[k] + r for r in path]
                                                      for k, (path, _) in enumerate(kept_rows)], in_place=True)
        for k, (s, tokens, (path, end)) in enumerate(zip(live, wins, kept_rows)):
            new = [tokens[r] for r in path[1:]] + [end]
            s.committed.extend(tokens[r] for r in path)
            s.counted(len(tokens))
            if s.snap is not None:                    # the kept rows' final states, for the head to absorb next
                s.snap.carry = Carry(hidden[starts[k] + path[0]:starts[k] + path[-1] + 1], new)
            s.take(new, self.eos)
        return done + [s for s in live if s.done]

    def _fits(self, s: Stream) -> bool:
        """Whether a drafting stream's caches fit the graphs' buffers (else it decodes eagerly in its own)."""

        return (s.snap is not None and self.graphs is not None and
                (s is self.resident or len(s.prompt) + s.count + self.depth <= self.graphs.capacity))

    def _alone(self, s: Stream) -> list[Stream]:
        """The one decoding stream's round as the one-stream engine runs it, in its graphs: the stream's state moves
        into their fixed buffers once and stays there, decoding eagerly with the others when streams join."""

        g, d = self.graphs, s.snap
        if self.resident is not s:
            s.st, d.cache = g.load(s.st, d.cache, min(g.capacity, len(s.prompt) + s.count + COPY_ROWS))
            self.resident = s
        tokens, path, new, d.carry = mtp_round(s.st, d.cache, d.carry, s.out[-1], s.count - len(s.out), s.sampling,
                                               s.context, s.copies, depth=self.depth, confidence=self.confidence,
                                               ids=self.head.ids, verify=g.verify, step=g.draft, eos=self.eos,
                                               in_place=True)
        s.committed.extend(tokens[r] for r in path)
        s.counted(len(tokens))
        s.take(new, self.eos)
        return [s] if s.done else []

    def _propose(self, live: list[Stream]) -> None:
        """Each drafting stream absorbs its carry (every stream in one head call), then proposes a copied continuation
        of its context or a chain of up to ``depth`` MTP drafts (chains a step at a time, every stream together)."""

        for s in live:
            s.drafts = []
        todo = [s for s in live if s.snap is not None]
        if not todo:
            return
        heads = [s.snap for s in todo]
        normed = self.head.forward_streams([d.cache for d in heads], [d.carry.states for d in heads],
                                           [d.carry.tokens for d in heads], [d.cache.pos for d in heads])
        last, row = [], 0
        for d in heads:
            row += len(d.carry.tokens)
            last.append(row - 1)
            d.cache.pos += len(d.carry.tokens)
        active = []
        for s, r in zip(todo, last):                  # an exact repeat of the context first: a long, likely window
            s.drafts = s.copies.propose(s.context, COPY_ROWS - 1)
            if not s.drafts:
                active.append((s, r))
        if not active:
            return
        rows = normed.index_select(0, torch.tensor([r for _, r in active], device=normed.device))
        while active:
            logits = self.head.logits(rows)
            chosen = picks(logits, [s.st.pos + 1 + len(s.drafts) for s, _ in active],
                           [s.sampling for s, _ in active], self.ids)
            going = []
            for k, ((s, _), (token, prob)) in enumerate(zip(active, chosen)):
                s.drafts.append(token)
                if prob >= self.confidence and len(s.drafts) < self.depth:
                    going.append((s, k))
            if not going:
                return
            rows = self.head.forward_streams(
                [s.snap.cache for s, _ in going], [rows[k:k + 1] for _, k in going], [[s.drafts[-1]] for s, _ in going],
                [s.snap.cache.pos + len(s.drafts) - 1 for s, _ in going])
            active = [(s, k) for k, (s, _) in enumerate(going)]

    @torch.no_grad()
    def warm(self, streams: int) -> None:
        """A synthetic request through prefill and a round, and forwards at the row counts ``streams`` windows bring,
        so no request compiles or loads a kernel; then forgotten."""

        s = Stream([0] * (min(300, self.context - self.depth - 2) if self.context else 300), 2, None, draft=True)
        self.admit(s)
        while not s.done:
            self.finish(self.round())
        self.finish([s])
        self.cache = PrefixCache(self.cache.keep)
        st = State(self.w)
        for rows in sorted({1, 16, 32, 64, 128, 16 * streams}):
            if rows > 16 * streams:
                continue
            n = -(-rows // 16)
            sizes = [rows // n + (i < rows % n) for i in range(n)]
            multi_tree_forward(self.w, [([0] * k, list(range(-1, k - 1)), st) for k in sizes], hidden=True)
            if self.head is not None:
                cache = Cache(self.w, 16)
                picks(self.head.logits(self.head.forward_streams([cache] * n, [torch.zeros(
                    (k, self.w.config.hidden), dtype=torch.bfloat16, device=self.w.norm.device) for k in sizes],
                    [[0] * k for k in sizes], [0] * n)), [1] * rows, [None] * rows, self.ids)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams (their prompt-end states joined the prefix cache at the end of their prefill)."""

        for s in done:
            self.streams.pop(s.sid, None)
            if s is self.resident:
                self.resident = None

    def drop(self) -> list[Stream]:
        """After an error in a round: forget the live streams and the queued prompts."""

        live = [s for s in self.streams.values() if not s.done]
        for s in live:
            del self.streams[s.sid]
        live += self.filling
        self.filling = []
        self.resident = None
        return live
