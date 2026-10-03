"""DeepSeek-V4.1's concurrent rounds (``tensorfold serve --parallel N``): N streams in N cache slots, each round's rows
of every stream verified in one forward (``SerialEngine.step_multi``), so the weights a round reads serve them all.

Rows are row-invariant, so a stream's tokens equal its serial decoding however many streams share its rounds. Rank 0
decides (admissions, prefill steps, each stream's draft count, completions) and sends every decision to rank 1 before
acting on it; both ranks then compute the same tokens (argmax or position-keyed samples of the gathered logits).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tensorfold.cuda.sampling import sample_rows
from tensorfold.cuda.streams import Stream, next_fill

from .serial import DRING, MAX_ROWS, WINDOW_ROWS, SerialEngine

ADMIT, FILL, ROUND, DONE, EVICT, GROW, MOVE = 1, 2, 3, 4, 5, 6, 7      # rank 0's messages
FRESH, TAKEOVER, COPY = 0, 1, 2            # how an admitted stream gets its extent (a kept prompt's, or new rows)
from .serial import PROMPT_ROWS as ROWS

KEEP_MIN = int(os.environ.get("TF_DSV41_KEEP_MIN") or 256)    # shorter states are not kept
# a stream's extent covers its prompt and this many reply tokens at first, and grows by as much while it decodes
# (in place, else moved to a free run, kept states evicted for it); 0: the whole reply up front
GROW_AHEAD = int(os.environ.get("TF_DSV41_GROW_AHEAD") or 4096)
RELEASE_AFTER = 4096                       # prompts prefilling more rows than this release cached blocks after
RELEASE_BELOW = int(float(os.environ.get("TF_DSV41_RELEASE_BELOW_GIB") or 2.5) * 2 ** 30)   # ... when memory is low


@dataclass(eq=False)
class Kept:
    """A kept prompt inside the shared pool: the per-token caches of ``ids`` in extent ``x``'s first rows and its
    window rings in bank entry ``bank``, which hold positions from ``vs`` on: a prompt sharing its first m tokens
    resumes there when the window before m (and the rows a short tail backs up into) is held.

    Only prefilled rows are kept: every prompt chunk gives a row the same values whatever the chunk's length or
    start (``SerialEngine.prefill``), so a resumed prompt's caches and reply equal a fresh prefill's. A reply's
    decoded rows are not kept (a decode graph's arithmetic differs in the last bits)."""

    kid: int
    x: Any
    ids: np.ndarray
    vs: int
    bank: int
    vd: int = 0         # from where every layer's window is exact (vs, or later after early bounded-tail chunks)

    @property
    def n(self) -> int:
        return len(self.ids)


def common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if len(diff) else n

SAMPLING_WORDS = 10


def pack_sampling(sampling) -> list[int]:
    """[on, seed lo, seed hi, top_k, temperature, top_p, min_p as int pairs]: SAMPLING_WORDS ints."""

    import struct

    if sampling is None or sampling.temperature <= 0:
        return [0] * SAMPLING_WORDS
    seed = sampling.seed & 0xFFFFFFFFFFFFFFFF
    out = [1, seed & 0xFFFFFFFF, seed >> 32, int(sampling.top_k)]
    for value in (sampling.temperature, sampling.top_p, sampling.min_p):
        out += list(struct.unpack("<ii", struct.pack("<d", float(value))))
    return out


def unpack_sampling(words: list[int]):
    import struct

    from tensorfold.engine.exact_sampling import Sampling

    if not words[0]:
        return None
    f = [struct.unpack("<d", struct.pack("<ii", words[4 + 2 * i], words[5 + 2 * i]))[0] for i in range(3)]
    return Sampling(seed=(words[2] << 32) | words[1], temperature=f[0], top_k=words[3], top_p=f[1], min_p=f[2])



class MultiDecoder:
    """The ``tensorfold.cuda.scheduler.Scheduler``'s decoder over ``SerialEngine`` slots (rank 0 or 1 of two)."""

    def __init__(self, e: SerialEngine, share: Callable[[list[int] | None], list[int]], *, rank: int,
                 drafts: int = 3, step: int = MAX_ROWS, pool=None) -> None:
        self.e, self.share, self.rank = e, share, rank
        # the shared cache pool (``pool.Pool``): each admitted stream gets an extent of it, or (None) its slot's
        # fixed extent; a request whose extent does not fit waits (NoRoom) until a stream finishes
        self.pool = pool
        # kept prompts inside the pool (``Kept``): a stream's extent outlives it while it holds kept states, which
        # are evicted (least recently used first) when an admission needs the rows
        banks = len(e.bank[0]) // DRING if pool is not None and getattr(e, "bank", None) else 0
        self.kept_on = banks > 0 and os.environ.get("TF_DSV41_POOL_KEEP", "1") != "0"
        self.kept: list[Kept] = []                 # least recently used first
        self.banks = list(range(banks))            # free bank entries, ascending
        self.next_kid = 0
        self.ext: dict[int, Any] = {}              # sid -> its extent
        self.yielded: list[Stream] = []            # rank 0: background streams giving up their rows (to replay)
        self.kstats = {"kept": 0, "hits": 0, "cached": 0, "takeovers": 0, "copies": 0, "evictions": 0}
        self.drafts = drafts if e.drafter is not None else 0

        self.check = os.environ.get("TF_MULTI_CHECK") == "1"
        if os.environ.get("TF_MULTI_PROF"):
            e._mprof = {}
        if os.environ.get("TF_MULTI_DRAFTS"):                    # tuning: drafts a stream in concurrent rounds
            self.drafts = min(self.drafts, int(os.environ["TF_MULTI_DRAFTS"]))
        self.prof = {"rounds": 0, "draft": 0.0, "verify": 0.0, "post": 0.0, "fill": 0.0, "round": 0.0, "rows": 0,
                     "streams": 0} \
            if os.environ.get("TF_MULTI_PROF") else None
        self.step_rows = step                      # prompt rows a fill step takes while other streams decode
        self.free = list(range(e.slots))
        self.streams: dict[int, Stream] = {}       # decoding, by sid
        self.filling: list[Stream] = []
        self.next_id = 0
        self.eos = (int(e.c.eos_token_id),)
        self.broken: Exception | None = None
        self.model_dir = None                      # rank 1 compiles a request's grammar from it
        self.costs: list[float] | None = None      # verify ms by rows (calibrate)
        self.draft_ms = 0.0
        self.draft_curve: list[float] = []
        self.overhead = 2.0                        # a round's host ms besides the forward and drafts
        self.prior = [0.6] * max(self.drafts, 1)   # acceptance by draft position, over every stream (new ones start here)
        import os as _os

        # while streams decode, prompt steps take this share of the time (the rest: decode rounds)
        self.fill_share = float(_os.environ.get("TF_FILL_SHARE") or "0.5")
        self.t_fill = self.t_decode = 0.0
        self.alone_rows = 8 * step                 # a prompt step's rows when nothing decodes (new arrivals join after)

    # -- costs and draft allocation -------------------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, gather: Callable[[list[int]], list[list[int]]]) -> None:
        """Verify ms for 1..ROWS rows of distinct tokens spread over the slots, and a draft's ms (the slower rank's,
        both ranks call this together). Caches written here are reset before any stream uses its slot."""

        import random

        e = self.e
        rng = random.Random(0)
        widths = [*e.widths, e.cap]                          # a cost curve for each key width the graphs have
        path = self._calib_path(widths)
        cached = None
        if path is not None:
            try:
                cached = [int(v) for v in json.loads(path.read_text())]
            except (OSError, ValueError):
                cached = None
        if all(f[0] for f in gather([int(cached is not None)])):
            self._curves(gather(cached), widths)             # both ranks timed this image and setting before
            return
        ms = []
        for w in widths:
            for R in range(1, ROWS + 1):
                if R not in e.graphs:
                    ms.append(ms[-1] if ms else 30.0)
                    continue
                g = e.graph_for(R, w)
                g["tok"].copy_(torch.tensor([rng.randrange(1000, 100000) for _ in range(R)]))
                g["pos"].copy_(torch.arange(R) + 200)
                g["sid"].copy_(torch.arange(R) % e.slots)
                best = float("inf")
                for _ in range(4):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    e._replay_free(g)
                    torch.cuda.synchronize()
                    best = min(best, time.perf_counter() - t)
                ms.append(1e3 * best)
        drafts = []                                          # ms of one drafting pass for 1.. streams at once
        if self.drafts:
            batched = getattr(e.drafter, "multi_graphs", None) or {}
            for M in range(1, max(batched, default=1) + 1):
                items = [(m % e.slots, rng.randrange(1000, 100000), 300) for m in range(M)]
                best = float("inf")
                for _ in range(4):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    if M in batched:
                        e.drafter.propose_multi(items)
                    else:
                        e.drafter.propose(items[0][1], 300)
                    best = min(best, time.perf_counter() - t)
                drafts.append(1e3 * best)
        raw = [int(1e3 * v) for v in ms + drafts]
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}")
                tmp.write_text(json.dumps(raw))
                os.replace(tmp, path)
            except OSError:
                pass
        self._curves(gather(raw), widths)
        for slot in range(e.slots):                          # the timing rows wrote every slot's caches
            e.select_slot(slot)
            e.reset()
        e.select_slot(0)

    def _calib_path(self, widths: list[int]):
        """Where this rank keeps its timings for this image and setting (TF_REVISION, set in the deploy image: the
        source snapshot's hash), or None (outside an image: always measure). The curves only steer draft counts,
        which never change a token."""

        import hashlib
        from pathlib import Path

        rev = os.environ.get("TF_REVISION", "")
        if rev in ("", "unknown") or os.environ.get("TF_DSV41_CALIB", "cached") != "cached":
            return None
        e = self.e
        knobs = sorted((k, v) for k, v in os.environ.items() if k.startswith("TF_") and k not in
                       ("TF_API_KEY", "TF_PORT", "TF_RANK", "TF_DSV41_LAUNCH_T0"))
        key = [rev, torch.cuda.get_device_name(), torch.__version__, e.cap, e.slots, ROWS, widths, self.drafts,
               sorted(getattr(e.drafter, "multi_graphs", None) or {}), knobs]
        h = hashlib.sha256(json.dumps(key, default=str).encode()).hexdigest()[:16]
        d = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "tensorfold" / "dsv41-calib"
        return d / f"r{self.rank}-{h}.json"

    def _curves(self, both: list[list[int]], widths: list[int]) -> None:
        worst = [max(a, b) / 1e3 for a, b in zip(*both)]
        n = ROWS * len(widths)
        self.curves = [(w, worst[i * ROWS:(i + 1) * ROWS]) for i, w in enumerate(widths)]
        self.costs = self.curves[-1][1]
        self.draft_curve = worst[n:]
        self.draft_ms = self.draft_curve[0] if self.draft_curve else 0.0

    def _expected(self, acc: list[float], k: int) -> float:
        total, run = 1.0, 1.0
        for j in range(k):
            run *= acc[j]
            total += run
        return total

    def _allocate(self, live: list[Stream]) -> list[int]:
        """Drafts a stream this round: one at a time to the stream whose next draft raises the round's expected
        tokens per ms the most, while any does (a draft is a verify row and, a stream's first, a drafting pass)."""

        ks = [0] * len(live)
        caps = []
        for s in live:
            room = min(self.e.limit, self.e.extents[s.slot][1]) - len(self.e.views[s.slot].ids) - 1
            ok = s.draft and s.constraint is None and self.drafts
            caps.append(max(0, min(self.drafts, room, s.count - len(s.out) - 1)) if ok else 0)
        if not any(caps) or self.costs is None:
            return ks
        need = max(len(self.e.views[s.slot].ids) for s in live) + ROWS   # the round's graph width (at most)
        self.costs = next((c for w, c in self.curves if need <= w), self.curves[-1][1])
        rows, drafting = len(live), 0
        tokens = float(len(live))
        rate = tokens / (self.costs[rows - 1] + self.overhead)
        while rows < ROWS:
            # a stream's next j drafts at once (j = 1 .. its cap): the cost curve is not convex (some row counts
            # cost much more than the next) and a stream's first draft also pays a drafting pass, so one draft at
            # a time stops early where several would pay (3 streams: almost no drafting)
            best = None
            for i, s in enumerate(live):
                base = self._expected(s.acc, ks[i])
                for j in range(1, min(caps[i] - ks[i], ROWS - rows) + 1):
                    gain = self._expected(s.acc, ks[i] + j) - base
                    cost = self.costs[rows + j - 1] + self.overhead + self._draft_cost(drafting + (ks[i] == 0))
                    r = (tokens + gain) / cost
                    if r > rate and (best is None or r > best[0]):
                        best = (r, i, j, gain)
            if best is None:
                break
            rate, i, j, gain = best
            drafting += ks[i] == 0
            ks[i] += j
            tokens += gain
            rows += j
        return ks

    def _draft_cost(self, streams: int) -> float:
        """ms of drafting for ``streams`` streams: one batched pass where captured, else a pass each."""

        if streams <= 0:
            return 0.0
        curve = self.draft_curve
        if streams <= len(curve) and len(curve) > 1:
            return curve[streams - 1]
        return streams * self.draft_ms

    def _learn(self, s: Stream, k: int, m: int) -> None:
        a = 0.15
        for j in range(min(k, m + 1)):                     # positions after the first rejection are unobserved
            hit = 1.0 if j < m else 0.0
            s.acc[j] = (1 - a) * s.acc[j] + a * hit
            self.prior[j] = (1 - a / 4) * self.prior[j] + a / 4 * hit

    # -- bookkeeping --------------------------------------------------------------------------------------------
    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def _send(self, values: list[int]) -> None:
        if self.broken is None:
            self.share(values)

    def _check(self) -> None:
        if self.broken is not None:
            raise RuntimeError("the two ranks are out of step after an error; restart both") from self.broken

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.eos if s.stop_eos else ()

    # -- admission and prompts ------------------------------------------------------------------------------------
    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        self._check()
        room = self.e.limit - len(s.prompt) - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.e.limit}-token context")
        s.count = min(s.count, room)
        from tensorfold.engine.grammar import pack

        packed = pack(s.constraint)                # before any pool decision (an eviction is sent at once)
        base = size = eid = -1
        k, m, mode = None, 0, FRESH
        if self.pool is not None:                  # the prompt, the reply and a verify window's rows past it
            if any(x.waiting for x in self.streams.values()):
                from tensorfold.cuda.memory_gate import NoRoom

                raise NoRoom("streams wait for their caches to grow; a new request waits until one finishes")
            size = self._first(s)
            if self.kept_on and s.draft:
                k, m = self._match(s.prompt)
            if k is not None and k.x.owner is None and size - k.x.size <= self.pool.room_after(k.x):
                mode, base, size = TAKEOVER, k.x.base, max(size, k.x.size)
            else:
                try:
                    base = self._room(size, [k.x] if k is not None else [])
                    mode = COPY if k is not None else FRESH
                except Exception:
                    if k is None:
                        raise
                    k, m, mode = None, 0, FRESH    # no room beside the kept rows: without them
                    base = self._room(size, [])
                eid = self.pool.next_eid
        s.sid = self.next_id
        self.next_id += 1
        self._send([ADMIT, s.sid, s.count, int(s.draft), int(s.stop_eos), *pack_sampling(s.sampling), len(packed),
                    base, size, eid, k.kid if k is not None else -1, m, mode])
        self._send(list(s.prompt))
        if packed:
            self._send(packed)
        self._queue(s, base, size, eid, mode, k, m)

    # -- extents: a stream's first rows, growth while it decodes ----------------------------------------------------
    def _most(self, s: Stream) -> int:
        """The rows a stream can ever write: its prompt, its reply and a verify window past it."""

        from .pool import align_up

        return min(align_up(len(s.prompt) + s.count + ROWS + 1), self.e.span)

    def _first(self, s: Stream) -> int:
        from .pool import align_up

        if GROW_AHEAD <= 0:
            return self._most(s)
        return min(self._most(s), align_up(len(s.prompt) + GROW_AHEAD + ROWS + 1))

    def _make_room(self, live: list[Stream]) -> list[Stream]:
        """Rank 0, before a round: each decoding stream's extent covers the rows its round can write, oldest first
        (grown in place, else moved to a free run, else after evicting kept states; GROW/MOVE/EVICT sent before the
        ROUND). A stream that cannot grow waits, and so do the newer ones. When even the oldest cannot, one other
        stream gives up its rows (they free when it finishes, before the next round): the newest background one,
        handed back to replay later (``yielded``), else the newest, ended with an error. Returns the streams ended."""

        from .pool import align_up

        if self.pool is None:
            return []
        live = sorted(live, key=lambda x: x.sid)
        blocked = False
        for s in live:
            x = self.ext[s.sid]
            most = self._most(s)
            need = min(len(self.e.views[s.slot].ids) + ROWS + 1, most)   # (never past the limit: _allocate)
            if need <= x.size:
                s.waiting = False
                continue
            size = min(most, max(align_up(need), align_up(need + GROW_AHEAD) if GROW_AHEAD > 0 else 0))
            s.waiting = blocked or not self._grow(s, x, size)
            blocked = blocked or s.waiting
        if not live or not live[0].waiting or self.yielded:
            return []                              # (a stream already giving way frees its rows first)
        others = [x for x in [*live[1:], *self.filling] if not x.done and x not in self.yielded]
        if not others:
            return []
        resumable = [x for x in others if x.background and x.constraint is None and x.vision is None]
        victim = max(resumable or others, key=lambda x: x.sid)
        victim.waiting = False
        if resumable:
            self.yielded.append(victim)            # the scheduler ends it and queues its replay
            return []
        victim.error = RuntimeError(
            f"The shared cache pool ran out of room with {len(live)} streams decoding, so the newest (this request, "
            f"after {len(victim.out)} tokens) was stopped for the older ones to finish. Retry it, shorten the prompt "
            "or max_tokens, or start the server with a smaller --parallel.")
        victim.done, victim.finished = True, time.perf_counter()
        return [victim]

    def _grow(self, s: Stream, x, size: int) -> bool:
        """Rank 0: extent ``x`` of stream ``s`` to ``size`` rows; False when no room opens."""

        if size - x.size <= self.pool.room_after(x):
            self._send([GROW, s.sid, size])
            self._resize(s.sid, size)
            return True
        base = self.pool.place(size, ignore=[x])
        if base is None:                           # kept states' extents out of the way, the fewest that do
            order = {id(k): i for i, k in enumerate(self.kept)}
            loose = [y for y in self.pool.extents if y.owner is None and y.kept]
            loose.sort(key=lambda y: max(order[id(k)] for k in y.kept))
            for j in range(1, len(loose) + 1):
                if self.pool.place(size, ignore=[x, *loose[:j]]) is not None:
                    for y in loose[:j]:
                        for k in sorted(y.kept, key=lambda k: order[id(k)]):
                            self._send([EVICT, k.kid])
                            self._drop(k)
                            self.kstats["evictions"] += 1
                    break
            else:
                return False
            if size - x.size <= self.pool.room_after(x):
                self._send([GROW, s.sid, size])
                self._resize(s.sid, size)
                return True
            base = self.pool.place(size, ignore=[x])
        self._send([MOVE, s.sid, base, size])
        self._move(s.sid, base, size)
        return True

    def _resize(self, sid: int, size: int) -> None:
        """Both ranks: stream ``sid``'s extent grown in place (its rows stay)."""

        s, x = self.streams[sid], self.ext[sid]
        self.pool.resize(x, size)
        self.e.bind(s.slot, x.base, x.size, ids=list(self.e.views[s.slot].ids))
        self.kstats["grows"] = self.kstats.get("grows", 0) + 1

    def _move(self, sid: int, base: int, size: int) -> None:
        """Both ranks: stream ``sid``'s extent moved to ``base`` with ``size`` rows (its committed rows, and the kept
        states' in them, copied along)."""

        s, x = self.streams[sid], self.ext[sid]
        ids = list(self.e.views[s.slot].ids)
        old = self.pool.move(x, base, size)
        self.e.copy_rows(old, base, len(ids))
        self.e.bind(s.slot, base, size, ids=ids)
        self.kstats["moves"] = self.kstats.get("moves", 0) + 1

    def yield_for(self, s: Stream) -> list[Stream]:
        """Rank 0, ``s`` (foreground) found no room: the fewest background streams, newest first, whose extents (with
        every kept state's) would open a run for its first rows; [] when even all of them would not. The scheduler
        ends and re-queues them (they replay from their prompts)."""

        if self.pool is None or any(x.waiting for x in self.streams.values()):
            return []                              # (growth comes first: a yield would not admit it)
        cands = [x for x in [*self.streams.values(), *self.filling]
                 if x.background and not x.done and x.constraint is None and x.vision is None
                 and len(x.out) < x.count and x.sid in self.ext]
        cands.sort(key=lambda x: -x.sid)
        loose = [y for y in self.pool.extents if y.owner is None and y.kept]
        size = self._first(s)
        for j in range(len(cands) + 1):           # (j = 0: evicting kept states makes room, no yield needed)
            if self.pool.place(size, ignore=[*loose, *(self.ext[x.sid] for x in cands[:j])]) is not None:
                return cands[:j]
        return []

    # -- kept prompts in the shared pool (both ranks make the same calls in the same order) ------------------------
    def _match(self, prompt: list[int]) -> tuple[Kept | None, int]:
        """The kept state to resume ``prompt`` from and the tokens it covers: the most (at least one left to
        prefill, and the rings holding the window there), then one whose extent is free (taken over without
        copying), then the most recent."""

        p = np.asarray(prompt, dtype=np.int64)
        best, key = None, None
        for i, k in enumerate(self.kept):
            if k.ids[0] != p[0]:
                continue
            m = min(common_prefix(k.ids, p), len(p) - 1, k.n)
            back = max(0, ROWS + 1 - (len(p) - m))   # a short tail backs up into rows before m (prefill)
            full = m - back - WINDOW_ROWS >= k.vd
            # (stale later-layer windows are fine far from the end: the bounded tail recomputes past their reach)
            early = m <= len(p) - self.e.tail_min and m - WINDOW_ROWS >= k.vs
            if m < KEEP_MIN or not (full or early):
                continue
            kk = (m, k.x.owner is None, i)
            if key is None or kk > key:
                best, key = k, kk
        return (best, key[0]) if best is not None else (None, 0)

    def _room(self, size: int, protect: list) -> int:
        """Rank 0: a base for a new ``size``-row extent, evicting kept states (least recently used first, none in a
        ``protect`` extent or a live stream's) until one fits; NoRoom when even all of them would not do."""

        from tensorfold.cuda.memory_gate import NoRoom

        base = self.pool.place(size)
        if base is not None:
            return base
        # whole extents of kept states only (dropping a shorter state beside a longer one frees nothing), least
        # recently used first; the fewest that open a run of ``size`` rows, found before anything is evicted
        order = {id(k): i for i, k in enumerate(self.kept)}
        loose = [x for x in self.pool.extents if x.owner is None and x.kept and all(x is not y for y in protect)]
        loose.sort(key=lambda x: max(order[id(k)] for k in x.kept))
        for j in range(1, len(loose) + 1):
            if self.pool.place(size, ignore=loose[:j]) is not None:
                for x in loose[:j]:
                    for k in sorted(x.kept, key=lambda k: order[id(k)]):
                        self._send([EVICT, k.kid])
                        self._drop(k)
                        self.kstats["evictions"] += 1
                base = self.pool.place(size)
                if base is not None:
                    return base
                break
        raise NoRoom(f"the shared cache pool has no {size}-token extent free ({self.pool.free_rows()} of "
                     f"{self.pool.rows} tokens free, largest run {self.pool.largest_gap()})")

    def _keep(self, s: Stream, x, ids: list[int], vs: int) -> None:
        """Keep stream ``s``'s state of ``ids`` (extent ``x``'s first rows, its slot's rings) as a kept prompt."""

        if not self.kept_on or len(ids) < KEEP_MIN:
            return
        a = np.asarray(ids, dtype=np.int64)
        for k in [k for k in self.kept if k.n == len(a) and np.array_equal(k.ids, a)]:
            self._drop(k)                          # the same tokens again: the newer state
        if not self.banks:
            self._drop(self.kept[0])
        bank = self.banks.pop(0)
        self.e.save_window(s.slot, bank)
        k = Kept(self.next_kid, x, a, vs, bank, max(vs, self.e.deep_from[s.slot]))
        self.next_kid += 1
        x.kept.append(k)
        self.kept.append(k)
        self.kstats["kept"] += 1

    def _drop(self, k: Kept) -> None:
        self.kept.remove(k)
        k.x.kept.remove(k)
        self.banks.append(k.bank)
        self.banks.sort()
        self._settle(k.x)

    def _settle(self, x) -> None:
        """An extent no stream writes: gone when it keeps nothing, else cut to its longest kept state's rows."""

        from .pool import align_up

        if x.owner is not None:
            return
        if not x.kept:
            self.pool.remove(x)
            return
        need = align_up(max(k.n for k in x.kept))
        if need < x.size:
            self.pool.resize(x, need)

    def _kid(self, kid: int) -> Kept:
        k = next((k for k in self.kept if k.kid == kid), None)
        if k is None:
            raise RuntimeError(f"the ranks disagree on the kept prompts (no kept state {kid})")
        return k

    def _queue(self, s: Stream, base: int = -1, size: int = -1, eid: int = -1, mode: int = FRESH,
               k: Kept | None = None, m: int = 0) -> None:
        s.acc = list(self.prior)
        s.slot = self.free.pop(0)
        s.pos = -1                                 # prompt tokens prefilled so far (-1: not started)
        if base >= 0:                              # the stream's extent of the shared pool (rank 1: rank 0's place)
            e = self.e
            if mode == TAKEOVER:                   # a kept state's free extent: its rows are the stream's prefix
                x = k.x
                if x.owner is not None or x.base != base:
                    raise RuntimeError("the ranks disagree on a kept prompt's extent")
                if size > x.size:
                    self.pool.resize(x, size)
                x.owner = s.sid
            else:
                x = self.pool.add(base, size, owner=s.sid, eid=eid)
            self.ext[s.sid] = x
            if k is not None:
                if mode == COPY:
                    e.copy_rows(k.x.base, x.base, m)
                    self.kstats["copies"] += 1
                else:
                    self.kstats["takeovers"] += 1
                e.bind(s.slot, x.base, x.size, ids=list(s.prompt[:m]))
                e.select_slot(s.slot)
                e.load_window(k.bank, s.slot)
                e.ring_from[s.slot] = k.vs
                e.deep_from[s.slot] = k.vd
                self.kept.remove(k)                # most recently used
                self.kept.append(k)
                for c in [c for c in x.kept if c.n > m]:
                    self._drop(c)                  # rows the stream overwrites (the hit too, its window loaded)
                s.pos = s.cached = m
                self.kstats["hits"] += 1
                self.kstats["cached"] += m
            else:
                e.bind(s.slot, x.base, x.size)
        self.filling.append(s)

    def _fill(self) -> list[Stream]:
        s = next_fill(self.filling)
        busy = any(not x.done for x in self.streams.values())
        rows = self.step_rows if busy else self.alone_rows
        self._send([FILL, s.sid, rows])
        first = self._step(s, rows)
        if first is None:
            return []
        s.take([first], self._ends(s))
        return [s] if s.done else []

    def _step(self, s: Stream, rows: int) -> int | None:
        """Prefill the next ``rows`` prompt tokens in the stream's slot (resuming a kept or live state first, so the
        step ends past what was resumed); at the prompt's end, keep its state and sample the first token."""

        e = self.e
        t0 = time.perf_counter()
        try:
            e.select_slot(s.slot)
            n = len(s.prompt)
            if s.pos < 0 and self.pool is not None:   # (a shared pool resumes kept states at admission)
                e.reset()
                s.pos = s.cached = 0
            if s.pos < 0:                          # the first step: resume what the slot or the pool holds
                cached = e.reusable(s.prompt) if s.draft else 0
                kept = e.pool.match(s.prompt) if s.draft and e.pool is not None else None
                if kept is not None and len(kept.ids) > cached:
                    e.load_prefix(kept.snapshot, kept.ids)
                    cached = len(kept.ids)
                elif cached:
                    del e.state.ids[cached:]
                else:
                    e.reset()
                s.pos, s.cached = cached, cached
            stop = min(n, s.pos + rows)
            # kept at the prompt's end and, for prompts that will share less, at its last chunk start (a long
            # document asked a long new question resumes there)
            point = (n - 1) // MAX_ROWS * MAX_ROWS
            keep = self.kept_on and s.draft
            split = (keep and point >= KEEP_MIN and s.cached < point < n - (DRING - WINDOW_ROWS)
                     and s.pos + ROWS < point < stop)        # (no <= ROWS-row call before it: prompt arithmetic)
            if split:
                stop = point
            # (a split call bounds its early chunks by the split point: the state kept there is exact in every layer)
            logits = e.prefill(s.prompt[s.pos:stop], final=point if split else n)
            s.pos = stop
            if split:
                self._keep(s, self.ext[s.sid], e.state.ids, e.ring_from[s.slot])
            if stop < n:
                return None
            if keep and n >= KEEP_MIN and s.cached < n:
                self._keep(s, self.ext[s.sid], e.state.ids, e.ring_from[s.slot])
            if s.draft and self.pool is None:
                e.keep_prompt(s.prompt)
            if n - s.cached > RELEASE_AFTER:       # a long prompt's transient buffers back to the system (both
                self._release()                    # ranks: the unified memory the OS and the next prompt share)
            last = logits[-1:]
            if s.constraint is not None:
                last = s.constraint.mask(last.float().clone())
            first = sample_rows(last, [n], s.sampling)[0]
        except Exception as exc:
            self.broken = exc
            raise
        finally:
            s.prefill_s += time.perf_counter() - t0
        s.context = list(s.prompt)
        s.started = time.perf_counter()
        self.filling = [x for x in self.filling if x is not s]
        self.streams[s.sid] = s
        return first

    def _release(self) -> None:
        """The allocator's cached free blocks back to the system (graph pools and live tensors stay), and the host
        heap's free pages (rank 0 tokenized the request's text: a long prompt leaves ~1 GiB in glibc's arenas)."""

        from .engine import available_bytes

        log = os.environ.get("TF_DSV41_MEMLOG") == "1"
        before = torch.cuda.memory_reserved() if log else 0
        t0 = time.perf_counter()
        # the CUDA allocator's blocks go back only when memory runs low: handed back, they return as scattered
        # pages, and the next prompt's ~1 GiB of buffers then waits on reclaim and compaction for large blocks
        # (8K prompts ran 650-1000 tok/s instead of ~1750 with a release after every long prompt)
        if available_bytes() < RELEASE_BELOW:
            torch.cuda.empty_cache()
        t1 = time.perf_counter()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
        t2 = time.perf_counter()
        if log:
            print(f"[memlog r{self.rank}] prompt released {(before - torch.cuda.memory_reserved()) / 2 ** 30:.2f} GiB; "
                  f"available {available_bytes() / 2 ** 30:.2f} GiB; empty_cache {1e3 * (t1 - t0):.0f} ms, "
                  f"malloc_trim {1e3 * (t2 - t1):.0f} ms", flush=True)

    # -- rounds ---------------------------------------------------------------------------------------------------
    def _plan(self, live: list[Stream]) -> list[tuple[int, int]]:
        """(sid, drafts) a stream: drafts while the round's rows fit (none for a grammar's stream yet)."""

        return [(s.sid, k) for s, k in zip(live, self._allocate(live))]

    @torch.no_grad()
    def round(self) -> list[Stream]:
        self._check()
        tr = time.perf_counter()
        busy = any(not x.done and not x.waiting for x in self.streams.values())   # (waiting ones: fill goes on)
        share = self.fill_share / max(1e-6, 1.0 - self.fill_share)
        done = []
        if self.filling and (not busy or self.t_fill <= share * self.t_decode):   # time-sliced while others decode
            done = self._fill()
            if busy:
                self.t_fill += time.perf_counter() - tr
        if self.prof is not None and self.rank == 0:
            self.prof["fill"] += time.perf_counter() - tr
        live = [s for s in self.streams.values() if not s.done]
        if self.rank == 0:
            done += self._make_room(live)
        live = [s for s in live if not s.done and not s.waiting and s not in self.yielded]
        if not live:
            return done
        plan = self._plan(live)
        self._send([ROUND, len(plan), *[x for item in plan for x in item]])
        td = time.perf_counter()
        news = self._verify(plan)
        if self.filling:
            self.t_decode += time.perf_counter() - td
        else:
            self.t_fill = self.t_decode = 0.0                  # nothing waits to fill: the shares start over
        for s, new in zip(live, news):
            if s.error is None:
                s.take(new, self._ends(s))
            else:
                s.done, s.finished = True, time.perf_counter()
        if self.prof is not None and self.rank == 0:
            self.prof["round"] += time.perf_counter() - tr
        return done + [s for s in live if s.done]

    def _verify(self, plan: list[tuple[int, int]]) -> list[list[int]]:
        """One forward over every planned stream's pending token and drafts; returns each stream's new tokens (kept
        drafts, then its next token) and rolls rejected rows back in its slot."""

        e = self.e
        try:
            rows, spans = [], []
            t0 = time.perf_counter()
            want = [(sid, k) for sid, k in plan if k]
            batched = getattr(e.drafter, "multi_graphs", None) if e.drafter is not None else None
            proposals: dict[int, list[int]] = {}
            if want and batched and len(want) in batched:     # one drafting pass for every drafting stream
                items = [(self.streams[sid].slot, self.streams[sid].out[-1], len(e.views[self.streams[sid].slot].ids))
                         for sid, _ in want]
                proposals = {sid: d for (sid, _), d in zip(want, e.drafter.propose_multi(items))}
            for sid, k in plan:
                s = self.streams[sid]
                pending = s.out[-1]
                drafts: list[int] = []
                if k and sid in proposals:
                    drafts = proposals[sid][:k]
                elif k:
                    e.select_slot(s.slot)
                    drafts = e.drafter.propose(pending, len(e.views[s.slot].ids))[:k]
                if s.constraint is not None:
                    s.constraint.advance([pending])
                spans.append((len(rows), len(drafts) + 1, len(e.views[s.slot].ids)))
                rows += [(s.slot, t) for t in [pending, *drafts]]
            t1 = time.perf_counter()
            logits, greedy = e.step_multi(rows)
            if self.check and len(rows) > 1:               # debug: row 0 of each stream alone, at the same position
                main = logits[:len(rows)].float().clone()
                for (sid, k), (r0, nrows, p0) in zip(plan, spans):
                    s = self.streams[sid]
                    ids = e.views[s.slot].ids
                    keep = ids[p0:]
                    del ids[p0:]
                    alone, g1 = e.step_multi([(s.slot, rows[r0][1])])
                    a = alone[0].float()
                    if self.rank == 0:
                        d = (a - main[r0]).abs().max().item()
                        top = torch.topk(main[r0], 2).values.tolist()
                        print(f"[check] sid {sid} pos {p0} rows {len(rows)}: max|diff| {d:.3g}, argmax multi "
                              f"{int(main[r0].argmax())} alone {g1[0]}, top2 gap {top[0] - top[1]:.3g}", flush=True)
                    del ids[p0:]
                    ids.extend(keep)
            if self.prof is not None and self.rank == 0:
                p = self.prof
                p["rounds"] += 1
                p["draft"] += t1 - t0
                p["verify"] += time.perf_counter() - t1
                p["rows"] += len(rows)
                p["streams"] += len(plan)
                if p["rounds"] % 50 == 0:
                    n = p["rounds"]
                    mp = getattr(e, "_mprof", None) or {}
                    if mp.get("n"):
                        print("[multi] step_multi: " + ", ".join(f"{k} {1e3 * mp[k] / mp['n']:.1f} ms" for k in
                                                                 ("hash", "gather0", "gather1", "wait")), flush=True)
                    print(f"[multi] {n} rounds: draft {1e3 * p['draft'] / n:.1f} ms, verify {1e3 * p['verify'] / n:.1f} ms, "
                          f"post {1e3 * p['post'] / n:.1f} ms, fill {1e3 * p['fill'] / n:.1f} ms, round "
                          f"{1e3 * p['round'] / n:.1f} ms, {p['rows'] / n:.1f} rows, {p['streams'] / n:.1f} streams"
                          + (f"; kept {len(self.kept)} {self.kstats}" if self.kept_on else ""), flush=True)
            t2 = time.perf_counter()
            news = []
            for (sid, k), (r0, nrows, p0) in zip(plan, spans):
                s = self.streams[sid]
                if s.sampling is not None and s.sampling.temperature > 0 or s.constraint is not None:
                    block = logits[r0:r0 + nrows].float()
                    if s.constraint is not None:
                        block = s.constraint.mask(block.clone(), s.constraint.window([rows[r0][1]], [-1]))
                    target = sample_rows(block, [p0 + 1 + j for j in range(nrows)], s.sampling)
                else:
                    target = greedy[r0:r0 + nrows]
                drafts = [t for _, t in rows[r0 + 1:r0 + nrows]]
                m = 0
                while m < len(drafts) and drafts[m] == target[m]:
                    m += 1
                del e.views[s.slot].ids[p0 + 1 + m:]           # rejected rows: overwritten later
                if k:
                    self._learn(s, k, m)
                if s.constraint is not None and m:
                    s.constraint.advance(drafts[:m])
                s.counted(nrows)
                new = drafts[:m] + [target[m]]
                ends = self._ends(s)                           # a kept draft can be the end token: stop at it
                cut = next((j + 1 for j, t in enumerate(new) if t in ends), len(new))
                news.append(new[:min(cut, max(1, s.count - len(s.out)))])
            if self.prof is not None and self.rank == 0:
                self.prof["post"] += time.perf_counter() - t2
            return news
        except Exception as exc:
            self.broken = exc
            raise

    # -- completion -----------------------------------------------------------------------------------------------
    def finish(self, done: list[Stream]) -> None:
        if done:
            self._send([DONE, len(done), *[s.sid for s in done]])
            for s in done:
                self._finish(s.sid)

    def _finish(self, sid: int) -> None:
        self.yielded = [x for x in self.yielded if x.sid != sid]
        s = self.streams.pop(sid, None)
        if s is None:
            s = next((x for x in self.filling if x.sid == sid), None)
            if s is not None:
                self.filling.remove(s)
        x = self.ext.pop(sid, None)
        if x is not None:                          # (a reply's decoded rows are not kept: a decode graph's rows
            x.owner = None                         # can differ from the prompt chunk's that would recompute them)
            self._settle(x)
        if s is not None and s.slot not in self.free:
            self.free.append(s.slot)
            self.free.sort()

    def drop(self) -> list[Stream]:
        live = list(self.streams.values()) + self.filling     # done ones too: their slots and extents go back
        self.kept_on = False                                  # (and nothing kept from a failed round)
        for s in live:
            self._finish(s.sid)
        for k in list(self.kept):
            self._drop(k)
        if self.broken is None:
            self.broken = RuntimeError("a round failed")
        return live

    @torch.no_grad()
    def follow(self) -> None:
        """Rank 1: mirror rank 0's admissions, prefill steps, rounds and completions, forever."""

        while True:
            msg = self.share(None)
            if not msg:                                # rank 0 has stopped (tests)
                return
            if msg[0] == ADMIT:
                sid, count, draft, stop_eos = msg[1:5]
                words, npacked = msg[5:5 + SAMPLING_WORDS], msg[5 + SAMPLING_WORDS]
                base, size, eid, kid, m, mode = msg[6 + SAMPLING_WORDS:12 + SAMPLING_WORDS]
                if (base >= 0) != (self.pool is not None):
                    raise RuntimeError("the ranks disagree on the shared cache pool")
                k = self._kid(kid) if kid >= 0 else None
                s = Stream(self.share(None), count, unpack_sampling(words), draft=bool(draft),
                           stop_eos=bool(stop_eos), sid=sid)
                if npacked:
                    from tensorfold.engine import grammar

                    s.constraint = grammar.compiler(self, self.model_dir, self.eos).follow(self.share(None))
                self._queue(s, base, size, eid, mode, k, m)    # rank 0's placement, replayed
            elif msg[0] == EVICT:
                self._drop(self._kid(msg[1]))
            elif msg[0] == GROW:
                self._resize(msg[1], msg[2])
            elif msg[0] == MOVE:
                self._move(msg[1], msg[2], msg[3])
            elif msg[0] == FILL:
                s = next(x for x in self.filling if x.sid == msg[1])
                first = self._step(s, msg[2])
                if first is not None:
                    s.out.append(first)
            elif msg[0] == ROUND:
                plan = [(msg[2 + 2 * i], msg[3 + 2 * i]) for i in range(msg[1])]
                for (sid, _), new in zip(plan, self._verify(plan)):
                    self.streams[sid].out.extend(new)
            elif msg[0] == DONE:
                for sid in msg[2:2 + msg[1]]:
                    self._finish(sid)


def stream_stats(s: Stream) -> dict[str, Any]:
    return s.stats()
