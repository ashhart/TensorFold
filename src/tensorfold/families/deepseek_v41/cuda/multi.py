"""DeepSeek-V4.1's concurrent rounds (``tensorfold serve --parallel N``): N streams in N cache slots, each round's rows
of every stream verified in one forward (``SerialEngine.step_multi``), so the weights a round reads serve them all.

Rows are row-invariant, so a stream's tokens equal its serial decoding however many streams share its rounds. Rank 0
decides (admissions, prefill steps, each stream's draft count, completions) and sends every decision to rank 1 before
acting on it; both ranks then compute the same tokens (argmax or position-keyed samples of the gathered logits).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from tensorfold.cuda.sampling import sample_rows
from tensorfold.cuda.streams import Stream, next_fill

from .serial import MAX_ROWS, SerialEngine

ADMIT, FILL, ROUND, DONE = 1, 2, 3, 4      # rank 0's messages
ROWS = 16                                  # a round's rows at most (above it, kernels switch to the prompt path)
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
                 drafts: int = 3, step: int = MAX_ROWS) -> None:
        self.e, self.share, self.rank = e, share, rank
        self.drafts = drafts if e.drafter is not None else 0
        import os

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
        ms = []
        for R in range(1, ROWS + 1):
            if R not in e.graphs:
                ms.append(ms[-1] if ms else 30.0)
                continue
            g = e.graphs[R]
            g["tok"].copy_(torch.tensor([rng.randrange(1000, 100000) for _ in range(R)]))
            g["pos"].copy_(torch.arange(R) + 200)
            g["sid"].copy_(torch.arange(R) % e.slots)
            best = float("inf")
            for _ in range(4):
                torch.cuda.synchronize()
                t = time.perf_counter()
                g["a0"].replay(), g["a1"].replay(), g["b"].replay()
                torch.cuda.synchronize()
                best = min(best, time.perf_counter() - t)
            ms.append(1e3 * best)
        draft = 0.0
        if self.drafts:
            best = float("inf")
            for _ in range(4):
                torch.cuda.synchronize()
                t = time.perf_counter()
                e.drafter.propose(rng.randrange(1000, 100000), 300)
                best = min(best, time.perf_counter() - t)
            draft = 1e3 * best
        both = gather([int(1e3 * v) for v in ms] + [int(1e3 * draft)])
        worst = [max(a, b) / 1e3 for a, b in zip(*both)]
        self.costs, self.draft_ms = worst[:ROWS], worst[ROWS]
        for slot in range(e.slots):                          # the timing rows wrote every slot's caches
            e.select_slot(slot)
            e.reset()
        e.select_slot(0)

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
            room = self.e.limit - len(self.e.views[s.slot].ids) - 1
            ok = s.draft and s.constraint is None and self.drafts
            caps.append(max(0, min(self.drafts, room, s.count - len(s.out) - 1)) if ok else 0)
        if not any(caps) or self.costs is None:
            return ks
        rows, drafting = len(live), 0
        tokens = float(len(live))
        rate = tokens / (self.costs[rows - 1] + self.overhead)
        while rows < ROWS:
            best = None
            for i, s in enumerate(live):
                if ks[i] >= caps[i]:
                    continue
                gain = self._expected(s.acc, ks[i] + 1) - self._expected(s.acc, ks[i])
                cost = self.costs[rows] + self.overhead + (drafting + (ks[i] == 0)) * self.draft_ms
                r = (tokens + gain) / cost
                if r > rate and (best is None or r > best[0]):
                    best = (r, i, gain)
            if best is None:
                break
            rate, i, gain = best
            drafting += ks[i] == 0
            ks[i] += 1
            tokens += gain
            rows += 1
        return ks

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
        s.sid = self.next_id
        self.next_id += 1
        from tensorfold.engine.grammar import pack

        packed = pack(s.constraint)
        self._send([ADMIT, s.sid, s.count, int(s.draft), int(s.stop_eos), *pack_sampling(s.sampling), len(packed)])
        self._send(list(s.prompt))
        if packed:
            self._send(packed)
        self._queue(s)

    def _queue(self, s: Stream) -> None:
        s.acc = list(self.prior)
        s.slot = self.free.pop(0)
        s.pos = -1                                 # prompt tokens prefilled so far (-1: not started)
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
            logits = e.prefill(s.prompt[s.pos:stop])
            s.pos = stop
            if stop < n:
                return None
            if s.draft:
                e.keep_prompt(s.prompt)
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

    # -- rounds ---------------------------------------------------------------------------------------------------
    def _plan(self, live: list[Stream]) -> list[tuple[int, int]]:
        """(sid, drafts) a stream: drafts while the round's rows fit (none for a grammar's stream yet)."""

        return [(s.sid, k) for s, k in zip(live, self._allocate(live))]

    @torch.no_grad()
    def round(self) -> list[Stream]:
        self._check()
        tr = time.perf_counter()
        busy = any(not x.done for x in self.streams.values())
        share = self.fill_share / max(1e-6, 1.0 - self.fill_share)
        done = []
        if self.filling and (not busy or self.t_fill <= share * self.t_decode):   # time-sliced while others decode
            done = self._fill()
            if busy:
                self.t_fill += time.perf_counter() - tr
        if self.prof is not None and self.rank == 0:
            self.prof["fill"] += time.perf_counter() - tr
        live = [s for s in self.streams.values() if not s.done]
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
            for sid, k in plan:
                s = self.streams[sid]
                pending = s.out[-1]
                drafts: list[int] = []
                if k:
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
                          f"{1e3 * p['round'] / n:.1f} ms, {p['rows'] / n:.1f} rows, {p['streams'] / n:.1f} streams",
                          flush=True)
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
        s = self.streams.pop(sid, None)
        if s is None:
            s = next((x for x in self.filling if x.sid == sid), None)
            if s is not None:
                self.filling.remove(s)
        if s is not None and s.slot not in self.free:
            self.free.append(s.slot)
            self.free.sort()

    def drop(self) -> list[Stream]:
        live = [s for s in self.streams.values() if not s.done] + self.filling
        for s in live:
            self._finish(s.sid)
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
                s = Stream(self.share(None), count, unpack_sampling(words), draft=bool(draft),
                           stop_eos=bool(stop_eos), sid=sid)
                if npacked:
                    from tensorfold.engine import grammar

                    s.constraint = grammar.compiler(self, self.model_dir, self.eos).follow(self.share(None))
                self._queue(s)
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
