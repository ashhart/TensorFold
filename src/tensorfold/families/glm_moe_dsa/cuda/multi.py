"""Full GLM-5.3's concurrent streams (Phase A of docs/design/glm-moe-dsa-concurrency.md): up to N requests decoded
together, MTP drafts, every stream's reply token-identical to the same request alone.

Each stream owns a slot of the Runner's caches (``fused.State(slots=N)``: N streams' rows back to back). A decode round
packs every live stream's [pending token, k MTP drafts] into one verify window (4 streams x 3 rows = 12 <= FAST_ROWS,
so the RoCE one-shot reductions, decode tilings and rank-order sums of a lone request apply unchanged); the fused
kernels read each row's position and cache base from int32 device tables (``fused.Rows``) copied in before the call or
graph replay, so a row computes exactly what it computes alone. The MTP chain runs batched too: the first step takes
every drafting stream's backlog rows (target hiddens of the rows its last round kept), steps 2..k one row a stream.
Prompts fill one chunk per round between decode rounds, with the chunking a lone request uses (prompt chunks are not
row-invariant, so the chunk boundaries must match) and the one-stream kernels on the stream's slot (``State.view``).
While other streams decode, a chunk goes TF_GLM53_FILL_LAYERS layers a step (default 8; 0: whole chunks) with a
decode round between steps, so a long fill stalls the others for a few layers' time, not a chunk's: the same rows
through the same layers and kernels, only paused between layers (the chunk's rows wait in the Runner's prompt
buffers, which decode rounds never touch). A prompt filling while no stream decodes takes whole chunks, as alone.

Graphs: keyed by the window's shape (rows, streams, MTP step, key bucket, pick), captured on first use; the position
and base tables are static buffers, so one graph serves every position mix of that shape.

TP: rank 0 decides (admission, fills, rounds) and samples; ranks 1..3 follow its messages - ADMIT (+ prompt), FILL
(chunk, layer range; + the first token), ROUND (+ each stream's kept tokens), DONE - one fixed-size all-gather each, and run the same GPU
work. Follower ranks never sample, so their host work is small and they cannot disagree with rank 0.

Credits: the follower protocol and round structure follow TensorFold's families/qwen3_5/cuda/multi.py; per-row
positions and per-stream cache bases follow MiaAI-Lab's GLM-5.3-Flash multi-stream patches (Apache-2.0: 0029
glm-multi-dsa, 0030 glm-multi-stream-engine, 0035 glm-multi-rounds, 0049 glm-multi-prefill).
"""

from __future__ import annotations

import os
import time
from typing import Callable

import torch

from tensorfold.cuda.streams import Stream

from . import fused

ADMIT, ROUND, DONE, FILL = 1, 2, 3, 4      # rank 0's messages (an empty message: stop following)
MSG = 128                                  # ints a one-shot message carries (longer ones take a second all-gather)
FILL_LAYERS = int(os.environ.get("TF_GLM53_FILL_LAYERS", "8"))   # layers a fill step while others decode (0: a chunk)


class GlmMultiDecoder:
    """The Scheduler's decoder (live/admit/round/finish/drop) on rank 0; ``follow`` on ranks 1..3."""

    def __init__(self, runner, *, rank: int, world: int, comm, limit: int, eos: tuple[int, ...],
                 sample: Callable | None = None) -> None:
        """``sample(logits [1, V], position, sampling) -> token`` (rank 0; sampling None: greedy)."""
        w = runner.w
        self.runner, self.w, self.st = runner, w, runner.st
        self.N, self.k = runner.st.slots, runner.k
        self.rank, self.world, self.comm = rank, world, comm
        self.limit, self.eos, self.sample = limit, tuple(eos), sample
        self.local = runner.st.local
        rows = self.N * (self.k + 1)
        if rows > fused.FAST_ROWS:
            raise ValueError(f"{self.N} streams x {self.k + 1} rows = {rows} > {fused.FAST_ROWS}: a window that wide "
                             "would leave the decode reductions and tilings a lone request uses (lower --parallel or "
                             "--mtp-drafts)")
        dev = w.device
        self.rows = rows
        self.vb = fused.Buffers(w, rows, runner.cols)
        self.mb = fused.Buffers(w, rows, runner.cols) if self.k else None
        D = w.cfg.hidden_size
        self.bh = torch.zeros((self.N * (self.k + 1), D), dtype=torch.bfloat16, device=dev)   # MTP backlog hiddens
        self.bt = torch.zeros((self.N * (self.k + 1),), dtype=torch.long, device=dev)        # ... and tokens
        # the round's tables: target (pos, base, ids) then per MTP step j (pos, base, last rows, verify slots), then
        # the backlog rows step 1 gathers
        R, N = rows, self.N
        o = {"vpos": 0, "vbase": R, "vids": 2 * R}
        at = 3 * R
        for j in range(1, self.k + 1):
            o[f"mpos{j}"], o[f"mbase{j}"], o[f"last{j}"], o[f"vdst{j}"] = at, at + R, at + 2 * R, at + 2 * R + N
            at += 2 * R + 2 * N
        o["src"] = at
        at += R
        self.off, self.tab_n = o, at
        self.t64 = torch.zeros((at,), dtype=torch.long, device=dev)
        self.t32 = torch.zeros((at,), dtype=torch.int32, device=dev)
        t32 = lambda name, n: self.t32[o[name]:o[name] + n]          # noqa: E731
        self.vrows = fused.Rows(self.st, t32("vpos", R), t32("vbase", R), None, None)
        self.mrows = {j: fused.Rows(self.st, None, None, t32(f"mpos{j}", R), t32(f"mbase{j}", R))
                      for j in range(1, self.k + 1)}
        self.views = [self.st.view(s) for s in range(self.N)]
        self.free = list(range(self.N))
        self.streams: dict[int, Stream] = {}      # decoding
        self.filling: list[Stream] = []           # admitted, prompt chunks left (oldest first)
        self.next_id = 0
        self.broken: Exception | None = None
        self.rounds = 0
        self.fill_layers = FILL_LAYERS
        self.mid_rounds = 0                       # decode rounds run while a prompt chunk was paused between layers

    # -------------------------------------------------------------------------------------------- messages ---
    def _share(self, values: list[int] | None) -> list[int]:
        """Rank 0's int list on every rank: one all-gather of MSG ints (a longer list: one more for the rest)."""
        dev = self.w.device
        buf = torch.zeros((MSG,), dtype=torch.long, device=dev)
        if self.rank == 0:
            n = len(values)
            head = [n] + list(values[:MSG - 1])
            buf[:len(head)] = torch.tensor(head, dtype=torch.long)
        if self.world == 1:
            got = buf
        else:
            got = torch.empty((self.world * MSG,), dtype=torch.long, device=dev)
            self.comm.all_gather(buf, got)
        first = got[:MSG].tolist()
        n = int(first[0])
        out = first[1:1 + min(n, MSG - 1)]
        rest = n - len(out)
        if rest > 0:
            tail = (torch.tensor(values[MSG - 1:], dtype=torch.long, device=dev) if self.rank == 0
                    else torch.zeros((rest,), dtype=torch.long, device=dev))
            if self.world > 1:
                allt = torch.empty((self.world * rest,), dtype=torch.long, device=dev)
                self.comm.all_gather(tail, allt)
                tail = allt[:rest]
            out += tail.tolist()
        return out

    def _send(self, values: list[int]) -> None:
        if self.world > 1 and self.broken is None:
            self._share(values)

    def _recv(self) -> list[int]:
        return self._share(None)

    def _check(self) -> None:
        if self.broken is not None:
            raise RuntimeError("the ranks are out of step after an error; restart all four") from self.broken

    # --------------------------------------------------------------------------------------------- streams ---
    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def nbytes_per_stream(self) -> int:
        return self.st.nbytes() // self.N

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Queue a request in a free slot; rounds fill its prompt a chunk at a time."""
        self._check()
        if len(s.prompt) >= self.limit:
            raise ValueError(f"prompt of {len(s.prompt)} tokens: this engine serves contexts up to {self.limit}")
        if not self.free:
            raise RuntimeError("no free stream slot (the scheduler admits at most --parallel streams)")
        s.count = max(1, min(int(s.count), self.limit - len(s.prompt)))
        s.sid = self.next_id
        self.next_id += 1
        temp = getattr(s.sampling, "temperature", 0.0) if s.sampling is not None else 0.0
        if temp <= 0:
            s.sampling = None
        slot = self.free.pop(0)
        self._send([ADMIT, s.sid, slot, int(s.draft and self.k > 0), int(s.sampling is not None)])
        self._send(list(s.prompt))
        self._queue(s, slot)

    def _queue(self, s: Stream, slot: int) -> None:
        s.slot = slot
        s.chunks = self.runner.chunks(len(s.prompt))
        s.ci = 0
        s.li = 0                                  # the current chunk's next layer
        s.toks = torch.tensor(s.prompt, dtype=torch.long, device=self.w.device)
        self.filling.append(s)

    def _d(self, s: Stream) -> int:
        return self.k if s.draft and self.k else 0

    # ------------------------------------------------------------------------------------------------ fill ---
    def _fill(self, alone: bool) -> list[Stream]:
        """The oldest queued prompt's next step: the rest of its current chunk (the chunking it has alone) when no
        stream decodes (``alone``), else the chunk's next fill_layers layers; at the prompt's end, its first token."""
        s = self.filling[0]
        a, e = s.chunks[s.ci]
        n, G = len(self.w.layers), self.fill_layers
        hi = n if alone or G <= 0 else min(n, s.li + G)
        self._send([FILL, s.sid, a, e, s.li, hi])
        t0 = time.perf_counter()
        try:
            first = self._chunk(s, a, e, s.li, hi)
            if first is not None:
                tok = int(self.sample(first, len(s.prompt), s.sampling))
                self._send([tok])
                self._start(s, tok)
        except Exception as exc:
            if self.world > 1:
                self.broken = exc
            raise
        finally:
            s.prefill_s += time.perf_counter() - t0
        if first is None:
            return []
        s.take([tok], self._ends(s))
        return [s] if s.done else []

    def _chunk(self, s: Stream, a: int, e: int, lo: int, hi: int):
        """Layers [lo, hi) of chunk a..e (all of them: the chunk in one go, as alone)."""
        n = len(self.w.layers)
        if lo != s.li:
            raise RuntimeError(f"fill step at layer {lo}, the chunk is at layer {s.li}")
        layers = None if (lo, hi) == (0, n) else (lo, hi)
        out = self.runner.prefill_chunk(self.views[s.slot], s.toks, a, e, len(s.prompt), layers=layers)
        if hi < n:
            s.li = hi
            return None
        s.li = 0
        s.ci += 1
        return out

    def _start(self, s: Stream, tok: int) -> None:
        """Prompt done: the pending token at P = len(prompt); the MTP backlog (carry = hidden P - 1, token)."""
        L0 = len(s.prompt)
        s.P, s.tok, s.m = L0, tok, 1
        if self.k:
            i = s.slot * (self.k + 1)
            self.bh[i].copy_(self.runner.carry)
            self.bt[i:i + 1].fill_(tok)
        s.toks = None
        s.started = time.perf_counter()
        self.filling = [x for x in self.filling if x is not s]
        self.streams[s.sid] = s

    # ----------------------------------------------------------------------------------------------- rounds ---
    @torch.no_grad()
    def round(self) -> list[Stream]:
        """A prompt chunk for the oldest queued prompt, then one decode round over the decoding streams; returns the
        streams that finished."""
        self._check()
        decoding = any(not s.done for s in self.streams.values())
        done = self._fill(not decoding) if self.filling else []
        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return done
        if self.filling and self.filling[0].li:
            self.mid_rounds += 1
        plan = [(s.sid, s.slot, s.P, s.tok, s.m, self._d(s), int(s.sampling is not None)) for s in live]
        self._send([ROUND, len(plan), *[x for item in plan for x in item]])
        try:
            offs, R = self._gpu(plan)
            results = self._picks(plan, live, offs, R)
            self._send([x for n, emit in results for x in (n, len(emit), *emit)])
            self._commit(plan, offs, results)
        except Exception as exc:
            if self.world > 1:
                self.broken = exc
            raise
        self.rounds += 1
        for s, item, (n, emit) in zip(live, plan, results):
            s.counted(1 + item[5])
            new = []
            for t in emit:                                   # a lone request's room: its count, its end tokens
                if len(s.out) + len(new) >= s.count:
                    break
                new.append(t)
                if t in self._ends(s):
                    break
            s.P, s.tok, s.m = s.P + n + 1, emit[-1], n + 1
            s.take(new, self._ends(s))
        return done + [s for s in live if s.done]

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.eos if s.stop_eos else ()

    def _gpu(self, plan) -> tuple[list[int], int]:
        """Every rank: the round's tables, the batched MTP chain, the verify window. Returns row offsets and R."""
        w, rn, k, o = self.w, self.runner, self.k, self.off
        vb, mb = self.vb, self.mb
        tab = [0] * self.tab_n
        offs, r = [], 0
        for sid, slot, P, tok, m, d, _ in plan:
            offs.append(r)
            for i in range(1 + d):
                tab[o["vpos"] + r + i] = P + i
                tab[o["vbase"] + r + i] = slot * self.local
            tab[o["vids"] + r] = tok
            r += 1 + d
        R = r
        drafting = [(item, off) for item, off in zip(plan, offs) if item[5]]
        S = len(drafting)
        M = 0
        if S:
            for si, ((sid, slot, P, tok, m, d, _), off) in enumerate(drafting):
                for i in range(m):
                    tab[o["mpos1"] + M] = P - m + i
                    tab[o["mbase1"] + M] = slot * self.local
                    tab[o["src"] + M] = slot * (k + 1) + i
                    M += 1
                tab[o["last1"] + si] = M - 1
                tab[o["vdst1"] + si] = off + 1
                for j in range(2, k + 1):
                    tab[o[f"mpos{j}"] + si] = P + j - 2
                    tab[o[f"mbase{j}"] + si] = slot * self.local
                    tab[o[f"last{j}"] + si] = si
                    tab[o[f"vdst{j}"] + si] = off + j
        self.t64.copy_(torch.tensor(tab, dtype=torch.long))
        self.t32.copy_(self.t64)
        vb.ids[:R].copy_(self.t64[o["vids"]:o["vids"] + R])
        if S:
            src = self.t64[o["src"]:o["src"] + M]
            torch.index_select(self.bh, 0, src, out=mb.hin[:M])
            torch.index_select(self.bt, 0, src, out=mb.ids[:M])
            Tm = rn._T(max(item[2] for item, _ in drafting) + k)
            for j in range(1, k + 1):
                self._mtp_step(j, M if j == 1 else S, S, Tm)
        T = rn._T(max(P + 1 + d for _, _, P, _, _, d, _ in plan))
        pick = "full" if any(item[6] for item in plan) else "argmax"
        rows = self.vrows
        rn.G.run(("mt", R, T, pick), lambda: fused.compute(w, rows, vb, R, T, logits="all", pick=pick))
        return offs, R

    def _mtp_step(self, j: int, n: int, S: int, Tm: int | None) -> None:
        w, rn, mb, vb, o = self.w, self.runner, self.mb, self.vb, self.off
        cn, full = rn.chain_normed, rn.draft_full
        rows = self.mrows[j]
        last = self.t64[o[f"last{j}"]:o[f"last{j}"] + S]
        vdst = self.t64[o[f"vdst{j}"]:o[f"vdst{j}"] + S]

        def fn():
            fused.mtp_compute(w, rows, mb, n, Tm, chain_normed=cn, draft_full=full, last=last)
            vb.ids.index_copy_(0, vdst, mb.argmax[:S])
            mb.ids[:S].copy_(mb.argmax[:S])
            torch.index_select(mb.hidden, 0, last, out=mb.hin[:S])
        rn.G.run(("mm", n, S, j, Tm, cn, full), fn)

    def _picks(self, plan, live, offs, R) -> list[tuple[int, list[int]]]:
        """Rank 0: each stream's target picks along its drafts -> (drafts kept, tokens emitted)."""
        vb = self.vb
        both = torch.cat([vb.argmax[:R], vb.ids[:R]]).tolist()
        amax, ids = both[:R], both[R:]
        out = []
        for s, (sid, slot, P, tok, m, d, sampled), off in zip(live, plan, offs):
            drafts = ids[off + 1:off + 1 + d]
            if not sampled:
                picks = amax[off:off + 1 + d]
            else:
                picks = []
                for i in range(1 + d):
                    picks.append(int(self.sample(vb.logits[off + i:off + i + 1], P + i + 1, s.sampling)))
                    if i >= d or drafts[i] != picks[-1]:
                        break
            n = 0
            while n < d and n < len(picks) - 1 and drafts[n] == picks[n]:
                n += 1
            out.append((n, picks[:n + 1]))
        return out

    def _commit(self, plan, offs, results) -> None:
        """Every rank: each drafting stream's next MTP backlog - the kept rows' target hiddens and tokens."""
        if not self.k:
            return
        idx, dst, toks = [], [], []
        for (sid, slot, P, tok, m, d, _), off, (n, emit) in zip(plan, offs, results):
            if not d:
                continue
            for i in range(n + 1):
                idx.append(off + i)
                dst.append(slot * (self.k + 1) + i)
                toks.append(emit[i])
        if not idx:
            return
        c = len(idx)
        t = torch.tensor(idx + dst + toks, dtype=torch.long).to(self.w.device)
        src = self.vb.hidden.index_select(0, t[:c])
        if self.runner.hid_normed:
            out = self.vb.normed[:c]
            from tensorfold.families.glm5_next.cuda import glue

            glue.rmsnorm(src, self.w.final_norm, self.w.cfg.rms_norm_eps, out)
        else:
            out = src
        self.bh.index_copy_(0, t[c:2 * c], out)
        self.bt.index_copy_(0, t[c:2 * c], t[2 * c:])

    @torch.no_grad()
    def prewarm(self) -> None:
        """Every rank (no messages): capture the windows of 1..N drafting streams with every backlog total (S..S(k+1)
        MTP rows), in every key bucket, both picks; other mixes (streams without drafts) capture on first use. Caches
        get scratch values at the positions used; every request writes a position before reading it."""
        t0 = time.perf_counter()
        rn, k = self.runner, self.k
        cap = rn.capacity - k - 2
        before = len(rn.G.graphs)
        for T in rn.buckets():
            P = 10 if T is None else min(T // 2 + 100, cap)
            if rn._T(P + k + 1) != T:
                continue
            for S in range(1, self.N + 1):
                for M in (range(S, S * (k + 1) + 1) if k else (S,)):
                    ms = [M // S + (i < M % S) for i in range(S)]
                    for sampled in ((0, 1) if M == S else (0,)):
                        self._gpu([(i, i, P, 1000, ms[i], k, sampled) for i in range(S)])
        torch.cuda.synchronize()
        print(f"[tensorfold] rank {self.w.rank}: {len(rn.G.graphs) - before} concurrent decode graphs captured in "
              f"{time.perf_counter() - t0:.1f}s", flush=True)

    # ------------------------------------------------------------------------------------------- endings ---
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
                self.filling = [x for x in self.filling if x is not s]
        if s is not None:
            s.toks = None
            if s.slot not in self.free:
                self.free.append(s.slot)
                self.free.sort()

    def drop(self) -> list[Stream]:
        """After an error in a round: forget the live and queued streams (the ranks may no longer agree)."""
        live = list(self.streams.values()) + self.filling
        for s in live:
            self._finish(s.sid)
        self.streams, self.filling = {}, []
        if self.world > 1 and self.broken is None:
            self.broken = RuntimeError("a round failed")
        return live

    def stop(self) -> None:
        """Rank 0: release the followers (an empty message)."""
        self._send([])

    # ------------------------------------------------------------------------------------------- followers ---
    @torch.no_grad()
    def follow(self) -> None:
        """Ranks 1..3: mirror rank 0's admissions, fills, rounds and endings until it sends an empty message."""
        while True:
            msg = self._recv()
            if not msg:
                return
            kind = msg[0]
            if kind == ADMIT:
                sid, slot, draft, sampled = msg[1:5]
                s = Stream(self._recv(), 1, None, draft=bool(draft), sid=sid)
                s.sampled = bool(sampled)
                self.free = [x for x in self.free if x != slot]
                self._queue(s, slot)
            elif kind == FILL:
                sid, a, e, lo, hi = msg[1:6]
                s = next(x for x in self.filling if x.sid == sid)
                if self._chunk(s, a, e, lo, hi) is not None:
                    self._start(s, self._recv()[0])
            elif kind == ROUND:
                n = msg[1]
                plan = [tuple(msg[2 + 7 * i:9 + 7 * i]) for i in range(n)]
                offs, R = self._gpu(plan)
                flat = self._recv()
                results, i = [], 0
                for _ in plan:
                    kept, c = flat[i], flat[i + 1]
                    results.append((kept, flat[i + 2:i + 2 + c]))
                    i += 2 + c
                self._commit(plan, offs, results)
                self.rounds += 1
            elif kind == DONE:
                for sid in msg[2:2 + msg[1]]:
                    self._finish(sid)
