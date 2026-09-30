"""Flash Next's concurrent rounds on one GPU: every stream keeps exactly its own accepted prefix."""

from __future__ import annotations

import time

import numpy as np
import torch

from tensorfold.cuda.sampling import sample_streams
from tensorfold.cuda.streams import Stream, accept
from tensorfold.engine.exact_sampling import MARGIN, choose_rows
from tensorfold.engine.grammar import GrammarError

from .decode import PREFILL_ROWS, WARM_TAIL, Engine, draft, prefill
from .forward import commit, compute, stage
from .mtp import mtp_compute, mtp_stage
from .state import Buffers, State
from ..cuda import CONFIDENCE, DEPTH

def _slot(w, st: State, buf: Buffers, mbuf: Buffers, pbuf: Buffers, capacity: int) -> Engine:
    """A one-sequence engine over a slot's state and the shared buffers (eager: no CUDA graphs)."""

    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows = w, capacity, buf.rows, pbuf.rows
    e.buf, e.mbuf, e.pbuf, e.st, e.graphs = buf, mbuf, pbuf, st, None
    return e


class MultiDecoder:
    """Rounds over the live streams; ``slots`` streams at most, each with ``capacity`` tokens of context."""

    def __init__(self, w, *, slots: int, capacity: int, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 stop_eos: bool = True, keep: int = 8, kv_dtype: str = "bf16") -> None:
        if w.comm is not None:
            raise ValueError("concurrent Flash Next runs on one GPU for now")
        self.w, self.depth, self.confidence, self.capacity = w, depth, confidence, capacity
        self.eos = tuple(w.cfg.eos) if stop_eos else ()
        rows = slots * (depth + 1)
        self.buf = Buffers(w, rows, capacity)
        self.mbuf = Buffers(w, rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, PREFILL_ROWS, capacity, prefill=True)
        self.free = [State(w, capacity, depth + 1, kv_dtype) for _ in range(slots)]   # sized by the startup admission
        self.slot_bytes = sum(t.numel() * t.element_size() for t in _tensors(self.free[0]))
        self.streams: dict[int, Stream] = {}
        self.next_id = 0
        self.draft_host = w.draft_ids.cpu().numpy() if w.draft_ids is not None else None
        self.kept: list[tuple[list[int], State, dict, torch.Tensor | None]] = []   # (ids, slot, snapshot, tail)
        self.keep = keep

    def _busy(self) -> set[int]:
        return {id(s.st) for s in self.streams.values()}

    def _drop_kept(self, st: State) -> None:
        self.kept = [k for k in self.kept if k[1] is not st]

    def _slot_for(self, prompt: list[int], reuse: bool):
        """The idle kept slot the prompt extends furthest, else a free slot, else the oldest idle kept one."""

        busy = self._busy()
        best = None
        for k in self.kept if reuse else []:
            ids, st = k[0], k[1]
            if id(st) not in busy and len(ids) < len(prompt) and prompt[:len(ids)] == ids and \
                    (best is None or len(ids) > len(best[0])):
                best = k
        if best is not None:
            self._drop_kept(best[1])
            return best[1], {"state": best[2], "tail": best[3]}, len(best[0])
        if not self.free:
            idle = next((k[1] for k in self.kept if id(k[1]) not in busy), None)
            if idle is None:
                raise RuntimeError("no free stream slot")
            self._drop_kept(idle)
            self.free.append(idle)
        return self.free.pop(), None, 0

    def _remember(self, ids: list[int], st: State, snap: dict, tail) -> None:
        gone = [k[1] for k in self.kept if k[0] == ids]
        self.kept = [k for k in self.kept if k[0] != ids] + [(ids, st, snap, tail)]
        while len(self.kept) > self.keep:
            gone.append(self.kept.pop(0)[1])
        busy = self._busy()
        for old in gone:           # a displaced idle slot no kept entry holds goes back to the free list
            if old is not st and id(old) not in busy and all(k[1] is not old for k in self.kept) and \
                    all(f is not old for f in self.free):
                self.free.append(old)

    def live(self) -> int:
        return len(self.streams)

    @torch.no_grad()
    def warm(self) -> None:
        """A synthetic greedy request through prefill, its drafts and one round, then forgotten, so no request compiles or loads a kernel."""

        s = Stream([0] * min(PREFILL_ROWS + WARM_TAIL, self.capacity - self.depth - 2), 2)
        self.admit(s)
        if not s.done:
            self.round()
        self.streams.pop(s.sid, None)
        self._drop_kept(s.st)
        if all(f is not s.st for f in self.free):
            self.free.append(s.st)

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Prefill a request in a free slot, draft its first chain and emit its first token."""

        room = self.capacity - len(s.prompt) - self.depth - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.capacity}-token context")
        s.count = max(1, min(s.count, room))
        t0 = time.perf_counter()
        st, resume, s.cached = self._slot_for(list(s.prompt), s.draft)
        e = _slot(self.w, st, self.buf, self.mbuf, self.pbuf, self.capacity)
        mtp = s.draft and self.depth > 0 and self.mbuf is not None
        try:
            first = prefill(e, s.prompt, s.sampling, mtp=mtp, resume=resume,
                            **({} if s.constraint is None else {"constraint": s.constraint}))
        except Exception:
            self.free.append(st)
            raise
        s.sid, s.st = self.next_id, st
        self.next_id += 1
        if s.draft:                    # the prompt's state; the MTP head has absorbed every position but the last
            self._remember(list(s.prompt), st, st.snapshot(), e.last_streams.clone() if mtp else None)
        s.context = list(s.prompt)
        s.drafts = draft(e, e.last_streams, [first], st.pos + 1, min(self.depth, s.count - 1), s.sampling,
                         self.confidence) if mtp and s.count > 1 else []
        s.prefill_s, s.started = time.perf_counter() - t0, time.perf_counter()
        self.streams[s.sid] = s
        s.take([first], self._ends(s))

    def _ends(self, s: Stream) -> tuple[int, ...]:
        """The end tokens that end this stream: none when its request ignores them (``ignore_eos``)."""

        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """One round over the live streams; returns the ones that finished."""

        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return []
        grammars, failed = {}, []
        for s in live:                                   # a grammar cuts the drafts no accepted path can hold
            if s.constraint is not None:
                tokens = [s.out[-1]] + list(s.drafts)
                try:
                    grammars[s.sid] = s.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
                except GrammarError as exc:              # this request ends with its error, the others go on
                    s.error, s.done = exc, True
                    failed.append(s)
                    continue
                s.drafts = grammars[s.sid].tokens[1:]
        live = [s for s in live if not s.done]
        if not live:
            return failed
        windows = [(s.st, [s.out[-1]] + list(s.drafts)) for s in live]
        segs = stage(self.w, self.buf, windows)
        logits = compute(self.w, segs, self.buf)
        starts = [a0 for _, a0, _ in segs] + [segs[-1][2]]
        for s, (_, a0, a1) in zip(live, segs):
            if s.sid in grammars:
                s.constraint.mask(logits[a0:a1], grammars[s.sid])
        positions = [[st.pos + 1 + r for r in range(a1 - a0)] for st, a0, a1 in segs]
        sampled = sample_streams(logits, starts, positions, [s.sampling for s in live])
        kept = []
        for s, (_, tokens), (st, a0, a1), rows in zip(live, windows, segs, sampled):
            path, end = accept(tokens, list(range(-1, len(tokens) - 1)), rows, s.count - len(s.out), self._ends(s))
            commit(self.w, st, self.buf, a1 - a0, len(path), at=a0)
            s.committed.extend(tokens[:len(path)])
            s.counted(len(tokens))
            new = [tokens[r] for r in path[1:]] + [end]
            if s.constraint is not None:
                try:
                    s.constraint.advance(new)
                except GrammarError as exc:
                    s.error = exc
            last = s.error is not None or len(s.out) + len(new) >= s.count or end in self._ends(s)
            kept.append((s, a0, rows[:len(path)], new, last))
        self._draft_all([(s, a0, keep) for s, a0, keep, _, last in kept if s.draft and not last])
        for s, _, _, new, _ in kept:
            if s.error is not None:
                s.done = True
                continue
            s.take(new, self._ends(s))
        return failed + [s for s in live if s.done]

    def _draft_all(self, streams: list) -> None:
        """Every drafting stream absorbs its kept rows and chains drafts, all streams in one step a depth."""

        for s, _, _ in streams:
            s.drafts = []
        room = {s.sid: min(self.depth, s.count - len(s.out) - len(keep)) for s, _, keep in streams}
        todo = [(s, a0, keep) for s, a0, keep in streams if room[s.sid] > 0 and self.mbuf is not None]
        if not todo:
            return
        for s, _, _ in todo:
            st = s.st
            if st.mtp_drafted:
                st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                st.mtp_drafted = 0
        windows = [(s.st, keep, self.buf.streams[a0:a0 + len(keep)]) for s, a0, keep in todo]
        segs = mtp_stage(self.w, self.mbuf, windows)
        logits = mtp_compute(self.w, segs, self.mbuf)
        for (s, _, keep), (st, a0, a1) in zip(todo, segs):
            st.set_mtp_len(st.mtp_len + len(keep))
        active = [(s, a1 - 1) for s, (_, _, a1) in zip([t[0] for t in todo], segs)]
        for j in range(self.depth):
            picks = self._picks(logits, [s.st.pos + 1 + j for s, _ in active], [s.sampling for s, _ in active])
            nxt = []
            for (s, row), (d, p) in zip(active, picks):
                low = self.confidence > 0 and p < self.confidence
                if low and j > 0:
                    continue
                s.drafts.append(d)
                if not low and j + 1 < room[s.sid]:
                    nxt.append((s, row, d))
            if not nxt:
                return
            windows = [(s.st, [d], self.mbuf.streams[row:row + 1]) for s, row, d in nxt]
            segs = mtp_stage(self.w, self.mbuf, windows)
            logits = mtp_compute(self.w, segs, self.mbuf)
            for s, _, _ in nxt:
                s.st.set_mtp_len(s.st.mtp_len + 1)
                s.st.mtp_drafted += 1
            active = [(s, a0) for (s, _, _), (_, a0, _) in zip(nxt, segs)]

    def _picks(self, logits: torch.Tensor, positions: list[int], samplings: list) -> list[tuple[int, float]]:
        """Each row's keyed draft and its probability at temperature 1, one read-back (drafts change speed only)."""

        row = logits.float()
        k = max([int(s.top_k) + MARGIN for s in samplings if s is not None and s.temperature > 0 and s.top_k] or [1])
        k = min(k, row.shape[1])
        vals, idx = torch.topk(row, k, dim=-1, sorted=False)
        top, col = row.max(dim=-1, keepdim=True)
        lse = torch.logsumexp(row, dim=-1, keepdim=True)
        got = torch.cat([vals, idx.float(), top, col.float(), lse], dim=1).cpu().numpy()
        out = []
        for i, (pos, smp) in enumerate(zip(positions, samplings)):
            g = got[i]
            lse_i = float(g[2 * k + 2])
            if smp is None or smp.temperature <= 0:
                c = int(g[2 * k + 1])
                out.append((int(self.draft_host[c]) if self.draft_host is not None else c,
                            float(np.exp(float(g[2 * k]) - lse_i))))
                continue
            cols = g[k:2 * k].astype(np.int64)
            ids = self.draft_host[cols] if self.draft_host is not None else cols
            tok = choose_rows(g[None, :k].astype(np.float32), ids[None, :], [pos], smp)[0]
            hit = np.nonzero(ids == tok)[0]
            out.append((int(tok), float(np.exp(float(g[hit[0]]) - lse_i)) if len(hit) else 0.0))
        return out

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams; a slot whose prompt end is kept stays with it, the rest are free again."""

        for s in done:
            self.streams.pop(s.sid, None)
            if not any(k[1] is s.st for k in self.kept):
                self.free.append(s.st)

    def drop(self) -> list[Stream]:
        live = [s for s in self.streams.values() if not s.done]
        for s in live:
            self.streams.pop(s.sid, None)
            self._drop_kept(s.st)
            self.free.append(s.st)
        return live


def _tensors(st: State):
    for value in vars(st).values():
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, torch.Tensor):
                yield v
            elif hasattr(v, "__dict__"):                  # scratch and KV cache objects, the MTP head's too
                yield from (t for t in vars(v).values() if isinstance(t, torch.Tensor))
