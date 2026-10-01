"""A lone stream's graph slot, with cache-pointer invalidation and deferred recurrence commits."""

from __future__ import annotations

import torch

from tensorfold.cuda.kernels import gdn
from tensorfold.cuda.logprobs import capture
from tensorfold.cuda.streams import Stream, accept

from .decode import Engine, draft
from .forward import commit
from .state import Buffers


def solo(w, st, capacity, depth, pbuf):
    from .graphs import Graphs

    if any(getattr(layer.moe.experts, "capturable", True) is False for layer in w.layers):
        return None
    rows = max(8, depth + 1)
    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows = w, capacity, rows, pbuf.rows
    e.kv_dtype, e.st = st.kv_dtype, st
    e.buf = Buffers(w, rows, capacity, moe_prefill=True)       # the serial engine and shared rounds use these bits
    e.mbuf, e.pbuf = Buffers(w, rows, capacity), pbuf
    e.graphs = Graphs(e, max_rows=rows)
    return e


class Alone:
    def _state_changed(self, st) -> None:
        """Drop graphs before reallocating a slot: graph pointers must never outlive its cache geometry."""

        if not self.planning and self.solo is not None and st is self.solo.st:
            from .graphs import Graphs

            self.solo.graphs = Graphs(self.solo, max_rows=self.solo.rows)

    def _flush(self, s) -> None:
        """Materialize the shared round's deferred recurrent rows before a graph reads or copies the slot."""

        rows = self.held.pop(s.sid, [])
        if not rows:
            return
        if self.planning:
            self.actions.append(["flush", s.sid])
            return
        sc, st = self.gdn, s.st
        parity = 1 - sc.parity
        if not sc.lin:
            return
        k, v, g, beta = [[getattr(sc, name)[parity, li] for li in range(sc.lin)]
                         for name in ("k", "v", "g", "beta")]
        ptrs = gdn.replay_table(k, v, g, beta, [[st.rec[st.cur[li], li] for li in range(sc.lin)]])
        table = gdn.to_device(ptrs, torch.int64, self.w.device)
        kept = gdn.to_device(rows, torch.int32, self.w.device).view(1, -1)
        counts = gdn.to_device([len(rows)], torch.int32, self.w.device)
        gdn.replay(table, sc.lin, 1, kept, counts, k[0], v[0], in_place=True)

    def _relocate_kept(self, target, avoid) -> bool:
        """Move the kept prompt end held in the graph slot into a free slot of its size (one GPU), so a different lone
        request can take the slot without dropping its graphs; False when no free slot fits without evicting."""

        if self.planning or getattr(self.w, "comm", None) is not None:
            return False                         # two ranks name kept ends by slot: they keep the slot swap
        spare = next((f for f in self.free if f is not target and f is not avoid), None)
        if spare is None:
            return False
        size = target.capacity
        if spare.capacity != size:
            self._shrink(spare, release=True)    # back to its first rows (a no-op when already there)
            need = spare.cache_bytes(size) - spare.cache_bytes() + spare.layer_bytes(size)
            if not self.memory_gate.fits(need):
                return False
            self.memory_gate.take(spare.resize(size))
        spare.copy_from(target)
        self.free = [f for f in self.free if f is not spare]
        self.kept = [(ids, spare if st is target else st, snap, tail) for ids, st, snap, tail in self.kept]
        return True

    def _move_to_solo(self, s) -> None:
        """Copy a lone stream into the graph slot after committing its pending rows and matching cache sizes."""

        target, old = self.solo.st, s.st
        self._flush(s)
        if any(k[1] is target for k in self.kept) and not self._relocate_kept(target, old):
            self.solo.st = old                   # preserve both prefix chains instead of evicting a kept slot
            if self.planning:
                self.actions.append(["solo", self._index(old)])
            else:
                self._state_changed(old)
            return
        self._drop_kept(target)
        self.free = [f for f in self.free if f is not target]
        self._shrink(target)
        if not self._grow(target, old.capacity, alone=True):
            raise RuntimeError("the lone stream's graph slot cannot hold its existing cache")
        target.copy_from(old)
        s.st = target
        if not any(k[1] is old for k in self.kept) and all(f is not old for f in self.free):
            self._shrink(old)
            self.free.append(old)

    def _solo_round(self, s: Stream) -> list[Stream]:
        """Verify a lone stream through its graphs, commit accepted rows and draft the next chain."""

        e, st = self.solo, s.st
        tokens = [s.out[-1]] + list(s.drafts)
        R = len(tokens)
        logits = e.forward(tokens)
        rows = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], s.sampling)
        path, end = accept(tokens, list(range(-1, R - 1)), rows, s.count - len(s.out), self._ends(s))
        if s.probabilities is not None:
            capture(logits, [tokens[r] for r in path[1:]] + [end],
                    [st.pos + 1 + r for r in path], s.probabilities, rows=path)
        commit(self.w, st, e.buf, R, len(path))
        s.committed.extend(tokens[:len(path)])
        s.counted(R)
        new = [tokens[r] for r in path[1:]] + [end]
        last = len(s.out) + len(new) >= s.count or end in self._ends(s)
        s.drafts = []
        room = min(self.depth, s.count - len(s.out) - len(path))
        if not last and room > 0:
            s.drafts = draft(e, e.buf.streams[:len(path)], rows[:len(path)], st.pos + 1, room, s.sampling,
                             self.confidence)
        s.take(new, self._ends(s))
        return [s] if s.done else []
