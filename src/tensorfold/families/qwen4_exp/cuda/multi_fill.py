"""Prompt pieces within Flash Next rounds, including the exact kept prompt end and first sampled token."""

from __future__ import annotations

import time

import torch

from tensorfold.cuda.logprobs import capture
from tensorfold.cuda.streams import Stream

from . import image_rows
from .decode import _gathered_fits, choose_gathered, entry_end, draft, tp_sample_rows
from .forward import Cut, commit, compute, cut_snapshot, stage
from .mtp import mtp_compute, mtp_stage
from .state import CAND, ENDS
from .multi_tp import OutOfStep
from .prompt_plan import pass_limit

PASS_MIN = 128
FILL_GUARD = 8


class PromptPasses:
    def _fill(self) -> list[Stream]:
        """Prompt passes over the filling prompts, oldest first, packed to the pass's rows."""

        ended: list[Stream] = []
        while self.filling:
            ended += self._pass()
            if any(not x.done and not x.waiting for x in self.streams.values()) or self._waiting_request():
                break
        return ended

    def _pass_rows(self) -> int:
        """A round's prompt rows: its decode (a round alone) takes ``share`` of the pass's time, by the last rounds."""

        live = any(not s.done for s in self.streams.values())
        return pass_limit(self.prefill_rows, live, self.share, self.round_s, self.row_s, PASS_MIN)

    def _waiting_request(self) -> bool:
        """Both ranks use the leader's sampled arrival decision; one GPU reads its queue directly."""

        paired = getattr(self, "link", None) is not None or getattr(self, "follower", None) is not None
        return self.fill_yield if paired else self.arrived()

    def _timed(self, seconds: float, rows: int) -> None:
        """A round's wall time: a round alone updates its estimate, a round with a pass the seconds a row adds."""

        if rows:
            extra = max(0.0, seconds - (self.round_s or 0.0)) / rows
            self.row_s = extra if self.row_s is None else 0.7 * self.row_s + 0.3 * extra
        else:
            self.round_s = seconds if self.round_s is None else 0.7 * self.round_s + 0.3 * seconds

    def _order(self) -> list[Stream]:
        """Order overdue prompts first, then foreground, fewest rows left and stable arrival order."""

        def key(s: Stream):
            passed = self.passed.get(s.sid, 0)
            due = passed >= FILL_GUARD
            return (not due, -passed if due else 0, s.background, len(s.prompt) - self.fills[s.sid][2])

        return sorted(self.filling, key=key)                         # stable: ties keep arrival order

    def _note_passed(self, pieces) -> None:
        """Count a pass against every filling prompt it left out; one it took starts over."""

        took = {s.sid for s, _, _ in pieces}
        for s in self.filling:
            self.passed[s.sid] = 0 if s.sid in took else self.passed.get(s.sid, 0) + 1
        for sid in [k for k in self.passed if k not in {s.sid for s in self.filling}]:
            del self.passed[sid]

    def _pieces(self, rows: int | None = None) -> list[tuple[Stream, int, int]]:
        """The next pass: rows from the filling prompts, oldest first, up to ``rows`` and ENDS ending prompts."""

        pieces, room = [], self._pass_rows() if rows is None else rows
        for s in self._order():
            e, mtp, start, _ = self.fills[s.sid]
            n = min(next((p for p in e.stops if p > start), len(s.prompt)) - start, room)
            ends = sum(1 for x, a, k in pieces if a + k == len(x.prompt))
            if n == 0 or (start + n == len(s.prompt) and ends == ENDS):
                break
            pieces.append((s, start, n))
            room -= n
        return pieces

    def _pass(self) -> list[Stream]:
        """One prompt pass alone; prompts that end sample their first token, draft and join the rounds."""

        expected = None if self.pass_plan is None else self.pass_plan[self.pass_index]
        pieces = self._pieces(None if expected is None else sum(n for _, _, n in expected))
        self._note_passed(pieces)
        if expected is not None:
            if [[s.sid, a, n] for s, a, n in pieces] != expected:
                raise OutOfStep("a prompt pass differs from the agreed round")
            self.pass_index += 1
        t0 = time.perf_counter()
        try:
            segs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
            ends, cuts = self._end_rows(pieces, segs), self._cuts(pieces, segs)
            logits = compute(self.w, segs, self.pbuf, logits=bool(ends), ends=ends, cuts=cuts)
            heads = logits[:len(ends)].clone() if ends else None
            candidates = self._prompt_candidates(len(ends))
            lasts = self._absorb(pieces, segs, cuts)
        except Exception as exc:                         # noqa: BLE001  (these requests fail, the others go on)
            return self._failed(pieces, exc)
        return self._joined(pieces, heads, lasts, (time.perf_counter() - t0) / len(pieces), candidates)

    @staticmethod
    def _end_rows(pieces, segs) -> list[int]:
        """The pass rows that end a prompt (each gets the head)."""

        return [a1 - 1 for (s, a, n), (_, _, a1) in zip(pieces, segs) if a + n == len(s.prompt)]

    @staticmethod
    def _keep_at(s: Stream) -> int | None:
        """Where a drafting stream's prompt state is kept: one token before its end, which a next turn extends."""

        return entry_end(s.prompt) if s.draft and s.st.image_positions is None else None

    def _point(self, s: Stream, start: int) -> int | None:
        """The next message-start or prompt-end snapshot this prompt piece can reach."""

        if not s.draft or s.st.image_positions is not None:
            return None
        return next((p for p in self.fills[s.sid][0].stops if p > start), self._keep_at(s))

    def _cuts(self, pieces, segs) -> list[Cut]:
        """The kept points strictly inside the pass's pieces, where their DeltaNet chains split."""

        return [Cut(k - a, at=a0) for (s, a, n), (_, a0, _) in zip(pieces, segs)
                if (k := self._point(s, a)) is not None and a < k < a + n]

    def _absorb(self, pieces, segs, cuts=()) -> list[torch.Tensor]:
        """After a pass's forward: each prompt's last row and kept point, the MTP head's absorb, the commits."""

        lasts = [self.pbuf.streams[a1 - 1:a1].clone() for _, _, a1 in segs]
        at, points = {cut.at: cut for cut in cuts}, []
        for (s, a, n), (st, a0, _) in zip(pieces, segs):      # before the MTP head writes the pass's streams
            k = self._point(s, a)
            if k is None or not a < k <= a + n:
                continue
            row, mtp = k - a, self.fills[s.sid][1]
            mtp_len = st.mtp_len + row - 1 if mtp else st.mtp_len       # every row but the point's last
            tail = self.pbuf.streams[a0 + row - 1:a0 + row].clone() if mtp else None
            cut = at.get(a0)
            snap = None if cut is None else cut_snapshot(self.w, st, self.pbuf, cut, mtp_len)
            points.append((s, mtp_len, tail, snap))
        absorb = [(s.st, s.prompt[a + 1:a + n + 1], self.pbuf.streams[a0:a0 + n])
                  for (s, a, n), (_, a0, _) in zip(pieces, segs) if self.fills[s.sid][1] and a + 1 < len(s.prompt)]
        if absorb:                   # the MTP head absorbs each prompt's rows (its cache in position order)
            absorb = [(st, nxt, streams[:len(nxt)]) for st, nxt, streams in absorb]
            mtp_compute(self.w, mtp_stage(self.w, self.pbuf, absorb), self.pbuf)
            for st, nxt, _ in absorb:
                st.set_mtp_len(st.mtp_len + len(nxt))
        for (s, a, n), (st, a0, _) in zip(pieces, segs):
            commit(self.w, st, self.pbuf, n, n, at=a0)
        for s, mtp_len, tail, snap in points:            # a point that ends its piece: the state as committed
            self.fills[s.sid][3] = (snap if snap is not None else {**s.st.snapshot(), "mtp_len": mtp_len}, tail)
        return lasts

    def _failed(self, pieces, exc: Exception) -> list[Stream]:
        failed = [s for s, _, _ in pieces]
        for s in failed:
            s.error, s.done = exc, True
            self.filling.remove(s)
            self.fills.pop(s.sid)
            self._drop_kept(s.st)
        return failed                                    # finish() frees their slots

    def _prompt_candidates(self, count):
        """Keep the verify head's gathered rows before the prompt buffer is reused by the MTP absorb."""

        if self.w.comm is None or not count:
            return None
        world, width = int(self.w.meta["world"]), 2 * CAND + 1
        return self.pbuf.cand_all[:world * count * width].view(world, count, width).clone()

    def _joined(self, pieces, heads, lasts, spent: float, candidates=None) -> list[Stream]:
        """Prompts that ended sample their first token, draft and join the rounds; returns those already done."""

        joined, head = [], 0
        for (s, a, n), last in zip(pieces, lasts):
            s.prefill_s += spent
            e, mtp, _, kept = self.fills[s.sid]
            self.fills[s.sid][2] = a + n
            if kept is not None and a < kept[0]["pos"] <= a + n:
                self._remember(list(s.prompt[:kept[0]["pos"]]), s.st, *kept)
            self.fills[s.sid][3] = None
            if a + n < len(s.prompt):
                continue
            self.filling.remove(s)
            self.fills.pop(s.sid)
            st, e.last_streams = s.st, last
            image_rows.finish(st)
            logits = heads[head:head + 1]
            if s.constraint is not None:                 # a reply's grammar: the first token too
                logits = s.constraint.mask(logits, None, self.w.meta.get("vocab_offset", 0))
            if self.w.comm is None:
                first = e.sample(logits, [len(s.prompt)], s.sampling)[0]
            elif _gathered_fits(s.sampling):
                gathered = candidates[:, head:head + 1].contiguous().view(-1)
                first = choose_gathered(self.w, gathered, 1, [len(s.prompt)], s.sampling)[0]
            else:
                first = tp_sample_rows(self.w, logits, [len(s.prompt)], s.sampling,
                                       offset=int(self.w.meta["vocab_offset"]))[0]
            if s.probabilities is not None:
                capture(logits, [first], [len(s.prompt)], s.probabilities)
            if s.constraint is not None:
                s.constraint.advance([first])
            head += 1
            s.context = list(s.prompt)
            s.drafts = draft(e, last, [first], st.pos + 1, min(self.depth, s.count - 1), s.sampling,
                             self.confidence) if mtp and s.count > 1 else []
            s.started = time.perf_counter()
            self.streams[s.sid] = s
            s.take([first], self._ends(s))
            if s.done:
                joined.append(s)
        self._joined_ranks()
        return joined
