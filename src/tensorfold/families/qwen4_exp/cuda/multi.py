"""Flash Next's concurrent rounds on one GPU or two ranks in step: every stream keeps exactly its own accepted prefix."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import timedelta

import numpy as np
import torch

from tensorfold.cuda.logprobs import capture

from tensorfold.cuda.capacity import LIMIT_ENV, available_bytes, cuda_limit_bytes
from tensorfold.cuda.markers import MIN_GAP
from tensorfold.cuda.memory_gate import MemoryGate, NoRoom, torch_live
from tensorfold.cuda.capacity import gather_ints
from tensorfold.cuda.sampling import sample_streams
from tensorfold.cuda.streams import Stream, accept
from tensorfold.engine.exact_sampling import MARGIN, choose_rows
from tensorfold.engine.grammar import GrammarError

from .decode import (PREFILL_ROWS, WARM_TAIL, Engine, _gathered_fits, choose_gathered_streams, draft, entry_end,
                     prefill, prefill_begin, tp_sample_rows)
from . import attn_multi, gdn_multi, image_rows, prefixes
from .forward import Cut, commit, compute, compute_mixed, converges, cut_snapshot, stage
from .mtp import mtp_compute, mtp_stage
from .prompt_plan import pass_limit
from .state import ENDS, Buffers, State
from ..cuda import CONFIDENCE, DEPTH

FIRST, STEP = 256, 8192          # rows an idle slot keeps; rows a stream's caches grow by at a time
GIB = 1024**3
SHARE = 0.0                      # --decode-share: a round alone takes this share of its pass's time (0: whole passes)
PASS_MIN = 512                   # live shared passes keep this floor within their admitted row ceiling
FILL_GUARD = 8                   # a prompt passed over this many passes takes the next one (no starvation), as on Macs


class Link:
    """Rank 0's ordered messages to rank 1: every admission, round and completion in the order rank 0 made it.
    Rank 1 blocks on a socket read between them, not inside a GPU collective. With ``host`` (rank 0's address on
    the link between the ranks) they cross one TCP connection whose port the rendezvous store publishes; without,
    they go through the store itself (wait, get, delete: three round trips a message; tests)."""

    KEY = "tensorfold/flashnext/multi/socket"

    def __init__(self, store, *, rank: int | None = None, host: str | None = None) -> None:
        import socket

        self.store, self.n, self.sock, self.server = store, 0, None, None
        if host is None:
            return
        if rank == 0:
            self.server = socket.create_server((host, 0))
            store.set(self.KEY, str(self.server.getsockname()[1]))
        else:
            store.wait([self.KEY], timedelta(hours=24))
            self.sock = socket.create_connection((host, int(store.get(self.KEY).decode())))
            self._fast(self.sock)

    @staticmethod
    def _fast(sock) -> None:
        import socket

        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _key(self) -> str:
        return f"tensorfold/flashnext/multi/{self.n}"

    def send(self, op: list) -> None:
        if self.server is not None:
            import struct

            if self.sock is None:                       # rank 1 connected at startup: this returns at once
                self.sock, _ = self.server.accept()
                self._fast(self.sock)
            data = json.dumps(op).encode()
            self.sock.sendall(struct.pack("!I", len(data)) + data)
            return
        self.store.set(self._key(), json.dumps(op))
        self.n += 1

    def _read(self, n: int) -> bytes | None:
        out = b""
        while len(out) < n:
            chunk = self.sock.recv(n - len(out))
            if not chunk:                               # rank 0 is gone
                return None
            out += chunk
        return out

    def receive(self) -> list | None:
        if self.sock is not None:
            import struct

            head = self._read(4)
            body = None if head is None else self._read(struct.unpack("!I", head)[0])
            return None if body is None else json.loads(body)
        from torch.distributed import DistNetworkError

        key = self._key()
        while True:
            try:
                self.store.wait([key], timedelta(hours=1))
                break
            except DistNetworkError:                        # rank 0 is gone
                return None
            except Exception:                               # noqa: BLE001, S112  (no message within the hour: wait on)
                continue
        text = self.store.get(key).decode()
        self.store.delete_key(key)
        self.n += 1
        return json.loads(text)


def _pack(sampling) -> list | None:
    """A request's sampling rule as exact JSON (floats round-trip bit for bit)."""

    if sampling is None:
        return None
    return [int(sampling.seed), float(sampling.temperature), int(sampling.top_k), float(sampling.top_p),
            float(sampling.min_p)]


def _unpack(values):
    from tensorfold.engine.exact_sampling import Sampling

    return None if values is None else Sampling(values[0], values[1], values[2], values[3], values[4])


def _slot(w, st: State, buf: Buffers, mbuf: Buffers, pbuf: Buffers, capacity: int, prefill_rows: int = PREFILL_ROWS) -> Engine:
    """A one-sequence engine over a slot's state and the shared buffers (eager: no CUDA graphs)."""

    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows = w, capacity, buf.rows, prefill_rows
    e.buf, e.mbuf, e.pbuf, e.st, e.graphs = buf, mbuf, pbuf, st, None
    e.stops = ()
    return e


class OutOfStep(RuntimeError):
    """The two ranks' plans for an admission or a round differ: both ranks raise it at the same point, before any
    collective of that step, so neither waits on the other; the requests it touches fail and serving goes on."""


class MultiDecoder:
    """Rounds over the live streams; ``slots`` streams at most, each with ``capacity`` tokens of context."""

    def __init__(self, w, *, slots: int, capacity: int, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 stop_eos: bool = True, keep: int = 8, kv_dtype: str = "bf16", prefill_rows: int = PREFILL_ROWS,
                 share: float = SHARE, points=None, vision=None, workspace_bytes: int = 0,
                 graphs: bool = True) -> None:
        self.link: Link | None = None
        self.w, self.depth, self.confidence, self.capacity = w, depth, confidence, capacity
        self.points = points                         # a prompt's message starts to keep states at, or None
        self.vision = vision
        self.eos = tuple(w.cfg.eos) if stop_eos else ()
        rows = slots * (depth + 1)
        # a round's window and a prompt pass share each layer's expert launch: the pass's buffers hold both
        self.converged, self.prefill_rows = converges(w), prefill_rows
        # rounds beside a filling prompt size its pass so decoding keeps ``share`` of the pass's time (0: whole passes)
        self.share, self.round_s, self.row_s = share, None, None
        self.buf = Buffers(w, rows, capacity, moe_prefill=True)
        self.mbuf = Buffers(w, rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, PREFILL_ROWS, capacity, prefill=True)
        # A stream decoding alone replays the one-stream CUDA graphs: the graphs' state is one of the slots (new
        # requests take it first), so a lone stream needs no copy; beside other streams it decodes eagerly like any.
        self.solo = _solo(w, capacity, depth, kv_dtype, self.pbuf) if graphs and depth > 0 and self.mbuf is not None \
            else None
        self.solo_on = self.solo is not None
        self.free = ([self.solo.st] if self.solo is not None else []) + \
            [State(w, capacity, depth + 1, kv_dtype) for _ in range(slots - (self.solo is not None))]
        self.slot_bytes = sum(t.numel() * t.element_size() for t in _tensors(self.free[0]))
        self.window_bytes = self.free[0].cache_bytes(capacity)          # one stream's caches at the full window
        free = torch_live(torch, available_bytes) if torch.cuda.is_available() else None
        # the mapped n-gram tables are not held back (they barely fit on a Spark); lookups page from disk instead
        live = free
        self.memory_gate = MemoryGate(live() if live is not None else 1 << 62,
                                      reserve=max(2 * GIB, workspace_bytes), live=live)
        self.streams: dict[int, Stream] = {}
        self.filling: list[Stream] = []                  # admitted, prompts still prefilling (oldest first)
        self.fills: dict[int, list] = {}                 # stream id -> [its engine, drafts?, next row, kept state]
        self.passed: dict[int, int] = {}                 # stream id -> prompt passes since its last rows
        self.arrived = lambda: False                     # a request waits to be admitted (the scheduler sets this)
        self.next_id = 0
        self.draft_host = w.draft_ids.cpu().numpy() if w.draft_ids is not None else None
        self.kept: list[tuple[list[int], State, dict, torch.Tensor | None]] = []   # (ids, slot, snapshot, tail)
        self.keep = keep
        self.slots = list(self.free)            # every slot by index: rank 0 names the slot a request takes

    def _busy(self) -> set[int]:
        return {id(s.st) for s in [*self.streams.values(), *self.filling]}

    def _drop_kept(self, st: State) -> None:
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st: State, rows: int, *, alone: bool = False, protect: State | None = None) -> bool:
        """Grow caches while the gate has room; one stream may use its startup allowance within an explicit cap."""

        if rows <= st.capacity or st.capacity >= st.limit:       # admission's count keeps a stream within its window
            return True
        size = min(st.limit, -(-rows // STEP) * STEP)
        before = st.cache_bytes()
        grow = st.cache_bytes(size) - before
        while not self.memory_gate.fits(grow + st.layer_bytes(size)):     # a layer's old buffers stay until its copy
            if not self._evict_kept(st, protect=protect):
                if alone:
                    limit = cuda_limit_bytes() if torch.cuda.is_available() else None
                    if limit is not None:
                        # A lone stream may use its startup reserve without exceeding the copy peak cap.
                        peak = int(torch.cuda.memory_allocated()) + grow + st.layer_bytes(st.capacity)
                        if peak > limit:
                            if self.filling:
                                return False             # a filling request can finish and release its slot
                            raise NoRoom(
                                f"Growing this request's attention caches to {size} tokens would exceed "
                                f"{LIMIT_ENV}={limit / GIB:g}: the estimated copy peak is {peak / GIB:.2f} GiB. "
                                "Shorten the prompt or max_tokens, reduce --context or --parallel, or raise "
                                f"{LIMIT_ENV} if more GPU memory is available.")
                    break
                return False
        try:
            added = st.resize(size)
        except Exception:
            self.memory_gate.take(st.cache_bytes() - before)
            raise
        self.memory_gate.take(added)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()             # the old buffers back to the system: MemAvailable stays true
        return True

    def _shrink(self, st: State, *, force: bool = False) -> None:
        """An idle slot back to its first rows: its caches' memory returns to the gate."""

        st.reset(self.w)
        if force or st.capacity > FIRST:
            self.memory_gate.give(-st.resize(min(FIRST, st.limit)))

    def _evict_kept(self, keep: State, *, protect: State | None = None) -> bool:
        """Free the oldest idle kept prompt end (never ``keep``); False when none is left."""

        busy = self._busy()
        for ids, st, _, _ in self.kept:
            if st is not keep and st is not protect and id(st) not in busy:
                self._drop_kept(st)
                self._shrink(st)
                if all(f is not st for f in self.free):
                    self.free.append(st)
                return True
        return False

    def _make_room(self) -> list[Stream]:
        """Before a round: grow each live window oldest-first; a stream that can't grow makes the newest end."""

        live = sorted((s for s in self.streams.values() if not s.done), key=lambda s: s.sid)
        blocked, ended = False, []
        for s in live:
            rows = max(s.st.pos, s.st.mtp_len) + len(s.drafts) + self.depth + 2
            try:
                s.waiting = rows > s.st.capacity if blocked else not self._grow(s.st, rows, alone=len(live) == 1)
            except NoRoom as exc:
                s.error, s.done, s.waiting = exc, True, False
                self.held.pop(s.sid, None)
                self._drop_kept(s.st)
                self.memory_gate.ends += 1
                ended.append(s)                          # finish() releases only this request's slot
                continue
            blocked = blocked or s.waiting
        if live and live[0].waiting and len(live) > 1:        # even the oldest can't grow: the newest ends
            newest = live[-1]
            newest.error = RuntimeError(
                f"This server ran out of memory with {len(live)} streams decoding, so the newest (this request, after "
                f"{len(newest.out)} tokens) was stopped for the older ones to finish. Retry it, shorten the prompt or "
                "max_tokens, or start the server with a smaller --parallel.")
            newest.done, newest.waiting = True, False
            self.memory_gate.ends += 1
            self.streams.pop(newest.sid, None)
            self.held.pop(newest.sid, None)
            self._drop_kept(newest.st)
            self._shrink(newest.st)
            self.free.append(newest.st)
            return [*ended, newest, *self._make_room()]
        self.memory_gate.waits += any(s.waiting for s in live)
        return ended

    def _slot_for(self, prompt: list[int], reuse: bool):
        """Reuse the longest kept point, copying a fork into a free slot when the memory gate permits it."""

        return prefixes.slot_for(self, prompt, reuse)

    def _index(self, st: State) -> int:
        return next(i for i, x in enumerate(self.slots) if x is st)

    def _slot_as_told(self, prompt: list[int], slot: int, cached: int):
        """Rank 1: the slot rank 0 chose, resumed from the kept prompt end rank 0 resumes from; None when this rank
        cannot (the admission then fails on both ranks)."""

        if not 0 <= slot < len(self.slots) or id(self.slots[slot]) in self._busy():
            return None
        st = self.slots[slot]
        if cached > 0:
            k = next((k for k in self.kept if k[1] is st and len(k[0]) == cached and list(prompt[:cached]) == k[0]),
                     None)
            if k is None:
                return None
            self._drop_kept(st)
            return st, {"state": k[2], "tail": k[3]}, cached
        self._drop_kept(st)
        self.free = [f for f in self.free if f is not st]
        return st, None, 0

    def _agree(self, what: str, plan: list) -> None:
        """Two ranks: both confirm the same plan in one small all-gather before the step's work, or both raise."""

        if self.w.comm is None:
            return
        digest = int.from_bytes(hashlib.sha256(json.dumps(plan).encode()).digest()[:8], "big", signed=True)
        mine, theirs = gather_ints(torch, self.w.comm.all_gather, [digest])
        if mine != theirs:
            raise OutOfStep(f"the two ranks planned different {what}s; the requests in it fail, serving goes on")

    def _remember(self, ids: list[int], st: State, snap: dict, tail) -> None:
        prefixes.remember(self, ids, st, snap, tail)

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    @torch.no_grad()
    def warm(self) -> None:
        """A synthetic greedy request through prefill (a full chunk, then a partial one cut at the kept point), its drafts and one round, then forgotten, so no request compiles or loads a kernel."""

        if self.solo is not None and self.solo.graphs is not None:    # every one-stream window, captured now
            self.solo.graphs.warm(self.depth + 1)
        self.solo_on = False                           # the synthetic request warms the eager rounds' kernels
        try:
            s = Stream([0] * min(PREFILL_ROWS + WARM_TAIL, self.capacity - self.depth - 2), 2)
            self.admit(s)
            if not s.done:
                self.round()
            self.streams.pop(s.sid, None)
            self._drop_kept(s.st)
            if all(f is not s.st for f in self.free):
                self.free.append(s.st)
        finally:
            self.solo_on = self.solo is not None

    @torch.no_grad()
    def admit(self, s: Stream, told: tuple[int, int] | None = None) -> None:
        """Prefill a request in a free slot, draft its first chain and emit its first token; told: on rank 1,
        the slot and resume point rank 0 chose."""

        room = self.capacity - len(s.prompt) - self.depth - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.capacity}-token context")
        s.count = max(1, min(s.count, room))
        if self.w.comm is not None and (s.constraint is not None or s.vision is not None):
            raise ValueError("concurrent Flash Next on two ranks serves text without a response grammar for now")
        t0 = time.perf_counter()
        taken = self._slot_for(list(s.prompt), s.draft) if told is None else self._slot_as_told(s.prompt, *told)
        slot, cached = (-1, -1) if taken is None else (self._index(taken[0]), taken[2])
        if self.link is not None:
            self.link.send(["admit", self.next_id, list(s.prompt), s.count, _pack(s.sampling), bool(s.draft),
                            bool(s.stop_eos), slot, cached])
        try:
            self._agree("admission", [self.next_id, slot, cached, list(s.prompt), _pack(s.sampling), bool(s.draft),
                                      s.count])
        except OutOfStep:
            if taken is not None:
                self.free.append(taken[0])
            raise
        st, resume, s.cached = taken
        e = _slot(self.w, st, self.buf, self.mbuf, self.pbuf, self.capacity, self.prefill_rows)
        mtp = s.draft and self.depth > 0 and self.mbuf is not None
        try:
            begin = prefill_begin(e, s.prompt, mtp=mtp, resume=resume)
            image_rows.begin(e, s, self.vision)
        except Exception:
            self._drop_kept(st)
            self.free.append(st)
            raise
        e.stops = sorted({p for p in self.points(s.prompt) if begin + MIN_GAP <= p < entry_end(s.prompt)}) \
            if s.draft and st.image_positions is None and self.points is not None else []
        s.sid, s.st = self.next_id, st
        self.next_id += 1
        s.prefill_s = time.perf_counter() - t0
        same = resume is not None and self._keep_at(s) == begin         # the same prompt again: its own point
        self.fills[s.sid] = [e, mtp, begin, (resume["state"], resume["tail"]) if same else None]
        self.filling.append(s)

    def _fill(self) -> list[Stream]:
        """Fill ordered prompt pieces until a stream decodes or a waiting request needs admission."""

        ended: list[Stream] = []
        while self.filling:
            ended += self._pass()
            if any(not x.done and not x.waiting for x in self.streams.values()) or self.arrived():
                break
        return ended

    def _pass_rows(self) -> int:
        """A round's prompt rows: its decode (a round alone) takes ``share`` of the pass's time, by the last rounds."""

        live = any(not s.done for s in self.streams.values())
        return pass_limit(self.prefill_rows, live, self.share, self.round_s, self.row_s, PASS_MIN)

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

    def _pieces(self, rows: int | None = None) -> list[tuple[Stream, int, int]]:
        """The next pass: rows from the filling prompts in ``_order``, up to ``rows`` and ENDS ending prompts."""

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

        pieces = self._pieces()
        self._note_passed(pieces)
        t0 = time.perf_counter()
        try:
            segs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
            ends, cuts = self._end_rows(pieces, segs), self._cuts(pieces, segs)
            logits = compute(self.w, segs, self.pbuf, logits=bool(ends), ends=ends, cuts=cuts)
            heads = logits[:len(ends)].clone() if ends else None
            lasts = self._absorb(pieces, segs, cuts)
        except Exception as exc:                         # noqa: BLE001  (these requests fail, the others go on)
            return self._failed(pieces, exc)
        if not self.converged and pieces:                # completed stand-alone passes calibrate the shared row cost
            rows = sum(n for _, _, n in pieces)
            extra = max(0.0, time.perf_counter() - t0) / max(1, rows)
            self.row_s = extra if self.row_s is None else 0.7 * self.row_s + 0.3 * extra
        return self._joined(pieces, heads, lasts, (time.perf_counter() - t0) / len(pieces))

    def _note_passed(self, pieces) -> None:
        """Count a pass against every filling prompt it left out; one it took starts over."""

        took = {s.sid for s, _, _ in pieces}
        for s in self.filling:
            self.passed[s.sid] = 0 if s.sid in took else self.passed.get(s.sid, 0) + 1
        for sid in [k for k in self.passed if k not in {s.sid for s in self.filling}]:
            del self.passed[sid]

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

    def _joined(self, pieces, heads, lasts, spent: float) -> list[Stream]:
        """Prompts that ended sample their first token, draft and join the rounds; returns those already done."""

        joined, head = [], 0
        for (s, a, n), last in zip(pieces, lasts):
            s.prefill_s += spent
            e, mtp, _, kept = self.fills[s.sid]
            self.fills[s.sid][2] = a + n
            if kept is not None and a < kept[0]["pos"] <= a + n:
                self._remember(list(s.prompt[:kept[0]["pos"]]), s.st, *kept)
            self.fills[s.sid][3] = None                    # only the bounded cache owns a stored snapshot
            if a + n < len(s.prompt):
                continue
            self.filling.remove(s)
            self.fills.pop(s.sid)
            st, e.last_streams = s.st, last
            logits = heads[head:head + 1]
            if s.constraint is not None:                 # a reply's grammar: the first token too
                logits = s.constraint.mask(logits, None, self.w.meta.get("vocab_offset", 0))
            first = e.sample(logits, [len(s.prompt)], s.sampling)[0]
            if s.probabilities is not None:
                capture(logits, [first], [len(s.prompt)], s.probabilities)
            if s.constraint is not None:
                s.constraint.advance([first])
            head += 1
            image_rows.finish(st)
            s.context = list(s.prompt)
            s.drafts = draft(e, last, [first], st.pos + 1, min(self.depth, s.count - 1), s.sampling,
                             self.confidence) if mtp and s.count > 1 else []
            s.started = time.perf_counter()
            self.streams[s.sid] = s
            s.take([first], self._ends(s))
            if s.done:
                joined.append(s)
        return joined

    def _ends(self, s: Stream) -> tuple[int, ...]:
        """The end tokens that end this stream: none when its request ignores them (``ignore_eos``)."""

        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """One round over the live streams, with the next prompt pass in the same forward while prompts fill."""

        live = [s for s in self.streams.values() if not s.done]
        if self.link is not None:            # streams rank 0 ended alone (a client that left) and not yet finished
            self.link.send(["round", [s.sid for s in self.streams.values() if s.done], self._step_check(live)])
        self._agree("round", [self._step_check(live), [self._index(s.st) for s in live], self.solo_on])
        if not live:
            return ended
        grammars, failed = {}, []
        for s in live:                                   # a grammar cuts the drafts no accepted path can hold
            if s.constraint is not None:
                tokens = [s.out[-1]] + list(s.drafts)
                try:
                    grammars[s.sid] = s.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
                except GrammarError as exc:              # this request ends with its error, the others go on
                    s.error, s.done = exc, True
                    self.held.pop(s.sid, None)
                    failed.append(s)
                    continue
                s.drafts = grammars[s.sid].tokens[1:]
        live = [s for s in live if not s.done]
        if not live:
            return failed
        if self.solo_on and len(live) == 1 and live[0].draft and live[0].constraint is None:
            if live[0].st is not self.solo.st:
                self._move_to_solo(live[0])
            return failed + self._solo_round(live[0])
        windows = [(s.st, [s.out[-1]] + list(s.drafts)) for s in live]
        segs = stage(self.w, self.buf, windows)
        # a pass shares the round's forward only where their experts share a launch; else _fill ran it between rounds
        pieces, psegs = (self._pieces(self._pass_rows()) if self.filling and self.converged else []), None
        cuts = []
        if pieces:
            self._note_passed(pieces)
            try:
                psegs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
                cuts = self._cuts(pieces, psegs)
            except Exception as exc:                     # noqa: BLE001  (the pass's requests fail, the round goes on)
                ended += self._failed(pieces, exc)
                pieces = []
        held = [self.held.pop(s.sid, []) for s in live]
        tables = self.buf.gdn_tables = gdn_multi.Tables(self.w, self.gdn, segs, held)
        self.buf.attn_step = attn_multi.Step(self.w, segs, mtp=False)
        try:
            if pieces:                                 # the window and the pass: each layer's experts once for both
                pends = self._end_rows(pieces, psegs)
                logits, heads = compute_mixed(self.w, segs, self.buf, psegs, self.pbuf, ends=pends, cuts=cuts)
                heads = heads[:len(pends)].clone() if pends else None
            else:
                logits = compute(self.w, segs, self.buf)
        finally:
            self.buf.gdn_tables = self.buf.attn_step = None
        lasts = self._absorb(pieces, psegs, cuts) if pieces else None
        starts = [a0 for _, a0, _ in segs] + [segs[-1][2]]
        for s, (_, a0, a1) in zip(live, segs):
            if s.sid in grammars:
                s.constraint.mask(logits[a0:a1], grammars[s.sid])
        positions = [[st.pos + 1 + r for r in range(a1 - a0)] for st, a0, a1 in segs]
        samplings = [s.sampling for s in live]
        if self.w.comm is None:
            sampled = sample_streams(logits, starts, positions, samplings)
        elif all(_gathered_fits(x) for x in samplings):    # the candidates gathered inside the forward, one read-back
            sampled = choose_gathered_streams(self.w, self.buf.cand_all, starts[-1], starts, positions, samplings)
        else:                                            # a rule wider than the gathered candidates: both ranks gather
            sampled = [tp_sample_rows(self.w, logits[a0:a1], pos, smp, offset=int(self.w.meta["vocab_offset"]))
                       for (_, a0, a1), pos, smp in zip(segs, positions, samplings)]
        kept = []
        for s, (_, tokens), (st, a0, a1), rows, (path, end) in zip(live, windows, segs, sampled, paths):
            commit(self.w, st, self.buf, a1 - a0, len(path), at=a0, states=False)
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
        spent = time.perf_counter() - t0
        self._timed(spent, sum(n for _, _, n in pieces))
        if pieces:                                     # prompts that ended in this round's pass join the next
            ended += self._joined(pieces, heads, lasts, spent / len(pieces))
        done = [s for s in live if s.done]
        for s in done:                                   # a finished stream's state is never read again
            self.held.pop(s.sid, None)
        return failed + done + ended

    def _move_to_solo(self, s: Stream) -> None:
        """A stream now decoding alone moves into the graphs' slot: its state copied in once (the same bits), the
        slot's kept prompt end, if any, given up; its old slot stays kept or goes back to the free list."""

        solo, old = self.solo.st, s.st
        self._drop_kept(solo)
        self.free = [f for f in self.free if f is not solo]
        solo.copy_from(old)
        s.st = solo
        if not any(k[1] is old for k in self.kept) and all(f is not old for f in self.free):
            self.free.append(old)

    def _solo_round(self, s: Stream) -> list[Stream]:
        """A lone stream's round in the one-stream engine (its graphs): verify, keep, commit and chain the next
        drafts as ``mtp_decode`` does, so the kept tokens are the ones every other path keeps."""

        e, st = self.solo, s.st
        tokens = [s.out[-1]] + list(s.drafts)
        R = len(tokens)
        logits = e.forward(tokens)
        rows = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], s.sampling)
        path, end = accept(tokens, list(range(-1, R - 1)), rows, s.count - len(s.out), self._ends(s))
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
        logits = self._mtp(segs)
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
            logits = self._mtp(segs)
            for s, _, _ in nxt:
                s.st.set_mtp_len(s.st.mtp_len + 1)
                s.st.mtp_drafted += 1
            active = [(s, a0) for (s, _, _), (_, a0, _) in zip(nxt, segs)]

    def _mtp(self, segs: list) -> torch.Tensor:
        """An MTP step over every drafting stream, its attention one launch a kernel for all of them."""

        self.mbuf.attn_step = attn_multi.Step(self.w, segs, mtp=True)
        try:
            return mtp_compute(self.w, segs, self.mbuf)
        finally:
            self.mbuf.attn_step = None

    def _picks(self, logits: torch.Tensor, positions: list[int], samplings: list) -> list[tuple[int, float]]:
        """Each row's keyed draft and its probability at temperature 1, one read-back (drafts change speed only)."""

        if self.w.comm is not None:
            return self._picks_tp(logits, positions, samplings)
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

    def _picks_tp(self, logits: torch.Tensor, positions: list[int], samplings: list) -> list[tuple[int, float]]:
        """Two ranks: each row's keyed draft and probability from the draft head's gathered candidates."""

        w, n = self.w, len(positions)
        if all(_gathered_fits(s) for s in samplings):
            chosen, probs = choose_gathered_streams(w, self.mbuf.cand_all, n, list(range(n + 1)),
                                                    [[p] for p in positions], samplings, with_prob=True)
            return [(c[0], p[0]) for c, p in zip(chosen, probs)]
        out = []
        for i, (pos, smp) in enumerate(zip(positions, samplings)):
            toks, probs = tp_sample_rows(w, logits[i:i + 1], [pos], smp, offset=int(w.meta["vocab_offset"]),
                                         id_map=w.draft_ids, with_prob=True)
            out.append((toks[0], probs[0]))
        return out

    @staticmethod
    def _step_check(live: list[Stream]) -> list:
        """What both ranks must agree on before a round: each live stream, its length and its drafts."""

        return [[s.sid, len(s.out), list(s.drafts)] for s in live]

    @torch.no_grad()
    def follow(self, link: Link) -> None:
        """Rank 1: replay rank 0's admissions, rounds and completions in order until rank 0 stops."""

        while True:
            op = link.receive()
            if op is None or op[0] == "stop":
                return
            kind = op[0]
            if kind == "admit":
                sid, prompt, count, smp, draft_, stop_eos, slot, cached = op[1:]
                if sid != self.next_id:              # the admission's collective check fails on both ranks
                    print(f"[tensorfold] rank 1: rank 0 admits stream {sid}, rank 1 expects {self.next_id}", flush=True)
                s = Stream(prompt, count, _unpack(smp), draft=draft_, stop_eos=stop_eos)
                try:
                    self.admit(s, told=(slot, cached))
                except (ValueError, OutOfStep) as exc:   # rank 0 raised at the same point on the same input
                    print(f"[tensorfold] rank 1: stream {sid} failed on both ranks: {exc}", flush=True)
            elif kind == "round":
                ended, check = op[1], op[2]
                for sid in ended:                        # ended on rank 0 alone: here too, finished with it later
                    if sid in self.streams:
                        self.streams[sid].done = True
                mine = self._step_check([s for s in self.streams.values() if not s.done])
                if mine != check:                        # the round's collective check fails on both ranks
                    print(f"[tensorfold] rank 1: out of step before a round: rank 0 {check}, rank 1 {mine}",
                          flush=True)
                try:
                    self.round()
                except Exception as exc:                 # noqa: BLE001  (rank 0 fails the same round and drops)
                    print(f"[tensorfold] rank 1: a round failed: {exc}", flush=True)
            elif kind == "finish":
                self.finish([self.streams[sid] for sid in op[1] if sid in self.streams])
            elif kind == "drop":
                self.drop()

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams; a slot whose prompt state is kept stays with it, the rest are free again."""

        if self.link is not None and done:
            self.link.send(["finish", [s.sid for s in done]])
        for s in done:
            self.streams.pop(s.sid, None)
            if not any(k[1] is s.st for k in self.kept) and all(f is not s.st for f in self.free):
                self._shrink(s.st)
                self.free.append(s.st)

    def drop(self) -> list[Stream]:
        if self.link is not None:
            self.link.send(["drop"])
        live = [s for s in self.streams.values() if not s.done]
        for s in live:
            self.streams.pop(s.sid, None)
            self.held.pop(s.sid, None)
            self._drop_kept(s.st)
            self._shrink(s.st)
            self.free.append(s.st)
        return live


def _solo(w, capacity: int, depth: int, kv_dtype: str, pbuf: Buffers) -> Engine | None:
    """The one-stream engine a lone stream decodes in: its own small window buffers and CUDA graphs, the
    concurrent decoder's prompt buffer; None when this checkpoint's experts can't be captured."""

    from .graphs import Graphs

    if any(getattr(layer.moe.experts, "capturable", True) is False for layer in w.layers):
        return None
    rows = max(8, depth + 1)
    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows, e.kv_dtype = w, capacity, rows, pbuf.rows, kv_dtype
    e.buf, e.mbuf, e.pbuf = Buffers(w, rows, capacity), Buffers(w, rows, capacity), pbuf
    e.st = State(w, capacity, depth + 1, kv_dtype)          # a slot's geometry: a stream's state copies in
    e.graphs = Graphs(e, max_rows=rows)
    return e


def _tensors(st: State):
    for value in vars(st).values():
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, torch.Tensor):
                yield v
            elif hasattr(v, "__dict__"):                  # scratch and KV cache objects, the MTP head's too
                yield from (t for t in vars(v).values() if isinstance(t, torch.Tensor))
