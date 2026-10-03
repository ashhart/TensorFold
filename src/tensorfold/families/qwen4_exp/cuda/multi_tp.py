"""Ordered two-rank admissions and round plans, checked before their GPU collectives."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import torch

from tensorfold.cuda.capacity import gather_ints
from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream

class Link:
    """Rank 0's ordered steps over TCP or the rendezvous store; rank 1 waits outside GPU collectives."""

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


class OutOfStep(RuntimeError):
    """Both ranks refuse a different admission or round plan before that step's collectives."""


def shape(dec) -> list:
    """The logical state both ranks must share; cache contents are checked when a prefix is selected."""

    return [dec.next_id, [[s.capacity, s.limit, s.pos, s.mtp_len, list(s.cur), s.kv_dtype] for s in dec.slots],
            [[s.sid, dec._index(s.st), len(s.out), s.out[-1] if s.out else None, list(s.drafts),
              bool(s.done), bool(s.waiting), s.background, s.draft, s.stop_eos, s.count, _pack(s.sampling)]
             for s in [*dec.streams.values(), *dec.filling]],
            [[s.sid, dec.fills[s.sid][2], list(dec.fills[s.sid][0].stops)] for s in dec.filling],
            [[dec._index(k[1]), len(k[0])] for k in dec.kept],
            [dec._index(st) for st in dec.free], sorted(dec.held.items()), bool(dec.solo_on), sorted(dec.passed.items()),
            [dec.depth, dec.confidence, dec.capacity, dec.prefill_rows, dec.converged, dec.keep,
             list(dec.eos), dec.gdn.parity]]


class TwoRanks:
    def _index(self, st) -> int:
        return next(i for i, x in enumerate(self.slots) if x is st)

    def _agree(self, what: str, plan: list) -> None:
        """One small all-gather checks the whole step before either rank enters its model collectives."""

        if self.w.comm is None:
            return
        digest = int.from_bytes(hashlib.sha256(json.dumps(plan).encode()).digest()[:8], "big", signed=True)
        mine, theirs = gather_ints(torch, self.w.comm.all_gather, [digest])
        if mine != theirs:
            raise OutOfStep(f"the two ranks planned different {what}s; the requests in it fail, serving goes on")

    def _joined_ranks(self) -> None:
        """A client may leave at its first token; both ranks then make the same next prompt-pass decision."""

        if self.follower is None:
            self.fill_yield = bool(self.arrived())
        if self.link is not None:
            self.link.send(["joined", [s.sid for s in self.streams.values() if s.done], self.fill_yield])
        elif self.follower is not None:
            op = self.follower.receive()
            if op is not None and op[0] == "drop":
                self.drop()
                raise OutOfStep("rank 0 aborted the prompt pass")
            if op is None or op[0] == "stop":
                self.follow_stopped = True
                raise OutOfStep("rank 0 stopped during a prompt pass")
            if op[0] != "joined":
                raise OutOfStep("expected the leader's prompt-pass completion")
            self.fill_yield = bool(op[2])
            for sid in op[1]:
                self.streams[sid].done = True

    def _prepare_admission(self, s, told):
        from .multi_plan import admission, apply, ready

        plan = admission(self, s) if told is None else told
        slot, cached = plan["slot"], plan["cached"]
        source = plan["resume_slot"]
        resume = next(({"state": k[2], "tail": k[3]} for k in self.kept
                       if self._index(k[1]) == source and len(k[0]) == cached and s.prompt[:cached] == k[0]), None)
        valid = 0 <= slot < len(self.slots) and (cached == 0 or resume is not None)
        if self.link is not None:
            self.link.send(["admit", self.next_id, list(s.prompt), s.count, _pack(s.sampling), bool(s.draft),
                            bool(s.stop_eos), bool(s.background), plan])
        fits = ready(self, plan)
        self._agree("admission", [shape(self), plan, valid, fits, list(s.prompt), s.count, _pack(s.sampling),
                                  bool(s.draft), bool(s.stop_eos), bool(s.background)])
        if not valid:
            raise OutOfStep("the agreed prefix is not available on both ranks")
        if not fits:
            if not self.streams and not self.filling:
                raise ValueError("the prompt cannot fit available memory; shorten it or lower the server context")
            raise NoRoom("the proposed prompt cache growth does not fit on both ranks")
        apply(self, plan)
        if plan["error"]:
            raise NoRoom(plan["error"])
        s.stops = list(plan["points"])
        return self.slots[slot], resume, cached

    def _prepare_round(self, told):
        from .multi_plan import apply, ready, round_plan

        plan = round_plan(self) if told is None else told
        if self.link is not None:
            self.link.send(["round", [s.sid for s in [*self.streams.values(), *self.filling] if s.done], plan])
        fits = ready(self, plan)
        self._agree("round", [shape(self), plan, fits])
        if not fits:
            raise NoRoom("the proposed round cache growth does not fit on both ranks")
        ended = [self.streams[sid] for sid in plan["ended"]]
        apply(self, plan)
        self.pass_plan, self.mixed_plan = plan["passes"], plan["mixed"]
        self.pass_index, self.pass_width = 0, plan["pass_width"]
        return ended, plan["solo"]

    @torch.no_grad()
    def follow(self, link: Link) -> None:
        """Replay the leader's complete plans, including prompt progress and growing-cache decisions."""

        self.follower, self.follow_stopped = link, False
        finished = {}
        while not self.follow_stopped:
            op = link.receive()
            if op is None or op[0] == "stop":
                return
            if op[0] == "admit":
                sid, prompt, count, smp, drafted, stop, background, plan = op[1:]
                s = Stream(prompt, count, _unpack(smp), draft=drafted, stop_eos=stop)
                s.background = background
                try:
                    self.admit(s, told=plan)
                except (ValueError, NoRoom, OutOfStep) as exc:
                    print(f"[tensorfold] rank 1: stream {sid} refused: {exc}", flush=True)
            elif op[0] == "round":
                for s in [*self.streams.values(), *self.filling]:
                    if s.sid in op[1]:
                        s.done = True
                try:
                    finished.update({s.sid: s for s in self.round(told=op[2])})
                except Exception as exc:                 # rank 0 drops the same failed round
                    print(f"[tensorfold] rank 1: a round failed: {exc}", flush=True)
            elif op[0] == "finish":
                known = {**self.streams, **finished}
                self.finish([known[sid] for sid in op[1] if sid in known])
                for sid in op[1]:
                    finished.pop(sid, None)
            elif op[0] == "drop":
                self.drop()
                finished.clear()
