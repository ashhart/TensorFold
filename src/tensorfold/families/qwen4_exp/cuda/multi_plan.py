"""Plan cache changes and prompt pieces on the host; the follower applies the leader's decisions verbatim."""

from __future__ import annotations

import copy

import torch

from tensorfold.cuda.markers import MIN_GAP
from .decode import entry_end
from .prompt_plan import pass_limit
from .multi_fill import PASS_MIN


class Shadow:
    """A slot's geometry and positions for planning, without copying or writing any of its tensors."""

    def __init__(self, st, index: int, actions: list) -> None:
        self.source, self.index, self.actions = st, index, actions
        self.capacity, self.limit, self.pos, self.mtp_len = st.capacity, st.limit, st.pos, st.mtp_len

    def cache_bytes(self, rows=None):
        return self.source.cache_bytes(self.capacity if rows is None else rows)

    def layer_bytes(self, rows):
        return self.source.layer_bytes(rows)

    def resize(self, rows):
        delta = self.cache_bytes(rows) - self.cache_bytes()
        self.actions.append(["resize", self.index, rows])
        self.capacity = rows
        return delta

    def reset(self, w):
        self.actions.append(["reset", self.index])
        self.pos = self.mtp_len = 0

    def copy_prefix(self, other, pos, mtp_len):
        self.actions.append(["prefix", self.index, other.index, pos, mtp_len])

    def copy_from(self, other):
        self.actions.append(["copy", self.index, other.index])
        self.pos, self.mtp_len = other.pos, other.mtp_len


def view(dec):
    """A host-only view runs the existing admission and memory policy, recording its slot operations."""

    p = copy.copy(dec)
    p.planning, p.actions = True, []
    p.slots = [Shadow(st, i, p.actions) for i, st in enumerate(dec.slots)]
    slots = {id(st): p.slots[i] for i, st in enumerate(dec.slots)}
    def stream(s):
        result = copy.copy(s)
        result.st = slots[id(s.st)]
        return result
    p.streams = {sid: stream(s) for sid, s in dec.streams.items()}
    p.filling = [stream(s) for s in dec.filling]
    p.all_streams = [*p.streams.values(), *p.filling]
    p.fills = {sid: list(fill) for sid, fill in dec.fills.items()}
    p.passed = dict(dec.passed)
    p.free = [slots[id(st)] for st in dec.free]
    p.kept = [(ids, slots[id(st)], snap, tail) for ids, st, snap, tail in dec.kept]
    p.held = {sid: list(rows) for sid, rows in dec.held.items()}
    p.memory_gate = copy.copy(dec.memory_gate)
    if dec.memory_gate.live is not None:
        free, held = dec.memory_gate.live(), dec.memory_gate.held
        p.memory_gate.live = lambda: free - (p.memory_gate.held - held)
    if dec.solo is not None:
        p.solo = copy.copy(dec.solo)
        p.solo.st = slots[id(dec.solo.st)]
    return p


def layout(p) -> dict:
    """The exact slot ownership and stream status after the planned memory operations."""

    return {"actions": p.actions, "free": [st.index for st in p.free],
            "kept": [[st.index, len(ids)] for ids, st, _, _ in p.kept],
            "streams": [[s.sid, s.st.index, bool(s.waiting), bool(s.done), str(s.error) if s.error else None]
                        for s in p.all_streams],
            "live": list(p.streams), "held": [[sid, rows] for sid, rows in p.held.items()],
            "waits": p.memory_gate.waits, "ends": p.memory_gate.ends, "passed": sorted(p.passed.items())}


def admission(dec, s) -> dict:
    p = view(dec)
    st, resume, cached = p._slot_for(list(s.prompt), s.draft)
    source = next((st.index for ids, st, snap, _ in p.kept
                   if resume is not None and snap is resume["state"] and ids == list(s.prompt[:cached])), None)
    error = None
    if not p._grow(st, len(s.prompt) + p.depth + 2, alone=not p.streams and not p.filling):
        if resume is None:
            p.free.append(st)
        else:
            p._remember(list(s.prompt[:cached]), st, resume["state"], resume["tail"])
        error = f"a {len(s.prompt)}-token prompt waits for memory until a live stream finishes"
    markers = p.points(s.prompt) if s.draft and p.points is not None else []
    points = sorted({n for n in markers if cached + MIN_GAP <= n < entry_end(s.prompt)})
    return {**layout(p), "slot": st.index, "cached": cached, "resume_slot": source, "error": error, "points": points}


def round_plan(dec) -> dict:
    p = view(dec)
    ended = [s.sid for s in p._make_room()]
    live = [s for s in p.streams.values() if not s.done and not s.waiting]
    solo = None
    if (p.solo_on and not ended and not p.filling and len(p.streams) == 1
            and len(live) == 1 and live[0].draft and live[0].constraint is None):
        s = live[0]
        if s.st is not p.solo.st:
            p._move_to_solo(s)
        else:
            p._flush(s)
        solo = s.sid
    result = layout(p)
    width = pass_limit(p.prefill_rows, True, p.share, p.round_s, p.row_s, PASS_MIN)
    passes, mixed = [], []
    while p.filling:
        mixed.append([[s.sid, a, n] for s, a, n in p._pieces(width)])
        pieces = p._pieces()
        passes.append([[s.sid, a, n] for s, a, n in pieces])
        p._note_passed(pieces)
        for s, a, n in pieces:
            p.fills[s.sid][2] = a + n
            if a + n == len(s.prompt):
                p.filling.remove(s)
    mixed.append([])
    return {**result, "ended": ended, "solo": solo, "passes": passes, "mixed": mixed, "pass_width": width}


def apply(dec, plan):
    """Apply agreed slot operations before prefill or decode, with no independent follower memory decisions."""

    kept = {(dec._index(st), len(ids)): (ids, st, snap, tail) for ids, st, snap, tail in dec.kept}
    all_streams = {s.sid: s for s in [*dec.streams.values(), *dec.filling]}
    for op in plan["actions"]:
        if op[0] == "flush":
            dec._flush(all_streams[op[1]])
            continue
        st = dec.slots[op[1]]
        if op[0] == "solo":
            dec.solo.st = st
            dec._state_changed(st)
        elif op[0] == "reset":
            st.reset(dec.w)
        elif op[0] == "resize":
            dec._state_changed(st)
            delta = st.resize(op[2])
            dec.memory_gate.take(delta) if delta > 0 else dec.memory_gate.give(-delta)
        elif op[0] == "copy":
            st.copy_from(dec.slots[op[2]])
        elif op[0] == "kept":
            moved = [k for k in kept if k[0] == op[1]]
            for key in moved:
                ids, _, snap, tail = kept.pop(key)
                kept[(op[2], key[1])] = (ids, dec.slots[op[2]], snap, tail)
        elif op[0] == "prefix":
            st.copy_prefix(dec.slots[op[2]], op[3], op[4])
    if plan["actions"] and torch.cuda.is_available():
        torch.cuda.empty_cache()
    dec.free = [dec.slots[i] for i in plan["free"]]
    dec.kept = [kept[tuple(key)] for key in plan["kept"]]
    for sid, slot, waiting, done, error in plan["streams"]:
        s = all_streams[sid]
        s.st, s.waiting, s.done = dec.slots[slot], waiting, done
        s.error = RuntimeError(error) if error else None
    dec.streams = {sid: all_streams[sid] for sid in plan["live"]}
    dec.held = dict(plan["held"])
    dec.memory_gate.waits, dec.memory_gate.ends = plan["waits"], plan["ends"]
    dec.passed = dict(plan["passed"])


def ready(dec, plan) -> bool:
    """Each rank checks that the proposed cache moves fit locally before the shared plan check."""

    p = view(dec)
    for op in plan["actions"]:
        if op[0] == "reset":
            p.slots[op[1]].pos = p.slots[op[1]].mtp_len = 0
        elif op[0] == "copy":
            dst, src = p.slots[op[1]], p.slots[op[2]]
            if dst.capacity < src.capacity or dst.source.kv_dtype != src.source.kv_dtype:
                return False
            dst.pos, dst.mtp_len = src.pos, src.mtp_len
        elif op[0] == "prefix":
            dst, src = p.slots[op[1]], p.slots[op[2]]
            if not 0 <= op[3] <= min(dst.capacity, src.pos) or not 0 <= op[4] <= min(dst.capacity, src.mtp_len):
                return False
        if op[0] != "resize":
            continue
        st, rows = p.slots[op[1]], op[2]
        delta = st.cache_bytes(rows) - st.cache_bytes()
        if delta > 0:
            peak = delta + st.layer_bytes(rows)
            if len(op) > 3 and op[3] == "alone":       # startup fitted one full window; keep the physical check
                if p.memory_gate.live is not None and p.memory_gate.live() < peak:
                    return False
            elif not p.memory_gate.fits(peak):
                return False
        st.capacity = rows
        p.memory_gate.take(delta) if delta > 0 else p.memory_gate.give(-delta)
    return True
