"""Two-rank round planning leaves live state untouched until the agreed plan is applied."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.memory_gate import MemoryGate
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.families.qwen4_exp.cuda.multi_plan import admission, apply, ready, round_plan


class Slot:
    def __init__(self, capacity=256):
        self.capacity, self.limit, self.pos, self.mtp_len = capacity, 1024, 0, 0
        self.cur, self.operations, self.kv_dtype = [0], [], "bf16"

    def cache_bytes(self, rows=None):
        return self.layer_bytes(rows) * 2

    def layer_bytes(self, rows=None):
        return (self.capacity if rows is None else rows) * 100

    def resize(self, rows):
        delta = self.cache_bytes(rows) - self.cache_bytes()
        self.operations.append(("resize", rows))
        self.capacity = rows
        return delta

    def reset(self, w):
        self.operations.append(("reset",))
        self.pos = self.mtp_len = 0


def decoder():
    d = object.__new__(MultiDecoder)
    d.slots = [Slot() for _ in range(3)]
    d.free, d.kept, d.streams, d.filling, d.fills, d.held = list(d.slots), [], {}, [], {}, {}
    d.solo, d.solo_on, d.planning, d.points = None, False, False, None
    d.passed, d.arrived, d.fill_yield, d.follower = {}, lambda: False, False, None
    d.link, d.confidence, d.eos, d.gdn = None, 0.3, (), SimpleNamespace(parity=0)
    d.capacity, d.depth, d.keep, d.next_id = 1024, 3, 8, 0
    d.w, d.memory_gate = SimpleNamespace(), MemoryGate(1 << 30, 0)
    d.prefill_rows, d.share, d.round_s, d.row_s, d.converged = 256, 0.5, 1.0, 1.0, True
    return d


def test_admission_plans_growth_without_writing_the_live_slot():
    d = decoder()
    plan = admission(d, Stream([7] * 310, 8))
    assert plan["slot"] == 2 and plan["actions"] == [["resize", 2, 1024, "alone"]]
    assert all(st.capacity == 256 and not st.operations for st in d.slots)
    assert len(d.free) == 3 and d.memory_gate.held == 0
    assert ready(d, plan)
    apply(d, plan)
    assert d.slots[2].capacity == 1024 and d.slots[2].operations == [("resize", 1024)]
    assert len(d.free) == 2 and d.memory_gate.held == 2 * (1024 - 256) * 100


def test_a_follower_can_refuse_growth_without_mutating_its_slot():
    d = decoder()
    plan = admission(d, Stream([7] * 310, 8))
    d.memory_gate.room, d.memory_gate.live = 0, lambda: 0
    assert not ready(d, plan)
    assert all(st.capacity == 256 and not st.operations for st in d.slots)


def test_round_plan_carries_pure_passes_and_mixed_pieces_at_each_boundary():
    d = decoder()
    live = Stream([7], 12, sid=0, st=d.slots[0], out=[3], drafts=[4])
    d.streams[0], d.free = live, []
    for sid, length in ((1, 300), (2, 310)):
        s = Stream([7] * length, 12, sid=sid, st=d.slots[sid])
        d.filling.append(s)
        d.fills[sid] = [SimpleNamespace(stops=[]), True, 0, None]
    plan = round_plan(d)
    assert plan["pass_width"] == 128
    assert plan["passes"] == [[[1, 0, 256]], [[1, 256, 44], [2, 0, 212]], [[2, 212, 98]]]
    assert plan["mixed"] == [[[1, 0, 128]], [[1, 256, 44], [2, 0, 84]], [[2, 212, 98]], []]
    assert [d.fills[sid][2] for sid in (1, 2)] == [0, 0]
    assert len(d.filling) == 2 and not any(st.operations for st in d.slots)


def test_waiting_owner_keeps_its_graph_slot_and_plans_recurrence_flush():
    d = decoder()
    old = Stream([7], 40, sid=0, st=d.slots[0], out=[3])
    waiting = Stream([8], 40, sid=1, st=d.slots[1], out=[4], drafts=[5])
    waiting.st.pos = 255
    d.streams, d.free = {0: old, 1: waiting}, [d.slots[2]]
    d.held, d.memory_gate.room = {1: [7, 8]}, 0
    d.solo, d.solo_on = SimpleNamespace(st=waiting.st), True
    plan = round_plan(d)
    assert plan["solo"] is None
    assert ["flush", 1] in plan["actions"] and plan["held"] == []
    assert next(s for s in plan["streams"] if s[0] == 1)[2] is True
    assert d.held == {1: [7, 8]} and not waiting.waiting
    assert not any(st.operations for st in d.slots)


def test_an_ended_streams_graph_slot_is_not_reused_before_finish():
    d = decoder()
    old = Stream([7], 40, sid=0, st=d.slots[0], out=[3], drafts=[4])
    newest = Stream([8], 40, sid=1, st=d.slots[1], out=[5], drafts=[6])
    old.st.pos = newest.st.pos = 255
    d.streams, d.free = {0: old, 1: newest}, [d.slots[2]]
    d.memory_gate.room = 0
    d.solo, d.solo_on = SimpleNamespace(st=newest.st), True
    plan = round_plan(d)
    assert plan["ended"] == [1] and plan["solo"] is None
    assert not old.done and not newest.done
    assert not any(st.operations for st in d.slots)


def test_plan_fingerprint_covers_execution_mode_and_exact_draft_threshold():
    from tensorfold.families.qwen4_exp.cuda.multi_tp import shape

    a, b = decoder(), decoder()
    for d in (a, b):
        d.confidence, d.eos, d.gdn = 0.7000001, (), SimpleNamespace(parity=0)
    assert shape(a) == shape(b)
    b.confidence = 0.7000002
    assert shape(a) != shape(b)                            # startup's six-decimal display is not an exact check
    b.confidence = a.confidence
    b.converged = False
    assert shape(a) != shape(b)
    b.converged = True
    b.gdn.parity = 1
    assert shape(a) != shape(b)


def test_leader_drop_during_first_token_delivery_clears_the_follower():
    from tensorfold.families.qwen4_exp.cuda.multi_tp import OutOfStep

    d = decoder()
    s = Stream([7], 12, sid=0, st=d.slots[0], out=[3])
    d.streams, d.free, d.link = {0: s}, d.slots[1:], None
    d.follower = SimpleNamespace(receive=lambda: ["drop"])
    with pytest.raises(OutOfStep, match="aborted"):
        d._joined_ranks()
    assert not d.streams and not d.filling and len(d.free) == 3


@pytest.mark.parametrize("message", [None, ["stop"]])
def test_stop_during_prompt_delivery_returns_without_waiting_for_another_message(message):
    d = decoder()
    d.link = None
    messages, reads = iter([["round", [], {}], message]), []
    def receive():
        reads.append(True)
        return next(messages)                            # a third read fails: the stop must end follow()
    d.round = lambda told=None: d._joined_ranks()
    d.follow(SimpleNamespace(receive=receive))
    assert len(reads) == 2


def test_a_lone_stream_can_use_physically_available_room_above_the_initial_gate():
    d = decoder()
    d.memory_gate.room = 0
    d.memory_gate.live = lambda: 1 << 30
    plan = admission(d, Stream([7] * 310, 8))
    assert plan["error"] is None
    assert ready(d, plan)                                # startup already fitted a lone stream's whole window
    assert all(st.capacity == 256 and not st.operations for st in d.slots)


def test_an_empty_decoder_refuses_instead_of_retrying_when_even_a_lone_prompt_cannot_fit():
    d = decoder()
    d.memory_gate.room, d.memory_gate.live = 0, lambda: 0
    d.w.comm, d.link = None, None                         # no collective is needed to test this refusal
    d.confidence, d.eos, d.gdn = 0.3, (), SimpleNamespace(parity=0)
    with pytest.raises(ValueError, match="available memory"):
        d._prepare_admission(Stream([7] * 310, 8), None)
    assert len(d.free) == 3 and not any(st.operations for st in d.slots)


def test_admission_carries_leader_message_points_to_a_different_follower():
    leader, follower = decoder(), decoder()
    leader.points = lambda prompt: [256, 512]
    follower.points = lambda prompt: pytest.fail("follower replanned the leader's points")
    request = Stream([7] * 800, 8)
    plan = admission(leader, request)
    assert plan["points"] == [256, 512]
    follower.w.comm = None
    slot, resume, cached = follower._prepare_admission(request, plan)
    assert request.stops == [256, 512] and resume is None and cached == 0
    assert slot is follower.slots[plan["slot"]]


def test_prompt_marker_changes_are_in_the_round_fingerprint():
    from tensorfold.families.qwen4_exp.cuda.multi_tp import shape

    d = decoder()
    stream = Stream([7] * 800, 8, sid=1, st=d.slots[0])
    d.filling = [stream]
    engine = SimpleNamespace(stops=[256, 512])
    d.fills[1] = [engine, True, 0, None]
    before = shape(d)
    engine.stops = [256]
    assert shape(d) != before


def test_lone_graph_rebinding_preserves_both_prefix_chains_on_both_ranks():
    leader, follower = decoder(), decoder()
    for d in (leader, follower):
        d.solo, d.solo_on = SimpleNamespace(st=d.slots[0]), True
        d.kept = [([7] * 256, d.slots[0], {}, None), ([8] * 256, d.slots[1], {}, None)]
        d.streams = {1: Stream([8] * 257, 8, sid=1, st=d.slots[1], out=[9])}
        d.free = [d.slots[2]]
        d._state_changed = lambda st: st.operations.append(("graphs",))
    plan = round_plan(leader)
    assert plan["actions"] == [["solo", 1]] and plan["solo"] == 1
    assert leader.solo.st is leader.slots[0] and not any(st.operations for st in leader.slots)
    for d in (leader, follower):
        apply(d, plan)
        assert d.solo.st is d.slots[1] and d.slots[1].operations == [("graphs",)]
        assert [ids for ids, _, _, _ in d.kept] == [[7] * 256, [8] * 256]
