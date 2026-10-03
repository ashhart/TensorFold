"""The lone-stream graph slot keeps its graphs once kept prompt ends fill every slot (host: slot logic only)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda.memory_gate import MemoryGate
from tensorfold.families.qwen4_exp.cuda import graphs as graphs_module
from tensorfold.families.qwen4_exp.cuda.multi import FIRST, MultiDecoder


class Graphs:
    made = 0

    def __init__(self, e, *, max_rows=8):
        self.slot = e.st
        Graphs.made += 1


class Slot:
    def __init__(self):
        self.capacity, self.limit, self.pos, self.copies = FIRST, 1024, 0, 0
        self.image_positions = None

    def cache_bytes(self, rows=None):
        return rows or self.capacity

    def layer_bytes(self, rows=None):
        return 0

    def resize(self, rows):
        before, self.capacity = self.capacity, rows
        return rows - before

    def reset(self, w):
        self.pos = 0

    def copy_from(self, other):
        self.copies += 1


@pytest.fixture
def decoder(monkeypatch):
    monkeypatch.setattr(graphs_module, "Graphs", Graphs)

    def make(slots):
        dec = object.__new__(MultiDecoder)
        dec.w, dec.planning, dec.link, dec.depth, dec.keep = SimpleNamespace(comm=None), False, None, 3, 8
        dec.free = [Slot() for _ in range(slots)]
        dec.streams, dec.filling, dec.held, dec.kept = {}, [], {}, []
        dec.memory_gate, dec.next_id = MemoryGate(1 << 40, 0), 0
        dec.solo = SimpleNamespace(st=dec.free[0], rows=4)
        dec.solo.graphs = Graphs(dec.solo)
        dec.solo.slot_graphs = {id(dec.free[0]): dec.solo.graphs}
        return dec
    return make


def _request(dec, prompt):
    """Admission, the kept prompt end, a first round alone and the finish, as the rounds call them."""

    s = SimpleNamespace(prompt=list(prompt), sid=dec.next_id, draft=True, vision=None)
    dec.next_id += 1
    s.st, _, s.cached = dec._slot_for(list(prompt), True)
    dec._remember(list(prompt[:-1]), s.st, {"pos": len(prompt) - 1, "mtp_len": len(prompt) - 2}, None)
    dec.streams[s.sid] = s
    if s.st is not dec.solo.st:
        dec._move_to_solo(s)
    graphs = dec.solo.graphs
    dec.finish([s])
    return s, graphs


def _prompts(n):
    return [[(13 * i + 7 * j + 1) % 3999 + 1 for i in range(5 + j)] for j in range(n)]


def test_eight_different_prompts_on_five_slots_keep_the_graph_slot_and_its_graphs(decoder):
    dec = decoder(5)
    slot, graphs, made = dec.solo.st, dec.solo.graphs, Graphs.made
    for prompt in _prompts(8):
        s, used = _request(dec, prompt)
        assert s.cached == 0 and s.st is slot and used is graphs
    assert Graphs.made == made and len(dec.kept) == 5      # every slot holds a kept end, none was evicted


def test_a_fresh_prompt_moves_the_graph_slots_kept_end_and_takes_the_slot(decoder):
    dec = decoder(2)
    first, _ = _request(dec, [5, 17, 99, 250])
    slot = dec.solo.st
    assert first.st is slot and any(k[1] is slot for k in dec.kept)
    second, _ = _request(dec, [1023, 7, 64, 300, 11])
    assert second.st is slot and second.cached == 0
    assert [k[1] is slot for k in dec.kept] == [False, True]  # the first prompt's end moved to the other slot
    again, _ = _request(dec, [5, 17, 99, 250, 40, 41])
    assert again.cached == 3                                 # and still resumes there


def test_two_conversations_taking_turns_on_a_full_pool_capture_once_per_slot(decoder):
    dec = decoder(3)
    for prompt in _prompts(8):
        _request(dec, prompt)
    a, b = _prompts(10)[8:]
    made = Graphs.made
    for turn in range(6):
        for i, prompt in enumerate((a, b)):
            s, _ = _request(dec, prompt)
            if turn:
                assert s.cached == len(prompt) - 4                    # resumed from its own kept end
            if turn == 1 and i == 0:
                made += 1                                             # the first conversation's slot, once
            assert Graphs.made <= made
            nxt = prompt + [50 + turn, 60 + turn, 70 + i]
            a, b = (nxt, b) if i == 0 else (a, nxt)


def test_resizing_a_slot_drops_the_graphs_it_kept(decoder):
    dec = decoder(3)
    slot = dec.solo.st
    other = dec.free[1]
    dec._graphs_to(other)
    kept = dec.solo.graphs
    dec._graphs_to(slot)
    dec._graphs_to(other)
    assert dec.solo.graphs is kept                                   # taken back, not captured again
    dec._graphs_to(slot)
    dec.free = [other, dec.free[2]]
    dec.kept = [([1, 2], slot, {}, None)]
    assert dec._relocate_kept(slot, None) is True
    assert id(other) in dec.solo.slot_graphs and other.copies == 1   # same size: its graphs stay
    slot.resize(512)
    dec.kept = [([1, 2], slot, {}, None)]
    dec.free = [other]
    assert dec._relocate_kept(slot, None) is True and other.capacity == 512
    assert id(other) not in dec.solo.slot_graphs                     # a resized spare loses its graphs
    dec._state_changed(slot)
    assert dec.solo.graphs is dec.solo.slot_graphs[id(slot)] and dec.solo.graphs.slot is slot
