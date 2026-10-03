"""A shared kept prefix is copied into a spare slot while its longer chain retains its own state."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder


class Slot:
    def __init__(self):
        self.copied = None

    def copy_prefix(self, source, pos, mtp_len):
        self.copied = (source, pos, mtp_len)


def decoder(*, busy=False, room=True):
    dec = object.__new__(MultiDecoder)
    source, spare = Slot(), Slot()
    prefix = list(range(300))
    snap = {"pos": 300, "mtp_len": 299}
    dec.kept = [(prefix, source, snap, "tail"), (prefix + [4, 5], source, {"pos": 302}, "later")]
    dec.free, dec.filling, dec.fills = [spare], [], {}
    dec.streams = {1: SimpleNamespace(st=source, waiting=False)} if busy else {}
    dec.depth, dec.capacity = 3, 1024
    growth = []
    def grow(st, rows, **kwargs):
        growth.append((st, rows, kwargs))
        return room
    dec._grow = grow
    return dec, source, spare, prefix, snap, growth


@pytest.mark.parametrize("busy", [False, True])
def test_a_spare_slot_reuses_the_shared_point_without_consuming_the_source_chain(busy):
    dec, source, spare, prefix, snap, growth = decoder(busy=busy)
    kept = list(dec.kept)
    st, resume, cached = dec._slot_for(prefix + [6, 7, 8], True)
    assert st is spare and cached == len(prefix)
    assert resume == {"state": snap, "tail": "tail"}
    assert spare.copied == (source, len(prefix), 299)
    assert dec.kept == kept and dec.free == []
    assert growth[0][2].get("protect") is source


def test_copy_reservation_refusal_keeps_the_source_and_returns_the_unadmitted_slot():
    dec, source, spare, prefix, _, _ = decoder(busy=True, room=False)
    kept = list(dec.kept)
    with pytest.raises(NoRoom):
        dec.admit(Stream(prefix + [6, 7, 8], 8))
    assert dec.free == [spare] and dec.kept == kept and spare.copied is None


def test_source_protection_excludes_it_from_growth_eviction():
    dec, source, spare, _, _, _ = decoder()
    other = Slot()
    dec.kept.append(([9], other, {}, None))
    shrunk = []
    dec._shrink = shrunk.append
    assert dec._evict_kept(spare, protect=source)
    assert shrunk == [other] and all(k[1] is source for k in dec.kept)


def test_copy_failure_returns_the_spare_and_discards_partial_allocations():
    dec, _, spare, prefix, _, _ = decoder()
    kept, shrunk = list(dec.kept), []
    def fail(*args):
        raise RuntimeError("copy failed")
    spare.copy_prefix = fail
    dec._shrink = lambda st, **kw: shrunk.append((st, kw))
    with pytest.raises(RuntimeError, match="copy failed"):
        dec._slot_for(prefix + [6, 7, 8], True)
    assert dec.free == [spare] and dec.kept == kept
    assert shrunk == [(spare, {"force": True})]


def test_a_partial_resize_failure_is_counted_before_the_spare_is_shrunk():
    from tensorfold.cuda.memory_gate import MemoryGate

    dec, _, spare, prefix, _, _ = decoder()
    del dec._grow
    dec.memory_gate = MemoryGate(10_000, 0)
    dec.memory_gate.held = 700
    spare.capacity, spare.limit, spare.allocated = 256, 1024, 100
    spare.cache_bytes = lambda size=None: spare.allocated if size is None else 900
    spare.layer_bytes = lambda size: 100
    def partial(size):
        spare.allocated = 300
        raise RuntimeError("partial allocation")
    spare.resize = partial
    held = []
    def shrink(st, **kwargs):
        held.append(dec.memory_gate.held)
        dec.memory_gate.give(st.allocated - 100)
        st.allocated = 100
    dec._shrink = shrink
    with pytest.raises(RuntimeError, match="partial allocation"):
        dec._slot_for(prefix + [6, 7, 8], True)
    assert held == [900] and dec.memory_gate.held == 700
    assert dec.free == [spare] and spare.allocated == 100
