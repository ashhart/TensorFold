"""The 27B's concurrent streams on one GPU hold their prompts and grow a step at a time while memory lasts (no GPU):
cached prompt ends go first, then the newest waits, then the newest ends; a lone stream always grows."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorfold.cuda.memory_gate import MemoryGate, NoRoom
from tensorfold.cuda.streams import PrefixCache
from tests.test_qwen27_prompt_end_cache_host import _decoder, _ids_of, _ids_state, cuda_modules  # noqa: F401

pytestmark = pytest.mark.torch


def _held(dec) -> int:
    """Bytes the streams' and cached ends' attention caches hold (a buffer viewed twice counts once)."""

    seen = {}
    states = [s.st for s in dec.streams.values() if s.st is not None] + [st for _, st, _ in dec.cache.entries]
    for st in states:
        for kv in st.kv:
            for t in kv or ():
                seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
    return sum(seen.values())


def _gated(m, total: int):
    dec = _decoder(m)
    dec.row_bytes = dec.layer_bytes = 8            # the stand-in state: one layer of float32 keys and values
    dec.memory_gate = MemoryGate(1 << 62, reserve=0, live=lambda: total - _held(dec))
    return dec


def _stream(m, sid: int, pos: int, count: int = 30000):
    ids = [(7 * sid + i) % 1000 for i in range(pos)]
    return SimpleNamespace(sid=sid, st=_ids_state(m, ids), prompt=ids[:100], count=count, done=False, waiting=False,
                           out=[1] * 5, error=None)


def test_the_oldest_grows_the_newest_waits_then_ends_and_a_lone_stream_grows(cuda_modules):  # noqa: F811
    m = cuda_modules
    held = 2 * 8190 * 8
    dec = _gated(m, held + (16384 - 8190) * 8 + 16384 * 8 + 1000)     # room for one stream's next step
    old, new = _stream(m, 0, 8190), _stream(m, 1, 8190)
    dec.streams = {0: old, 1: new}
    ids = _ids_of(old.st)
    assert dec._make_room([old, new]) == [] and not old.waiting and new.waiting
    assert old.st.kv[0][0].shape[0] == 16384 and _ids_of(old.st) == ids and old.st.limit == 16384
    old.st.pos = 16380                                                # the oldest needs its next step now
    ended = dec._make_room([old, new])
    assert ended == [new] and new.done and "ran out of memory" in str(new.error) and new.st is None
    assert 1 not in dec.streams and dec.memory_gate.ends == 1
    assert not old.waiting and old.st.kv[0][0].shape[0] == 24576        # alone, it grows regardless


def test_cached_prompt_ends_go_before_a_stream_waits(cuda_modules):  # noqa: F811
    m = cuda_modules
    dec = _gated(m, 8190 * 8 + 16 * 8 + 4096 * 8 + (16384 - 8190) * 8 + 16384 * 8 - 10000)   # fits once it goes
    dec.cache.add([1, 2, 3], _ids_state(m, list(range(4096))), None)  # a cached prompt end holding 4,096 rows
    live, other = _stream(m, 0, 8190), _stream(m, 1, 16, count=10)
    dec.streams = {0: live, 1: other}
    assert dec._make_room([live, other]) == [] and not live.waiting and dec.cache.entries == []


def test_a_request_waits_for_room_while_others_decode_and_starts_alone_regardless(cuda_modules):  # noqa: F811
    m = cuda_modules
    dec = _gated(m, 8190 * 8 + 100)                                   # no room beyond the live stream
    live = _stream(m, 0, 8190)
    dec.streams = {0: live}
    s = SimpleNamespace(prompt=list(range(300)), count=10, draft=False, sampling=None)
    with pytest.raises(NoRoom, match="waits for memory"):
        dec._room(s)
    live.waiting = True
    with pytest.raises(NoRoom, match="already wait"):
        dec._room(s)
    dec.streams = {}
    dec._room(s)                                                      # alone: startup fitted one whole window
    assert dec._first(s) == dec._most(s) == 310                        # a short request: its prompt and reply


def test_eviction_takes_the_entry_add_would_drop_next():
    cache = PrefixCache(4)
    for ids in ([1], [2], [3]):
        cache.add(ids, None, None)
    cache.longest([1, 9])                                             # [1] is resumed from: kept longer
    assert cache.evict() and [e[0] for e in cache.entries] == [[3], [1]]
    assert cache.evict() and cache.evict() and not cache.evict()


def test_a_stream_near_its_end_does_not_grow_past_its_prompt_and_reply(cuda_modules):  # noqa: F811
    m = cuda_modules
    dec = _gated(m, 1 << 30)
    s = _stream(m, 0, 150, count=50)                  # prompt 100 of its 150 rows, reply 50: it never holds past 150
    dec.streams = {0: s}
    assert dec._make_room([s]) == [] and s.st.kv[0][0].shape[0] == 150 and not s.waiting
