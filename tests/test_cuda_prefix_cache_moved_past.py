"""Kept prompt states past ``keep``: a state its own conversation has moved past goes before the newest of another."""

from types import SimpleNamespace

from tensorfold.cuda.streams import KVRoom, PrefixCache

HEADER = [7] * 8                                          # a system block the conversations share
X1, Y1 = HEADER + [1] * 4, HEADER + [2] * 4
X2, Y2 = X1 + [3] * 4, Y1 + [4] * 4
X3 = X2 + [5] * 4


def _ids(cache):
    return [e[0] for e in cache.entries]


def _turn(cache, prompt, kept):
    """A request: resume from the longest kept prefix, then keep the state its prefill ends with."""

    hit = cache.longest(prompt)
    cache.add(kept, None, None)
    return hit[0] if hit else []


def test_a_conversation_resumes_from_its_newest_state_while_another_starts():
    cache = PrefixCache(3)
    assert _turn(cache, X1 + [0], HEADER) == []            # the first prefill keeps the shared block ...
    cache.add(X1, None, None)                              # ... and its own end
    assert _turn(cache, X2 + [0], X2) == X1                # the next turn resumes from it
    assert _turn(cache, Y1 + [0], Y1) == HEADER            # a sub-agent starts from the shared block
    assert _ids(cache) == [X2, HEADER, Y1]                 # X1 went: X2 extends it, nothing else does
    assert _turn(cache, X3 + [0], X3) == X2                # so the main conversation's next turn finds X2


def test_the_old_rule_dropped_the_newest_state_of_the_waiting_conversation():
    class Old(PrefixCache):
        _moved_past = staticmethod(lambda entry: False)

    cache = Old(3)
    _turn(cache, X1 + [0], HEADER)
    cache.add(X1, None, None)
    _turn(cache, X2 + [0], X2)
    _turn(cache, Y1 + [0], Y1)
    assert _ids(cache) == [X1, HEADER, Y1]                 # X2, never resumed yet, went first
    assert _turn(cache, X3 + [0], X3) == X1                # the next turn re-reads X2's tokens


def test_a_block_two_diverging_conversations_extend_stays():
    cache = PrefixCache(3)
    for ids in (HEADER, X1, Y1, X2):                       # HEADER is extended by X1 and Y1, which diverge
        cache.add(ids, None, None)
    assert _ids(cache) == [HEADER, Y1, X2]                 # X1 went (X2 extends it); HEADER stayed
    cache.add(Y2, None, None)
    assert _ids(cache) == [HEADER, X2, Y2]                 # then Y1 (Y2 extends it); HEADER still stays


def test_without_extensions_the_never_resumed_entry_goes_first_then_the_oldest():
    cache = PrefixCache(2)
    for ids in ([1], [2]):
        cache.add(ids, None, None)
    cache.longest([1, 9])                                  # [1] is resumed from
    cache.add([3], None, None)
    assert _ids(cache) == [[1], [3]]                       # [2], never resumed, went
    cache.longest([3, 9])
    cache.add([4], None, None)
    assert _ids(cache) == [[3], [4]]                       # both resumed: the oldest went


def test_eviction_for_memory_takes_a_moved_past_entry_among_those_offered():
    cache = PrefixCache(4)
    _turn(cache, X1 + [0], X1)
    _turn(cache, X2 + [0], X2)                             # resumes from X1, then keeps X2
    cache.add(Y1, None, None)
    assert cache.evict([cache.entries[0], cache.entries[2]]) and _ids(cache) == [X2, Y1]   # X1, not Y1 (never resumed)


class _Buffer:
    """A stand-in tensor with storage of its own (as tests/test_cuda_kv_room.py's)."""

    made = 0

    def __init__(self, nbytes: int) -> None:
        _Buffer.made += 1
        self.ptr, self.bytes = _Buffer.made, nbytes

    def data_ptr(self) -> int:
        return self.ptr

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: self.bytes)


def _state(nbytes: int) -> SimpleNamespace:
    return SimpleNamespace(kv=[(_Buffer(nbytes), _Buffer(nbytes))])


def test_one_gpus_room_frees_a_moved_past_state_before_another_conversations_newest():
    live, x1, x2, y1 = _state(100), _state(100), _state(100), _state(100)
    cache = PrefixCache(4)
    cache.add(X1, x1, None)
    cache.longest(X2 + [0])                                # X2's request resumes from X1 ...
    cache.add(X2, x2, None)                                # ... and keeps X2
    cache.add(Y1, y1, None)
    room = KVRoom(cache, 4 * 200)                          # the live state and three kept buffers fit
    room(live, 200)                                        # one more buffer: one kept state goes
    assert _ids(cache) == [X2, Y1]                         # X1, which X2 extends: not X2 or Y1, never resumed yet
    room(live, 400)                                        # one more: no moved-past state left, the old rule (oldest)
    assert _ids(cache) == [Y1]


def test_a_state_resumed_from_after_its_extension_was_kept_is_not_moved_past():
    cache = PrefixCache(keep=2)                            # as tests/cuda/test_qwen27_multi.py's prefix-cache case
    cache.add([1, 2], "a", None)
    cache.add([1, 2, 3], "b", None)
    assert cache.longest([1, 2, 9])[1] == "a"             # a fork back to [1, 2], after [1, 2, 3] was kept
    cache.add([7], "c", None)
    assert [e[1] for e in cache.entries] == ["a", "c"]     # the old rule: "b", the least recently used, goes
