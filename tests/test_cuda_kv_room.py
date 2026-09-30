"""One GPU's attention-cache budget: other conversations' buffers go oldest first, and only for room."""

from types import SimpleNamespace

import pytest

from tensorfold.cuda import capacity, geometry
from tensorfold.cuda.streams import KVRoom, PrefixCache
from tensorfold.families.qwen3_5.cuda.affine_memory import draft_weights, weight_transform

from cuda_27b_headers import DRAFT, TEXT, drafter, target

ROW = 16 * 2 * 4 * 256 * 2                               # the 27B's key and value bytes a row, every attention layer


class Buffer:
    """A stand-in tensor with storage of its own."""

    made = 0

    def __init__(self, nbytes: int) -> None:
        Buffer.made += 1
        self.ptr, self.bytes = Buffer.made, nbytes

    def data_ptr(self) -> int:
        return self.ptr

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: self.bytes)


def _state(rows: int) -> SimpleNamespace:
    return SimpleNamespace(kv=[pair for _ in range(16) for pair in (None, (Buffer(rows * ROW // 32),
                                                                          Buffer(rows * ROW // 32)))])


def _cache(*states) -> PrefixCache:
    cache = PrefixCache(4)
    for i, st in enumerate(states):
        cache.add([i], st, None)
    return cache


def test_nothing_goes_while_it_fits():
    live, a, b = _state(4096), _state(8192), _state(1024)
    cache = _cache(a, b)
    KVRoom(cache, (4096 + 8192 + 1024 + 8192) * ROW)(live, 8192 * ROW)
    assert [e[1] for e in cache.entries] == [a, b]


def test_other_buffers_go_least_recently_used_first_and_the_resumed_list_stays():
    live, a, b, c = _state(4096), _state(8192), _state(8192), _state(1024)
    shared = SimpleNamespace(kv=live.kv)                  # the entry the live state resumed: same list
    cache = _cache(a, b, shared, c)
    cache.longest([2, 9])                                 # resuming touches it: newest, and hit
    room = KVRoom(cache, (4096 + 8192 + 1024 + 8192) * ROW)
    room(live, 8192 * ROW)                                # 8,192 rows past the budget: a, the oldest, goes
    assert [e[1] for e in cache.entries] == [b, c, shared]
    room(live, 17000 * ROW)                               # then b, then c; never the shared entry
    assert [e[1] for e in cache.entries] == [shared]
    room(live, 10 ** 6 * ROW)                             # only this conversation is left: nothing more to free
    assert [e[1] for e in cache.entries] == [shared]


def test_a_gb10_budget_evicts_nothing_today_keeps(tmp_path):
    """A GB10's 100 GiB keeps three native-window conversations beside the live one (the old estimate, two)."""

    folder, draft = target(tmp_path / "target"), drafter(tmp_path / "drafter")
    weights = capacity.estimate_weights(folder, weight_transform(folder, one_gpu=True))
    side = draft_weights(draft)
    main = geometry.gdn_geometry(TEXT, 1, 128, rows=128, prompt=4096, evicts=True)
    rider = geometry.draft_geometry(DRAFT, 1, 128, bounded=True)
    both = capacity.Geometry(lambda slots: main.bytes_at(slots) + rider.bytes_at(slots), 128)
    budget = 100 * capacity.GIB
    plan = capacity.make_plan(262144, None, False, budget, capacity.Weights(weights.resident + side.resident, 0),
                              both)
    assert plan.fitting == 262144
    spare = budget - weights.resident - side.resident - both.needed(plan.fitting)
    room = KVRoom(_cache(*(_state(262144) for _ in range(3))), spare + geometry.live_kv(TEXT, 1, 262144))
    before = list(room.cache.entries)
    room(_state(262144 - 1024), 1024 * ROW // 32)          # the live one's last layer grows to the window
    assert room.cache.entries == before


@pytest.mark.parametrize("gib,window", [(19.16, 39936), (100, 262144)])
def test_one_live_window_is_admitted(gib, window, tmp_path):
    """An RTX 4090's default budget admits a 40k window with DFlash2 (four copies gave 8,192); a GB10 its native one."""

    folder, draft = target(tmp_path / "target"), drafter(tmp_path / "drafter")
    weights = capacity.estimate_weights(folder, weight_transform(folder, one_gpu=True))
    side = draft_weights(draft)
    rows = 128 if gib == 100 else 12
    main = geometry.gdn_geometry(TEXT, 1, rows, rows=rows, prompt=4096 if gib == 100 else 2048, evicts=True)
    rider = geometry.draft_geometry(DRAFT, 1, rows, bounded=True)
    both = capacity.Geometry(lambda slots: main.bytes_at(slots) + rider.bytes_at(slots), rows)
    plan = capacity.make_plan(262144, None, False, int(gib * capacity.GIB),
                              capacity.Weights(weights.resident + side.resident, 0), both)
    assert plan.fitting // 1024 * 1024 == window
