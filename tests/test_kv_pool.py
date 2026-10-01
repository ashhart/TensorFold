"""tensorfold.cuda.kv_pool: prefix matching, LRU eviction within the budget, identical decisions for equal inputs."""

from tensorfold.cuda.kv_pool import PrefixPool


def make(n: int):
    return lambda: ({"n": n}, n * 10)


def test_match_takes_the_longest_proper_prefix():
    pool = PrefixPool(budget=10 ** 6, min_tokens=2)
    pool.add([1, 2, 3], make(3))
    pool.add([1, 2, 3, 4, 5], make(5))
    pool.add([9, 9, 9], make(3))
    assert pool.match([1, 2, 3, 4, 5, 6]).ids == [1, 2, 3, 4, 5]
    assert pool.match([1, 2, 3, 4]).ids == [1, 2, 3]
    assert pool.match([1, 2, 3, 4, 5]).ids == [1, 2, 3]           # a full match leaves no row to prefill
    assert pool.match([7, 1, 2, 3]) is None


def test_eviction_is_least_recently_used_and_deterministic():
    def run():
        pool = PrefixPool(budget=100, min_tokens=2)
        for ids in ([1, 1, 1], [2, 2, 2], [3, 3, 3]):            # 30 bytes each
            pool.add(ids, make(3))
        pool.match([1, 1, 1, 0])                                 # touch the oldest
        pool.add([4, 4, 4, 4], make(4))                          # 40 bytes: evicts [2, 2, 2]
        return [e.ids for e in pool.entries], pool.used

    first = run()
    assert first == run()
    assert first[0] == [[3, 3, 3], [1, 1, 1], [4, 4, 4, 4]] and first[1] == 100


def test_short_or_oversized_states_are_not_kept_and_repeats_refresh():
    pool = PrefixPool(budget=50, min_tokens=3)
    assert pool.add([1, 2], make(2)) is None
    assert pool.add([1, 2, 3, 4, 5, 6], make(6)) is None          # 60 bytes > budget
    calls = []
    pool.add([5, 5, 5], lambda: (calls.append(1) or {"n": 3}, 30))
    pool.add([5, 5, 5], lambda: (calls.append(1) or {"n": 3}, 30))
    assert len(calls) == 1 and len(pool.entries) == 1
