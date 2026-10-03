"""Two ranks meet on the store after loading: a rank that never arrives is named instead of waited on in NCCL."""

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.comm import NCCL


class Store:
    """The TCPStore calls ``ready`` makes; ``wait`` times out like torch's store while a key is missing."""

    def __init__(self, keys=()):
        self.keys, self.waits = set(keys), 0

    def set(self, key, value):
        self.keys.add(key)

    def wait(self, keys, timeout):
        self.waits += 1
        if not set(keys) <= self.keys:
            raise RuntimeError(f"wait timeout after {timeout.total_seconds() * 1000:.0f}ms, keys: {keys}")


def comm(store, rank=0):
    c = object.__new__(NCCL)
    c.rank, c.world, c.store = rank, 2, store
    return c


def test_both_ranks_loaded_passes_at_once():
    store = Store({"tf_ready/loading/1"})
    comm(store).ready("loading", every=0.01, timeout=1.0)
    assert "tf_ready/loading/0" in store.keys and store.waits == 1


def test_a_missing_rank_is_named_after_the_timeout(capsys):
    store = Store()
    with pytest.raises(RuntimeError, match="rank 0 finished loading but rank 1 has not"):
        comm(store).ready("loading", every=0.01, timeout=0.05)
    assert "waiting for rank 1" in capsys.readouterr().out and store.waits >= 2


def test_a_store_failure_other_than_its_timeout_goes_up():
    class Broken(Store):
        def wait(self, keys, timeout):
            raise RuntimeError("connection reset by peer")

    with pytest.raises(RuntimeError, match="connection reset"):
        comm(Broken()).ready("loading", every=0.01, timeout=1.0)
