import pytest

from tensorfold.server.app import CheckpointStore


@pytest.mark.parametrize("pinned", [False, True])
def test_an_oversized_checkpoint_cannot_exceed_the_store_budget(pinned):
    store = CheckpointStore(2, copier=lambda c: c, budget_bytes=1000, sizer=lambda c: c[0])
    store.insert([1, 2], [1200], last_prompt=[1, 2], pinned=pinned)
    assert store.nbytes <= 1000
    assert len(store) == 0


def test_rejecting_an_oversized_checkpoint_preserves_a_usable_prefix():
    store = CheckpointStore(2, copier=lambda c: c, budget_bytes=1000, sizer=lambda c: c[0])
    store.insert([1], [800], last_prompt=[1])
    store.insert([2], [1200], last_prompt=[2])
    assert store.match([1, 3]) == (1, [800], [1])
    assert store.nbytes == 800
