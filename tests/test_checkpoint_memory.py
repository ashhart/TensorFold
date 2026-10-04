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


def _order(store):
    return [entry.cache[0] for entry in store._entries]


def test_a_conversation_that_moved_on_loses_its_older_checkpoint_before_another_conversations_newest():
    store = CheckpointStore(3, copier=lambda c: c)
    store.insert([5, 6], ["b"], last_prompt=[5, 6, 7])                  # conversation b, the least recently used
    store.insert([1, 2], ["a1"], last_prompt=[1, 2, 3])
    store.insert([1, 2, 3, 4], ["a2"], last_prompt=[1, 2, 3, 4, 5])     # a's next turn continues a1
    store.insert([9], ["c"], last_prompt=[9, 9])                         # over the three slots
    assert _order(store) == ["c", "a2", "b"]
    assert store.evict_one() and _order(store) == ["c", "a2"]           # then the least recently used
    store.insert([1, 2, 3, 4, 5, 6], ["a3"], last_prompt=[1, 2, 3, 4, 5, 6, 7])
    assert store.evict_one() and _order(store) == ["a3", "c"]           # memory pressure: a2 first, c stays


def test_one_turns_checkpoints_are_all_its_newest():
    store = CheckpointStore(3, copier=lambda c: c)
    store.insert([5, 6], ["b"], last_prompt=[5, 6, 7])
    store.insert([1, 2], ["stable"], last_prompt=[1, 2, 3, 4])           # a turn's stable prefix and its history
    store.insert([1, 2, 3], ["history"], last_prompt=[1, 2, 3, 4])
    store.insert([9], ["c"], last_prompt=[9, 9])
    assert _order(store) == ["c", "history", "stable"]                   # the next turn may diverge before 3


def test_a_finished_reply_matches_at_its_own_position_under_a_plan():
    store = CheckpointStore(4, copier=lambda c: list(c))
    starts = {0, 8}
    usable = lambda n: n in starts                      # the prompt's plan points
    prompt = [1] * 10
    reply = [2] * 6
    store.insert(prompt, ["prompt-end"], last_prompt=prompt)                    # a plan-start checkpoint
    store.insert(prompt + reply, ["reply-end"], last_prompt=prompt, any_position=True)
    nxt = prompt + reply + [3] * 12                       # the next turn's prompt, tool message and all
    entry = store.peek(nxt, usable=usable)
    assert entry is not None and len(entry.tokens) == 16  # the reply end, not a plan start
    store2 = CheckpointStore(4, copier=lambda c: list(c))
    store2.insert(prompt + reply, ["reply-end"], last_prompt=prompt)            # unmarked: the old behaviour
    assert store2.peek(nxt, usable=usable) is None


def test_a_finished_reply_never_extends_a_diverging_prompt():
    store = CheckpointStore(4, copier=lambda c: list(c))
    store.insert([1] * 16, ["reply-end"], last_prompt=[1] * 10, any_position=True)
    assert store.peek([1] * 10 + [9] * 10) is None        # strict prefix still rules
