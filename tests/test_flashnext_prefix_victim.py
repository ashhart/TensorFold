"""A fork with no free slot goes into the idle slot that loses the fewest kept tokens, not over its source's chain."""

import pytest

from tensorfold.families.qwen4_exp.cuda import prefixes

SYSTEM = list(range(300))


class Slot:
    def __init__(self, name):
        self.name, self.copied = name, None

    def copy_prefix(self, source, pos, mtp_len):
        self.copied = (source, pos, mtp_len)


class Owner:
    def __init__(self, kept, free=(), busy=(), room=True):
        self.kept, self.free, self.busy, self.room = kept, list(free), set(busy), room
        self.depth, self.grown, self.shrunk = 3, [], []

    def _busy(self):
        return {id(st) for st in self.busy}

    def _drop_kept(self, st):
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st, rows, **kwargs):
        self.grown.append((st, rows, kwargs))
        return self.room

    def _shrink(self, st, **kwargs):
        self.shrunk.append((st, kwargs))


def entry(ids, slot, tag):
    return (ids, slot, {"pos": len(ids), "mtp_len": len(ids) - 1, "tag": tag}, tag)


def two_conversations(other_len=200, busy_source=False, room=True, free=()):
    """Slot a holds the shared system prompt and a long chain on it; slot c holds an unrelated chain."""

    a, c = Slot("a"), Slot("c")
    chain = SYSTEM + list(range(1000, 1500))
    unrelated = list(range(5000, 5000 + other_len))
    kept = [entry(SYSTEM, a, "system"), entry(chain, a, "a-chain"), entry(unrelated, c, "other")]
    return Owner(kept, free=free, busy=[a] if busy_source else [], room=room), a, c, chain


def test_a_fork_claims_the_idle_slot_that_loses_less_than_its_source_chain():
    owner, a, c, chain = two_conversations()
    st, resume, cached = prefixes.slot_for(owner, SYSTEM + [7, 8, 9], True)
    assert st is c and cached == len(SYSTEM) and resume == {"state": owner.kept[0][2], "tail": "system"}
    assert c.copied == (a, len(SYSTEM), len(SYSTEM) - 1)
    assert [k[3] for k in owner.kept] == ["system", "a-chain"] and all(k[1] is a for k in owner.kept)
    assert owner.grown[0][2] == {"protect": a} and owner.free == []


def test_the_idle_slot_with_the_fewest_kept_tokens_is_the_victim():
    owner, a, c, _ = two_conversations(other_len=200)
    d = Slot("d")
    owner.kept.append(entry(list(range(7000, 7050)), d, "short"))
    st, _, cached = prefixes.slot_for(owner, SYSTEM + [7], True)
    assert st is d and cached == len(SYSTEM)
    assert [k[3] for k in owner.kept] == ["system", "a-chain", "other"]


def test_resuming_in_place_stays_when_the_idle_slot_costs_more():
    owner, a, c, _ = two_conversations(other_len=900)           # a's chain beyond the system prompt is 500 tokens
    st, _, cached = prefixes.slot_for(owner, SYSTEM + [7], True)
    assert st is a and cached == len(SYSTEM) and c.copied is None
    assert [k[3] for k in owner.kept] == ["system", "other"]    # a's longer chain is cut, as before


def test_a_free_slot_is_still_preferred_to_an_idle_victim():
    spare = Slot("spare")
    owner, a, c, _ = two_conversations(free=[spare])
    st, _, cached = prefixes.slot_for(owner, SYSTEM + [7], True)
    assert st is spare and cached == len(SYSTEM) and any(k[1] is c for k in owner.kept)


def test_a_busy_source_forks_into_the_idle_slot_instead_of_dropping_it_for_a_cold_start():
    owner, a, c, _ = two_conversations(busy_source=True, other_len=900)
    st, _, cached = prefixes.slot_for(owner, SYSTEM + [7], True)
    assert st is c and cached == len(SYSTEM) and c.copied[0] is a
    assert all(k[1] is a for k in owner.kept)


def test_a_refused_growth_leaves_the_victims_chains_kept_and_the_slot_unfree():
    owner, a, c, _ = two_conversations(room=False)
    before = [k[3] for k in owner.kept]
    st, _, cached = prefixes.slot_for(owner, SYSTEM + [7], True)
    assert st is a and cached == len(SYSTEM)                    # the old in-place resume, after the refusal
    assert c.copied is None and owner.free == []
    assert [k[3] for k in owner.kept if k[1] is c] == ["other"] and before.count("other") == 1


def test_a_failed_copy_returns_the_claimed_slot_and_raises():
    owner, a, c, _ = two_conversations()
    def fail(*args):
        raise RuntimeError("copy failed")
    c.copy_prefix = fail
    with pytest.raises(RuntimeError, match="copy failed"):
        prefixes.slot_for(owner, SYSTEM + [7], True)
    assert owner.free == [c] and owner.shrunk == [(c, {"force": True})]


def test_without_an_idle_slot_to_claim_nothing_changes():
    a = Slot("a")
    owner = Owner([entry(SYSTEM, a, "system"), entry(SYSTEM + list(range(1000, 1500)), a, "a-chain")])
    st, _, cached = prefixes.slot_for(owner, SYSTEM + [7], True)
    assert st is a and cached == len(SYSTEM)
    assert [k[3] for k in owner.kept] == ["system"]
