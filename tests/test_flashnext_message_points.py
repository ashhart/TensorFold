"""Two-rank prompt pieces use the leader's message markers even when follower assets differ."""

from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine


@pytest.mark.parametrize("draft", [False, True])
def test_leader_shares_exact_message_points(draft):
    leader = object.__new__(FlashNextEngine)
    records = {}
    leader.comm = SimpleNamespace(store=SimpleNamespace(set=records.__setitem__))
    leader.served = 0
    leader.points = lambda prompt: [300, 700]
    request = leader._share([1] * 800, 12, None, draft, 0)
    assert request[-1] == ([300, 700] if draft else [])
    assert FlashNextEngine._unpack(next(iter(records.values()))) == request


def test_follower_decodes_the_shared_points_without_its_own_tokenizer():
    follower = object.__new__(FlashNextEngine)
    follower.served, follower.cache = 0, []
    follower.points = lambda prompt: pytest.fail("follower replanned prompt boundaries")
    requests = iter([([1] * 800, 12, None, True, 0, [], True, [300, 700]), None])
    follower._receive = lambda: next(requests)
    seen = []
    follower._decode = lambda *args, **kwargs: seen.append(kwargs["points"])
    follower.follow()
    assert seen == [[300, 700]] and follower.served == 1
