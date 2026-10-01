"""A stream's context is its prompt then its committed tokens, kept as one list that commits grow (no MLX)."""

from __future__ import annotations

from tensorfold.engine.lane_engine import LaneStream


def _stream(prompt, max_new=64, eos=()):
    return LaneStream(stream_id="s", prompt_ids=list(prompt), max_new_tokens=max_new, eos_ids=frozenset(eos))


def test_context_is_prompt_then_commits_and_reused():
    s = _stream(range(10))
    first = s.context
    assert first == list(range(10)) and s.context_len == 10
    s.commit([100, 101])
    assert s.context == [*range(10), 100, 101] and s.context_len == 12
    assert s.context is first                     # grown in place, not rebuilt
    s.commit([102])
    assert s.context[-3:] == [100, 101, 102] and s.context is first


def test_context_follows_a_reassigned_reply_or_prompt():
    s = _stream([1, 2, 3])
    s.commit([7, 8])
    s.emitted = []                                # a prefill restart forgets the reply
    assert s.context == [1, 2, 3] and s.context_len == 3
    s.commit([9])
    assert s.context == [1, 2, 3, 9]
    s.prompt_ids.append(4)                        # a prompt grown in place
    assert s.context == [1, 2, 3, 4, 9]
    s.prompt_ids = [5]
    assert s.context == [5, 9] and s.context_len == 2


def test_context_matches_a_fresh_list_through_a_reply():
    s = _stream(range(1000), max_new=300, eos=(-1,))
    for step in range(100):
        s.commit([2000 + step, 3000 + step][: 1 + step % 2])
        assert s.context == [*s.prompt_ids, *s.emitted]
        assert s.context_len == len(s.prompt_ids) + len(s.emitted)
