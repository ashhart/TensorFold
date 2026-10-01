"""Flash Next's concurrent decoder fills prompts as the Mac scheduler does: fewest rows left first, foreground before
background, and a prompt passed over FILL_GUARD passes takes the next one (no starvation)."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.torch


def _decoder(prompts):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    m = object.__new__(MultiDecoder)
    m.filling = [SimpleNamespace(sid=i, prompt=[0] * n, background=bg) for i, (n, bg) in enumerate(prompts)]
    m.fills = {s.sid: [None, False, done, None] for s, (_, _, done) in zip(m.filling, [(*p, 0) for p in prompts])}
    m.passed = {}
    return m


def test_fewest_rows_left_first_then_arrival():
    m = _decoder([(120_000, False), (2_000, False), (2_000, False), (30_000, False)])
    assert [s.sid for s in m._order()] == [1, 2, 3, 0]


def test_background_prompts_after_foreground_ones():
    m = _decoder([(2_000, True), (50_000, False)])
    assert [s.sid for s in m._order()] == [1, 0]


def test_a_prompt_passed_over_fill_guard_passes_goes_first():
    from tensorfold.families.qwen4_exp.cuda.multi import FILL_GUARD

    m = _decoder([(120_000, False), (2_000, False)])
    short = m.filling[1]
    for _ in range(FILL_GUARD):
        m._note_passed([(short, 0, 2_000)])
    assert m.passed == {0: FILL_GUARD, 1: 0}
    assert [s.sid for s in m._order()][0] == 0                     # the long one is due
    m._note_passed([(m.filling[0], 0, 2_048)])
    assert m.passed == {0: 0, 1: 1}


def test_counts_drop_with_prompts_that_left():
    m = _decoder([(4_000, False), (2_000, False)])
    m._note_passed([(m.filling[1], 0, 2_000)])
    m.filling = m.filling[:1]
    m._note_passed([])
    assert m.passed == {0: 2}


def test_a_waiting_request_stops_a_lone_prompts_passes():
    """With no stream decoding, passes stop for a request that waits to be admitted (it then fills beside them)."""
    m = _decoder([(32_000, False)])
    m.streams, passes = {}, []

    def one_pass():
        passes.append(1)
        if len(passes) == 16:
            m.filling = []                     # the prompt ended
        return []

    m._pass = one_pass
    m.arrived = lambda: len(passes) >= 3
    m._fill()
    assert len(passes) == 3
    m.arrived = lambda: False
    m._fill()
    assert len(passes) == 16
