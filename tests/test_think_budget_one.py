"""thinking_budget 1: the first reply token is the budget's cut, but the engine feeds the sampled token."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from test_family_streams import END, NL, NLNL, StreamsModel, after  # noqa: E402

from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402

PROMPT, MAX_NEW = [8, 2], 12


def _run(drafts: bool, budget: int, head: bool = True, gpu: bool = False) -> list[int]:
    engine = LaneEngine(StreamsModel(head=head, gpu_tokens=gpu))
    stream = LaneStream(stream_id="s", prompt_ids=list(PROMPT), max_new_tokens=MAX_NEW, think_budget=budget,
                        think_close=(NL, END, NLNL), think_end=END, think_open=budget > 0, drafts=drafts)
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    return stream.emitted


def _meant(budget: int) -> list[int]:
    """The budget-th reply token is the close; the model then continues after the close."""

    out, last = [], PROMPT[-1]
    while len(out) < MAX_NEW:
        if budget and len(out) + 1 == budget:
            out.extend([NL, END, NLNL])
            last = NLNL
            continue
        last = after(last)
        out.append(last)
    return out[:MAX_NEW]


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_drafted_equals_serial(budget):
    assert _run(True, budget) == _run(False, budget)


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_serial_continues_after_the_close(budget):
    assert _run(False, budget) == _meant(budget)


@pytest.mark.parametrize("gpu,head", [(False, True), (True, True), (True, False), (False, False)])
@pytest.mark.parametrize("budget", [1, 2, 3, 7])
def test_close_continuation_matches_in_every_mode(budget, gpu, head):
    assert _run(True, budget, head, gpu) == _run(False, budget, head, gpu) == _meant(budget)
