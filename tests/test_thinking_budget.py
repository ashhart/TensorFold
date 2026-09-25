"""The thinking budget: the engine closes the think block after ``think_budget`` reply tokens, the same way in
drafted rounds (any draft depth, right or wrong drafts) as in one-token rounds."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine.family_engine import SerialEngine  # noqa: E402
from tensorfold.engine.lane_engine import LaneStream  # noqa: E402

V = 97
NL, END, NLNL = 91, 90, 92          # "\n", "</think>", "\n\n"


def after(token: int) -> int:
    """The chain every forward follows; it never writes END by itself."""

    value = (5 * int(token) + 3) % V
    return value if value not in (NL, END, NLNL) else (value + 4) % V


class _Cache:
    def __init__(self) -> None:
        self.fed: list[int] = []
        self.state = None


class ChainModel:
    """A synchronously drafting model whose logits pick ``after(token)``; its drafts are wrong every
    ``wrong_every``-th time (0: never)."""

    multi_row_exact = True
    gpu_tokens = False
    gpu_sampling = False

    def __init__(self, drafts: int, wrong_every: int = 0) -> None:
        self.mtp = object()
        self.drafts = drafts
        self.wrong_every = wrong_every
        self.calls = 0
        self.last_streams = mx.zeros((0,))

    def make_cache(self) -> list[_Cache]:
        return [_Cache()]

    def hidden(self, inputs, cache):
        tokens = [int(t) for t in np.array(inputs).reshape(-1)]
        cache[0].fed.extend(tokens)
        self.last_streams = mx.array(tokens)
        return mx.array(tokens, dtype=mx.float32).reshape(1, -1, 1)          # [1, R, D]: a row's token

    def head(self, hidden):
        tokens = np.array(hidden).reshape(-1).astype(np.int64)
        logits = np.zeros((1, len(tokens), V), dtype=np.float32)
        for i, t in enumerate(tokens):
            logits[0, i, after(t)] = 10.0
        return mx.array(logits)

    def keep_rows(self, cache, rows: int, keep: int) -> None:
        del cache[0].fed[len(cache[0].fed) - (rows - keep):]

    def absorb_draft_context(self, hidden, next_tokens, cache) -> None:
        pass

    def draft(self, cache, streams, tokens, position, sampling, count=None):
        out, t = [], int(tokens[-1])
        for _ in range(self.drafts if count is None else int(count)):
            t = after(t)
            self.calls += 1
            out.append((t + 1) % V if self.wrong_every and self.calls % self.wrong_every == 0 else t)
        return out


def run(model: ChainModel, budget: int, max_new: int = 30) -> tuple[list[int], list[int]]:
    engine = SerialEngine(model)
    assert engine.sync_drafts
    stream = LaneStream(stream_id="s", prompt_ids=[3, 14, 15], max_new_tokens=max_new, think_budget=budget,
                        think_close=(NL, END, NLNL), think_end=END, think_open=budget > 0)
    engine.add_stream(stream)
    cache = engine._live[0][1]
    while engine.active_count:
        engine.step()
    return stream.emitted, cache[0].fed


def expected(budget: int, max_new: int = 30) -> list[int]:
    out = [after(15)]
    while len(out) < max_new:
        if len(out) + 1 == budget:
            out.extend([NL, END, NLNL])
        else:
            out.append(after(out[-1]))
    return out[:max_new]


@pytest.mark.parametrize("drafts,wrong_every", [(1, 0), (3, 0), (3, 2), (2, 3), (1, 1)])
@pytest.mark.parametrize("budget", [2, 7, 12])
def test_budget_closes_thinking_at_the_same_place_with_any_drafts(drafts, wrong_every, budget):
    emitted, fed = run(ChainModel(drafts, wrong_every), budget)
    assert emitted == expected(budget)
    # the cache read the prompt and every emitted token but the last (the pending one); a final round may have
    # read rows past the length limit
    assert fed[:2 + len(emitted)] == [3, 14, 15, *emitted[:-1]]


def test_no_budget_and_a_natural_close_are_left_alone():
    emitted, _ = run(ChainModel(3), 0)
    assert emitted == expected(10**6)
    stream = LaneStream(stream_id="t", prompt_ids=[1], max_new_tokens=9, think_budget=4, think_close=(NL, END, NLNL),
                        think_end=END, think_open=True)
    assert stream.think_cut([7, END, 8, 9]) is None          # closed by the model before the budget
    assert stream.think_cut([7, 8, 9, 10]) == 3
    stream.commit([7, END])
    assert not stream.think_open and stream.think_cut([1, 2, 3, 4]) is None


# -- the pipelined serial engine (a model that takes its tokens as GPU arrays, with copy windows) -------------
class PipelinedChain(ChainModel):
    gpu_tokens = True
    multi_row_exact = True

    def __init__(self) -> None:
        super().__init__(drafts=0)
        self.mtp = None

    def draft(self, *args, **kwargs):
        raise AssertionError("no MTP head here")


class CopyAhead:
    """Proposes the chain's true continuation (as a suffix copy would), every ``every``-th round a wrong one."""

    def __init__(self, every: int = 0) -> None:
        self.every = every
        self.calls = 0
        self.last_match = 99

    def propose(self, context, max_draft):
        self.calls += 1
        out, t = [], int(context[-1])
        for _ in range(max_draft):
            t = after(t)
            out.append(t)
        if self.every and self.calls % self.every == 0:
            out[len(out) // 2] = (out[len(out) // 2] + 1) % V
        return out

    def observe(self, proposed, accepted):
        pass


@pytest.mark.parametrize("every", [0, 2, 3])
@pytest.mark.parametrize("budget", [2, 9, 13])
def test_pipelined_engine_forces_the_close_at_the_budget(every, budget):
    engine = SerialEngine(PipelinedChain())
    assert engine.pipelined and engine.windows and not engine.sync_drafts
    stream = LaneStream(stream_id="p", prompt_ids=[3, 14, 15], max_new_tokens=30, think_budget=budget,
                        think_close=(NL, END, NLNL), think_end=END, think_open=True, proposer=CopyAhead(every))
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    assert stream.emitted == expected(budget)


# -- the lane engine: the close replaces a round's own sample, one token a round ------------------------------
from lane_fakes import FakeEngine, PatternProposer, fake_next  # noqa: E402

LANE_END = 90


def lane_expected(prompt, max_new, budget, close):
    history, out, open_, forcing = list(prompt), [], True, []
    while len(out) < max_new:
        if forcing:
            token = forcing.pop(0)
        elif open_ and len(out) + 1 >= budget:
            open_, forcing, token = False, list(close[1:]), close[0]
        else:
            token = fake_next(history)
            if token == LANE_END:
                open_ = False
        out.append(token)
        history.append(token)
    return out


@pytest.mark.parametrize("pattern", [[0], [3, 1], [6, 2, 0, 5]])
@pytest.mark.parametrize("budget", [2, 8, 17])
def test_lane_engine_forces_the_close_at_the_budget(pattern, budget):
    close = (91, LANE_END, 92)
    prompt = [5, 11, 23, 42]
    engine = FakeEngine(max_rows=16, max_draft=6, pending_cap=8)
    stream = LaneStream(stream_id="l", prompt_ids=list(prompt), max_new_tokens=40, think_budget=budget,
                        think_close=close, think_end=LANE_END, think_open=True, proposer=PatternProposer(pattern))
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    assert stream.emitted == lane_expected(prompt, 40, budget, close)
