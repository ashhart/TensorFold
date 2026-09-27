"""The thinking budget: the engine closes the think block after ``think_budget`` reply tokens, the same way in
drafted rounds (any draft depth, right or wrong drafts) as in one-token rounds."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402

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
    """A family model whose logits pick ``after(token)``, with a draft head whose drafts are wrong every
    ``wrong_every``-th time (0: never)."""

    lane_family = True
    exact_width = 16
    gpu_tokens = False

    def __init__(self, drafts: int, wrong_every: int = 0) -> None:
        self.mtp = object()
        self.drafts = drafts
        self.wrong_every = wrong_every
        self.calls = 0
        self.last: list[int] = []
        self._spec: list[int] | None = None

    def make_cache(self) -> list[_Cache]:
        return [_Cache()]

    def hidden(self, inputs, cache):
        tokens = [int(t) for t in np.array(inputs).reshape(-1)]
        cache[0].fed.extend(tokens)
        self.last = tokens
        return mx.array(tokens, dtype=mx.float32).reshape(1, -1, 1)          # [1, R, D]: a row's token

    def head(self, hidden):
        tokens = np.array(hidden).reshape(-1).astype(np.int64)
        logits = np.zeros((1, len(tokens), V), dtype=np.float32)
        for i, t in enumerate(tokens):
            logits[0, i, after(t)] = 10.0
        return mx.array(logits)

    def keep_rows(self, cache, rows: int, keep: int) -> None:
        del cache[0].fed[len(cache[0].fed) - (rows - keep):]

    def absorb_draft_context(self, hidden, next_tokens, cache, start=0) -> None:
        pass

    def _guess(self, token: int) -> int:
        self.calls += 1
        guess = after(token)
        return (guess + 1) % V if self.wrong_every and self.calls % self.wrong_every == 0 else guess

    def speculate(self, cache, tokens, position, sampling, start=0, last_only=False):
        # row start + i is followed by tokens[i]: its first draft is the token after that one
        follow = [int(t) for t in np.array(tokens).reshape(-1)]
        self._spec = follow
        follow = follow[-1:] if last_only else follow
        return mx.array([self._guess(t) for t in follow], dtype=mx.uint32)

    def settle(self, cache, keep, first, position, sampling, count):
        self._spec = None
        if count <= 0:
            return []
        out = [int(first.item()) if isinstance(first, mx.array) else int(first)]
        while len(out) < count:
            out.append(self._guess(out[-1]))
        return out

    def unspeculate(self, cache):
        self._spec = None


def run(model: ChainModel, budget: int, max_new: int = 30, early: bool = True) -> tuple[list[int], list[int]]:
    model.speculate_early = early
    engine = LaneEngine(model)
    assert engine.family and engine.family_mtp and engine.speculate_early == early
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


@pytest.mark.parametrize("early", [True, False])
@pytest.mark.parametrize("drafts,wrong_every", [(1, 0), (3, 0), (3, 2), (2, 3), (1, 1)])
@pytest.mark.parametrize("budget", [2, 7, 12])
def test_budget_closes_thinking_at_the_same_place_with_any_drafts(drafts, wrong_every, budget, early):
    emitted, fed = run(ChainModel(drafts, wrong_every), budget, early=early)
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

    def __init__(self) -> None:
        super().__init__(drafts=0)
        self.mtp = None

    def speculate(self, *args, **kwargs):
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
    engine = LaneEngine(PipelinedChain())
    assert engine.family and engine.pipelined and not engine.family_mtp
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
    engine = FakeEngine(max_rows=16, max_draft=6)
    stream = LaneStream(stream_id="l", prompt_ids=list(prompt), max_new_tokens=40, think_budget=budget,
                        think_close=close, think_end=LANE_END, think_open=True, proposer=PatternProposer(pattern))
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    assert stream.emitted == lane_expected(prompt, 40, budget, close)


# -- a model that takes GPU tokens, with a draft head: every round verifies the head's drafts ------------------
class PipelinedMTPChain(ChainModel):
    gpu_tokens = True

    def __init__(self, wrong_every: int = 0) -> None:
        super().__init__(drafts=3, wrong_every=wrong_every)

    def settle(self, cache, keep, first, position, sampling, count):
        # drafts as an unread GPU array, as Nemotron's head returns them
        out = super().settle(cache, keep, first, position, sampling, count)
        return mx.array(out, dtype=mx.uint32) if out else out


@pytest.mark.parametrize("early", [True, False])
@pytest.mark.parametrize("every,wrong", [(0, 0), (0, 2), (3, 3)])
@pytest.mark.parametrize("budget", [2, 9, 13])
def test_mtp_rounds_force_the_close_at_the_budget(every, wrong, budget, early):
    model = PipelinedMTPChain(wrong_every=wrong)
    model.speculate_early = early
    engine = LaneEngine(model)
    assert engine.family and engine.family_mtp
    stream = LaneStream(stream_id="m", prompt_ids=[3, 14, 15], max_new_tokens=30, think_budget=budget,
                        think_close=(NL, END, NLNL), think_end=END, think_open=True, proposer=CopyAhead(every))
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    assert stream.emitted == expected(budget)
