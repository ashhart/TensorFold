"""Several streams in the lane engine's shared family rounds: each stream emits exactly what it emits alone, whatever
the other streams draft, keep, cut or finish (fake models; the real kernels are checked on the model)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine.lane_engine import LaneEngine, LaneStream  # noqa: E402

V = 97
NL, END, NLNL = 91, 90, 92


def after(token: int) -> int:
    value = (5 * int(token) + 3) % V
    return value if value not in (NL, END, NLNL) else (value + 4) % V


def chain(last: int, budget: int, max_new: int) -> list[int]:
    out = [after(last)]
    while len(out) < max_new:
        if len(out) + 1 == budget:
            out.extend([NL, END, NLNL])
        else:
            out.append(after(out[-1]))
    return out[:max_new]


class _Cache:
    def __init__(self) -> None:
        self.fed: list[int] = []
        self.state = None


class StreamsModel:
    """A family model whose logits pick ``after(token)``; its head drafts wrong every ``wrong_every``-th time, so
    streams see different drafts alone and together."""

    lane_family = True
    streams_exact = True
    exact_width = 16
    speculate_early = False

    def __init__(self, drafts: int = 3, wrong_every: int = 0, gpu_tokens: bool = False, head: bool = True) -> None:
        self.mtp = object() if head else None
        self.drafts = drafts
        self.wrong_every = wrong_every
        self.gpu_tokens = gpu_tokens
        self.calls = 0
        self.rows_calls = 0

    def make_cache(self):
        return [_Cache()]

    def _feed(self, tokens, cache):
        cache[0].fed.extend(tokens)
        return tokens

    def hidden(self, inputs, cache, parents=None):
        tokens = self._feed([int(t) for t in np.array(inputs).reshape(-1)], cache)
        self.last = tokens
        return mx.array(tokens, dtype=mx.float32).reshape(1, -1, 1)

    def hidden_rows(self, windows, caches, parents=None):
        self.rows_calls += 1
        tokens = []
        for window, cache in zip(windows, caches):
            tokens += self._feed([int(t) for t in np.array(window).reshape(-1)], cache)
        self.last = tokens
        return mx.array(tokens, dtype=mx.float32).reshape(1, -1, 1)

    def head(self, hidden):
        tokens = np.array(hidden).reshape(-1).astype(np.int64)
        logits = np.zeros((1, len(tokens), V), dtype=np.float32)
        for i, t in enumerate(tokens):
            logits[0, i, after(t)] = 10.0
        return mx.array(logits)

    def keep_rows(self, cache, rows, keep):
        if isinstance(keep, int):
            del cache[0].fed[len(cache[0].fed) - (rows - keep):]
            return
        window = cache[0].fed[len(cache[0].fed) - rows:]           # a tree: keep the accepted path's rows
        del cache[0].fed[len(cache[0].fed) - rows:]
        cache[0].fed.extend(window[r] for r in keep)

    def keep_rows_streams(self, caches, lengths, keeps):
        for cache, n, keep in zip(caches, lengths, keeps):
            self.keep_rows(cache, n, keep)

    def absorb_draft_context(self, hidden, next_tokens, cache, start=0):
        pass

    def _guess(self, token):
        self.calls += 1
        guess = after(token)
        return (guess + 1) % V if self.wrong_every and self.calls % self.wrong_every == 0 else guess

    def speculate(self, cache, tokens, position, sampling, start=0, last_only=False, rows=None):
        follow = [int(t) for t in np.array(tokens).reshape(-1)]
        follow = follow[-1:] if last_only else follow
        return mx.array([self._guess(t) for t in follow], dtype=mx.uint32)

    def settle(self, cache, keep, first, position, sampling, count):
        if count <= 0:
            return []
        out = [int(first.item()) if isinstance(first, mx.array) else int(first)]
        while len(out) < count:
            out.append(self._guess(out[-1]))
        return mx.array(out, dtype=mx.uint32)

    def unspeculate(self, cache):
        pass


SPECS = [  # prompt, thinking budget, max tokens, drafts on
    ([3, 14, 15], 7, 30, True),
    ([8, 2], 0, 22, True),
    ([40, 41, 42, 43], 12, 35, True),
    ([5], 3, 18, False),
    ([60, 61], 0, 27, True),
]


def _stream(i, spec):
    prompt, budget, max_new, drafts = spec
    return LaneStream(stream_id=f"s{i}", prompt_ids=list(prompt), max_new_tokens=max_new, think_budget=budget,
                      think_close=(NL, END, NLNL), think_end=END, think_open=budget > 0, drafts=drafts)


@pytest.mark.parametrize("wrong_every", [0, 2, 3])
@pytest.mark.parametrize("count", [2, 3, 5])
def test_streams_together_emit_what_they_emit_alone(wrong_every, count):
    model = StreamsModel(wrong_every=wrong_every)
    engine = LaneEngine(model)
    assert engine.family and engine.family_streams
    streams = [_stream(i, spec) for i, spec in enumerate(SPECS[:count])]
    for stream in streams:
        engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    assert model.rows_calls > 0
    for stream, (prompt, budget, max_new, _) in zip(streams, SPECS):
        assert stream.emitted == chain(prompt[-1], budget, max_new)
        if stream.drafts:
            assert stream.min_rows >= 2


def test_the_first_token_is_out_before_the_first_drafts_settle():
    model = StreamsModel(wrong_every=3)
    settled = []
    settle = model.settle
    model.settle = lambda *args: settled.append(args[3]) or settle(*args)
    engine = LaneEngine(model)
    stream = _stream(0, SPECS[1])
    engine.add_stream(stream)
    assert stream.emitted == chain(SPECS[1][0][-1], SPECS[1][1], SPECS[1][2])[:1] and settled == []
    engine.step()
    assert settled[:1] == [len(SPECS[1][0]) + 1]      # settled once, at the first round, for the position after it
    while engine.active_count:
        engine.step()
    assert stream.emitted == chain(SPECS[1][0][-1], SPECS[1][1], SPECS[1][2])


def test_streams_join_and_leave_between_rounds():
    model = StreamsModel(wrong_every=3)
    engine = LaneEngine(model)
    streams = [_stream(i, spec) for i, spec in enumerate(SPECS)]
    engine.add_stream(streams[0])
    for i, stream in enumerate(streams[1:], start=1):
        for _ in range(i):
            engine.step()
        engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    for stream, (prompt, budget, max_new, _) in zip(streams, SPECS):
        assert stream.emitted == chain(prompt[-1], budget, max_new)


def test_a_stream_running_ahead_lands_its_step_before_sharing_rounds():
    model = StreamsModel(gpu_tokens=True, head=False)
    engine = LaneEngine(model)
    assert engine.pipelined and not engine.family_mtp
    first = _stream(0, ([3, 14, 15], 7, 30, True))
    engine.add_stream(first)
    for _ in range(4):
        engine.step()                            # alone: steps queued ahead
    second = _stream(1, ([8, 2], 0, 22, True))
    engine.add_stream(second)
    while engine.active_count:
        engine.step()
    assert first.emitted == chain(15, 7, 30)
    assert second.emitted == chain(2, 0, 22)


def test_the_row_budget_trims_drafts_then_waits_streams():
    model = StreamsModel(drafts=8)
    engine = LaneEngine(model)
    engine.batch_rows = 6
    streams = [_stream(i, spec) for i, spec in enumerate(SPECS)]
    for stream in streams:
        engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    # drafts are cut first; a stream whose pending row and forced close (never cut) do not fit waits a round
    assert all(r.rows <= 6 for r in engine.round_stats if r.streams > 1)
    assert any(r.rows == 6 for r in engine.round_stats if r.streams > 1)
    for stream, (prompt, budget, max_new, _) in zip(streams, SPECS):
        assert stream.emitted == chain(prompt[-1], budget, max_new)


def test_more_streams_than_a_forward_takes_share_rounds_in_turn():
    model = StreamsModel(wrong_every=2)
    model.max_streams, model.batch_rows = 2, 5
    engine = LaneEngine(model)
    assert (engine.batch_streams, engine.batch_rows) == (2, 5)
    streams = [_stream(i, spec) for i, spec in enumerate(SPECS)]
    for stream in streams:
        engine.add_stream(stream)
    shared = []
    while engine.active_count:
        before = {s.stream_id: len(s.emitted) for s in streams}
        engine.step()
        moved = sum(len(s.emitted) > before[s.stream_id] for s in streams)
        if engine.round_stats[-1].streams > 1:
            shared.append((engine.round_stats[-1].streams, moved))
    assert shared and all(n <= 2 and moved <= 2 for n, moved in shared)
    for stream, (prompt, budget, max_new, _) in zip(streams, SPECS):
        assert stream.emitted == chain(prompt[-1], budget, max_new)


def _alone(make, spec, i):
    engine = LaneEngine(make())
    stream = _stream(i, spec)
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    return stream.emitted


@pytest.mark.parametrize("pipelined", [False, True])
def test_joiners_started_together_emit_what_they_emit_alone(pipelined):
    """The server starts every waiting request before the next round, each prefilled on its own; with fewer rows a
    call than streams they take turns."""

    def make():
        model = StreamsModel(wrong_every=2, gpu_tokens=pipelined, head=not pipelined)
        model.batch_rows = 5
        return model

    alone = [_alone(make, spec, i) for i, spec in enumerate(SPECS)]
    engine = LaneEngine(make())
    streams = [_stream(i, spec) for i, spec in enumerate(SPECS)]
    for stream in streams:
        engine.add_stream(stream)
    assert all(len(s.emitted) == 1 for s in streams)
    while engine.active_count:
        engine.step()
    assert [s.emitted for s in streams] == alone


class TreeModel(StreamsModel):
    """Drafts a small tree each round: two candidates for the next token (the chain's and a wrong sibling), and the
    chain continued below the first; with ``wrong_every`` the first candidate is sometimes the wrong one."""

    def settle(self, cache, keep, first, position, sampling, count):
        if count <= 0:
            return []
        a1 = int(first.item()) if isinstance(first, mx.array) else int(first)
        sibling = (after(a1) + 7) % V if a1 == after(0) else (a1 + 11) % V
        tokens, parents = [a1, sibling], [-1, -1]
        t = a1
        for _ in range(max(0, count - 1)):
            t = self._guess(t)
            tokens.append(t)
            parents.append(len(tokens) - 2 if len(tokens) > 3 else 0)
        return (tokens, parents)


@pytest.mark.parametrize("wrong_every", [0, 2])
@pytest.mark.parametrize("count", [1, 3, 5])
def test_draft_trees_keep_the_accepted_path(wrong_every, count):
    model = TreeModel(wrong_every=wrong_every)
    engine = LaneEngine(model)
    streams = [_stream(i, spec) for i, spec in enumerate(SPECS[:count])]
    for stream in streams:
        engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    for stream, (prompt, budget, max_new, _) in zip(streams, SPECS):
        assert stream.emitted == chain(prompt[-1], budget, max_new)


class OwnSamplerModel(StreamsModel):
    """A family with its own sampler (host argmax): every draw of every stream goes through it."""

    draws = 0

    def sample(self, logits, sampling, positions):
        self.draws += 1
        return [int(t) for t in np.argmax(np.array(logits.reshape(-1, logits.shape[-1])), axis=-1)]


class StreamsSamplerModel(OwnSamplerModel):
    """Draws a shared round's streams in one call (``sample_streams``)."""

    batches = 0

    def sample_streams(self, logits, samplings, positions):
        self.batches += 1
        return [self.sample(x, s, p) for x, s, p in zip(logits, samplings, positions)]


def test_a_shared_round_draws_its_streams_in_one_call():
    model = StreamsSamplerModel()
    engine = LaneEngine(model)
    streams = [_stream(i, spec) for i, spec in enumerate(SPECS[:3])]
    for stream in streams:
        engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    assert model.batches == sum(1 for r in engine.round_stats if r.streams > 1) > 0
    for stream, (prompt, budget, max_new, _) in zip(streams, SPECS):
        assert stream.emitted == chain(prompt[-1], budget, max_new)


def test_a_models_own_sampler_draws_every_token():
    model = OwnSamplerModel()
    engine = LaneEngine(model)
    streams = [_stream(i, spec) for i, spec in enumerate(SPECS[:3])]
    for stream in streams:
        engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    assert model.draws >= sum(stream.rounds for stream in streams) // 3
    for stream, (prompt, budget, max_new, _) in zip(streams, SPECS):
        assert stream.emitted == chain(prompt[-1], budget, max_new)


class _TreeCache(_Cache):
    def __init__(self) -> None:
        super().__init__()
        self.pending = None


class DeferredTreeModel(StreamsModel):
    """Commits as the Qwen3.8 family does: a chain window at once, a tree window only in ``keep_rows``. Its head drafts
    a chain of three and a sibling branch that the row allocator trims (a millisecond a row, the branch's chance
    0.001), leaving a chain-shaped tree."""

    window_costs = {w: 10.0 + w for w in range(1, 17)}

    def make_cache(self):
        return [_TreeCache()]

    def hidden(self, inputs, cache, parents=None):
        assert cache[0].pending is None, "a tree window was never committed"
        tokens = [int(t) for t in np.array(inputs).reshape(-1)]
        if parents is None:
            cache[0].fed.extend(tokens)
        else:
            cache[0].pending = tokens
        self.last = tokens
        return mx.array(tokens, dtype=mx.float32).reshape(1, -1, 1)

    def keep_rows(self, cache, rows, keep):
        if cache[0].pending is None:
            return super().keep_rows(cache, rows, keep)
        window, cache[0].pending = cache[0].pending, None
        cache[0].fed.extend(window[r] for r in (range(keep) if isinstance(keep, int) else keep))

    def settle(self, cache, keep, first, position, sampling, count):
        if count <= 0:
            return []
        tokens = [int(first.item()) if isinstance(first, mx.array) else int(first)]
        while len(tokens) < min(count, 3):
            tokens.append(self._guess(tokens[-1]))
        parents = list(range(-1, len(tokens) - 1))
        return (tokens + [(tokens[0] + 11) % V], parents + [-1])

    def draft_probabilities(self, cache):
        return [0.99, 0.98, 0.97, 0.001]


def test_a_tree_trimmed_to_a_chain_and_kept_whole_is_committed():
    """A one-stream round whose tree the allocator trims to a chain, every row accepted: the family still gets
    ``keep_rows`` (its recurrent state is committed there), and the stream emits its chain."""

    model = DeferredTreeModel(drafts=4)
    engine = LaneEngine(model)
    stream = _stream(0, SPECS[1])
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    prompt, budget, max_new, _ = SPECS[1]
    assert stream.emitted == chain(prompt[-1], budget, max_new)
    assert engine.accepted > 0
