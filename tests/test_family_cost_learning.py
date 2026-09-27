"""First decode initialization is timed, but does not estimate steady draft costs."""

from types import SimpleNamespace

import pytest

from tensorfold.engine.lane_family import FamilyRounds


def policy():
    engine = FamilyRounds()
    # Real 64k calibration and acceptance at the first affected depth decision.
    engine.family_costs = {1: 10.994, 2: 14.116, 3: 17.223, 4: 21.167}
    engine.mtp_step_ms = 0.891
    engine.most_drafts = 3
    engine._round_ms = {}
    engine._depth_state = {"s": {"p": [0.891625, 0.691875, 0.7], "rounds": 2}}
    return engine


def test_measured_64k_initialization_does_not_poison_the_next_depth():
    engine = policy()
    engine._observe_cost(2, 725.5441249581054, initializing=True)
    assert engine._round_ms == {}
    assert engine._depth(SimpleNamespace(stream_id="s", draft_room=128)) == 2
    engine._observe_cost(2, 22.990375058725476)
    assert engine._round_ms[2] == pytest.approx(22.990375058725476)


def test_later_slow_rounds_still_train_the_steady_estimator():
    engine = policy()
    engine._observe_cost(2, 22.990375058725476)
    engine._observe_cost(2, 725.5441249581054)
    engine._observe_cost(2, 725.5441249581054)
    assert engine._round_ms[2] == pytest.approx(275.90972502250224)
    assert engine._depth(SimpleNamespace(stream_id="s", draft_room=128)) == 1


@pytest.fixture
def timed_engine(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from tensorfold.engine import lane_family
    from tensorfold.engine.lane_engine import LaneEngine, LaneStream
    from test_thinking_budget import ChainModel, after

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(lane_family.time, "perf_counter", lambda: clock.now)

    class TimedChain(ChainModel):
        window_costs = {1: 10.994, 2: 14.116, 3: 17.223, 4: 21.167}
        mtp_step_ms = 0.891
        elapsed_ms = 0.0

        def hidden(self, inputs, cache):
            clock.now += self.elapsed_ms / 1000.0
            return super().hidden(inputs, cache)

    model = TimedChain(drafts=3)
    engine = LaneEngine(model)

    def start(stream_id):
        model.elapsed_ms = 0.0
        stream = LaneStream(stream_id=stream_id, prompt_ids=[3, 14, 15], max_new_tokens=128)
        engine.add_stream(stream)
        assert stream.rounds == 0  # Prefill commits the first token, but is not a decode round.
        return stream

    def round_on(stream, depth, ms, *, copied=None, forced=None):
        token = stream.pending[-1]
        drafts = []
        for _ in range(depth):
            token = after(token)
            drafts.append(token)
        engine._next[stream.stream_id] = drafts
        stream.force = [] if forced is None else list(forced)
        model.elapsed_ms = ms
        if copied is None:
            engine.step()
        else:
            engine._family_round(stream, engine._live[0][1], copied=copied)

    yield engine, start, round_on, after
    mx.set_default_device(previous)


def test_first_round_is_retained_in_actual_stats_then_steady_sample_trains(timed_engine):
    engine, start, round_on, _ = timed_engine
    stream = start("s")
    round_on(stream, 2, 725.5441249581054)
    assert stream.rounds == 1
    assert engine.round_stats[-1].total_ms == pytest.approx(725.5441249581054)
    assert engine._round_ms == {}
    round_on(stream, 2, 22.990375058725476)
    assert stream.rounds == 2
    assert engine._round_ms[2] == pytest.approx(22.990375058725476)
    assert sum(r.total_ms for r in engine.round_stats) == pytest.approx(748.5345000168309)


def test_new_stream_keeps_steady_history_without_inheriting_its_cold_sample(timed_engine):
    engine, start, round_on, _ = timed_engine
    first = start("first")
    round_on(first, 2, 725.5441249581054)
    round_on(first, 2, 22.990375058725476)
    # The scheduler retires the first request before adding the next one.
    engine._live = []
    second = start("second")
    round_on(second, 1, 766.0605839919299)
    assert engine._round_ms == {2: pytest.approx(22.990375058725476)}
    assert engine.round_stats[-1].total_ms == pytest.approx(766.0605839919299)
    round_on(second, 1, 19.17724998202175)
    assert engine._round_ms[1] == pytest.approx(19.17724998202175)


@pytest.mark.parametrize("kind", ["copy", "forced"])
def test_first_nonhead_round_does_not_skip_the_later_genuine_head_sample(timed_engine, kind):
    engine, start, round_on, after = timed_engine
    stream = start("s")
    proposal = [after(stream.pending[-1]), after(after(stream.pending[-1]))]
    round_on(stream, 2, 700.0, copied=proposal if kind == "copy" else None,
             forced=proposal if kind == "forced" else None)
    assert stream.rounds == 1
    assert engine._round_ms == {}
    round_on(stream, 2, 22.990375058725476)
    assert stream.rounds == 2
    assert engine._round_ms[2] == pytest.approx(22.990375058725476)
