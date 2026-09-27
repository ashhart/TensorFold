"""Shared forwards require exact windows and honor stream-check results when exposed."""

from types import SimpleNamespace

import pytest

from tensorfold.engine.lane_engine import LaneEngine, LaneStream


@pytest.mark.parametrize("width", [1, 2, 8])
@pytest.mark.parametrize("checked", ["absent", None, False, True])
def test_shared_rounds_require_exact_windows_and_honor_exposed_checks(width, checked):
    model = SimpleNamespace(lane_family=True, exact_width=width, hidden_rows=lambda *args: None, max_streams=32)
    if checked != "absent":
        model.streams_exact = checked
    engine = LaneEngine(model)
    enabled = width >= 2 and (checked == "absent" or checked is True)
    assert engine.family_streams is enabled
    assert engine.batch_streams == (32 if enabled else 1)


def test_an_engine_row_limit_also_disables_shared_rounds():
    model = SimpleNamespace(lane_family=True, exact_width=8, streams_exact=True, hidden_rows=lambda *args: None)
    assert not LaneEngine(model, max_rows=1).family_streams


@pytest.mark.parametrize("checked", [True, False, "absent"])
def test_stream_check_controls_rounds_without_changing_outputs(checked):
    model = SimpleNamespace(lane_family=True, exact_width=8, hidden_rows=lambda *args: None)
    if checked != "absent":
        model.streams_exact = checked
    engine = LaneEngine(model)
    streams = [LaneStream(str(i), [i + 1], 3, pending=[i + 1]) for i in range(2)]
    engine._live = [(stream, []) for stream in streams]
    engine.streams = list(streams)
    seen, shared_calls = [], []

    def advance(stream):
        token = stream.pending[-1] * 2
        stream.pending = [token]
        return stream.commit([token]), 1, 1

    def solo(stream, cache):
        seen.append(stream.stream_id)
        return advance(stream)

    def shared(entries):
        shared_calls.append([s.stream_id for s, _ in entries])
        return {s.stream_id: advance(s) for s, _ in entries}

    engine._family_round, engine._family_round_streams = solo, shared
    while engine.active_count:
        engine.step()
    assert [s.emitted for s in streams] == [[2, 4, 8], [4, 8, 16]]
    if checked is False:
        assert seen.count("0") == seen.count("1") == 3
        assert not shared_calls
        assert len(engine.round_stats) == 6 and all(stat.streams == 1 for stat in engine.round_stats)
    else:
        assert not seen
        assert shared_calls == [["0", "1"]] * 3


def test_flash_without_fused_decode_serves_streams_separately(monkeypatch):
    from tests.test_qwen4_exp_family import tiny
    from tensorfold.families.qwen4_exp.runtime import FlashNext

    monkeypatch.setenv("TF_FLASH_FUSED", "0")
    model = FlashNext(tiny(), None, drafts=0)
    assert model.fused is None and model.exact_width == 1
    # the tiny model's DeltaNet heads are below mlx_lm's Metal kernel minimum, so only the gate is checked here
    engine = LaneEngine(model)
    assert engine.family_streams is False and engine.batch_streams == 1
