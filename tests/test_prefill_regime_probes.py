"""The prefill fit probes sit in the regime long prompts run in (#228)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from tensorfold.engine.memory import measure, probe_premise


class FakeCore:
    """The few mlx.core calls measure() makes; the engine's fake drives peak and active."""

    def __init__(self) -> None:
        self.state = {"active": 0, "peak": 0}

    def eval(self, *arrays) -> None:
        pass

    def zeros(self, shape):
        return ("zeros", shape)

    def get_active_memory(self) -> int:
        return self.state["active"]

    def reset_peak_memory(self) -> None:
        self.state["peak"] = 0

    def get_peak_memory(self) -> int:
        return self.state["peak"]

    def clear_cache(self) -> None:
        pass


class ProbeEngine:
    """Captures the probe lengths measure() fits from; a peak of a byte a token, linear in length."""

    prefill_plan = SimpleNamespace(step=2048)

    def __init__(self, core: FakeCore, model=None) -> None:
        self.model = model
        self.core = core
        self.probes: list[int] = []

    def round_working_set(self) -> int:
        return 0

    def prefill_prefix(self, tokens, *, cache=None, cached_tokens=0):
        n = len(tokens)
        self.probes.append(n)
        self.core.state["peak"] = n                     # a byte a token: b fits to 1.0 / chunk
        return []


class SwitchModel:
    """A model declaring its regime switch as a spec into its config, as GLM-5.3 declares index_topk."""

    def __init__(self, topk: int, spec: str = "args.index_topk") -> None:
        self.prefill_regime_switch = spec
        self.args = SimpleNamespace(index_topk=topk)


@pytest.fixture()
def fake_mlx(monkeypatch):
    """Install a fake mlx.core backed by a caller-built core; reverted after the test."""

    def install(core: FakeCore) -> FakeCore:
        core_module = ModuleType("mlx.core")
        for name in ("eval", "zeros", "get_active_memory", "reset_peak_memory", "get_peak_memory", "clear_cache"):
            setattr(core_module, name, getattr(core, name))
        mlx_module = ModuleType("mlx")
        mlx_module.core = core_module
        monkeypatch.setitem(sys.modules, "mlx", mlx_module)
        monkeypatch.setitem(sys.modules, "mlx.core", core_module)
        return core

    return install


def test_probes_stay_put_without_a_declared_regime_switch(fake_mlx):
    engine = ProbeEngine(fake_mlx(FakeCore()))
    measure(engine)
    assert engine.probes == [64, 64, 2112, 4160]   # measure() warms up at probe[0] before the loop

    engine = ProbeEngine(fake_mlx(FakeCore()), model=object())      # a model that declares nothing
    measure(engine)
    assert engine.probes == [64, 64, 2112, 4160]   # measure() warms up at probe[0] before the loop


def test_probes_move_past_the_declared_switch(fake_mlx):
    engine = ProbeEngine(fake_mlx(FakeCore()), model=SwitchModel(2048))
    stream = measure(engine)
    assert engine.probes == [64, 64, 4096, 6144]            # a whole width past index_topk, both fit probes
    assert stream.prefill_b == pytest.approx(1 / 2048)  # the fit still sees a byte-a-token growth
    assert stream.prefill_a == pytest.approx(0.03125)     # t2/chunk - b*(n2-64), on the relocated probes


def test_explicit_probes_are_the_callers_choice(fake_mlx):
    engine = ProbeEngine(fake_mlx(FakeCore()), model=SwitchModel(2048))
    measure(engine, probe=(64, 4160, 6208))
    assert engine.probes == [64, 64, 4160, 6208]


def test_probe_premise_moves_both_fit_probes_a_width_past_the_switch():
    assert probe_premise(None, (64, 2112, 4160)) == (64, 2112, 4160)
    assert probe_premise(SwitchModel(0), (64, 2112, 4160)) == (64, 2112, 4160)
    assert probe_premise(SwitchModel(2048, spec="config.missing.path"), (64, 2112, 4160)) == (64, 2112, 4160)
    assert probe_premise(SwitchModel(2048, spec="args.wrong_name"), (64, 2112, 4160)) == (64, 2112, 4160)
    assert probe_premise(SwitchModel(2048), (64, 2112, 4160)) == (64, 4096, 6144)
    assert probe_premise(SwitchModel(5000), (64, 2112, 4160)) == (64, 7048, 9096)
    assert probe_premise(SwitchModel(200), (64, 2112, 4160)) == (64, 2248, 4296)   # never below the short probe
    assert probe_premise(SwitchModel(2048), (64, 4160, 6208)) == (64, 4160, 6208)  # already past: unchanged


def test_a_switch_past_the_default_probes_still_crosses_the_regime(fake_mlx):
    engine = ProbeEngine(fake_mlx(FakeCore()), model=SwitchModel(5000))
    measure(engine)
    assert engine.probes == [64, 64, 7048, 9096]


class StepPeakEngine(ProbeEngine):
    """A prefill peak that steps up at the switch and stays flat: the regime the fit must see."""

    GIAB = 1024**3

    def prefill_prefix(self, tokens, *, cache=None, cached_tokens=0):
        n = len(tokens)
        self.probes.append(n)
        self.core.state["peak"] = int((4.302 if n < 4096 else 6.81) * self.GIAB)  # step at topk + chunk
        return []


def test_a_step_peak_is_the_artifact_on_default_probes_and_flat_on_relocated_ones(fake_mlx):
    # the whole bug in one fake: peak steps at index_topk + chunk and stays flat after
    engine = StepPeakEngine(fake_mlx(FakeCore()), model=SwitchModel(2048))
    shipped = measure(engine, probe=(64, 2112, 4160))   # the shipped probes straddle the step
    assert shipped.prefill_b == pytest.approx(642.25, rel=0.001)   # the step read as growth per token
    assert shipped.prefill_bytes(113_000) > 100 * 1024**3

    engine = StepPeakEngine(fake_mlx(FakeCore()), model=SwitchModel(2048))
    stream = measure(engine)                            # the fix: both fit probes past the step
    assert engine.probes[-2:] == [4096, 6144]
    assert stream.prefill_b == pytest.approx(0.0)       # the step is not read as growth per token
    assert stream.prefill_bytes(200_000) == pytest.approx(6.81 * 1024**3, rel=0.01)


def test_a_declared_spec_that_stops_resolving_is_loud_and_falls_back(fake_mlx, capsys):
    engine = ProbeEngine(fake_mlx(FakeCore()), model=SwitchModel(2048, spec="args.gone_after_an_update"))
    measure(engine)
    assert engine.probes == [64, 64, 2112, 4160]        # the shipped probes, unchanged
    assert "does not resolve" in capsys.readouterr().out   # and the fallback says so


def test_explicit_probes_pass_through_even_when_a_switch_is_declared(fake_mlx):
    engine = ProbeEngine(fake_mlx(FakeCore()), model=SwitchModel(2048))
    measure(engine, probe=(64, 3000, 5048))             # a caller probing across the switch on purpose
    assert engine.probes == [64, 64, 3000, 5048]        # stays the caller's choice


def test_a_spec_resolving_to_a_non_number_is_loud_and_falls_back(fake_mlx):
    class ListSwitch:
        prefill_regime_switch = "args.index_topk"

        def __init__(self) -> None:
            self.args = SimpleNamespace(index_topk=[512, 1536])

    engine = ProbeEngine(fake_mlx(FakeCore()), model=ListSwitch())
    measure(engine)
    assert engine.probes == [64, 64, 2112, 4160]        # the shipped probes, unchanged


def test_the_flat_price_stays_inside_150_percent_of_the_measured_200k_peak(fake_mlx):
    # phase1 ground truth on the M5 Ultra: a 200k cold prefill peaks 8.58 GiB above active
    # memory; the fit prices the 6,144-token level and is a disclosed ~21% under there.
    # This pins the shape: the flat model may not drift past 1.5x of the measured peak.
    engine = StepPeakEngine(fake_mlx(FakeCore()), model=SwitchModel(2048))
    stream = measure(engine)
    assert stream.prefill_bytes(200_000) <= 1.5 * 8.58 * 1024**3
