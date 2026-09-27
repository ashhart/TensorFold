"""The Qwen3.8 dense load gate: which checkpoints the lane decoders take, and the refusal of the others."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold import families  # noqa: E402
from tensorfold.families import qwen3_5  # noqa: E402
from tensorfold.families.qwen3_5 import family as qwen_family  # noqa: E402


def _model(widths):
    """bf16 linear layers quantized at ``widths`` in groups of 64 (None: left unquantized)."""

    def linear(bits):
        layer = nn.Linear(256, 64, bias=False)
        layer.set_dtype(mx.bfloat16)
        return layer if bits is None else nn.QuantizedLinear.from_linear(layer, group_size=64, bits=bits)

    model = nn.Module()
    model.layers = [linear(bits) for bits in widths]
    return model


class _Family:
    def __init__(self, model, *, drafter=None, widest=32, rows=False):
        self.inner, self.rows = model, rows
        self.exact_width, self.window_costs = widest, {1: 1.0}


@pytest.fixture
def gate(monkeypatch):
    """load() with the model, config and GPU faked: returns (run, calls)."""

    calls = {"loaded": [], "lanes": [], "rows": []}

    def run(top, widths, *, units=True, lane_kernels="auto"):
        model = _model(widths)
        monkeypatch.setattr(qwen3_5, "load_lane_model", lambda path: calls["loaded"].append(model) or (model, "tok"))
        monkeypatch.setattr(families, "read_config",
                            lambda path: {"quantization": {"bits": top[0], "group_size": top[1]}})
        monkeypatch.setattr(qwen3_5, "tensor_units", lambda: units)
        monkeypatch.setattr(qwen3_5, "install_lane_kernels", calls["lanes"].append)
        monkeypatch.setattr(qwen3_5, "install_row_decoder", lambda m: calls["rows"].append(m) or True)
        monkeypatch.setattr(qwen_family, "Qwen35Family", _Family)
        return qwen3_5.load("unused", lane_kernels=lane_kernels)

    return run, calls


@pytest.mark.parametrize("top,widths", [((4, 64), (4, 4)), ((3, 64), (3, 3)), ((2, 64), (2, 2)),
                                        ((3, 64), (3, 2, 4)), ((2, 64), (2, 3, 4))])
def test_lanes_take_4_3_and_2_bit_checkpoints(gate, top, widths):
    run, calls = gate
    family, _ = run(top, widths)
    assert family.inner._tensorfold_lanes is True and not family.rows
    assert calls["lanes"] == [family.inner] and calls["rows"] == []


@pytest.mark.parametrize("top,widths,named", [
    ((3, 64), (3, 6, 2), "1 6-bit g64"),
    ((3, 64), (3, None), "1 unquantized"),
    ((4, 64), (2, 6, 2), "1 6-bit g64"),                     # mlx_lm's mixed_2_6
])
def test_projections_the_lanes_do_not_take_are_refused(gate, top, widths, named):
    run, calls = gate
    with pytest.raises(SystemExit, match=named):
        run(top, widths)
    assert calls["lanes"] == []                               # nothing served with MLX's row-count-dependent kernels


def test_other_top_level_widths_are_refused_before_loading(gate):
    run, calls = gate
    with pytest.raises(SystemExit, match="2/3/4-bit weights in groups of 64"):
        run((4, 32), (4, 4))
    assert calls["loaded"] == []


@pytest.mark.parametrize("top,units,lane_kernels,decoder", [
    ((4, 64), False, "auto", True),                           # M1 to M4: the lane decoder without tensor units
    ((4, 64), True, "off", True),
    ((3, 64), False, "auto", False),
    ((2, 64), False, "auto", False),
    ((3, 64), True, "off", False),
])
def test_without_the_lane_kernels_only_4_bit_checkpoints_load(gate, top, units, lane_kernels, decoder):
    run, calls = gate
    if decoder:
        family, _ = run(top, (top[0], top[0]), units=units, lane_kernels=lane_kernels)
        assert family.rows and calls["rows"] == [family.inner] and calls["lanes"] == []
    else:
        with pytest.raises(SystemExit, match="4-bit weights in groups of 64 without tensor units"):
            run(top, (top[0], top[0]), units=units, lane_kernels=lane_kernels)
        assert calls["loaded"] == [] and calls["rows"] == []
