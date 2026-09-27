"""The Qwen3.8 dense load gate: which checkpoints the lane decoders take, and the refusal of the others."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold import families  # noqa: E402
from tensorfold.families import qwen3_5  # noqa: E402
from tensorfold.families.qwen3_5 import family as qwen_family  # noqa: E402


def _model(widths):
    """bf16 linears quantized at ``widths``: bits or (bits, group size), groups of 64 by default; None: float."""

    def linear(width):
        layer = nn.Linear(256, 64, bias=False)
        layer.set_dtype(mx.bfloat16)
        if width is None:
            return layer
        bits, group = width if isinstance(width, tuple) else (width, 64)
        return nn.QuantizedLinear.from_linear(layer, group_size=group, bits=bits)

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

    def run(top, widths, *, units=True, lane_kernels="auto", config=None, tied=False):
        model = _model(widths)
        if tied:                                               # mlx_lm's args, as a tied checkpoint loads them
            model.args = SimpleNamespace(tie_word_embeddings=True)
        monkeypatch.setattr(qwen3_5, "load_lane_model", lambda path: calls["loaded"].append(model) or (model, "tok"))
        monkeypatch.setattr(families, "read_config",
                            lambda path: {"quantization": {"bits": top[0], "group_size": top[1]}, **(config or {})})
        monkeypatch.setattr(qwen3_5, "tensor_units", lambda: units)
        monkeypatch.setattr(qwen3_5, "install_lane_kernels", calls["lanes"].append)
        monkeypatch.setattr(qwen3_5, "install_row_decoder", lambda m: calls["rows"].append(m) or True)
        monkeypatch.setattr(qwen_family, "Qwen35Family", _Family)
        return qwen3_5.load("unused", lane_kernels=lane_kernels)

    return run, calls


@pytest.mark.parametrize("top,widths", [((4, 64), (4, 4)), ((3, 64), (3, 3)), ((2, 64), (2, 2)),
                                        ((3, 64), (3, 2, 4)), ((2, 64), (2, 3, 4)), ((8, 64), (8, 8)),
                                        ((4, 64), (4, 5, 6)),              # Vontra oQ4: 5- and 6-bit layers
                                        ((2, 64), (2, 5, 6, 3)),           # Vontra oQ2
                                        ((4, 64), (2, 6, 2))])             # mlx_lm's mixed_2_6
def test_lanes_take_every_mlx_width_in_groups_of_64(gate, top, widths):
    run, calls = gate
    family, _ = run(top, widths)
    assert family.inner._tensorfold_lanes is True and not family.rows
    assert calls["lanes"] == [family.inner] and calls["rows"] == []


@pytest.mark.parametrize("top,widths,named", [
    ((3, 64), (3, (3, 32), 2), "1 3-bit g32"),
    ((3, 64), (3, None), "1 unquantized"),
    ((4, 64), (4, (6, 32), None), "1 6-bit g32, 1 unquantized"),
])
def test_projections_the_lanes_do_not_take_are_refused(gate, top, widths, named):
    run, calls = gate
    with pytest.raises(SystemExit, match=named):
        run(top, widths)
    assert calls["lanes"] == []                               # nothing served with MLX's row-count-dependent kernels


def test_other_top_level_widths_are_refused_before_loading(gate):
    run, calls = gate
    with pytest.raises(SystemExit, match="2/3/4/5/6/8-bit weights in groups of 64"):
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


OQ4 = {"bits": 4, "group_size": 64, "model.layers.0.linear_attn.out_proj": {"bits": 5, "group_size": 64},
       "model.layers.3.self_attn.v_proj": {"bits": 6, "group_size": 64}}


@pytest.mark.parametrize("units", [True, False])
@pytest.mark.parametrize("config", [{"tie_word_embeddings": True}, {"text_config": {"tie_word_embeddings": True}}])
def test_a_tied_head_is_refused_before_loading(gate, units, config):
    run, calls = gate
    with pytest.raises(SystemExit, match="tied embedding"):
        run((4, 64), (4, 4), units=units, config=config)
    assert calls["loaded"] == []


def test_a_tied_head_the_config_does_not_name_is_refused_after_loading(gate):
    run, calls = gate
    with pytest.raises(SystemExit, match="1 tied embedding head"):
        run((4, 64), (4, 4), tied=True)
    assert calls["lanes"] == []


def test_per_layer_widths_the_decoder_does_not_read_are_refused_before_loading(gate):
    run, calls = gate
    with pytest.raises(SystemExit, match="layers at 5-bit g64, 6-bit g64"):
        run((4, 64), (4, 4), units=False, config={"quantization": OQ4})
    assert calls["loaded"] == []
    family, _ = run((4, 64), (4, 5, 6), config={"quantization": OQ4})
    assert calls["lanes"] == [family.inner]


def test_lane_kernels_on_needs_tensor_units(gate):
    run, calls = gate
    with pytest.raises(SystemExit, match="needs Metal 4 tensor units"):
        run((4, 64), (4, 4), units=False, lane_kernels="on")
    assert calls["loaded"] == []


@pytest.mark.parametrize("config,units,named", [
    ({}, True, r"none \(unquantized weights\)"),
    ({"quantization": {"bits": 4, "group_size": 64}, "tie_word_embeddings": True}, True, "tied embedding"),
    ({"quantization": {"bits": 4, "group_size": 64}, "text_config": {"tie_word_embeddings": True}}, False,
     "tied embedding"),
    ({"quantization": {"bits": 4, "group_size": 64, "a": {"bits": 3, "group_size": 32}}}, True, "layers at 3-bit g32"),
    ({"quantization": OQ4}, False, "layers at 5-bit g64, 6-bit g64"),
    ({"quantization": {"bits": 2, "group_size": 64}}, False, "MLX 2-bit, groups of 64"),
    ({"quantization": {"bits": 8, "group_size": 32}}, True, "MLX 8-bit, groups of 32"),
    ({"quantization": {"bits": 4, "group_size": 64}}, True, None),
    ({"quantization": {"bits": 4, "group_size": 64}}, False, None),
    ({"quantization": OQ4}, True, None),
    ({"quantization": {"bits": 2, "group_size": 64}}, True, None),
    ({"quantization": {"bits": 4, "group_size": 64, "model.embed_tokens": {"bits": 8, "group_size": 64}}}, False,
     None),                                                    # a lookup, not a matmul
    ({"quantization": {"bits": 4, "group_size": 64, "model.embed_tokens": {"bits": 6, "group_size": 32}}}, True, None),
    ({"quantization": {"bits": 4, "group_size": 64, "language_model.lm_head": {"group_size": 64}}}, False, None),
    ({"quantization": {"bits": 4, "group_size": 64, "language_model.lm_head": {"bits": 5}}}, True, None),
])
def test_check_refuses_from_the_config_before_the_download(tmp_path, monkeypatch, config, units, named):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", **config}))
    monkeypatch.setattr(qwen3_5, "tensor_units", lambda: units)
    if named is None:
        qwen3_5.check(tmp_path)
    else:
        with pytest.raises(ValueError, match=named):
            qwen3_5.check(tmp_path)


def test_check_leaves_cuda_to_its_own_rules(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.setattr(sys, "platform", "linux")
    qwen3_5.check(tmp_path)                                    # require_readable checks CUDA_QUANTIZATION there
