"""Qwen format admission and decoder selection use metadata doubles without accelerator imports."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest

from tensorfold import families
from tensorfold.families import qwen3_5


class Linear(dict):
    pass


class QuantizedLinear(Linear):
    def __init__(self, bits, group=64, dtype="bfloat16"):
        super().__init__(weight=NS(dtype="uint32", ndim=2, shape=(64, 256 * bits // 32)),
                         scales=NS(dtype=dtype, shape=(64, 256 // group)),
                         biases=NS(dtype=dtype, shape=(64, 256 // group)))
        self.bits, self.group_size, self.mode = bits, group, "affine"


class Model:
    def __init__(self, widths, *, tied=False, dtype="bfloat16"):
        self.layers = [Linear() if width is None else QuantizedLinear(
            *(width if isinstance(width, tuple) else (width, 64)), dtype=dtype) for width in widths]
        self.args = NS(tie_word_embeddings=tied)

    def named_modules(self):
        return [(f"layers.{index}", layer) for index, layer in enumerate(self.layers)]


class Family:
    def __init__(self, model, *, drafter=None, widest=32, rows=False, first_copy_rows=None):
        self.inner, self.rows = model, rows
        self.exact_width, self.window_costs = widest, {1: 1.0}
        self.first_copy_rows = first_copy_rows


@pytest.fixture(autouse=True)
def block_accelerators(monkeypatch):
    for name in ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "torch", "triton"):
        monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture
def gate(monkeypatch):
    """Run the actual load gate and lane coverage check with metadata-only model objects."""
    core, nn, mlx = ModuleType("mlx.core"), ModuleType("mlx.nn"), ModuleType("mlx")
    core.bfloat16, core.uint32 = "bfloat16", "uint32"
    nn.QuantizedLinear, nn.Linear = QuantizedLinear, Linear
    mlx.core, mlx.nn = core, nn
    for name, module in (("mlx", mlx), ("mlx.core", core), ("mlx.nn", nn)):
        monkeypatch.setitem(sys.modules, name, module)
    inputs = ModuleType("tensorfold.kernels.inputs")
    inputs.ints = lambda *args, **kwargs: pytest.fail("a load gate must not allocate kernel inputs")
    monkeypatch.setitem(sys.modules, inputs.__name__, inputs)
    package = importlib.import_module("tensorfold.kernels.qwen.dense.v1")
    source = Path(package.__file__).with_name("lane_qmm.py")
    spec = importlib.util.spec_from_file_location("quant_gate_lane_qmm", source)
    lane = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lane)
    monkeypatch.setattr(package, "lane_qmm", lane, raising=False)
    row = ModuleType(package.__name__ + ".row_matmul")
    row.WINDOW_ROWS = 16
    monkeypatch.setattr(package, "row_matmul", row, raising=False)
    family = ModuleType("tensorfold.families.qwen3_5.family")
    family.Qwen35Family = Family
    monkeypatch.setitem(sys.modules, family.__name__, family)
    calls = {"loaded": [], "lanes": [], "rows": []}

    def run(top, widths, *, units=True, lane_kernels="auto", config=None, tied=False, dtype="bfloat16",
            row_supported=True):
        model = Model(widths, tied=tied, dtype=dtype)
        monkeypatch.setattr(qwen3_5, "load_lane_model", lambda path: calls["loaded"].append(model) or (model, "tok"))
        monkeypatch.setattr(families, "read_config",
                            lambda path: {"quantization": {"bits": top[0], "group_size": top[1]}, **(config or {})})
        monkeypatch.setattr(qwen3_5, "tensor_units", lambda: units)
        monkeypatch.setattr(qwen3_5, "install_lane_kernels", calls["lanes"].append)
        monkeypatch.setattr(qwen3_5, "install_row_decoder", lambda m: calls["rows"].append(m) or row_supported)
        return qwen3_5.load("unused", lane_kernels=lane_kernels)

    return run, calls


@pytest.mark.parametrize("top,widths", [((4, 64), (4, 4)), ((3, 64), (3, 3)), ((2, 64), (2, 2)),
                                        ((3, 64), (3, 2, 4)), ((2, 64), (2, 3, 4)), ((8, 64), (8, 8)),
                                        ((4, 64), (4, 5, 6)), ((2, 64), (2, 5, 6, 3)),
                                        ((4, 32), ((4, 32), (4, 32)))])
def test_native_lane_formats_keep_tensor_unit_routing(gate, top, widths):
    run, calls = gate
    family, _ = run(top, widths)
    assert family.inner._tensorfold_lanes is True and not family.rows
    assert calls["lanes"] == [family.inner] and calls["rows"] == []


@pytest.mark.parametrize("top", [(bits, group) for bits in (2, 3, 4, 5, 6, 8) for group in (32, 64, 128)])
@pytest.mark.parametrize("units,lane_kernels", [(False, "auto"), (True, "off")])
def test_packed_row_decoder_accepts_every_affine_format(gate, top, units, lane_kernels, monkeypatch):
    monkeypatch.delenv("TF_COPY_ROWS", raising=False)
    run, calls = gate
    family, _ = run(top, (top, top), units=units, lane_kernels=lane_kernels)
    assert family.rows and not family.inner._tensorfold_lanes
    assert calls["rows"] == [family.inner] and calls["lanes"] == []
    assert (family.exact_width, family.first_copy_rows) == (128, 16)


@pytest.mark.parametrize("value,lanes,rows", [(None, 32, 128), ("0", 32, 16), ("128", 128, 128), ("64", 64, 64),
                                               ("8", 32, 16), ("999", 128, 128)])
def test_copy_rows_sets_the_checked_window_above_the_first_copy_width(gate, monkeypatch, value, lanes, rows):
    run, _ = gate
    if value is None:
        monkeypatch.delenv("TF_COPY_ROWS", raising=False)
    else:
        monkeypatch.setenv("TF_COPY_ROWS", value)
    family, _ = run((4, 64), (4, 4))
    assert (family.exact_width, family.first_copy_rows) == (lanes, 32)
    family, _ = run((4, 64), (4, 4), units=False)
    assert (family.exact_width, family.first_copy_rows) == (rows, 16)


@pytest.mark.parametrize("top", [(8, 32), (3, 32), (4, 128), (5, 128)])
def test_tensor_unit_auto_uses_packed_rows_for_other_groups(gate, top):
    run, calls = gate
    family, _ = run(top, (top, top), units=True)
    assert family.rows and calls["rows"] == [family.inner] and calls["lanes"] == []


@pytest.mark.parametrize("top", [(8, 32), (3, 128), (4, 128)])
def test_forced_tensor_unit_kernels_refuse_other_groups_before_loading(gate, top):
    run, calls = gate
    with pytest.raises(ValueError, match="packed affine row kernels.*auto or off"):
        run(top, (top, top), lane_kernels="on")
    assert calls["loaded"] == [] and calls["lanes"] == calls["rows"] == []


@pytest.mark.parametrize("top,widths,named", [((3, 64), (3, (3, 32), 2), "1 3-bit g32"),
                                             ((3, 64), (3, None), "1 unquantized")])
def test_unadvertised_unsupported_native_projections_are_refused(gate, top, widths, named):
    run, calls = gate
    with pytest.raises(SystemExit, match=named):
        run(top, widths)
    assert calls["lanes"] == calls["rows"] == []


@pytest.mark.parametrize("top,mode", [((7, 64), "affine"), ((4, 16), "affine"), ((4, 32), "mxfp4")])
def test_invalid_top_level_formats_are_refused_before_loading(gate, top, mode):
    run, calls = gate
    value = {"quantization": {"bits": top[0], "group_size": top[1], "mode": mode}}
    with pytest.raises(SystemExit, match="affine"):
        run(top, (4, 4), config=value)
    assert calls["loaded"] == []


@pytest.mark.parametrize("units", [True, False])
@pytest.mark.parametrize("config", [{"tie_word_embeddings": True}, {"text_config": {"tie_word_embeddings": True}}])
def test_tied_heads_are_refused_before_loading(gate, units, config):
    run, calls = gate
    with pytest.raises(SystemExit, match="tied embedding"):
        run((4, 64), (4, 4), units=units, config=config)
    assert calls["loaded"] == []


def test_unadvertised_tied_head_is_refused_after_loading(gate):
    run, calls = gate
    with pytest.raises(SystemExit, match="1 tied embedding head"):
        run((4, 64), (4, 4), tied=True)
    assert len(calls["loaded"]) == 1 and calls["lanes"] == []


def test_native_kernels_refuse_non_bf16_scales(gate):
    run, calls = gate
    with pytest.raises(SystemExit, match="float16 scales"):
        run((8, 64), (8, 8), dtype="float16")
    assert calls["lanes"] == []


def test_row_installation_failure_does_not_create_a_family(gate):
    run, calls = gate
    with pytest.raises(SystemExit, match="does not take these weights"):
        run((8, 128), ((8, 128),), row_supported=False)
    assert len(calls["rows"]) == 1 and calls["lanes"] == []


@pytest.mark.parametrize("units", [False, True])
def test_mixed_affine_overrides_route_without_rejecting_supported_widths(gate, units):
    run, calls = gate
    value = {"quantization": {"bits": 4, "group_size": 64,
                              "model.layers.0.linear_attn.out_proj": {"bits": 5},
                              "model.layers.3.self_attn.v_proj": {"bits": 6}}}
    family, _ = run((4, 64), (4, 5, 6), units=units, config=value)
    assert family.rows is (not units)
    assert (calls["lanes"] if units else calls["rows"]) == [family.inner]


def test_mixed_group_override_routes_entire_target_to_packed_rows(gate):
    run, calls = gate
    value = {"quantization": {"bits": 4, "group_size": 64,
                              "model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 128}}}
    family, _ = run((4, 64), (4, (8, 128)), config=value)
    assert family.rows and calls["lanes"] == [] and calls["rows"] == [family.inner]


def test_lane_kernels_on_needs_tensor_units(gate):
    run, calls = gate
    with pytest.raises(SystemExit, match="needs Metal 4 tensor units"):
        run((4, 64), (4, 4), units=False, lane_kernels="on")
    assert calls["loaded"] == []


@pytest.mark.parametrize("bits,group", [(bits, group) for bits in (2, 3, 4, 5, 6, 8) for group in (32, 64, 128)])
def test_check_accepts_affine_metadata_before_the_download(tmp_path, monkeypatch, bits, group):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5",
                                                   "quantization": {"bits": bits, "group_size": group}}))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(qwen3_5, "tensor_units", lambda: False)
    qwen3_5.check(tmp_path)


@pytest.mark.parametrize("value", [{}, {"quantization": {"bits": 7, "group_size": 64}},
                                  {"quantization": {"bits": 4, "group_size": 16}},
                                  {"quantization": {"bits": 4, "group_size": 64, "mode": "mxfp4"}}])
def test_check_refuses_unsupported_metadata_before_the_download(tmp_path, monkeypatch, value):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", **value}))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(qwen3_5, "tensor_units", lambda: True)
    with pytest.raises(ValueError, match="affine"):
        qwen3_5.check(tmp_path)


def test_check_leaves_cuda_to_its_metadata_hook(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.setattr(sys, "platform", "linux")
    qwen3_5.check(tmp_path)
