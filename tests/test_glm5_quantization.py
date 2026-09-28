"""GLM-5.3-Flash's loader reads each module's format from the config, checks the shapes, refuses what it can't read."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from glm5_fakes import write_checkpoint  # noqa: E402
from tensorfold.families import glm5_next  # noqa: E402
from tensorfold.families.glm5_next import linear, weights  # noqa: E402

DOWN = "model.language_model.layers.0.mlp.down_proj"


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_a_module_at_half_the_group_is_read_as_the_config_says(tmp_path):
    # 4-bit weights in groups of 32 have the shapes of 2-bit weights in groups of 64 at twice the input width
    model = weights.load_backbone(write_checkpoint(tmp_path / "g", overrides={DOWN: {"bits": 4, "group_size": 32}}))
    down = model.layers[0].mlp.down
    assert (down.bits, down.group, down.ins) == (4, 32, 128)
    x = mx.random.normal((3, 128)).astype(mx.bfloat16)
    want = mx.quantized_matmul(x, down.weight, down.scales, down.biases, transpose=True, group_size=32, bits=4)
    assert bool(mx.array_equal(linear.project(x, down, rows_exact=False), want).item())


def test_a_stated_format_that_does_not_fit_the_shapes_is_refused(tmp_path):
    folder = write_checkpoint(tmp_path / "s", stated={DOWN: {"bits": 8, "group_size": 64}})
    with pytest.raises(ValueError, match="layers.0.mlp.down_proj"):
        weights.load_backbone(folder)


@pytest.mark.parametrize("entry", [{"bits": 4, "group_size": 64, "mode": "mxfp4"}, {"bits": 7, "group_size": 64}, False])
def test_unsupported_formats_are_refused_before_and_at_load(tmp_path, entry, monkeypatch):
    folder = write_checkpoint(tmp_path / "u", stated={DOWN: entry})
    monkeypatch.setattr("sys.platform", "darwin")
    with pytest.raises(ValueError, match="layers.0.mlp.down_proj"):
        glm5_next.check(folder)
    with pytest.raises(ValueError, match="layers.0.mlp.down_proj"):
        weights.load_backbone(folder)


def test_projections_of_one_input_at_two_formats_stay_separate_calls(tmp_path):
    gate = "model.language_model.layers.0.mlp.gate_proj"
    folder = write_checkpoint(tmp_path / "m", overrides={gate: {"bits": 8, "group_size": 64}})
    gate_up = weights.load_backbone(folder).layers[0].mlp.gate_up
    assert isinstance(gate_up, linear.QSplit) and [p.bits for p in gate_up.parts] == [8, 4]
    x = mx.random.normal((3, gate_up.parts[0].ins)).astype(mx.bfloat16)
    want = mx.concatenate([p(x) for p in gate_up.parts], axis=-1)
    for rows_exact in (False, True):
        assert bool(mx.array_equal(linear.project(x, gate_up, rows_exact=rows_exact), want).item())


def test_a_mixed_checkpoints_entries_are_read_by_the_layouts_names(tmp_path):
    from glm5_fakes import TEXT

    from tensorfold.families.glm5_next.config import quant_formats

    # mlx-lm's keys: a stated entry without bits takes MLX's default for its mode (to_quantized), as the 27B reads it
    config = {"text_config": TEXT, "quantization": {"bits": 4, "group_size": 64, "language_model.lm_head": {"bits": 8},
                                                    "language_model.model.layers.1.self_attn.forget_gate.f_b_proj":
                                                        {"group_size": 64}, "language_model.mtp.0.eh_proj": False}}
    default, stated = quant_formats(config)
    assert default == (4, 64, "affine")
    assert stated == {"language_model.lm_head": (8, 64, "affine"),
                      "language_model.model.layers.1.self_attn.forget_gate.f_b_proj": (4, 64, "affine"),
                      "language_model.mtp.0.eh_proj": None}
