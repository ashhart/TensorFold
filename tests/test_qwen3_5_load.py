"""The Qwen3.8 dense load gate: which checkpoints get the lane kernels, and the note on layers they do not take."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold import families  # noqa: E402
from tensorfold.families import qwen3_5  # noqa: E402


def _model(widths):
    """bf16 linear layers quantized at ``widths`` in groups of 64 (None: left unquantized)."""

    def linear(bits):
        layer = nn.Linear(256, 64, bias=False)
        layer.set_dtype(mx.bfloat16)
        return layer if bits is None else nn.QuantizedLinear.from_linear(layer, group_size=64, bits=bits)

    model = nn.Module()
    model.layers = [linear(bits) for bits in widths]
    return model


@pytest.mark.parametrize("top,widths,lanes,note", [
    ((4, 64), (4, 4), True, None),
    ((3, 64), (3, 3), True, None),
    ((3, 64), (3, 6, 2), True, "lane kernels do not take 1 6-bit g64 projections"),
    ((3, 64), (3, None), True, "lane kernels do not take 1 unquantized projections"),
    ((2, 64), (2, 2), True, None),
    ((4, 64), (2, 6, 2), True, "lane kernels do not take 1 6-bit g64 projections"),   # mlx_lm mixed_2_6
    ((4, 32), (4, 4), False, "lane kernels need 4-, 3- or 2-bit weights in groups of 64"),
])
def test_lanes_for_4_3_and_2_bit_checkpoints_and_the_note_on_other_layers(monkeypatch, capsys, top, widths, lanes,
                                                                        note):
    model = _model(widths)
    installed = []
    monkeypatch.setattr(qwen3_5, "load_lane_model", lambda path: (model, "tokenizer"))
    monkeypatch.setattr(families, "read_config", lambda path: {"quantization": {"bits": top[0], "group_size": top[1]}})
    monkeypatch.setattr(qwen3_5, "tensor_units", lambda: True)
    monkeypatch.setattr(qwen3_5, "install_lane_kernels", installed.append)
    fallback = []                                    # the row-exact / MLX path (it patches global state)
    monkeypatch.setattr(qwen3_5, "install_mlx_lanes", lambda m: fallback.append(m) or (1, 1))
    got, _ = qwen3_5.load("unused", lane_kernels="auto")
    out = capsys.readouterr().out
    assert got._tensorfold_lanes is lanes and installed == ([model] if lanes else [])
    assert fallback == ([] if lanes else [model])
    assert (note in out) if note else "lane kernels" not in out, out


@pytest.mark.parametrize("top,units,lane_kernels,fallback_called,note", [
    ((4, 64), False, "auto", True, "they need an M5-generation GPU"),      # upstream's row-exact decoder
    ((3, 64), False, "auto", False, "drafts without the lane kernels need 4-bit weights"),
    ((2, 64), False, "auto", False, "drafts without the lane kernels need 4-bit weights"),
    ((3, 64), True, "off", False, "lane kernels off (--lane-kernels off)"),
    ((4, 64), True, "off", True, "lane kernels off (--lane-kernels off)"),
])
def test_without_lanes_only_4_bit_checkpoints_take_the_row_decoder(monkeypatch, capsys, top, units, lane_kernels,
                                                                     fallback_called, note):
    model = _model((top[0], top[0]))
    installed, fallback = [], []
    monkeypatch.setattr(qwen3_5, "load_lane_model", lambda path: (model, "tokenizer"))
    monkeypatch.setattr(families, "read_config", lambda path: {"quantization": {"bits": top[0], "group_size": top[1]}})
    monkeypatch.setattr(qwen3_5, "tensor_units", lambda: units)
    monkeypatch.setattr(qwen3_5, "install_lane_kernels", installed.append)
    monkeypatch.setattr(qwen3_5, "install_mlx_lanes", lambda m: fallback.append(m) or (1, 1))
    got, _ = qwen3_5.load("unused", lane_kernels=lane_kernels)
    out = capsys.readouterr().out
    assert got._tensorfold_lanes is False and installed == []
    assert fallback == ([model] if fallback_called else [])
    assert note in out, out
    assert ("no verify window" in out) == fallback_called, out

