"""Expert-down lanes must not read past the activation for narrow quantized widths."""
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
from tools.native_legacy import flash


def ones_weights(shape):
    # q=0, bias=1 gives exactly one at every dequantized element.
    return SimpleNamespace(
        weight=mx.zeros((*shape[:-1], shape[-1] // 8), dtype=mx.uint32),
        scales=mx.ones((*shape[:-1], shape[-1] // 32), dtype=mx.bfloat16),
        biases=mx.ones((*shape[:-1], shape[-1] // 32), dtype=mx.bfloat16),
    )


@pytest.mark.parametrize("width", [32, 64, 128, 480, 512, 544, 768, 1024])
@pytest.mark.parametrize("rows", [1, 3])
def test_expert_down_width_boundaries(width, rows):
    down = ones_weights((32, 8, width))
    shared = ones_weights((8, width))
    logits = mx.zeros((rows, 33), dtype=mx.float32)
    group = flash.expert_group(logits, 2, 32)
    picks, weights = group[:2]
    act = mx.ones((rows, 3, width), dtype=mx.bfloat16)
    # An independent exact oracle: each dot product is width ones summed.
    for actual in (flash.expert_down_y(act, picks, down, shared),
                   flash.grouped_down(act, group, down, shared)):
        assert actual.shape == (rows, 3, 8)
        assert bool(mx.all(actual == width).item())
    combined = flash.expert_down(act, picks, weights, logits, 2, 32, down, shared)
    assert bool(mx.all(combined == 1.5 * width).item())  # unit routing sum + sigmoid(0)
    plain = flash.expert_down(act[:, :2], picks, weights, logits, 2, 32, down)
    assert bool(mx.all(plain == width).item())


@pytest.mark.parametrize("width", [0, 16, 1056])
def test_expert_down_rejects_unsupported_widths(width):
    down = ones_weights((32, 8, 32))
    shared = ones_weights((8, 32))
    logits = mx.zeros((1, 33), dtype=mx.float32)
    group = flash.expert_group(logits, 2, 32)
    act = mx.ones((1, 3, width), dtype=mx.bfloat16)
    for run in (
        lambda: flash.expert_down_y(act, group[0], down, shared),
        lambda: flash.grouped_down(act, group, down, shared),
        lambda: flash.expert_down(act, group[0], group[1], logits, 2, 32, down, shared),
    ):
        with pytest.raises(ValueError, match="NI"):
            run()
