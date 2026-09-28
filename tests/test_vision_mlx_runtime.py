"""Small CPU-array checks against the installed MLX rotary and embedding operations."""
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
nn = pytest.importorskip('mlx.nn')

from tensorfold.vision.rotary import frequency_axes, install_rotary, vision_positions
from tensorfold.vision.qwen_mlx import QwenVisionFrontend
from tensorfold.vision.qwen_processing import PreparedVisionPrompt


@pytest.fixture(autouse=True)
def cpu_arrays():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_three_axis_rotation_matches_per_token_native_rotations():
    rope = nn.RoPE(8, base=10000)
    attention = SimpleNamespace(rope=rope)
    core = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    install_rotary(core, [2, 1, 1])
    x = mx.array(np.arange(60, dtype=np.float32).reshape(1, 2, 3, 10) / 20)
    positions = mx.array([[[5, 5, 5]], [[5, 6, 7]], [[5, 7, 6]]], dtype=mx.int32)
    with vision_positions(positions):
        actual = attention.rope(x)
    references = [np.array(rope(x.transpose(2, 1, 0, 3), offset=positions[a, 0]).transpose(2, 1, 0, 3))
                  for a in range(3)]
    expected = np.array(x)
    for column, axis in enumerate(frequency_axes(8, [2, 1, 1]) * 2):
        expected[..., column] = references[axis][..., column]
    assert np.array_equal(np.array(actual), expected)
    assert np.array_equal(np.array(attention.rope(x, offset=9)), np.array(rope(x, offset=9)))
    pieces = []
    for start, end in ((0, 1), (1, 3)):
        with vision_positions(positions[:, :, start:end]):
            pieces.append(attention.rope(x[:, :, start:end]))
    assert np.array_equal(np.array(mx.concatenate(pieces, axis=2)), expected)


def test_indexed_visual_embeddings_preserve_text_and_cast_dtype():
    class Tower:
        patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=mx.zeros((1,), dtype=mx.float32)))

        def __call__(self, pixels, grid):
            return mx.array([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=mx.float32), None

    frontend = QwenVisionFrontend.__new__(QwenVisionFrontend)
    frontend.mx, frontend.tower = mx, Tower()
    frontend.embed_tokens = lambda tokens: mx.broadcast_to(tokens[..., None], (*tokens.shape, 4)).astype(mx.bfloat16)
    prepared = PreparedVisionPrompt((1, 99, 99, 2), np.zeros((8, 1)), np.array([[1, 2, 4]]),
                                    np.arange(12).reshape(3, 1, 4), -1, ((1, 3),), ('test',))
    result = frontend.encode(prepared)
    assert result.inputs_embeds.dtype == mx.bfloat16
    assert np.array_equal(np.array(result.inputs_embeds.astype(mx.float32)),
                          [[[1]*4, [7, 8, 9, 10], [11, 12, 13, 14], [2]*4]])
    assert result.rope_delta == -1
