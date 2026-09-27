"""The draft head's prologue keeps serial bits across the supported copy windows."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.families.qwen4_exp.runtime import FlashNext


def head_fixture():
    dims, streams = 512, 4
    embedding = nn.Embedding(32, dims)
    embedding.weight = mx.random.normal((32, dims), key=mx.random.key(41)).astype(mx.bfloat16)
    embedding = nn.QuantizedEmbedding.from_embedding(embedding, group_size=32, bits=4)

    def linear(key):
        layer = nn.Linear(dims, dims, bias=False)
        layer.weight = (0.05 * mx.random.normal((dims, dims), key=mx.random.key(key))).astype(mx.bfloat16)
        return nn.QuantizedLinear.from_linear(layer, group_size=32, bits=4)

    flash = FlashNext.__new__(FlashNext)
    flash.model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding))
    flash.mtp = SimpleNamespace(streams=streams, fc_embedding=linear(42), fc_hidden=linear(43))
    fused = SimpleNamespace(eps=mx.array([1e-5], dtype=mx.float32))

    def run(value, tokens, cache):
        fused.last_streams = value
        return value[None]

    fused.run = run
    flash.mtp_fused = fused
    flash._mtp_scales = [mx.ones((dims,)), mx.ones((streams * dims,))]
    return flash, dims * streams


@pytest.mark.parametrize("rows", [8, 9, 16])
def test_head_prologue_copy_rows_match_separate_steps(rows):
    flash, width = head_fixture()
    streams = mx.random.normal((rows, width), key=mx.random.key(44)).astype(mx.bfloat16)
    tokens = list(range(rows))
    _, window = flash._mtp_step(tokens, streams, None)
    serial = mx.concatenate([flash._mtp_step([token], streams[i:i + 1], None)[1]
                             for i, token in enumerate(tokens)])
    mx.eval(window, serial)
    assert bool(mx.array_equal(window, serial).item())


def test_projection_tail_preserves_leading_shape_and_serial_bits():
    from tensorfold.families.qwen4_exp.decode import project

    flash, _ = head_fixture()
    inputs = mx.random.normal((3, 11, 512), key=mx.random.key(45)).astype(mx.bfloat16)
    linear = flash.mtp.fc_hidden
    window = project(inputs, linear)
    flat = inputs.reshape(33, 512)
    serial = mx.concatenate([project(flat[i:i + 1], linear) for i in range(33)])
    assert window.shape == inputs.shape
    assert bool(mx.array_equal(window.reshape(serial.shape), serial).item())
