"""H3 video decoder: tile layout, clip chunking and output geometry on a tiny random decoder."""

import mlx.core as mx
import pytest

from tensorfold.families.h3.vae_video import DecoderConfig, VideoDecoder, _blend


def tiny(**changes):
    return VideoDecoder(DecoderConfig(layers=1, heads=2, head_dim=48, **changes))


def test_tile_layout_matches_the_release_geometry():
    decoder = tiny()
    assert decoder._tiles(480) == ([0, 112, 224], [144, 144])
    assert decoder._tiles(864) == ([0, 144, 288, 448, 608], [112, 112, 96, 96])
    assert decoder._tiles(256) == ([0], [])


def test_blend_cross_fades_the_overlap_only():
    a, b = mx.zeros((1, 4, 1)), mx.ones((1, 6, 1))
    out = _blend(a, b, 4, axis=1)
    assert out.shape == (1, 6, 1)
    assert out[0, :, 0].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0, 1.0])


@pytest.mark.parametrize("latent_frames, frames", [(7, 22), (12, 39), (37, 124)])
def test_decode_geometry(latent_frames, frames):
    decoder = tiny()
    out = decoder.decode(mx.random.normal((1, 24, latent_frames, 2, 3)))
    assert out.shape == (1, 3, frames, 32, 48)


def test_too_few_latent_frames_are_refused():
    with pytest.raises(ValueError):
        tiny().decode(mx.zeros((1, 24, 1, 2, 2)))
