"""H3 scaffold: checkpoint detection and sequence geometry, with no weights and no MLX."""

import json

import pytest

from tensorfold import families
from tensorfold.families.h3 import config as h3


def test_family_is_listed_without_an_mlx_loader():
    family = families.families()["minimax_h3"]
    assert family.title == "MiniMax H3"
    assert not family.lanes
    assert families.backends_of(family) == ()


def test_frame_lattice():
    assert [h3.latent_frames(n) for n in (56, 73, 124, 243, 362)] == [17, 22, 37, 72, 107]
    assert h3.align_frames(120) == 107 and h3.align_frames(124) == 124
    with pytest.raises(ValueError):
        h3.latent_frames(120)


def test_generated_rows_at_the_benchmark_canvas():
    assert h3.generated_rows(864, 480, 124) == (37 * 15 * 27, 414)
    assert h3.generated_rows(768, 448, 124) == (37 * 14 * 24, 414)
    with pytest.raises(ValueError):
        h3.generated_rows(860, 480, 124)


def test_pipeline_root_finds_the_fl2va_partition(tmp_path):
    part = tmp_path / "FL2VA"
    (part / "transformer").mkdir(parents=True)
    (part / "model_index.json").write_text(json.dumps({"_class_name": "MiniMaxH3Pipeline"}))
    (part / "transformer" / "config.json").write_text(json.dumps({"num_layers": 50, "patch_size": [1, 2, 2]}))
    assert h3.pipeline_root(tmp_path) == part
    assert h3.DiTConfig.from_checkpoint(tmp_path).inner_dim == 7168
    assert h3.pipeline_root(tmp_path / "missing") is None
