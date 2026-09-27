"""--prefill-grid sets the prefill chunks and the resumable checkpoints together."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold import cli
from tensorfold.engine.lane_engine import LaneEngine


def parse(*extra):
    return cli.build_parser().parse_args(["serve", "some/model", *extra])


def test_the_grid_defaults_to_2048_and_takes_the_listed_sizes():
    assert parse().prefill_grid == 2048
    assert parse("--prefill-grid", "512").prefill_grid == 512
    for bad in ("300", "4096", "grid"):
        with pytest.raises(SystemExit):
            parse("--prefill-grid", bad)


def test_serving_puts_chunks_and_checkpoints_on_the_chosen_grid(monkeypatch):
    class Loaded(Exception):
        pass

    def load(model_dir, **options):
        raise Loaded

    monkeypatch.setattr(LaneEngine, "prefill_step", LaneEngine.prefill_step)
    monkeypatch.setattr(LaneEngine, "prefill_align", LaneEngine.prefill_align)
    family = SimpleNamespace(title="fake", model_type="fake", package=SimpleNamespace(load=load))
    args = parse("--prefill-grid", "512", "--no-drafts")
    with pytest.raises(Loaded):
        cli._serve_mlx(args, family, Path("some/model"), 0, [], 1 << 30)
    assert LaneEngine.prefill_step == LaneEngine.prefill_align == 512
    engine = LaneEngine.__new__(LaneEngine)
    engine.lane_prefill = 0
    assert engine.on_grid(1500) == 1024
