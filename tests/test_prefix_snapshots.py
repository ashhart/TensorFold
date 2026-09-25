"""Prefix snapshots: host-side state survives the disk round trip; warming never crosses models."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine import prefix_snapshots as ps  # noqa: E402


class Layer:
    def __init__(self) -> None:
        self.state_a = mx.arange(6).reshape(2, 3)
        self.slots = [None, mx.ones((2,))]
        self.history = np.array([[11, 12]], dtype=np.int64)
        self.offset = 7


def test_numpy_state_round_trips(tmp_path):
    ps.save_snapshot(tmp_path, "model-a|mlx=1", [1, 2, 3], [Layer()])
    [(tokens, cache)] = list(ps.load_snapshots(tmp_path, "model-a|mlx=1"))
    layer = cache[0]
    assert tokens == [1, 2, 3] and layer.offset == 7
    assert isinstance(layer.history, np.ndarray) and layer.history.tolist() == [[11, 12]]
    assert layer.slots[0] is None and mx.array_equal(layer.slots[1], mx.ones((2,)))


def test_warming_only_takes_blocks_of_the_same_model(tmp_path):
    ps.save_snapshot(tmp_path, "/models/qwen|mlx=1", [5, 6, 7], [Layer()])
    ps.save_snapshot(tmp_path, "/models/nemotron|mlx=1", [8, 9], [Layer()])
    assert ps.blocks_to_warm(tmp_path, "/models/qwen|mlx=2") == [[5, 6, 7]]
    assert ps.blocks_to_warm(tmp_path, "/models/other|mlx=1") == []


def test_keeping_the_newest_counts_one_model_only(tmp_path):
    for i in range(3):
        ps.save_snapshot(tmp_path, "/models/qwen|mlx=1", [i, 1], [Layer()], keep=2)
    ps.save_snapshot(tmp_path, "/models/nemotron|mlx=1", [9], [Layer()], keep=1)
    models = sorted(ps.read_metadata(p)["model"] for p in tmp_path.glob("*.safetensors"))
    assert models == ["/models/nemotron|mlx=1", "/models/qwen|mlx=1", "/models/qwen|mlx=1"]
