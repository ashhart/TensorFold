"""Scaled RoPE frequencies must be ready before inference moves to a worker thread."""

import threading

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from mlx_lm.models.rope_utils import YarnRoPE  # noqa: E402
from tensorfold.families.qwen3_5 import _materialize_rope_frequencies  # noqa: E402


def test_yarn_rope_can_run_on_scheduler_thread_after_loading():
    model = nn.Module()
    model.rope = YarnRoPE(64, base=10_000_000, scaling_factor=4,
                          original_max_position_embeddings=262_144)
    _materialize_rope_frequencies(model)
    results, errors = [], []

    def infer():
        try:
            for position in (0, 262_144, 999_999):
                output = model.rope(mx.ones((1, 1, 1, 256)), offset=position)
                mx.async_eval(output)
                results.append(bool(mx.all(mx.isfinite(output)).item()))
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=infer)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert not errors
    assert results == [True, True, True]


def test_model_without_private_rope_frequencies_is_unchanged():
    model = nn.Linear(4, 4)
    before = model.weight
    _materialize_rope_frequencies(model)
    assert model.weight is before
