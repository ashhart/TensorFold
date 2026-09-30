"""An n-gram table's weight_scale (oMLX's oQ checkpoints store the rows scaled up) is applied at lookup, not refused."""

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.qwen4_exp.model import sanitize  # noqa: E402
from tensorfold.kernels.qwen.flash_next.v1.embed import scaled_rows  # noqa: E402

KEY = "language_model.model.layers.3.ple.ple_embedding.ngram_embedding.weight_scale"


def test_sanitize_collects_the_table_scale_by_embedding_path():
    scales = {}
    out, _ = sanitize({KEY: mx.array([0.0002], dtype=mx.bfloat16)}, scales)
    assert not any("weight_scale" in k for k in out)
    assert list(scales) == ["model.layers.3.ple.ple_embedding"]
    assert scales["model.layers.3.ple.ple_embedding"] == pytest.approx(0.0002, rel=1e-2)


def test_sanitize_without_a_scale_dict_still_refuses_a_scale_other_than_one():
    with pytest.raises(ValueError):
        sanitize({KEY: mx.array([0.5], dtype=mx.bfloat16)})
    out, _ = sanitize({KEY: mx.array([1.0], dtype=mx.bfloat16)})     # MLX conversions: 1, dropped
    assert out == {}


def test_scaled_rows_rounds_once_and_is_the_identity_at_one():
    rows = (mx.random.normal((5, 64)) * 3000).astype(mx.bfloat16)
    assert scaled_rows(rows, 1.0) is rows
    got = scaled_rows(rows, 0.0002)
    want = (rows.astype(mx.float32) * 0.0002).astype(mx.bfloat16)
    assert got.dtype == mx.bfloat16 and bool(mx.array_equal(got, want).item())
    # every row the same function of its own values: row count does not change the bits (drafted == undrafted)
    assert bool(mx.array_equal(scaled_rows(rows[2:3], 0.0002), got[2:3]).item())
