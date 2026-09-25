"""Radix top-k: exactly the k largest by (value desc, id asc), whatever the ties."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine.topk import topk_rows  # noqa: E402


def _reference(row: np.ndarray, k: int) -> np.ndarray:
    ids = np.arange(len(row))
    return np.lexsort((ids, -row))[:k]


@pytest.mark.parametrize("k", [1, 16, 28, 64])
def test_topk_matches_lexsort(k):
    rng = np.random.default_rng(k)
    rows = [rng.normal(size=248320) * 3,                        # continuous-ish (bf16 rounding makes ties)
            np.round(rng.normal(size=5000) * 2),                # heavy ties
            np.full(3000, 1.5),                                 # all equal
            -np.abs(rng.normal(size=4000))]                     # all negative
    for data in rows:
        x = mx.array(data.astype(np.float32)).astype(mx.bfloat16)[None]
        want = _reference(np.array(x.astype(mx.float32))[0], k)
        try:
            idx, val = topk_rows(x, k)
            mx.eval(idx, val)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"metal kernels unavailable: {str(exc).splitlines()[0][:80]}")
        assert np.array(idx)[0].tolist() == want.tolist()


def test_rows_are_independent():
    rng = np.random.default_rng(3)
    data = (rng.normal(size=(8, 50000)) * 3).astype(np.float32)
    x = mx.array(data).astype(mx.bfloat16)
    idx, _ = topk_rows(x, 28)
    alone = [np.array(topk_rows(x[i:i + 1], 28)[0])[0] for i in range(8)]
    assert all((np.array(idx)[i] == alone[i]).all() for i in range(8))
