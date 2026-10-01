"""GLM-5.3-Flash's dense EXL3 pack (``glm5_next/cuda/dense_exl3.py``) on the CPU: a rank's cut of an EXL3 group is the
matching block of the full layer's weight, for every split rule the BF16 twin takes (row, col, rep) and for the
head's vocabulary columns, so the ranks' outputs (concatenated, or summed by the all-reduce) are the full layer's.
Weights come from ``format.dequantize`` (float64, diag(suh) H W_q H diag(svh)) on synthetic groups."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.families.glm5_next.cuda import dense_exl3


def _group(kt: int, nt: int, bits: float = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(kt, nt, fmt.tile_words(bits))).astype(np.int16))
    suh = torch.from_numpy(rng.choice([-1.0, 1.0], kt * 16) * rng.uniform(0.5, 1.5, kt * 16)).half()
    svh = torch.from_numpy(rng.uniform(0.01, 0.02, nt * 16)).half()
    return trellis, suh, svh


def _w(trellis, suh, svh, bits=4, codebook="mul1"):
    return fmt.dequantize(trellis, suh, svh, bits, codebook)


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_row_cut_is_the_rank_columns(bits):
    tr, suh, svh = _group(16, 32, bits)
    full = _w(tr, suh, svh, bits)
    n = full.shape[1]
    for rank in (0, 1):
        part = _w(*dense_exl3.cut(tr, suh, svh, "row", rank, 2), bits)
        np.testing.assert_array_equal(part, full[:, rank * n // 2:(rank + 1) * n // 2])


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_col_cut_rows_sum_to_the_full_output(bits):
    tr, suh, svh = _group(32, 16, bits, seed=1)
    full = _w(tr, suh, svh, bits)
    k = full.shape[0]
    x = np.random.default_rng(2).standard_normal((3, k))
    y = sum(x[:, r * k // 2:(r + 1) * k // 2] @ _w(*dense_exl3.cut(tr, suh, svh, "col", r, 2), bits) for r in (0, 1))
    np.testing.assert_allclose(y, x @ full, rtol=1e-9, atol=1e-9)


def test_rep_and_vocabulary_columns():
    tr, suh, svh = _group(16, 48)
    full = _w(tr, suh, svh)
    np.testing.assert_array_equal(_w(*dense_exl3.cut(tr, suh, svh, "rep", 1, 2)), full)
    np.testing.assert_array_equal(_w(*dense_exl3.cut(tr, suh, svh, "rep", 1, 2, cols=(384, 768))), full[:, 384:768])


def test_cuts_off_the_128_grid_are_refused():
    tr, suh, svh = _group(16, 24)                     # N = 384: halves of 192 columns
    with pytest.raises(ValueError):
        dense_exl3.cut(tr, suh, svh, "row", 0, 2)
    with pytest.raises(ValueError):
        dense_exl3.cut(tr, suh, svh, "rep", 0, 2, cols=(0, 200))
    with pytest.raises(ValueError):
        dense_exl3.cut(tr, suh, svh, "drop", 0, 2)


def test_pack_paths():
    assert dense_exl3.pack_paths({"TF_GLM_DENSE_EXL3": " a.safetensors:b.safetensors "}) == ["a.safetensors", "b.safetensors"]
    assert dense_exl3.pack_paths({}) == []
