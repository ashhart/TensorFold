"""Which prompt GEMM a GPU takes in checkpoint math: bulk-copy tiles from sm_90, gemm_ck's same tile below (host)."""

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.nvfp4.checkpoint import WS, bulk_tile  # noqa: E402


def test_bulk_copy_tiles_from_sm_90_only():
    for cap in ((9, 0), (10, 0), (12, 0), (12, 1)):
        assert bulk_tile(WS + 1, cap) and bulk_tile(WS + 3, cap)
        assert not bulk_tile(2, cap) and not bulk_tile(0, cap)
    assert not bulk_tile(WS + 1, (8, 9))
