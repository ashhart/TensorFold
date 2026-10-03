"""Flash Next's rank shares: whole quantization groups a rank, the lower ranks one block more."""

import pytest

from tensorfold.families.qwen4_exp.cuda.ranks import share


def test_width_shares_are_whole_groups_lower_ranks_first():
    assert [share(640, r, 4) for r in range(4)] == [(0, 192), (192, 384), (384, 512), (512, 640)]
    assert [share(640, r, 2) for r in range(2)] == [(0, 320), (320, 640)]
    assert [share(320, r, 4) for r in range(4)] == [(0, 128), (128, 192), (192, 256), (256, 320)]
    assert [share(512, r, 4, 32) for r in range(4)] == [(0, 128), (128, 256), (256, 384), (384, 512)]
    assert share(640, 0, 1) == (0, 640)


def test_a_width_that_leaves_a_rank_nothing_is_refused():
    with pytest.raises(ValueError, match="no 64-wide block"):
        share(192, 3, 4)
    with pytest.raises(ValueError, match="not whole 64-wide blocks"):
        share(100, 0, 2)
