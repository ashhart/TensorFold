"""The checkpoint lane matmul's block by rows, width, K slices and SM count (host; blocks never change bits)."""

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.nvfp4.checkpoint import FILL, WIDE, lane_tile  # noqa: E402

# the 27B's projections: (n, K slices) as qmm.split_k gives them
SHAPES = {"qkv": (10240, 2), "z": (6144, 2), "out": (5120, 4), "q": (12288, 1), "k": (1024, 8), "gate": (17408, 1),
          "down": (5120, 4), "head": (248320, 1)}
BUILT = {16, 32, 64, 64128, 128064, 128128}


def test_small_row_counts_keep_the_64_wide_tiles():
    for n, sk in SHAPES.values():
        for sms in (48, 188, 1):
            assert [lane_tile(m, n, sk, sms) for m in (1, 16, 17, 32)] == [16, 16, 32, 32]


def test_wide_blocks_where_they_fill_the_gpu():
    for name, (n, sk) in SHAPES.items():
        if name != "k":
            assert lane_tile(33, n, sk, 188) == lane_tile(64, n, sk, 188) == 64128, name
            assert lane_tile(65, n, sk, 188) == lane_tile(128, n, sk, 188) == 128128, name
    n, sk = SHAPES["k"]                                      # 8 column blocks x 8 slices: too few on 188 SMs
    assert lane_tile(64, n, sk, 188) == 64 and lane_tile(128, n, sk, 188) == 128064
    assert lane_tile(64, n, sk, 96) == 64128 and lane_tile(128, n, sk, 96) == 128128


def test_under_wide_sms_64_by_64_blocks_past_32_rows():
    for n, sk in SHAPES.values():                            # a GB10 has 48 SMs
        assert {lane_tile(m, n, sk, sms) for m in range(33, 257) for sms in (48, WIDE - 1)} == {64}


def test_any_sm_count_and_row_count_gets_a_built_tile():
    for n, sk in SHAPES.values():
        for sms in (1, 20, 48, 84, 95, 96, 128, 132, 170, 188, 1000):
            for m in range(1, 513):
                tile = lane_tile(m, n, sk, sms)
                assert tile in BUILT
                rows = 128 if tile in (128064, 128128) else 64 if tile in (64, 64128) else tile
                assert rows >= min(m, 128 if sms >= WIDE else 64)    # one row tile up to its height, then side by side
                if tile in (64128, 128128):
                    assert -(-m // rows) * -(-n // 128) * sk * FILL >= sms
