"""The grouped 4-bit lane matmul's block by rows and chip (host): wide blocks on SM 12.0 from 96 SMs only."""

from tensorfold.cuda.kernels.qmm_tiles import WIDE_SMS, group_tile


def test_other_gpus_keep_the_kernels_own_pick():
    for chip in ((12, 1, 48), (12, 0, WIDE_SMS - 1), (12, 0, 84), (8, 9, 128), (9, 0, 132), (10, 0, 148)):
        assert {group_tile(m, *chip) for m in range(1, 513)} == {0}, chip


def test_a_wide_sm_12_0_gpu_takes_the_wide_blocks_past_16_rows():
    for sms in (WIDE_SMS, 170, 188):
        tiles = [group_tile(m, 12, 0, sms) for m in range(1, 257)]
        assert set(tiles[:16]) == {0}                                     # 1-16 rows: today's tiles
        assert set(tiles[16:32]) == {12} and set(tiles[32:64]) == {11}    # 64 x 128 on 2 x 4, then on 1 x 8 warps
        assert set(tiles[64:96]) == {10} and set(tiles[96:]) == {11}      # 128 x 128, then 64-row blocks again
