"""Concurrent Flash Next sizes the prompt pass a round carries so the round's decoding keeps its share of the time."""

import importlib

import pytest

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)

pytestmark = pytest.mark.torch


def test_a_rounds_pass_keeps_decoding_its_share(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.prefill_rows, dec.share, dec.round_s, dec.row_s = 2048, 0.25, None, None
    assert dec._pass_rows() == 2048                                   # nothing timed yet: whole passes
    dec._timed(0.1, 0)                                                # a round alone: 0.1 s
    dec._timed(0.1 + 512 * 4e-4, 512)
    assert dec.round_s == pytest.approx(0.1) and dec.row_s == pytest.approx(4e-4)
    assert dec._pass_rows() == 960                                    # 0.1 s is a quarter of 960 rows' 0.38 s
    dec._timed(0.5, 0)                                                # busier rounds: whole passes again
    assert dec._pass_rows() == 2048
    dec.round_s, dec.row_s = 0.001, 1e-3
    assert dec._pass_rows() == multi.PASS_MIN                         # never below the floor
    dec.share = 0.0
    assert dec._pass_rows() == 2048                                   # share 0: whole passes
