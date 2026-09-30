"""Minimal quant checks: real stored layout against independent pinned C vectors.

All 16 fixed blocks still cover zero/negative/subnormal scales, int8 extremes,
Q2 subblocks and min offsets. Expected values come from oracle.c, not this
Python decoder. Exhaustive mutation campaigns and duplicated semantic checks
are deliberately omitted. These are CPU fixture checks, not GPU qualification.
"""
import json
from pathlib import Path
import numpy as np
import pytest

QUANTS = Path(__file__).parent / "fixtures/deepseek_v4/quants"


def half(block, offset):
    return np.frombuffer(block, dtype="<f2", count=1, offset=offset)[0].astype(np.float32)


def decode_q8_0(block):
    assert len(block) == 34
    return np.frombuffer(block, dtype=np.int8, count=32, offset=2).astype(np.float32) * half(block, 0)


def decode_q2_k(block):
    assert len(block) == 84
    scales = block[:16]
    qs = np.frombuffer(block, dtype=np.uint8, count=64, offset=16)
    d, dmin = half(block, 80), half(block, 82)
    out = np.empty(256, dtype=np.float32)
    for h in range(2):
        for group in range(8):
            sc = scales[h * 8 + group]
            dl, dm = d * np.float32(sc & 15), dmin * np.float32(sc >> 4)
            start = h * 32 + (group & 1) * 16
            codes = (qs[start:start + 16] >> (2 * (group // 2))) & 3
            out[h * 128 + group * 16:h * 128 + (group + 1) * 16] = dl * codes.astype(np.float32) - dm
    return out


@pytest.mark.parametrize("name,decode", [("q8_0", decode_q8_0), ("q2_k", decode_q2_k)])
def test_pinned_blocks_decode_exactly(name, decode):
    for case in json.loads((QUANTS / f"{name}.json").read_text())["cases"]:
        got = decode(bytes.fromhex(case["block"]))
        want = np.asarray(case["expected"], dtype=np.float32)
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32), err_msg=case["id"])


@pytest.mark.parametrize("name,decode,case_id,offset", [
    ("q8_0", decode_q8_0, "q8_ramp_extremes", 2),
    ("q2_k", decode_q2_k, "q2_varied_nibbles", 16),
])
def test_nonzero_payload_corruption_is_detected(name, decode, case_id, offset):
    cases = json.loads((QUANTS / f"{name}.json").read_text())["cases"]
    case = next(c for c in cases if c["id"] == case_id)
    block = bytearray.fromhex(case["block"])
    block[offset] ^= 1
    want = np.asarray(case["expected"], dtype=np.float32)
    assert not np.array_equal(decode(bytes(block)), want)
