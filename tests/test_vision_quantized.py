"""Contratti per la conversione delle proiezioni visuali Qwen."""

import numpy as np
import pytest

from tensorfold.vision.qwen_quantized import dequantize_linear


def test_dequantizes_modelopt_nvfp4_using_direct_scale2():
    weight = np.full((1, 8), 0x10, dtype=np.uint8)
    scale = np.full((1, 1), 0x38, dtype=np.uint8)

    result = dequantize_linear(weight, scale, 2.0, scheme="nvfp4", group_size=16)

    np.testing.assert_array_equal(result, np.tile([0.0, 1.0], 8).reshape(1, 16))
    assert result.dtype == np.float32


def test_dequantizes_modelopt_mxfp8_by_groups_of_32():
    weight = np.resize(np.array([0x38, 0xB8], dtype=np.uint8), (2, 32))
    scale = np.full((2, 1), 127, dtype=np.uint8)

    result = dequantize_linear(weight, scale, scheme="mxfp8", group_size=32)

    np.testing.assert_array_equal(result, np.resize(np.array([1.0, -1.0], dtype=np.float32), (2, 32)))


def test_rejects_invalid_shapes_groups_and_schemes():
    invalid = [
        (np.zeros((1, 7), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), 1.0, "nvfp4", 16),
        (np.zeros((1, 8), dtype=np.uint8), np.zeros((1, 2), dtype=np.uint8), 1.0, "nvfp4", 16),
        (np.zeros((1, 32), dtype=np.uint8), np.zeros((1, 2), dtype=np.uint8), None, "mxfp8", 32),
        (np.zeros((1, 32), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), None, "mxfp8", 16),
        (np.zeros((1, 32), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8), None, "fp8", 32),
    ]
    for weight, scale, scale_2, scheme, group_size in invalid:
        with pytest.raises(ValueError):
            dequantize_linear(weight, scale, scale_2, scheme=scheme, group_size=group_size)


def test_rejects_nonfinite_scales_and_wrong_array_dtypes():
    weight = np.full((1, 8), 0x10, dtype=np.uint8)
    scale = np.full((1, 1), 0x7F, dtype=np.uint8)
    with pytest.raises(ValueError):
        dequantize_linear(weight, scale, 1.0, scheme="nvfp4", group_size=16)

    with pytest.raises(ValueError):
        dequantize_linear(weight.astype(np.int16), np.zeros((1, 1), dtype=np.uint8), 1.0,
                          scheme="nvfp4", group_size=16)
