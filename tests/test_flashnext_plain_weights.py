"""The Flash Next loader refuses quantized bytes where it reads bf16 values, instead of casting them to garbage."""

import pytest

pytestmark = pytest.mark.torch


def test_real_values_pass_and_quantized_bytes_are_refused():
    import torch

    from tensorfold.families.qwen4_exp.cuda.weights import _plain

    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        w = torch.ones(2, 3, dtype=dtype)
        assert _plain("x.weight", w) is w
    for dtype in (torch.uint8, torch.int8, torch.float8_e4m3fn):
        with pytest.raises(ValueError, match="x.weight: .* without a scale"):
            _plain("x.weight", torch.zeros(2, 3, dtype=dtype))
