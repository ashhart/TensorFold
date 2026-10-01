"""The NVFP4 / FP8 reader against an independent dequantizer (torch's float8 types and the e2m1 value list)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfold.cuda.nvfp4 import format as fmt

E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def _codes(rng, n, k):
    return rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)


@pytest.mark.torch
def test_nvfp4_matches_torch_float8_and_the_e2m1_list():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    n, k = 5, 64
    packed = _codes(rng, n, k)
    scale = rng.integers(0, 0x7F, size=(n, k // 16), dtype=np.uint8)          # finite positive e4m3
    got = fmt.dequant("nvfp4", packed, scale, 0.0625)
    bs = torch.from_numpy(scale).view(torch.float8_e4m3fn).float().numpy()
    want = np.empty((n, k), dtype=np.float32)
    for r in range(n):
        for c in range(k):
            byte = int(packed[r, c // 2])
            code = byte & 0xF if c % 2 == 0 else byte >> 4
            value = E2M1_VALUES[code & 7] * (-1 if code & 8 else 1)
            want[r, c] = value * bs[r, c // 16] * 0.0625
    assert np.array_equal(got, want)


@pytest.mark.torch
def test_fp8_and_mxfp8_match_torch_float8():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    w = rng.integers(0, 256, size=(4, 64), dtype=np.uint8)
    w[(w & 0x7F) == 0x7F] = 0x10                                                 # no NaN bytes
    e4 = torch.from_numpy(w).view(torch.float8_e4m3fn).float().numpy()
    assert np.array_equal(fmt.dequant("fp8", w, np.array([0.25], dtype=np.float32)), e4 * np.float32(0.25))
    s = rng.integers(100, 140, size=(4, 2), dtype=np.uint8)
    e8 = torch.from_numpy(s).view(torch.float8_e8m0fnu).float().numpy()
    assert np.array_equal(fmt.dequant("mxfp8", w, s), e4 * np.repeat(e8, 32, axis=1))


def test_compressed_tensors_global_scale_is_a_reciprocal():
    packed = np.array([[0x21, 0x07]], dtype=np.uint8)                          # codes 1, 2, 7, 0
    scale = np.full((1, 1), 0x38, dtype=np.uint8)                              # e4m3 1.0
    a = fmt.dequant("nvfp4", np.pad(packed, ((0, 0), (0, 6))), scale, 4.0, reciprocal_global=True)
    assert list(a[0, :4]) == [0.125, 0.25, 1.5, 0.0]


def test_schemes_from_tensor_storage():
    assert fmt.scheme({"weight": ("U8", [8, 32]), "weight_scale": ("F8_E4M3", [8, 4]),
                       "weight_scale_2": ("F32", [])}) == "nvfp4"
    assert fmt.scheme({"weight_packed": ("U8", [8, 32]), "weight_scale": ("F8_E4M3", [8, 4]),
                       "weight_global_scale": ("F32", [1])}) == "nvfp4"
    assert fmt.scheme({"weight": ("F8_E4M3", [8, 64]), "weight_scale": ("F32", [])}) == "fp8"
    assert fmt.scheme({"weight": ("F8_E4M3", [8, 64]), "weight_scale": ("U8", [8, 2])}) == "mxfp8"
    assert fmt.scheme({"weight": ("BF16", [8, 64])}) == "bf16"


def test_config_gate_takes_nvfp4_and_fp8_and_refuses_integer_weights():
    kw = dict(where="CUDA", tested="x", help="")
    ok = {"quantization_config": {"quant_method": "modelopt", "config_groups": {
        "a": {"weights": {"num_bits": 8, "type": "float"}}, "b": {"weights": {"num_bits": 4, "type": "float",
                                                                               "group_size": 16}}}}}
    fmt.require_config(ok, **kw)
    assert fmt.config_block(ok)["quant_method"] == "modelopt"
    ct = {"quantization_config": {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
                                  "config_groups": {"g": {"weights": {"num_bits": 4, "type": "float"}}}}}
    fmt.require_config(ct, **kw)
    awq = {"quantization_config": {"quant_method": "compressed-tensors", "format": "pack-quantized",
                                   "config_groups": {"g": {"weights": {"num_bits": 4, "type": "int"}}}}}
    with pytest.raises(ValueError, match="4-bit int"):
        fmt.require_config(awq, **kw)


@pytest.mark.torch
def test_block_fp8_matches_torch_float8_and_its_block_scales():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    w = rng.integers(0, 256, size=(200, 256), dtype=np.uint8)
    w[(w & 0x7F) == 0x7F] = 0x10
    s = rng.random((2, 2)).astype(np.float32) + 0.5                                # [ceil(200/128), 256/128]
    e4 = torch.from_numpy(w).view(torch.float8_e4m3fn).float().numpy()
    want = e4 * np.repeat(np.repeat(s, 128, axis=0)[:200], 128, axis=1)
    assert np.array_equal(fmt.dequant("fp8block", w, s), want)


def test_block_fp8_scheme_from_tensor_storage():
    assert fmt.scheme({"weight": ("F8_E4M3", [200, 256]), "weight_scale_inv": ("F32", [2, 2])}) == "fp8block"
    assert fmt.scheme({"weight": ("F8_E4M3", [200, 256]), "weight_scale": ("F32", [])}) == "fp8"
