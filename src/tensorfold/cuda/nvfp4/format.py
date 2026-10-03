"""NVFP4 / FP8 / MXFP8 checkpoints (ModelOpt, compressed-tensors): each linear's scheme, and a numpy dequantizer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METHODS = ("modelopt", "compressed-tensors")
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                dtype=np.float32)
SCHEMES = ("nvfp4", "fp8", "fp8block", "mxfp8", "bf16")


def config_block(config: dict) -> dict | None:
    """The quantization block of config.json (top level or text config) when it names ModelOpt or compressed-tensors."""

    for source in (config, config.get("text_config") or {}):
        found = source.get("quantization_config")
        if isinstance(found, dict) and str(found.get("quant_method", "")).lower() in METHODS:
            return found
    return None


def require_config(config: dict, *, where: str, tested: str, help: str) -> None:
    """Refuse a ModelOpt / compressed-tensors checkpoint whose weights are not NVFP4, FP8 or MXFP8."""

    block = config_block(config) or {}
    method = str(block.get("quant_method", "")).lower()
    formats = {str(block.get("format", "") or "")}
    groups = block.get("config_groups") or {}
    for group in groups.values():
        weights = group.get("weights") or {}
        formats.add(str(group.get("format", "") or ""))
        bits, kind = weights.get("num_bits"), str(weights.get("type", "float"))
        if kind != "float" or bits not in (4, 8):
            raise ValueError(f"this {method} checkpoint stores {bits}-bit {kind} weights; {where} reads NVFP4 and FP8 "
                             f"weights of this format. Tested checkpoints: {tested}. {help}")
    bad = {f for f in formats if f and f not in ("nvfp4-pack-quantized", "float-quantized", "naive-quantized",
                                                 "mixed-precision")}
    if bad:
        raise ValueError(f"this {method} checkpoint's weights are stored as {', '.join(sorted(bad))}; {where} reads "
                         f"NVFP4 and FP8 weights. Tested checkpoints: {tested}. {help}")


def is_quantized(model_dir: str | Path) -> bool:
    """Whether config.json names ModelOpt or compressed-tensors quantization."""

    config = Path(model_dir) / "config.json"
    return config.is_file() and config_block(json.loads(config.read_text())) is not None


def scheme(tensors: dict[str, tuple[str, list[int]]]) -> str:
    """One linear's scheme from its tensors' (dtype, shape) by suffix: nvfp4, fp8, mxfp8 or bf16."""

    w = tensors.get("weight") or tensors.get("weight_packed")
    s = tensors.get("weight_scale")
    if w is None:
        raise ValueError("a linear without weight or weight_packed")
    if w[0] == "U8" and s is not None and s[0] == "F8_E4M3":
        return "nvfp4"
    if w[0] == "F8_E4M3" and s is not None and s[0] == "U8":
        return "mxfp8"
    si = tensors.get("weight_scale_inv")
    if w[0] == "F8_E4M3" and si is not None and si[0] == "F32" and len(si[1]) == 2:
        return "fp8block"                      # ModelOpt FP8_PB_WO / DeepSeek: an fp32 scale per 128x128 block
    if w[0] == "F8_E4M3":
        return "fp8"
    if w[0] in ("BF16", "F16", "F32"):
        return "bf16"
    raise ValueError(f"unknown weight storage {w[0]} with scale {s[0] if s else None}")


def e4m3(bits: np.ndarray) -> np.ndarray:
    """e4m3fn bytes as float32 (no infinities; 0x7f and 0xff are NaN)."""

    b = bits.astype(np.int32)
    sign = np.where(b & 0x80, -1.0, 1.0).astype(np.float32)
    exp, man = (b >> 3) & 0xF, b & 0x7
    value = np.where(exp == 0, man / 8.0 * 2.0 ** -6, (1 + man / 8.0) * 2.0 ** (exp - 7.0)).astype(np.float32)
    return np.where((b & 0x7F) == 0x7F, np.float32(np.nan), sign * value)


def e8m0(bits: np.ndarray) -> np.ndarray:
    """e8m0 exponent bytes as float32 powers of two (255 is NaN)."""

    b = bits.astype(np.int32)
    return np.where(b == 255, np.float32(np.nan), np.ldexp(np.float32(1.0), b - 127).astype(np.float32))


def dequant(scheme_name: str, weight: np.ndarray, scale: np.ndarray | None = None, global_scale: float = 1.0,
            *, reciprocal_global: bool = False) -> np.ndarray:
    """[N, K] float32 of one stored weight: e2m1 pairs with e4m3 per 16 x global, e4m3 x scale, or e4m3 x e8m0 per 32."""

    if scheme_name == "nvfp4":
        n, half = weight.shape
        codes = np.empty((n, 2 * half), dtype=np.uint8)
        codes[:, 0::2], codes[:, 1::2] = weight & 0xF, weight >> 4
        g = 1.0 / global_scale if reciprocal_global else global_scale
        return E2M1[codes] * np.repeat(e4m3(scale), 16, axis=1) * np.float32(g)
    if scheme_name == "fp8":
        return e4m3(weight) * np.float32(np.asarray(scale, dtype=np.float32).reshape(-1)[0])
    if scheme_name == "mxfp8":
        return e4m3(weight) * np.repeat(e8m0(scale), 32, axis=1)
    if scheme_name == "fp8block":              # ``scale`` = weight_scale_inv [ceil(N/128), K/128]
        n, k = weight.shape
        s = np.asarray(scale, dtype=np.float32)
        return e4m3(weight) * np.repeat(np.repeat(s, 128, axis=0)[:n], k // s.shape[1], axis=1)
    return weight.astype(np.float32)
