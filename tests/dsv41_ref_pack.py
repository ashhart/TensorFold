"""The pack's tensors on the CPU for the fp32 reference: read by name through the index, dense formats to fp32."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from tensorfold.families.deepseek_v41.cuda.convert import BLOCK, fp8_block_rows

FLOATS = (torch.bfloat16, torch.float16, torch.float32)
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
POW2 = torch.tensor([math.ldexp(1.0, u - 127) for u in range(255)] + [math.nan])   # E8M0 byte -> 2^(u-127), 0xFF NaN


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _scaled(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 [n, K] times 2^(e - 127) per 1x32 input block (e: E8M0 bytes [n, K/32]); every product is exact."""

    n, k = values.shape
    if k % BLOCK or tuple(scale.shape) != (n, k // BLOCK):
        raise ValueError(f"values {tuple(values.shape)} do not match 1x{BLOCK} scales {tuple(scale.shape)}")
    return (values.view(n, -1, BLOCK) * POW2[scale.long()][:, :, None]).view(n, k)


class RefPack:
    """One checkpoint directory's tensors by name, each read alone from its shard (symlinked shards resolved)."""

    def __init__(self, model_dir) -> None:
        self.dir = Path(model_dir)
        self.where = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]

    def has(self, name: str) -> bool:
        return name in self.where

    def path(self, name: str) -> Path:
        return (self.dir / self.where[name]).resolve()

    def shape(self, name: str) -> tuple[int, ...]:
        with safe_open(str(self.path(name)), framework="pt", device="cpu") as f:
            return tuple(f.get_slice(name).get_shape())

    def tensor(self, name: str, rows: slice | None = None) -> torch.Tensor:
        """The tensor as stored (dtype and shape), or only ``rows`` of its first dimension."""

        with safe_open(str(self.path(name)), framework="pt", device="cpu") as f:
            return f.get_tensor(name) if rows is None else f.get_slice(name)[rows]

    def dense_fp32(self, name: str, rows: slice | None = None) -> torch.Tensor:
        """Module ``name``'s weight (or tensor ``name``) as fp32 [out, in]: FP8 32x32, MXFP8, MXFP4 or plain float.

        The format follows the stored dtypes and the scale's shape: e4m3 with ``[ceil(N/32), K/32]`` scales is a
        32x32 block, e4m3 with ``[N, K/32]`` is MXFP8, packed e2m1 (I8) with ``[N, K/32]`` is MXFP4 (low nibble first).
        """

        key = f"{name}.weight" if f"{name}.weight" in self.where else name
        scale = next((f"{name}.{s}" for s in ("scale", "weight_scale") if f"{name}.{s}" in self.where), None)
        w = self.tensor(key, rows)
        if scale is None:
            if w.dtype not in FLOATS:
                raise ValueError(f"{key}: {w.dtype} has no scale tensor")
            return w.to(torch.float32)
        n = self.shape(key)[0]
        take = slice(*(rows or slice(None)).indices(n))
        if take.step != 1:
            raise ValueError(f"{key}: rows {rows} must be contiguous")
        if w.dtype == torch.float8_e4m3fn:
            k = w.shape[1]
            if self.shape(scale) == (n, -(-k // BLOCK)):
                return _scaled(w.to(torch.float32), _bytes(self.tensor(scale, take)))
            return _scaled(w.to(torch.float32), fp8_block_rows(self.tensor(scale), n)[take])
        if w.dtype in (torch.int8, torch.uint8):
            b = _bytes(w).long()
            codes = torch.stack([b & 0xF, b >> 4], dim=-1).view(b.shape[0], -1)
            return _scaled(E2M1[codes], _bytes(self.tensor(scale, take)))
        raise ValueError(f"{key}: no dequantization for {w.dtype} with {scale}")
