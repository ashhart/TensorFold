"""The CUDA chunk pass of decode attention over the NVFP4 compressed KV cache (mqa_fp4.cu; adapted from FlashInfer's
Cake DeepSeek-V4.1 decode, Apache-2.0: THIRD_PARTY_NOTICES.md). kernels.mqa uses it for rows <= FULL_ROWS when
``kernels.CUDA_MQA`` is set (serial: fp4 KV and TF_DSV41_CUDA_MQA=1); the Triton merge finishes the rows.

Splits: ``SPLIT`` compressed candidates each (in idx order), then the window in splits of 32 positions. The split
partition depends only on the layer's index count, so a row's result never depends on the other rows of a call."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

SPLIT = int(os.environ.get("TF_DSV41_MQA_SPLIT") or 64)      # compressed candidates a split (32 or 64), fixed a process
WSPLIT = 32                                                  # window positions a split
TILES = int(os.environ.get("TF_DSV41_MQA_TILES") or 1)       # 16-head tiles a CTA (1: twice the CTAs, gathers twice)
if SPLIT not in (32, 64) or TILES not in (1, 2):
    raise ValueError(f"TF_DSV41_MQA_SPLIT={SPLIT}: 32 or 64; TF_DSV41_MQA_TILES={TILES}: 1 or 2")


def nsplit(n_idx: int, window: int = 128) -> int:
    return -(-n_idx // SPLIT) + window // WSPLIT


@lru_cache(maxsize=2)
def ext(lut: bool = False):
    """The extension (None off sm_12x: Triton runs); ``lut``: a build with the exact byte tables instead of the
    CUDA 13.2+ cvt instructions (tests compare both)."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        return None
    from tensorfold.cuda.build import load

    flags = ["-O3", "-lineinfo"] + (["-DTF_MQA4_LUT"] if lut else [])
    return load("tf_dsv41_mqa_fp4_lut_v3" if lut else "tf_dsv41_mqa_fp4_v3",
                [str(Path(__file__).with_name("mqa_fp4.cu"))], arch_specific=True, extra_cuda_cflags=flags)
