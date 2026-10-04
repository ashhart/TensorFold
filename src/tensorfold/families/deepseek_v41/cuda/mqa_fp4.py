"""Decode attention over the NVFP4 compressed KV cache in CUDA (mqa_fp4.cu; adapted from FlashInfer's Cake DeepSeek-V4.1
decode, Apache-2.0: THIRD_PARTY_NOTICES.md). kernels.mqa uses it for rows <= FULL_ROWS when ``kernels.CUDA_MQA`` is
set (serial: fp4 KV and TF_DSV41_CUDA_MQA=1).

A row's candidates come in stages of 32: the compressed entries (idx order), then the window. Its result is one fixed
reduction tree: each stage attended alone, the stages of each group of ``GROUP`` folded left to right, the groups
folded onto the sink. Up to ``FEW`` rows a call, every stage runs in its own CTA and the merge folds everything (the
most parallel: a decode row); above, a CTA folds a whole group (few partials in memory: 32 rows). The arithmetic is
the same either way, so a row's result never depends on the other rows of a call."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

GROUP = int(os.environ.get("TF_DSV41_MQA_GROUP") or 5)      # stages a group: the tree (fixed for the process)
FEW = int(os.environ.get("TF_DSV41_MQA_FEW") or 1)          # rows a call up to which every stage gets its own CTA


def stages(n_idx: int, window: int = 128) -> int:
    return -(-n_idx // 32) + window // 32


def parts(rows: int, n_idx: int, window: int = 128) -> tuple[int, int]:
    """(stages a CTA, partials a row) for a call of ``rows`` rows."""

    per = 1 if rows <= FEW else GROUP
    return per, -(-stages(n_idx, window) // per)


def scratch_rows(rows: int, n_idx: int) -> int:
    """Partials (of [32, 512] fp32) the calls of up to ``rows`` rows need at most."""

    return max(r * parts(r, n_idx)[1] for r in {min(rows, FEW), rows})


@lru_cache(maxsize=2)
def ext(lut: bool = False):
    """The extension (None off sm_12x: Triton runs); ``lut``: a build with the exact byte tables instead of the
    CUDA 13.2+ cvt instructions (tests compare both)."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        return None
    from tensorfold.cuda.build import load

    flags = ["-O3", "-lineinfo"] + (["-DTF_MQA4_LUT"] if lut else [])
    return load("tf_dsv41_mqa_fp4_lut_v6" if lut else "tf_dsv41_mqa_fp4_v6",
                [str(Path(__file__).with_name("mqa_fp4.cu"))], arch_specific=True, extra_cuda_cflags=flags)
