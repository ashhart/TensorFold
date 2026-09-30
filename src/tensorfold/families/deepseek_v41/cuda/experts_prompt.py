"""The prompt-chunk grouped expert kernel (CUDA): one trellis decode shared by up to 64 member rows."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def ext():
    from tensorfold.cuda import exl3
    from tensorfold.cuda.build import load

    inc = str(Path(exl3.__file__).parent)
    return load("tensorfold_dsv41_experts_prompt_v1", [str(Path(__file__).with_name("experts_prompt.cu"))],
                extra_include_paths=[inc], extra_cuda_cflags=["-O3", "-lineinfo"])
