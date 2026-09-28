"""Build the CUDA extensions for the GPU this process runs on, whatever architecture list the container sets."""

from __future__ import annotations

from typing import Any

MIN_CAPABILITY = (9, 0)         # the kernels use thread-block clusters and FP8 MMA


def arch_flags() -> list[str]:
    """nvcc flags for the current GPU alone; a GPU older than the kernels need is refused by name."""

    import torch

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < MIN_CAPABILITY:
        raise RuntimeError(f"TensorFold's CUDA kernels need compute capability {MIN_CAPABILITY[0]}.{MIN_CAPABILITY[1]} "
                           f"or newer (thread-block clusters and FP8 MMA); this GPU ({torch.cuda.get_device_name()}) "
                           f"is {major}.{minor}")
    return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]


def load(**kwargs: Any) -> Any:
    """torch's JIT ``load`` built for this GPU only: NVIDIA's containers list every architecture back to sm_80."""

    from torch.utils.cpp_extension import load as torch_load

    kwargs["extra_cuda_cflags"] = [*kwargs.get("extra_cuda_cflags", []), *arch_flags()]
    return torch_load(**kwargs)


__all__ = ["MIN_CAPABILITY", "arch_flags", "load"]
