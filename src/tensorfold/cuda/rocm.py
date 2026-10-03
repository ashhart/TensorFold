"""ROCm (HIP) builds of PyTorch run the CUDA backend on AMD GPUs; these switches keep them on portable kernels."""

from __future__ import annotations

import torch

HIP = bool(getattr(torch.version, "hip", None))     # a ROCm build: torch.cuda is the HIP device


# nvcc flags and their hipcc meaning; None drops a flag with no HIP counterpart
NVCC_TO_HIP = {
    "--fmad=false": "-ffp-contract=off",      # no fused multiply-add contraction: the kernels' bits depend on it
    "--expt-relaxed-constexpr": None,         # HIP-Clang allows constexpr host functions in device code already
    "-lineinfo": None,
}


def hip_flags(flags: list[str]) -> list[str]:
    """``flags`` written for nvcc, as hipcc takes them."""

    out = []
    for flag in flags:
        mapped = NVCC_TO_HIP.get(flag, flag)
        if mapped is not None:
            out.append(mapped)
    return out


def use_pip_sdk() -> None:
    """Build with the ROCm SDK installed next to this torch (``rocm[devel]``), not an older system ROCm.

    torch's cpp_extension reads ROCM_HOME when it is first imported, so this runs before that import."""

    import importlib.util
    import os

    if os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH"):
        return
    spec = importlib.util.find_spec("_rocm_sdk_devel")
    if spec is None or not spec.submodule_search_locations:
        return
    root = list(spec.submodule_search_locations)[0]
    if os.path.exists(os.path.join(root, "bin", "hipcc")):
        os.environ["ROCM_HOME"] = os.environ["ROCM_PATH"] = root
        os.environ["PATH"] = os.path.join(root, "bin") + os.pathsep + os.environ.get("PATH", "")


def offload_arch() -> str:
    """The present AMD GPU's target, e.g. ``gfx1151`` (feature suffixes such as ``:xnack-`` dropped)."""

    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


__all__ = ["HIP", "NVCC_TO_HIP", "hip_flags", "offload_arch", "use_pip_sdk"]
