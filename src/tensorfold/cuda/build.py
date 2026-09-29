"""Build the CUDA extensions for the GPU present, whatever architecture list the container sets, saying when a build runs or waits on a lock."""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any

MIN_CAPABILITY = (9, 0)         # the kernels use thread-block clusters and FP8 MMA
# stop first: when the lock goes, a waiting start imports whatever module is there without building, even an old one
HINT = "if no other build is running, a killed build left it: stop this start, delete the lock and start again"
LOCK_WAIT_SECONDS = 60.0        # a start still waiting on the same lock this long says so again

# nvcc flags and their clang (hipcc) counterparts; None drops a flag that has no HIP meaning
_HIP_FLAGS = {"--fmad=false": "-ffp-contract=off", "--expt-relaxed-constexpr": None, "-lineinfo": None}


@lru_cache(maxsize=1)
def hip() -> bool:
    """Whether torch runs on ROCm: its ``torch.cuda`` then drives an AMD GPU and extensions build with hipcc."""

    try:
        import torch
    except ImportError:
        return False
    return getattr(torch.version, "hip", None) is not None


def hip_arch() -> str:
    """The AMD GPU's architecture as hipcc names it, e.g. ``gfx1201`` (feature suffixes dropped); without a visible
    GPU, the first of ``PYTORCH_ROCM_ARCH`` (a build-only host)."""

    import torch

    if not torch.cuda.is_available() and os.environ.get("PYTORCH_ROCM_ARCH"):
        return os.environ["PYTORCH_ROCM_ARCH"].replace(",", ";").split(";")[0].strip()
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def arch_flags() -> list[str]:
    """Compiler flags for the current GPU alone; an NVIDIA GPU older than the kernels need is refused by name."""

    import torch

    if hip():
        return [f"--offload-arch={hip_arch()}"]
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < MIN_CAPABILITY:
        raise RuntimeError(f"TensorFold's CUDA kernels need compute capability {MIN_CAPABILITY[0]}.{MIN_CAPABILITY[1]} "
                           f"or newer (thread-block clusters and FP8 MMA); this GPU ({torch.cuda.get_device_name()}) "
                           f"is {major}.{minor}")
    return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]


def device_flags(flags: list[str]) -> list[str]:
    """``flags`` as the device compiler takes them: nvcc's as given, or their hipcc equivalents on ROCm."""

    if not hip():
        return list(flags)
    return [_HIP_FLAGS.get(f, f) for f in flags if _HIP_FLAGS.get(f, f) is not None]


def load(name: str, sources: str | list[str], **kwargs: Any) -> Any:
    """torch's JIT ``load`` for this GPU only (NVIDIA's containers list every architecture back to sm_80), with a line when it compiles or waits on a lock."""

    from torch.utils import cpp_extension

    kwargs["extra_cuda_cflags"] = [*device_flags(kwargs.get("extra_cuda_cflags", [])), *arch_flags()]
    held = _announce(cpp_extension, name, sources, kwargs.get("build_directory"))
    timer = None
    if held is not None:
        timer = threading.Timer(LOCK_WAIT_SECONDS, _still_waiting, held)
        timer.daemon = True
        timer.start()
    try:
        return cpp_extension.load(name=name, sources=sources, **kwargs)
    finally:
        if timer is not None:
            timer.cancel()


def _announce(cpp_extension: Any, name: str, sources: str | list[str],
              directory: str | None) -> tuple[str, tuple[int, int]] | None:
    """Say whether ``name`` builds or waits on a lock; return (lock path, (inode, mtime)) when it waits."""

    try:
        directory = directory or cpp_extension._get_build_directory(name, verbose=False)   # private in torch
    except Exception:  # noqa: BLE001 - no lookup in this torch: build without the lines
        return None
    lock = os.path.join(directory, "lock")          # torch's FileBaton for this extension
    try:
        seen = os.stat(lock)
    except OSError:
        seen = None
    if seen is not None:
        _say(f"CUDA extension {name} waits on the build lock {lock}; {HINT}")
        return lock, (seen.st_ino, seen.st_mtime_ns)
    if _needs_build(os.path.join(directory, name + getattr(cpp_extension, "LIB_EXT", ".so")), sources):
        _say(f"building CUDA extension {name} (first start after an install or update; later starts reuse it)")
    return None


def _needs_build(module: str, sources: str | list[str]) -> bool:
    """No built module, or a source newer than it (ninja then compiles again)."""

    try:
        built = os.path.getmtime(module)
    except OSError:
        return True
    for source in [sources] if isinstance(sources, str) else sources:
        try:
            if os.path.getmtime(source) > built:
                return True
        except OSError:
            continue
    return False


def _still_waiting(lock: str, identity: tuple[int, int]) -> None:
    """Repeat the hint while the lock seen before the build is still the one there."""

    try:
        now = os.stat(lock)
    except OSError:
        return
    if (now.st_ino, now.st_mtime_ns) == identity:
        _say(f"still waiting after {LOCK_WAIT_SECONDS:g} s on the build lock {lock}; {HINT}")


def _say(text: str) -> None:
    """One ``[tensorfold]`` line, flushed like the CLI's."""

    print(f"[tensorfold] {text}", flush=True)


__all__ = ["MIN_CAPABILITY", "arch_flags", "device_flags", "hip", "hip_arch", "load"]
