"""Build the CUDA extensions for the GPU present, whatever architecture list the container sets, saying when a build runs or waits on a lock."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import shutil
import sys
import threading
from typing import Any

MIN_CAPABILITY = (8, 9)         # FP8 MMA and e4m3 conversions (Ada); kernels with clusters use them from 9.0
CLUSTERS = (9, 0)               # extensions built only on thread-block clusters (NVFP4) need Hopper or newer
# stop first: when the lock goes, a waiting start imports whatever module is there without building, even an old one
HINT = "if no other build is running, a killed build left it: stop this start, delete the lock and start again"
LOCK_WAIT_SECONDS = 60.0        # a start still waiting on the same lock this long says so again


def arch_flags(need: tuple[int, int] = MIN_CAPABILITY) -> list[str]:
    """nvcc flags for the current GPU alone; a GPU older than ``need`` is refused by name."""

    import torch

    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < need:
        why = "thread-block clusters" if need >= CLUSTERS else "FP8 MMA"
        raise RuntimeError(f"TensorFold's CUDA kernels need compute capability {need[0]}.{need[1]} or newer ({why}"
                           f"{' for these weights' if need > MIN_CAPABILITY else ''}); this GPU "
                           f"({torch.cuda.get_device_name()}) is {major}.{minor}")
    return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]


def refuse_old_gpu(need: tuple[int, int] = MIN_CAPABILITY) -> None:
    """At startup, before any weight loads: a GPU older than ``need`` is refused by name (no GPU: skipped)."""

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        try:
            arch_flags(need)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from None


def load(name: str, sources: str | list[str], need: tuple[int, int] = MIN_CAPABILITY, **kwargs: Any) -> Any:
    """torch's JIT ``load`` for this GPU only (NVIDIA's containers list every architecture back to sm_80), with a line when it compiles or waits on a lock."""

    from torch.utils import cpp_extension

    kwargs["extra_cuda_cflags"] = [*kwargs.get("extra_cuda_cflags", []), *arch_flags(need)]
    links = _toolkit()
    if links:
        kwargs["extra_ldflags"] = [*kwargs.get("extra_ldflags", []), *links]
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


def _toolkit() -> list[str]:
    """The pip route's tools: the venv's ninja on PATH, NVIDIA's pip toolkit, and the link flags it needs."""

    venv = Path(sys.executable).parent              # an inactive venv: ninja is beside its python, not on PATH
    if shutil.which("ninja") is None and (venv / "ninja").is_file():
        os.environ["PATH"] = f"{venv}{os.pathsep}{os.environ.get('PATH', '')}"
    return list(_pip_flags())


@lru_cache(maxsize=1)
def _pip_flags() -> tuple[str, ...]:
    """Once a process: NVIDIA's pip toolkit when torch found none (the module keeps it after the first call)."""

    import torch
    from torch.utils import cpp_extension

    return tuple(pip_toolkit(cpp_extension, torch))


def pip_toolkit(cpp_extension: Any, torch: Any) -> list[str]:
    """No toolkit found (CUDA_HOME, nvcc, /usr/local/cuda): NVIDIA's pip one beside torch, its cudart linked by name."""

    if cpp_extension.CUDA_HOME is not None or not getattr(torch.version, "cuda", None):
        return []
    home = Path(torch.__file__).resolve().parents[1] / "nvidia" / f"cu{torch.version.cuda.split('.')[0]}"
    if not (home / "bin" / "nvcc").is_file():
        return []
    cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"] = str(home)    # torch reads its CUDA_HOME at every build
    os.environ["PATH"] = f"{home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    _say(f"CUDA compiler: NVIDIA's pip toolkit for CUDA {torch.version.cuda} at {home}")
    versioned = sorted((home / "lib").glob("libcudart.so.*"))
    if (home / "lib" / "libcudart.so").exists() or not versioned:
        return []
    root = os.environ.get("TORCH_EXTENSIONS_DIR") or cpp_extension.get_default_build_root()
    links = Path(root) / "tensorfold_cudart"                         # -lcudart wants the bare name
    links.mkdir(parents=True, exist_ok=True)
    bare = links / "libcudart.so"
    if not bare.is_symlink() or bare.resolve() != versioned[0].resolve():
        bare.unlink(missing_ok=True)
        try:
            bare.symlink_to(versioned[0])
        except FileExistsError:                                      # another start made the same link first
            pass
    return [f"-L{links}"]


def _say(text: str) -> None:
    """One ``[tensorfold]`` line, flushed like the CLI's."""

    print(f"[tensorfold] {text}", flush=True)


__all__ = ["CLUSTERS", "MIN_CAPABILITY", "arch_flags", "load", "pip_toolkit"]
