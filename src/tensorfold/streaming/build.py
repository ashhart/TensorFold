"""Build the host-sync extension on first use, against the running Python and MLX, into TensorFold's cache."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).resolve().parent / "hostsync"
NANOBIND = "2.15.0"          # MLX's own nanobind: another version cannot take mlx.core.array


def _key() -> str:
    import mlx.core as mx

    digest = hashlib.sha256()
    for path in sorted(p for p in SOURCE.glob("*") if p.suffix in (".cpp", ".txt")):
        digest.update(path.name.encode() + path.read_bytes())
    digest.update(f"{mx.__version__}|{sys.version}|{NANOBIND}".encode())
    return digest.hexdigest()[:16]


def load(cache: Path | None = None) -> Any:
    """The compiled ``_hostsync`` module, building it with cmake and nanobind when this MLX has none yet."""

    folder = (cache or Path.home() / ".cache" / "tensorfold" / "ext") / f"hostsync-{_key()}"
    built = next(iter(folder.glob("_hostsync*.so")), None) if folder.is_dir() else None
    if built is None:
        _build(folder)
        built = next(iter(folder.glob("_hostsync*.so")))
    spec = importlib.util.spec_from_file_location("_hostsync", built)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(folder: Path) -> None:
    try:
        import nanobind
    except ImportError:
        nanobind = None
    cmake = shutil.which("cmake") or shutil.which("cmake", path=str(Path(sys.executable).parent))
    if nanobind is None or nanobind.__version__ != NANOBIND or cmake is None:
        raise RuntimeError(f"SSD expert streaming builds a small MLX extension on first use: it needs cmake and "
                           f"nanobind {NANOBIND} (`pip install \"tensorfold[ssd]\"`) and the Xcode command line "
                           "tools (`xcode-select --install`)")
    work = folder.with_name(folder.name + ".build")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    run = dict(cwd=work, check=True, capture_output=True, text=True)
    try:
        subprocess.run([cmake, str(SOURCE), f"-DPython_EXECUTABLE={sys.executable}", "-DCMAKE_BUILD_TYPE=Release"],
                       **run)
        subprocess.run([cmake, "--build", ".", "-j8"], **run)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"building the host-sync extension failed:\n{error.stdout[-2000:]}{error.stderr[-2000:]}") from None
    folder.mkdir(parents=True, exist_ok=True)
    for so in work.glob("_hostsync*.so"):
        shutil.copy2(so, folder / so.name)
    shutil.rmtree(work, ignore_errors=True)


__all__ = ["NANOBIND", "load"]
