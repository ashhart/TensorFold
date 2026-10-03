"""Every CUDA source and data file under src ships in the wheel: pyproject's package-data names it (#66)."""

import fnmatch
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = {".cu", ".cuh", ".cpp", ".hip", ".hpp", ".json", ".txt"}   # what the engines read or build at run time


def hipified(path: Path) -> bool:
    """torch's JIT build on ROCm writes hipified copies at run time, with a ``cuda`` folder of the path renamed ``hip``
    (``cuda/kernels/gdn.cu`` -> ``hip/kernels/gdn.cpp``, ``qwen3_5/cuda/b16.cu`` -> ``qwen3_5/hip/b16.hip``): not sources."""

    parts = path.parts
    for i, part in enumerate(parts[:-1]):
        if part == "hip":
            cuda = Path(*parts[:i], "cuda", *parts[i + 1:-1])
            if any((cuda / (path.stem + ext)).exists() for ext in (".cu", ".cpp", ".cuh")):
                return True
    return False


def unlisted(pyproject: str, src: Path) -> list[str]:
    table = tomllib.loads(pyproject)["tool"]["setuptools"]["package-data"]
    missing = []
    for path in sorted((src / "tensorfold").rglob("*")):     # not a build's egg-info
        if path.suffix not in SHIPPED or hipified(path):
            continue
        rel = path.relative_to(src)
        # a package's patterns name its own files, or (like setuptools) paths into non-package folders below it
        listed = any(fnmatch.fnmatch("/".join(rel.parts[i:]), p)
                     for i in range(1, len(rel.parts))
                     for p in table.get(".".join(rel.parts[:i]), []))
        if not listed:
            missing.append(str(rel))
    return missing


def test_package_data_covers_every_runtime_file():
    assert unlisted((ROOT / "pyproject.toml").read_text(), ROOT / "src") == []
