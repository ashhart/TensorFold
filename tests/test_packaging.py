"""Every CUDA source and data file under src ships in the wheel: pyproject's package-data names it (#66)."""

import fnmatch
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = {".cu", ".cuh", ".cpp", ".json", ".txt"}       # what the engines read or build at run time


def unlisted(pyproject: str, src: Path) -> list[str]:
    table = tomllib.loads(pyproject)["tool"]["setuptools"]["package-data"]
    missing = []
    for path in sorted((src / "tensorfold").rglob("*")):     # not a build's egg-info
        patterns = table.get(".".join(path.parent.relative_to(src).parts), [])
        if path.suffix in SHIPPED and not any(fnmatch.fnmatch(path.name, p) for p in patterns):
            missing.append(str(path.relative_to(src)))
    return missing


def test_package_data_covers_every_runtime_file():
    assert unlisted((ROOT / "pyproject.toml").read_text(), ROOT / "src") == []
