"""The native (Zig) engine a platform wheel carries: binary, gate manifest, bundle digest; a pure wheel has none."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

ENGINES = ("auto", "zig", "python")
ROOT = Path(__file__).parent
BINARY = Path("bin") / "tensorfold-native"
MANIFEST = "gate.json"


def binary() -> Path | None:
    """The installed native binary, or None for an install without one (the pure wheel, a source checkout)."""

    path = ROOT / BINARY
    return path if path.is_file() and os.access(path, os.X_OK) else None


def bundle_files(root: Path) -> list[Path]:
    """The bundle: every file below a folder of ``root`` (bin/, kernels/ ...), never this package's own files."""

    found = (path.relative_to(root) for path in root.rglob("*") if path.is_file())
    return sorted(path for path in found if len(path.parts) > 1 and path.parts[0] != "__pycache__")


def bundle_digest() -> str:
    """sha256 over each bundle file's relative path and sha256: the identity a gate entry names."""

    digest = hashlib.sha256()
    for path in bundle_files(ROOT):
        with open(ROOT / path, "rb") as file:
            digest.update(f"{path.as_posix()}\0{hashlib.file_digest(file, 'sha256').hexdigest()}\n".encode())
    return digest.hexdigest()
