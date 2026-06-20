#!/usr/bin/env python3
"""Build and install TensorFold, then run the no-model release smoke."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and install TensorFold in a temporary venv, then run demo/inspect/pack checks.",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="TensorFold source checkout.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary build directory.")
    parser.add_argument("--skip-scrub", action="store_true", help="Skip tools/check_public_scrub.py.")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not (root / "pyproject.toml").is_file():
        print(f"smoke: missing pyproject.toml under {root}", file=sys.stderr)
        return 2

    temp_dir = Path(tempfile.mkdtemp(prefix="tensorfold-release-smoke-"))
    try:
        return _run_smoke(root, temp_dir, skip_scrub=args.skip_scrub)
    finally:
        if args.keep_temp:
            print(f"smoke: kept temp directory {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _run_smoke(root: Path, temp_dir: Path, *, skip_scrub: bool) -> int:
    dist_dir = temp_dir / "dist"
    dist_dir.mkdir()

    if not skip_scrub:
        _run([sys.executable, "tools/check_public_scrub.py"], cwd=root)

    _run([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "-w", str(dist_dir), "."], cwd=root)
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in {dist_dir}, found {len(wheels)}")

    venv_dir = temp_dir / "venv"
    _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=root)
    python = venv_dir / "bin" / "python"
    tensorfold = venv_dir / "bin" / "tensorfold"
    _run([str(python), "-m", "pip", "install", "-q", str(wheels[0])], cwd=root)

    demo_dir = temp_dir / "demo"
    pack_dir = temp_dir / "packs"
    _run([str(tensorfold), "--version"], cwd=root)
    _run([str(tensorfold), "doctor"], cwd=root)
    _run([str(tensorfold), "demo", "create", str(demo_dir)], cwd=root)
    _run([str(tensorfold), "inspect", str(demo_dir / "toy.safetensors")], cwd=root)
    _run([str(tensorfold), "inspect", str(demo_dir / "toy-moe" / "model-00001-of-00001.safetensors")], cwd=root)
    _run([str(tensorfold), "pack", str(demo_dir / "toy-moe"), "--out", str(pack_dir)], cwd=root)
    _run([str(tensorfold), "selftest"], cwd=root)

    packs = sorted(pack_dir.glob("*.pack"))
    if len(packs) != 1:
        raise RuntimeError(f"expected one pack artifact, found {len(packs)}")

    print("tensorfold release smoke: ok")
    return 0


def _run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, check=False)
    if completed.returncode != 0:
        quoted = " ".join(command)
        raise RuntimeError(f"command failed with exit {completed.returncode}: {quoted}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"smoke: {exc}", file=sys.stderr)
        raise SystemExit(1)
