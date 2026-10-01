"""Resolve immutable ds4 build inputs without bundling the native source tree."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

PIN = "d183482b413ecd2e3b540b290e6497437e9fbb73"
PACKAGE = Path(__file__).parent


def source_manifest():
    manifest = json.loads((PACKAGE / "ds4-source.json").read_text())
    if manifest["revision"] != PIN:
        raise ValueError("unsupported ds4 native source revision")
    return manifest


def verify_sources(source):
    """Use TensorFold's packaged hashes, never a manifest supplied by the checkout."""
    source = Path(source)
    manifest = source_manifest()
    for name, expected in manifest["sha256"].items():
        path = source / name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned native source missing or changed: {name}")
    return manifest


def _archive(repository, destination, manifest):
    archive = subprocess.run(
        ["git", "-C", str(repository), "archive", "--format=tar", PIN, "--", *manifest["sha256"]],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    # Copy only expected regular files; never extract archive paths or symlinks.
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for name in manifest["sha256"]:
            member = tar.getmember(name)
            if not member.isfile():
                raise ValueError(f"pinned native source is not a regular file: {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream, target.open("wb") as output:
                shutil.copyfileobj(stream, output)


def resolve_sources(source=None, *, cache_dir=None, offline=False):
    """Reuse verified cache/local inputs; fetch only when no local input is selected."""
    cache = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache/tensorfold/ds4" / PIN
    cache = cache.expanduser().resolve()
    cache.parent.mkdir(parents=True, exist_ok=True)
    with (cache.parent / f".{cache.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            verify_sources(cache)  # A damaged cache fails closed; do not silently replace it.
            return cache
        manifest = source_manifest()
        selected = source if source is not None else os.environ.get("TENSORFOLD_DS4_SOURCE")
        with tempfile.TemporaryDirectory(prefix=f".{cache.name}-", dir=cache.parent) as temporary:
            root = Path(temporary)
            stage = root / "sources"
            stage.mkdir()
            if selected:
                local = Path(selected).expanduser().resolve(strict=True)
                try:
                    verify_sources(local)
                except ValueError:
                    # A newer/modified checkout may still hold the pinned Git object.
                    # Read it without checking out, fetching or modifying that repository.
                    _archive(local, stage, manifest)
                else:
                    for name in manifest["sha256"]:
                        target = stage / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(local / name, target)
            elif offline:
                raise RuntimeError("ds4 source cache missing; supply --ds4-source or TENSORFOLD_DS4_SOURCE offline")
            else:
                repository = root / "repository"
                subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
                subprocess.run(
                    ["git", "-C", str(repository), "fetch", "--depth=1", "--no-tags", manifest["repository"], PIN],
                    check=True,
                )
                _archive(repository, stage, manifest)
            verify_sources(stage)
            stage.rename(cache)
        return cache
