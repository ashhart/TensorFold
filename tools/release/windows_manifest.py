#!/usr/bin/env python3
"""Bind a Windows package to its source checkout, produced binary and explicit fatbin inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def record(binary, fatbins, host_only):
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    dirty = bool(git("status", "--porcelain=v1", "--untracked-files=all"))
    if dirty and not host_only:
        raise ValueError("production source checkout must be clean")
    inputs = []
    if fatbins:
        for path in sorted(fatbins.iterdir()):
            if path.suffix != ".fatbin":
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("fatbin inputs must be regular files")
            inputs.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
    return {"schema": 1, "target": "windows-x86_64", "source_commit": git("rev-parse", "HEAD"),
            "source_tree": git("rev-parse", "HEAD^{tree}"), "source_dirty": dirty,
            "binary_sha256": sha(binary), "zig": subprocess.check_output(["zig", "version"], text=True).strip(),
            "fatbin_inputs": inputs, "host_only": host_only,
            "qualification": "CPU build/package identity only; GPU, model and performance checks are separate"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fatbins", type=Path)
    parser.add_argument("--host-only", action="store_true")
    args = parser.parse_args()
    try:
        args.output.write_text(json.dumps(record(args.binary, args.fatbins, args.host_only), indent=2) + "\n")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Windows source manifest refused ({type(error).__name__})", file=sys.stderr)
        sys.exit(1)
