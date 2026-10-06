#!/usr/bin/env python3
"""CPU regression for repeated packaging; keeps its receipt directory for inspection."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

root = Path(__file__).resolve().parents[2]
script = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else root / "tools/release/package.sh"
(root / ".zig-cache").mkdir(exist_ok=True)
receipt = Path(tempfile.mkdtemp(prefix="package-repeat-", dir=root / ".zig-cache"))
output = receipt / "archives"
output.mkdir()
binary = receipt / "tensorfold-native"
binary.write_text("synthetic packaging fixture, not an executable\n")
version = "0.6.5"  # Existing manifest version, not a release choice.
name = f"tensorfold-{version}-linux-x86_64"

def capture(dirname, sets):
    directory = receipt / dirname
    for sm, cubins in sets.items():
        target = directory / sm
        (target / "cubins").mkdir(parents=True)
        (target / "aot.json").write_text('{"synthetic":true}\n')
        for file in cubins:
            (target / "cubins" / file).write_bytes(b"synthetic cubin fixture\n")
    return directory

full = capture("full", {"sm121": ["keep.cubin", "stale.cubin"], "sm120": ["older.cubin"]})
smaller = capture("smaller", {"sm121": ["keep.cubin"]})

def pack(tree=None):
    args = ["sh", str(script), version, "linux-x86_64", str(binary), str(root / "LICENSE"),
            str(root / "NOTICE"), str(root / "packaging/RUNTIME.md"), str(root / "LICENSES"),
            str(root / "THIRD_PARTY_NOTICES.md"), str(output)]
    if tree is not None:
        args.append(str(tree))
    subprocess.run(args, check=True)
    archive = output / (name + ".tar.gz")
    companion = archive.with_name(archive.name + ".sha256")
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == companion.read_text().split()[0]
    with tarfile.open(archive) as tar:
        files = {m.name.removeprefix(name + "/"): tar.extractfile(m).read()
                 for m in tar.getmembers() if m.isfile()}
    for line in files["SHA256SUMS"].decode().splitlines():
        digest, path = line.split("  ", 1)
        assert hashlib.sha256(files[path.removeprefix("./")]).hexdigest() == digest
    return files

first = pack(full)
assert "share/tensorfold/cuda/sm121/cubins/stale.cubin" in first
second = pack(smaller)
assert "share/tensorfold/cuda/sm121/cubins/keep.cubin" in second
assert not any("stale.cubin" in p or "/sm120/" in p for p in second), "stale cubin survived a smaller capture set"
third = pack()
assert not any(p.startswith("share/") for p in third), "capture files survived removal of the AOT input"
for file in ["LICENSES/MIT.txt", "LICENSES/Apache-2.0.txt", "LICENSES/MiaAI-Lab-MIT.txt", "THIRD_PARTY_NOTICES.md"]:
    assert third[file] == (root / file).read_bytes(), file
print(f"PASS same-version full -> smaller -> no capture, all hashes and notice files; receipt {receipt}")
