#!/usr/bin/env python3
"""CPU regression for repeated packaging; keeps its receipt directory for inspection."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--path-checker", type=Path, required=True)
parser.add_argument("script", nargs="?", type=Path, default=root / "tools/release/package.sh")
options = parser.parse_args()
script = options.script.resolve()
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

overlay = receipt / "overlay"
(overlay / "lib").mkdir(parents=True)
(overlay / "LICENSES").mkdir()
(overlay / "lib/libtfgrammar.so").write_bytes(b"synthetic grammar library fixture\n")
(overlay / "LICENSES/xgrammar-Apache-2.0.txt").write_text("synthetic license fixture\n")

def command(tree=None, extra=None, license_tree=None):
    args = ["sh", str(script), version, "linux-x86_64", str(binary), str(root / "LICENSE"),
            str(root / "NOTICE"), str(root / "packaging/RUNTIME.md"), str(license_tree or root / "LICENSES"),
            str(root / "THIRD_PARTY_NOTICES.md"), str(output),
            f"--path-checker={options.path_checker.resolve()}", f"--build-root={root}"]
    if extra is not None:
        args.append(f"--overlay={extra}")
    if tree is not None:
        args.append(str(tree))
    return args

def pack(tree=None, extra=None):
    subprocess.run(command(tree, extra), check=True)
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

first = pack(full, overlay)
assert first["lib/libtfgrammar.so"] == b"synthetic grammar library fixture\n", "the overlay's library is missing"
assert "LICENSES/xgrammar-Apache-2.0.txt" in first and "LICENSES/MIT.txt" in first
assert "share/tensorfold/cuda/sm121/cubins/stale.cubin" in first
second = pack(smaller)
assert "share/tensorfold/cuda/sm121/cubins/keep.cubin" in second
assert not any("stale.cubin" in p or "/sm120/" in p for p in second), "stale cubin survived a smaller capture set"
third = pack()
assert not any(p.startswith("share/") for p in third), "capture files survived removal of the AOT input"
assert "lib/libtfgrammar.so" not in third, "overlay files survived removal of the overlay input"
for file in ["LICENSES/MIT.txt", "LICENSES/Apache-2.0.txt", "LICENSES/MiaAI-Lab-MIT.txt", "THIRD_PARTY_NOTICES.md"]:
    assert third[file] == (root / file).read_bytes(), file

archive = output / (name + ".tar.gz")
companion = archive.with_name(archive.name + ".sha256")
clean_archive = archive.read_bytes()
clean_hash = companion.read_bytes()
for file, tree, extra in [(binary, None, None), (overlay / "lib/libtfgrammar.so", None, overlay),
                          (smaller / "sm121/cubins/keep.cubin", smaller, None)]:
    original = file.read_bytes()
    try:
        file.write_bytes(original + b"\0" + str(root).encode() + b"/zig/kernels/cuda/exl3_block\0")
        run = subprocess.run(command(tree, extra), capture_output=True, text=True)
        assert run.returncode != 0 and "PrivateBuildPath" in run.stderr, run.stderr
        assert str(root) not in run.stderr, "path check leaked the private build root in its diagnostic"
        assert archive.read_bytes() == clean_archive, "a rejected package replaced the clean archive"
        assert companion.read_bytes() == clean_hash, "a rejected package replaced the clean checksum"
    finally:
        file.write_bytes(original)
licenses = receipt / "license-links"
shutil.copytree(root / "LICENSES", licenses)
safe_license = receipt / "private-build-root/clean-license.txt"
safe_license.parent.mkdir()
safe_license.write_text("clean license bytes, no path\n")
for target in [safe_license, safe_license.parent, safe_license.parent / "absent"]:
    link = licenses / "example.txt"
    link.symlink_to(target)
    try:
        run = subprocess.run(command(license_tree=licenses), capture_output=True, text=True)
        assert run.returncode != 0 and "symlink refused" in run.stderr, run.stderr
        assert str(target) not in run.stderr and str(root) not in run.stderr
        assert archive.read_bytes() == clean_archive and companion.read_bytes() == clean_hash
    finally:
        link.unlink()

def archive_case(label, member_name, data=b"clean payload\n", target="", kind=tarfile.REGTYPE, pax=None):
    fixture = receipt / (label + ".tar.gz")
    with tarfile.open(fixture, "w:gz") as tar:
        member = tarfile.TarInfo(member_name)
        member.type = kind
        member.linkname = target
        member.pax_headers = pax or {}
        member.size = len(data) if kind == tarfile.REGTYPE else 0
        tar.addfile(member, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    return fixture

archive_checker = root / "tools/release/check_archive.py"
cases = [
    (archive_case("clean", "package/LICENSES/example.txt"), 0),
    (archive_case("private-name", str(root) + "/zig/kernels/cuda/source.cu"), 1),
    (archive_case("private-link", "package/example.txt", target=str(safe_license), kind=tarfile.SYMTYPE), 1),
    (archive_case("private-hardlink", "package/example.txt", target=str(safe_license), kind=tarfile.LNKTYPE), 1),
    (archive_case("relative-link", "package/example.txt", target="clean.txt", kind=tarfile.SYMTYPE), 1),
    (archive_case("private-pax", "package/example.txt", pax={"comment": str(root)}), 1),
    (archive_case("private-contents", "package/bin/native", data=str(root).encode()), 1),
]
for fixture, expected in cases:
    run = subprocess.run([sys.executable, str(archive_checker), str(options.path_checker.resolve()), str(root), str(fixture)],
                         capture_output=True, text=True)
    assert run.returncode == expected, run.stderr
    assert str(root) not in run.stderr, "archive check leaked the private root in its diagnostic"

fake_bin = receipt / "late-link-tar"
fake_bin.mkdir()
fake_tar = fake_bin / "tar"
fake_tar.write_text(f'''#!/usr/bin/env python3
import io, os, subprocess, sys, tarfile
real_tar = {shutil.which("tar")!r}
if sys.argv[1:] == ["--version"]:
    os.execv(real_tar, [real_tar, "--version"])
subprocess.run([real_tar, *sys.argv[1:]], check=True)
archive = sys.argv[sys.argv.index("-czf") + 1]
with tarfile.open(archive) as source:
    entries = [(m, source.extractfile(m).read() if m.isfile() else None) for m in source.getmembers()]
with tarfile.open(archive, "w:gz") as target:
    for member, data in entries:
        target.addfile(member, io.BytesIO(data) if data is not None else None)
    link = tarfile.TarInfo("package/late-link")
    link.type = tarfile.SYMTYPE
    link.linkname = {str(safe_license)!r}
    target.addfile(link)
''')
fake_tar.chmod(0o755)
run = subprocess.run(command(), env=dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ["PATH"]),
                     capture_output=True, text=True)
assert run.returncode != 0 and "archive refused" in run.stderr, run.stderr
assert str(root) not in run.stderr
assert archive.read_bytes() == clean_archive and companion.read_bytes() == clean_hash
print(f"PASS repeated packaging, hashes, notices, contents and LICENSES symlink refusal; receipt {receipt}")
