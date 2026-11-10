#!/usr/bin/env python3
"""CPU regression for repeated packaging; keeps its receipt directory for inspection."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

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

def pe_fixture(dll=False):
    data = bytearray(256)
    data[:2] = b"MZ"
    data[60:64] = (128).to_bytes(4, "little")
    data[128:132] = b"PE\0\0"
    data[132:134] = (0x8664).to_bytes(2, "little")
    data[150:152] = (0x2002 if dll else 2).to_bytes(2, "little")
    return bytes(data) + b"synthetic nonfunctional packaging fixture\n"


windows_binary = receipt / "native-serving-fixture.exe"
windows_binary.write_bytes(pe_fixture())
windows_runtime = receipt / "windows-runtime"
(windows_runtime / "lib").mkdir(parents=True)
(windows_runtime / "LICENSES").mkdir()
windows_dll = windows_runtime / "lib/tfgrammar.dll"
windows_dll.write_bytes(pe_fixture(dll=True))
windows_license = windows_runtime / "LICENSES/windows-fixture.txt"
windows_license.write_text("Synthetic nonfunctional DLL fixture; no third-party runtime.\n")
(windows_runtime / "WINDOWS-RUNTIME-NOTICES.txt").write_text("Synthetic runtime packaging test, not a redistributable library.\n")
runtime_manifest = windows_runtime / "WINDOWS-RUNTIMES.json"
runtime_record = {"schema": 1, "target": "windows-x86_64", "libraries": [{
    "path": "lib/tfgrammar.dll", "sha256": hashlib.sha256(windows_dll.read_bytes()).hexdigest(),
    "version": "synthetic-fixture", "source": "https://example.invalid/runtime",
    "licenses": ["LICENSES/windows-fixture.txt"],
}]}
runtime_manifest.write_text(json.dumps(runtime_record))
source_manifest = receipt / "windows-source.json"
source_manifest.write_text(json.dumps({"schema": 1, "target": "windows-x86_64", "source_commit": "0" * 40,
    "source_tree": "1" * 40, "source_dirty": False,
    "binary_sha256": hashlib.sha256(windows_binary.read_bytes()).hexdigest(),
    "qualification": "CPU packaging fixture only; no executable or GPU qualification"}))
windows_name = f"tensorfold-{version}-windows-x86_64"


def windows_command(runtime=windows_runtime, source=source_manifest, host_only=False):
    args = command()
    args[3] = "windows-x86_64-host-only" if host_only else "windows-x86_64"
    args[4] = str(windows_binary)
    if runtime is not None:
        args.append(f"--overlay={runtime}")
    if source is not None:
        args.append(f"--source-manifest={source}")
    return args


def windows_files():
    archive = output / (windows_name + ".zip")
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == archive.with_name(archive.name + ".sha256").read_text().split()[0]
    with zipfile.ZipFile(archive) as zipped:
        assert not zipped.comment and all(not entry.extra and not entry.comment for entry in zipped.infolist())
        files = {entry.filename.removeprefix(windows_name + "/"): zipped.read(entry)
                 for entry in zipped.infolist() if not entry.is_dir()}
    assert files["bin/tensorfold.exe"] == windows_binary.read_bytes()
    assert not any(path.startswith("bin/tensorfold-native") for path in files)
    assert files["lib/tfgrammar.dll"] == windows_dll.read_bytes()
    assert files["SOURCE.json"] == source_manifest.read_bytes()
    assert "WINDOWS-RUNTIMES.json" in files and "WINDOWS-RUNTIME-NOTICES.txt" in files
    for line in files["SHA256SUMS"].decode().splitlines():
        digest, path = line.split("  ", 1)
        assert hashlib.sha256(files[path.removeprefix("./")]).hexdigest() == digest
    return archive


subprocess.run(windows_command(), check=True)
windows_archive = windows_files()
windows_clean = windows_archive.read_bytes()
windows_hash = windows_archive.with_name(windows_archive.name + ".sha256").read_bytes()


def windows_refuses(args):
    run = subprocess.run(args, capture_output=True, text=True)
    assert run.returncode != 0, "unreviewed Windows package input was accepted"
    assert windows_archive.read_bytes() == windows_clean, "rejected input replaced the clean Windows archive"
    assert windows_archive.with_name(windows_archive.name + ".sha256").read_bytes() == windows_hash


windows_refuses(windows_command(runtime=None))
windows_refuses(windows_command(source=None))
unlisted = windows_runtime / "model.safetensors"
unlisted.write_bytes(b"synthetic unreviewed weights")
windows_refuses(windows_command())
unlisted.unlink()
original_dll = windows_dll.read_bytes()
windows_dll.write_bytes(original_dll + b"changed")
windows_refuses(windows_command())
windows_dll.write_bytes(original_dll)
license_bytes = windows_license.read_bytes()
windows_license.unlink()
windows_refuses(windows_command())
windows_license.write_bytes(license_bytes)
runtime_record["libraries"][0]["path"] = "../escape.dll"
runtime_manifest.write_text(json.dumps(runtime_record))
windows_refuses(windows_command())
runtime_record["libraries"][0]["path"] = "lib/tfgrammar.dll"
runtime_manifest.write_text(json.dumps(runtime_record))
windows_license.unlink()
windows_license.symlink_to(safe_license)
windows_refuses(windows_command())
windows_license.unlink()
windows_license.write_bytes(license_bytes)
subprocess.run(windows_command(), check=True)
windows_files()
print("PASS Windows canonical serving filename, ZIP/checksums, manifests, licensed DLL overlay and refusal controls")
if shutil.which("pwsh"):
    subprocess.run(["pwsh", "-NoProfile", "-File", str(root / "tools/release/check_windows_smoke.ps1"),
                    "-FixtureArchive", str(windows_archive)], check=True)
else:
    print("SKIP Windows PowerShell fixture checks: pwsh is not installed")
