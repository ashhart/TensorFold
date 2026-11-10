"""Fetch pinned public runtime sources and the pinned inspection tool without running upstream build scripts."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def files(root):
    result = []
    for path in sorted(root.rglob("*")):
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ValueError("source symlink requires review")
        if path.is_file():
            result.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                           "sha256": digest(path)})
    return sorted(result, key=lambda row: row["path"])


def capture(argv, log, cwd=None):
    with log.open("ab") as output:
        run = subprocess.run([str(part) for part in argv], cwd=cwd, stdout=output,
                             stderr=subprocess.STDOUT, timeout=300)
    if run.returncode:
        raise ValueError("pinned source operation failed")


def fetch(work):
    lock = json.loads((HERE / "runtime-inputs.lock.json").read_text())
    downloads = work / "runtime-downloads"
    downloads.mkdir()
    sources = work / "runtime-sources"
    sources.mkdir()
    manifest = {"schema": 1, "sources": [], "gitleaks": None}
    log = work / "public-source-fetch.private.log"
    xgrammar = None
    for row in lock["source_archives"]:
        archive = downloads / (row["name"] + ".tar.gz")
        urllib.request.urlretrieve(row["archive_url"], archive)
        if archive.stat().st_size != row["archive_bytes"] or digest(archive) != row["archive_sha256"]:
            raise ValueError("public source archive digest mismatch")
        temporary = sources / (row["name"] + "-archive")
        temporary.mkdir()
        with tarfile.open(archive, "r:gz") as package:
            entries = package.getmembers()
            if len(entries) > 10000 or sum(entry.size for entry in entries) > 128 << 20:
                raise ValueError("public source archive extraction limit")
            for entry in entries:
                parts = Path(entry.name).parts
                if (entry.issym() or entry.islnk() or not (entry.isfile() or entry.isdir())
                        or entry.name.startswith(("/", "\\")) or ".." in parts
                        or any(":" in part for part in parts)):
                    raise ValueError("unsafe public source archive member")
            package.extractall(temporary, filter="data")
        roots = list(temporary.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError("public source archive root")
        expected = files(roots[0])
        expected_digest = hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if expected_digest != row["archive_file_manifest_sha256"]:
            raise ValueError("public source file closure changed")
        target = sources / "xgrammar" if row["name"] == "xgrammar" else xgrammar / "3rdparty" / "dlpack"
        if target.exists():
            if list(target.iterdir()):
                raise ValueError("nonempty source checkout destination")
            target.rmdir()
        target.mkdir(parents=True)
        capture(["git", "-c", "core.autocrlf=false", "-c", "core.hooksPath=NUL",
                 "init", str(target)], log)
        capture(["git", "-C", target, "config", "core.autocrlf", "false"], log)
        capture(["git", "-C", target, "config", "core.hooksPath", "NUL"], log)
        capture(["git", "-C", target, "remote", "add", "origin", row["repository"] + ".git"], log)
        if row["name"] == "xgrammar":
            capture(["git", "-C", target, "fetch", "--depth=1", "origin",
                     "refs/tags/" + lock["xgrammar_tag"] + ":refs/tags/" + lock["xgrammar_tag"]], log)
            tag = subprocess.check_output(
                ["git", "-C", str(target), "rev-parse", lock["xgrammar_tag"]], text=True).strip()
            if tag != lock["xgrammar_tag_object"]:
                raise ValueError("public annotated source tag changed")
        else:
            capture(["git", "-C", target, "fetch", "--depth=1", "origin", row["commit"]], log)
        capture(["git", "-C", target, "checkout", "--detach", row["commit"]], log)
        if files(target) != expected:
            raise ValueError("public Git source differs from the pinned source archive")
        manifest["sources"].append({**row, "files": expected})
        if row["name"] == "xgrammar":
            xgrammar = target
        shutil.rmtree(temporary)
    # Only dlpack is compiled; cpptrace is disabled and googletest is not built.
    dlpack = lock["source_archives"][1]["commit"]
    actual = subprocess.check_output(
        ["git", "-C", str(xgrammar), "ls-tree", "HEAD", "3rdparty/dlpack"], text=True).split()
    if len(actual) < 3 or actual[2] != dlpack:
        raise ValueError("dlpack is not the xgrammar tag's exact gitlink")
    tool = lock["gitleaks"]
    archive = downloads / "gitleaks.zip"
    urllib.request.urlretrieve(tool["url"], archive)
    if archive.stat().st_size != tool["bytes"] or digest(archive) != tool["sha256"]:
        raise ValueError("secret scanner archive digest mismatch")
    directory = work / "gitleaks"
    directory.mkdir()
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            if ("/" in entry.filename or "\\" in entry.filename or ":" in entry.filename
                    or entry.filename in ("", ".", "..") or entry.is_dir()):
                raise ValueError("unexpected secret scanner archive layout")
        package.extractall(directory)
    executable = directory / "gitleaks.exe"
    if not executable.is_file():
        raise ValueError("pinned secret scanner executable missing")
    manifest["gitleaks"] = {**tool, "executable_sha256": digest(executable), "files": files(directory)}
    path = work / "runtime-source-manifest.private.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return xgrammar, executable, path
