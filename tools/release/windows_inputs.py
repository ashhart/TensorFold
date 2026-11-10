#!/usr/bin/env python3
"""Validate and copy explicitly inventoried Windows runtime inputs without loading them."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from urllib.parse import urlsplit


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("nonregular input")
    if path.stat().st_size > 1 << 30:
        raise ValueError("oversized input")
    return path


def pe(path, dll=False):
    with regular(path).open("rb") as source:
        header = source.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            raise ValueError("not PE")
        offset = int.from_bytes(header[60:64], "little")
        if offset < 64 or offset > 1 << 20:
            raise ValueError("bad PE offset")
        source.seek(offset)
        coff = source.read(24)
    if len(coff) != 24 or coff[:4] != b"PE\0\0" or coff[4:6] != b"\x64\x86":
        raise ValueError("not AMD64 PE")
    if bool(int.from_bytes(coff[22:24], "little") & 0x2000) != dll:
        raise ValueError("wrong PE kind")


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in (".", "..") for part in path.parts):
        raise ValueError("nonrelative path")
    for part in path.parts:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part.endswith("."):
            raise ValueError("invalid Windows filename")
        if re.fullmatch(r"(?i)(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part):
            raise ValueError("reserved Windows filename")
    return value


def manifest(path):
    if regular(path).stat().st_size > 1 << 20:
        raise ValueError("oversized manifest")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("schema") != 1 or result.get("target") != "windows-x86_64":
        raise ValueError("wrong manifest target")
    return result


def runtime_files(root):
    expected = {"WINDOWS-RUNTIMES.json", "WINDOWS-RUNTIME-NOTICES.txt"}
    record = manifest(root / "WINDOWS-RUNTIMES.json")
    libraries = record.get("libraries")
    if not isinstance(libraries, list) or not libraries:
        raise ValueError("empty runtime inventory")
    names = set()
    for entry in libraries:
        name = relative(entry["path"])
        parts = PurePosixPath(name).parts
        if len(parts) != 2 or parts[0] not in ("bin", "lib") or not parts[1].lower().endswith(".dll"):
            raise ValueError("runtime must be a DLL")
        folded = parts[1].lower()
        if folded in names or folded in {"nvcuda.dll", "kernel32.dll", "ntdll.dll", "user32.dll", "ws2_32.dll"}:
            raise ValueError("duplicate or system DLL")
        names.add(folded)
        pe(root / name, dll=True)
        if not re.fullmatch("[0-9a-f]{64}", entry["sha256"]) or digest(root / name) != entry["sha256"]:
            raise ValueError("runtime digest mismatch")
        source = urlsplit(entry["source"])
        if source.scheme != "https" or not source.hostname or source.username or source.password or source.query or source.fragment:
            raise ValueError("runtime needs a public source URL")
        if not isinstance(entry.get("version"), str) or not entry["version"].strip():
            raise ValueError("runtime version missing")
        licenses = entry.get("licenses")
        if not isinstance(licenses, list) or not licenses:
            raise ValueError("runtime licenses missing")
        for license_name in licenses:
            license_name = relative(license_name)
            if not re.fullmatch(r"LICENSES/windows-[A-Za-z0-9_.-]+\.(?:txt|md)", license_name):
                raise ValueError("runtime license must have its own Windows name")
            if regular(root / license_name).stat().st_size == 0:
                raise ValueError("runtime license empty")
            expected.add(license_name)
        expected.add(name)
    if "tfgrammar.dll" not in names:
        raise ValueError("structured output runtime missing")
    if regular(root / "WINDOWS-RUNTIME-NOTICES.txt").stat().st_size == 0:
        raise ValueError("runtime notices empty")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("nonregular runtime tree")
        if path.is_file():
            actual.add(relative(path.relative_to(root).as_posix()))
    if actual != expected or len({name.lower() for name in actual}) != len(actual):
        raise ValueError("runtime tree differs from inventory")
    return expected


def check(binary, overlay, source_manifest, stage, host_only):
    pe(binary)
    if source_manifest:
        source = Path(source_manifest)
        record = manifest(source)
        if record.get("host_only") is True and not host_only:
            raise ValueError("host-only source cannot produce an inference archive")
        if any(not re.fullmatch("[0-9a-f]{40}", record.get(key, "")) for key in ("source_commit", "source_tree")):
            raise ValueError("source identity missing")
        if (record.get("source_dirty") is not False and not host_only) or record.get("binary_sha256") != digest(binary):
            raise ValueError("source manifest does not identify this clean build")
        shutil.copyfile(source, stage / "SOURCE.json")
    elif not host_only:
        raise ValueError("source manifest required")
    if overlay:
        root = Path(overlay)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("runtime root must be a real directory")
        for name in sorted(runtime_files(root)):
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)
    elif not host_only:
        raise ValueError("inventoried runtime overlay required")


if __name__ == "__main__":
    try:
        if len(sys.argv) != 6:
            raise ValueError("arguments")
        check(Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), sys.argv[5] == "1")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Windows package: input refused ({type(error).__name__})", file=sys.stderr)
        sys.exit(1)
