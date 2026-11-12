import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile

MAX_ARCHIVE = 512 * 1024 ** 2
MAX_EXPANDED = 2 * 1024 ** 3


def safe(name):
    return len(name) <= 240 and all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part)
        and not part.endswith((".", " "))
        and not re.match(r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)", part, re.I)
        for part in name.split("/")
    )


def digest(file):
    with file.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def unpack(archive, root, prefix=""):
    info = archive.lstat()
    if not archive.is_file() or archive.is_symlink() or info.st_nlink != 1 or not 0 < info.st_size <= MAX_ARCHIVE:
        raise ValueError("Expected a bounded regular runtime archive")
    names, records, expanded, count = set(), [], 0, 0
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            count += 1
            if count > 1024:
                raise ValueError("Too many runtime entries")
            name = member.name.removesuffix("/")
            if prefix:
                if name == prefix and member.isdir():
                    continue
                if not name.startswith(prefix + "/"):
                    raise ValueError("Wrong native archive root")
                name = name[len(prefix) + 1:]
            if not safe(name) or name.lower() in names or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe or duplicate runtime entry")
            names.add(name.lower())
            if member.isdir():
                continue
            expanded += member.size
            if member.size < 0 or expanded > MAX_EXPANDED or len(records) >= 512 or name == "HOST-ONLY":
                raise ValueError("Runtime bounds or qualification invalid")
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as reader, target.open("xb") as writer:
                copied = 0
                while block := reader.read(1024 * 1024):
                    copied += len(block)
                    if copied > member.size:
                        raise ValueError("Runtime member grew")
                    writer.write(block)
            if copied != member.size:
                raise ValueError("Truncated runtime entry")
            executable = name == "bin/tensorfold-native"
            os.chmod(target, 0o755 if executable else 0o644)
            records.append({"path": name, "size": copied, "sha256": digest(target), "executable": executable})
    return sorted(records, key=lambda record: record["path"])


def package(version, archive, output):
    if not re.fullmatch(r"1\.\d{1,5}\.\d{1,5}", version):
        raise ValueError("Expected a stable 1.x engine")
    source_name = f"tensorfold-{version}-macos-arm64"
    root = output / "tree"
    root.mkdir(mode=0o700)
    files = unpack(archive, root, source_name)
    required = {"bin/tensorfold-native", "lib/libtfgrammar.dylib", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "VERSION"}
    if not required.issubset({record["path"] for record in files}):
        raise ValueError("Required desktop runtime files are missing")
    version_path = root / "VERSION"
    if version_path.stat().st_size != len(version) + 1 or version_path.read_bytes() != (version + "\n").encode():
        raise ValueError("Native archive version mismatch")
    for path in ["bin/tensorfold-native", "lib/libtfgrammar.dylib"]:
        with (root / path).open("rb") as source:
            header = source.read(8)
        if header != b"\xcf\xfa\xed\xfe\x0c\x00\x00\x01":
            raise ValueError("Expected a native arm64 Mach-O runtime")
    identifier = f"tensorfold-engine-{version}-darwin-arm64-metal"
    destination = output / f"{identifier}.tar.gz"
    with destination.open("xb") as raw, gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as target:
            for record in files:
                member = tarfile.TarInfo(record["path"])
                member.size = record["size"]
                member.mode = 0o755 if record["executable"] else 0o644
                with (root / record["path"]).open("rb") as stream:
                    target.addfile(member, stream)
    metadata = {"id": identifier, "version": version, "platform": "darwin", "arch": "arm64", "backend": "metal",
                "protocol": 1, "minAppVersion": "1.0.0", "notes": "",
                "url": f"https://github.com/ashhart/TensorFold/releases/download/v{version}/{identifier}.tar.gz",
                "sha256": digest(destination), "bytes": destination.stat().st_size,
                "entrypoint": "bin/tensorfold-native", "files": files}
    with (output / "package.json").open("x") as target:
        json.dump(metadata, target, indent=2)
        target.write("\n")
    shutil.rmtree(root)
    return metadata


def verify(version, directory):
    pkg = json.loads((directory / "package.json").read_text())
    archive = directory / f"tensorfold-engine-{version}-darwin-arm64-metal.tar.gz"
    if digest(archive) != pkg["sha256"] or archive.stat().st_size != pkg["bytes"]:
        raise ValueError("Prepared archive changed")
    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp)
        expected = package(version, directory / f"tensorfold-{version}-macos-arm64.tar.gz", temporary)
        identity = lambda value: {key: item for key, item in value.items() if key not in {"sha256", "bytes"}}
        if identity(expected) != identity(pkg):
            raise ValueError("Prepared files disagree with the authenticated native archive")
        root = temporary / "flat"
        root.mkdir()
        if unpack(archive, root) != pkg["files"]:
            raise ValueError("Prepared archive inventory changed")
    return pkg


def verify_inventory(directory):
    pkg = json.loads((directory / "package.json").read_text())
    archive = directory / "runtime.tar.gz"
    if digest(archive) != pkg["sha256"] or archive.stat().st_size != pkg["bytes"]:
        raise ValueError("Downloaded runtime changed")
    with tempfile.TemporaryDirectory() as tmp:
        if unpack(archive, Path(tmp)) != sorted(pkg["files"], key=lambda record: record["path"]):
            raise ValueError("Downloaded runtime inventory changed")


if __name__ == "__main__":
    try:
        operation, version, directory = sys.argv[1:]
        output = Path(directory)
        if operation == "prepare":
            package(version, output / f"tensorfold-{version}-macos-arm64.tar.gz", output)
        elif operation == "verify":
            verify(version, output)
        elif operation == "inventory":
            verify_inventory(output)
        else:
            raise ValueError("Unknown archive operation")
    except (ValueError, OSError, tarfile.TarError):
        sys.exit("Runtime archive verification failed; staging was retained for review.")
