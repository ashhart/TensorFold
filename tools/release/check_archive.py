#!/usr/bin/env python3
"""Inspect the finished archive's metadata and decoded file contents before publication."""
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def scan(checker, build_root, file):
    subprocess.run([checker, build_root, str(file)], check=True)


def check(checker, build_root, archive):
    with tempfile.TemporaryDirectory(prefix=".archive-privacy-", dir=archive.parent) as temporary:
        directory = Path(temporary)
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            listing = directory / "listing"
            with listing.open("wb") as out:
                for key, value in tar.pax_headers.items():
                    out.write(f"{key}={value}\0".encode("utf-8", "surrogateescape"))
                for member in members:
                    for value in [member.name, member.linkname, member.uname, member.gname]:
                        out.write(value.encode("utf-8", "surrogateescape") + b"\0")
                    for key, value in member.pax_headers.items():
                        out.write(f"{key}={value}\0".encode("utf-8", "surrogateescape"))
            scan(checker, build_root, listing)
            for member in members:
                name = member.name.replace("\\", "/")
                if name.startswith("/") or ".." in name.split("/") or (len(name) > 1 and name[1] == ":"):
                    raise ValueError("nonrelative member name")
                if not (member.isfile() or member.isdir()):
                    raise ValueError("nonregular archive member")
                if member.isdir():
                    continue
                if member.size > 1 << 30:
                    raise ValueError("oversized archive member")
                payload = directory / "payload"
                with tar.extractfile(member) as source, payload.open("wb") as out:
                    while block := source.read(1 << 20):
                        out.write(block)
                scan(checker, build_root, payload)


def main():
    if len(sys.argv) != 4:
        print("usage: check_archive.py CHECKER BUILD_ROOT ARCHIVE", file=sys.stderr)
        return 2
    try:
        check(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
    except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as error:
        print(f"release privacy: archive refused ({type(error).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
