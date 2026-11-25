import hashlib
import re
import stat
import tarfile
from pathlib import Path

from .pins import InstallError, MAX_EXTRACTED_BYTES


class BoundedTarInfo(tarfile.TarInfo):
    def _metadata(self, archive, reader):
        total = getattr(archive, "_metadata_bytes", 0) + self.size
        depth = getattr(archive, "_metadata_depth", 0) + 1
        if self.size > 1024 * 1024 or total > 16 * 1024 * 1024 or depth > 32:
            raise InstallError("Archive metadata exceeds its size or nesting limit.")
        archive._metadata_bytes = total
        archive._metadata_depth = depth
        try:
            return reader(archive)
        finally:
            archive._metadata_depth -= 1

    def _proc_pax(self, archive):
        return self._metadata(archive, super()._proc_pax)

    def _proc_gnulong(self, archive):
        return self._metadata(archive, super()._proc_gnulong)

    def _proc_sparse(self, archive):
        raise InstallError("Sparse archive files are refused.")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path, expected):
    if not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise InstallError("Expected SHA-256 is malformed.")
    actual = file_hash(path)
    if actual != expected:
        raise InstallError(f"SHA-256 mismatch for {Path(path).name}: {actual} != {expected}")


def relative_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name or ":" in name:
        raise InstallError("Unsafe archive path.")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise InstallError(f"Unsafe archive path: {name!r}")
    return parts


def parse_sums(text):
    sums = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([a-fA-F0-9]{64}) [ *](.+)", line)
        if not match:
            raise InstallError("Malformed SHA-256 checksum line.")
        name = match[2]
        if name.startswith("./"):
            name = name[2:]
        relative_name(name)
        if name in sums:
            raise InstallError(f"Duplicate checksum entry: {name}")
        sums[name] = match[1].lower()
    if not sums:
        raise InstallError("Empty checksum manifest.")
    return sums


def extract_archive(archive, destination, root_name, max_bytes=MAX_EXTRACTED_BYTES):
    relative_name(root_name)
    if "/" in root_name:
        raise InstallError("Expected archive root must be one directory.")
    destination = Path(destination)
    destination.mkdir(mode=0o700)
    seen = set()
    files = set()
    total = 0
    try:
        with tarfile.open(archive, "r:*", tarinfo=BoundedTarInfo) as tar:
            for member in tar:
                name = member.name.rstrip("/") if member.isdir() else member.name
                parts = relative_name(name)
                if parts[0] != root_name:
                    raise InstallError("Archive member is outside its expected root.")
                if name in seen:
                    raise InstallError(f"Duplicate archive member: {name}")
                seen.add(name)
                if len(seen) > 100000:
                    raise InstallError("Archive has too many members.")
                if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE) or member.sparse is not None:
                    raise InstallError("Archive links, sparse files and special files are refused.")
                if member.size < 0 or (member.isdir() and member.size):
                    raise InstallError("Archive member has an invalid size.")
                total += member.size
                if member.size > 1024 * 1024 * 1024 or total > max_bytes:
                    raise InstallError("Archive exceeds its unpacked size limit.")
                if len(parts) == 1 and not member.isdir():
                    raise InstallError("Archive root is not a directory.")
                target = destination.joinpath(*parts)
                if member.isdir():
                    target.mkdir(mode=0o755, parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("xb") as output:
                    remaining = member.size
                    while remaining:
                        block = source.read(min(1024 * 1024, remaining))
                        if not block:
                            raise InstallError("Archive member is truncated.")
                        output.write(block)
                        remaining -= len(block)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
                files.add("/".join(parts[1:]))
    except (tarfile.TarError, EOFError, OSError) as error:
        raise InstallError(f"Could not safely extract archive: {error}") from error
    root = destination / root_name
    if not files or not root.is_dir():
        raise InstallError("Archive is empty.")
    return root, files


def verify_engine(root, files, version):
    if "HOST-ONLY" in files:
        raise InstallError("A host-only verification archive cannot run CUDA inference.")
    required = {"bin/tensorfold-native", "SHA256SUMS", "VERSION", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "RUNTIME.md"}
    if not required.issubset(files):
        raise InstallError(f"Engine archive is missing required files: {sorted(required - files)}")
    sums_file = root / "SHA256SUMS"
    if sums_file.stat().st_size > 4 * 1024 * 1024:
        raise InstallError("Engine checksum manifest is too large.")
    sums = parse_sums(sums_file.read_text(encoding="utf-8"))
    if set(sums) != files - {"SHA256SUMS"}:
        raise InstallError("Engine archive and internal checksums name different files.")
    for name, digest in sums.items():
        verify_hash(root / name, digest)
    if (root / "VERSION").read_text(encoding="ascii").strip() != version:
        raise InstallError("Engine archive VERSION differs from its release.")
    executable = root / "bin/tensorfold-native"
    if not executable.stat().st_mode & stat.S_IXUSR:
        raise InstallError("Engine binary is not executable.")
