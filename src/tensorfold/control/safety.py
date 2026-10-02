"""Bounded, literal terminal text and private atomic files; no subprocess uses a shell."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import re
import stat
import tempfile
import unicodedata
from typing import Iterator


class ControlError(RuntimeError):
    """A recoverable user-facing control-plane error."""


# Remove OSC/DCS (including hyperlinks/clipboard sequences), CSI and short escapes before C0 filtering.
_ESCAPE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\|$)|"
                     r"\x1b[P^_].*?(?:\x1b\\|$)|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-_]", re.S)
_SECRET = re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+|"
                     r"((?:hf_token|api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*)"
                     r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)|\b(?:hf_[A-Za-z0-9]{12,}|sk-[A-Za-z0-9_-]{12,})")


def clean(value: object, limit: int = 4096) -> str:
    """Display data literally; never pass remote/log/config strings to a markup parser."""
    text = _ESCAPE.sub("", str(value)[:max(0, limit) * 2])
    return "".join(c for c in text if c in "\n\t" or unicodedata.category(c) not in {"Cc", "Cf", "Cs"})[:limit]


def redact(value: object, limit: int = 4096) -> str:
    def replace(match: re.Match) -> str:
        return (match.group(1) or match.group(2) or "") + "[REDACTED]"
    return _SECRET.sub(replace, clean(value, limit))


def absolute(path: str | Path) -> Path:
    """Keep a venv's interpreter symlink; resolving it would silently leave the venv."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def no_symlinks(path: Path) -> None:
    """Reject existing symlink components, including a dangling leaf. No mutations here."""
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ControlError(f"refusing symlink: {item}")


def private_dir(path: Path) -> None:
    no_symlinks(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = path.stat()
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise ControlError(f"directory is not owned by this user: {path}")
    path.chmod(0o700)


def private_read(path: Path, maximum: int = 1 << 20, *, owner_only: bool = True) -> bytes:
    """Use O_NOFOLLOW where available, and validate the opened descriptor, not just its name."""
    no_symlinks(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ControlError(f"not a regular file: {path}")
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            raise ControlError(f"file is not owned by this user: {path}")
        if owner_only and os.name != "nt" and st.st_mode & 0o077:
            raise ControlError(f"private file must be mode 0600: {path}")
        if st.st_size > maximum:
            raise ControlError(f"file exceeds {maximum} bytes: {path}")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(maximum + 1)
        if len(data) > maximum:
            raise ControlError(f"file exceeds {maximum} bytes: {path}")
        return data
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes) -> None:
    private_dir(path.parent)
    no_symlinks(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600) if hasattr(os, "fchmod") else None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Serialize local service mutations across CLI and TUI processes (macOS/POSIX)."""
    import fcntl
    private_dir(path.parent)
    no_symlinks(path)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ControlError("invalid service lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControlError("another TensorFold service operation is in progress") from exc
        yield
    finally:
        os.close(fd)
