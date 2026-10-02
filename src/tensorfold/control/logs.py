"""Bounded, rotation-aware log reads. All terminal control sequences and common credentials are removed."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import stat

from .safety import ControlError, no_symlinks, redact


@dataclass
class Tail:
    path: Path
    limit: int = 400
    byte_limit: int = 128 << 10
    _identity: tuple[int, int] | None = None
    _position: int = 0
    _partial: bytes = b""
    _lines: list[str] = field(default_factory=list)
    _added: list[str] = field(default_factory=list)

    def read_new(self) -> list[str]:
        """Only newly completed records, preserving repeated identical lines during --follow."""
        self.read()
        return self._added[-self.limit:]

    def _append(self, line: str) -> None:
        self._lines.append(line)
        self._added.append(line)

    def read(self) -> list[str]:
        self._added = []
        try:
            no_symlinks(self.path)
            fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return list(self._lines)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ControlError("log is not a regular file")
            identity = (info.st_dev, info.st_ino)
            fresh = identity != self._identity or info.st_size < self._position
            if fresh:
                self._partial = b""
                self._position = max(0, info.st_size - self.byte_limit)
                if self._identity is not None:
                    self._append("— log rotated / truncated —")
                self._identity = identity
            skipped = self._position < max(0, info.st_size - self.byte_limit)
            if skipped:
                self._position = max(0, info.st_size - self.byte_limit)
                self._partial = b""
                self._append("— log burst truncated to keep the UI responsive —")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                stream.seek(self._position)
                data = stream.read(self.byte_limit)
                self._position = stream.tell()
            pieces = (self._partial + data).split(b"\n")
            self._partial = pieces.pop()[-8192:]
            if (fresh or skipped) and self._position - len(data) > 0 and pieces:
                pieces.pop(0)  # the first chunk started inside a line
            for piece in pieces:
                self._append(redact(piece.decode("utf-8", errors="replace"), 8192))
            self._lines = self._lines[-self.limit:]
            partial = [redact(self._partial.decode("utf-8", errors="replace"), 8192)] if self._partial else []
            return (self._lines + partial)[-self.limit:]
        finally:
            os.close(fd)
