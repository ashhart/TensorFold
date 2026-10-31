"""One TensorFold Web per user at a time: the shared state (settings, living copies, versions, journals) is recovered and
changed by one process, whose gates are its own. A second TensorFold Web started beside a running service stops before it
recovers, adopts or touches anything."""

from __future__ import annotations

import os
import time
from pathlib import Path

from . import settings as settings_mod

try:
    import fcntl
except ImportError:                                # Windows: msvcrt's byte lock instead
    fcntl = None
    import msvcrt


def _try_lock(f) -> bool:
    try:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        return True
    except OSError:
        return False


def lock_path() -> Path:
    return settings_mod.config_dir() / "web.lock"


def hold(wait: float = 15.0, path: Path | None = None):
    """Take the lock, waiting up to `wait` seconds for a TensorFold Web that is stopping (a service restart); the open file
    is returned and must be kept for the life of the process. None when another TensorFold Web keeps it."""

    if path is None:
        settings_mod.secure_config_dir()
    path = path or lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")  # noqa: SIM115 - kept open for the life of the process: the lock is the open file
    deadline = time.monotonic() + wait
    while not _try_lock(f):
        if time.monotonic() >= deadline:
            f.close()
            return None
        time.sleep(0.5)
    if fcntl is not None:                          # Windows locks the first byte: the pid is not written over it
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()))
        f.flush()
    return f


def holder(path: Path | None = None) -> str:
    try:
        return (path or lock_path()).read_text().strip()
    except OSError:
        return ""
