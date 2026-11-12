"""Engine child."""

from __future__ import annotations

import contextlib
import subprocess

import psutil


class _Child:

    def __init__(self, popen: subprocess.Popen | None = None, found: psutil.Process | None = None) -> None:
        self.popen, self.found = popen, found
        self.pid = popen.pid if popen else found.pid if found else 0

    def alive(self) -> bool:
        if self.popen is not None:
            return self.popen.poll() is None
        try:
            return self.found is not None and self.found.is_running() and self.found.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    @property
    def returncode(self) -> int | None:
        return self.popen.returncode if self.popen is not None else None     # unknown for a process found again

    def terminate(self) -> None:
        with contextlib.suppress(ProcessLookupError, psutil.Error):
            (self.popen or self.found).terminate()

    def kill(self) -> None:
        with contextlib.suppress(ProcessLookupError, psutil.Error):
            (self.popen or self.found).kill()

    def wait(self, timeout: float) -> None:
        """Waits for the end; raises subprocess.TimeoutExpired as Popen does."""

        if self.popen is not None:
            self.popen.wait(timeout=timeout)
            return
        try:
            self.found.wait(timeout=timeout)
        except psutil.TimeoutExpired as exc:
            raise subprocess.TimeoutExpired("engine", timeout) from exc
        except psutil.Error:
            pass
