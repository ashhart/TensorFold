"""launchd child supervisor: no shell, bounded logs, signal forwarding, graceful exit, no inference imports."""
from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .config import Profile, read_environment
from .safety import ControlError, no_symlinks, private_dir, private_read, redact


def logger_for(path: Path, maximum: int, backups: int) -> logging.Logger:
    private_dir(path.parent)
    for candidate in [path, *(Path(f"{path}.{i}") for i in range(1, backups + 1))]:
        no_symlinks(candidate)
    log = logging.Logger(f"tensorfold.service.{path.name}", logging.INFO)
    handler = RotatingFileHandler(path, maxBytes=maximum, backupCount=backups, encoding="utf-8")
    path.chmod(0o600)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(handler)
    return log


def supervise(argv: list[str], environment: dict[str, str], log: logging.Logger,
              stop: threading.Event, *, grace: float = 15) -> int:
    """Run one child, not a restart loop. launchd owns retry policy and process-group cleanup."""
    child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env=environment, bufsize=0, close_fds=True)
    log.info("[control] server process started pid=%d", child.pid)
    assert child.stdout is not None

    def pump() -> None:
        pending = b""
        try:
            while data := child.stdout.read(8192):
                pending += data
                while b"\n" in pending or len(pending) >= 16384:
                    end = pending.find(b"\n")
                    take = min(end if end >= 0 else 16384, 16384)
                    line, pending = pending[:take], pending[take + (end == take):]
                    log.info("%s", redact(line.decode("utf-8", errors="replace"), 16384))
            if pending:
                log.info("%s", redact(pending.decode("utf-8", errors="replace"), 16384))
        except (OSError, ValueError):
            log.warning("[control] log stream closed")

    reader = threading.Thread(target=pump, name="tensorfold-log", daemon=True)
    reader.start()
    stopping_at: float | None = None
    try:
        while child.poll() is None:
            if stop.is_set() and stopping_at is None:
                log.info("[control] forwarding SIGTERM; allowing %.1fs to exit", grace)
                child.terminate()
                stopping_at = time.monotonic()
            if stopping_at is not None and time.monotonic() - stopping_at >= grace:
                log.warning("[control] graceful-stop deadline reached; killing child")
                child.kill()
                break
            stop.wait(0.1) if not stop.is_set() else time.sleep(0.05)
        result = child.wait(timeout=max(grace, 1))
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=max(grace, 1))
        reader.join(timeout=2)
        child.stdout.close()
    log.info("[control] server exited status=%d", result)
    return 0 if stop.is_set() else (128 - result if result < 0 else min(result, 255))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    # Open diagnostics first, so a malformed profile is still visible to service logs/doctor.
    log = logger_for(args.log, 8 << 20, 4)
    stop = threading.Event()
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, lambda *_: stop.set())
    try:
        profile = Profile.decode(private_read(args.profile))
        for handler in log.handlers:
            handler.maxBytes = profile.log_bytes
            handler.backupCount = profile.log_backups
        environment = {**os.environ, **read_environment(profile)}
        if not profile.allow_download:
            log.info("[control] offline cache only; pull the model explicitly before first start")
        return supervise(profile.command(), environment, log, stop)
    except (ControlError, OSError, ValueError, subprocess.SubprocessError) as exc:
        log.error("[control] %s", redact(str(exc)))
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        for handler in log.handlers:
            handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
