"""Per-user launchd jobs. Never sudo, never a daemon, never mutate an unrelated job."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import time
from typing import Callable

from .config import Paths, Profile, Store, read_environment
from .safety import ControlError, atomic_write, file_lock, no_symlinks, private_dir, private_read, redact


@dataclass(frozen=True)
class Result:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def execute(argv: list[str], timeout: float = 20) -> Result:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=timeout, check=False, env={**os.environ, "LC_ALL": "C"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ControlError(f"launchctl could not complete: {type(exc).__name__}") from exc
    return Result(result.returncode, result.stdout, result.stderr)


@dataclass(frozen=True)
class Status:
    name: str
    loaded: bool
    state: str
    pid: int | None = None
    last_exit: int | None = None
    path: str | None = None
    detail: str = ""


def parse_status(name: str, text: str) -> Status:
    fields: dict[str, str] = {}
    matches = list(re.finditer(r"(?m)^([ \t]+)(state|pid|last exit code|path) = (.*?)\s*$", text))
    level = min((len(m[1]) for m in matches), default=0)
    for match in matches:
        if len(match[1]) == level:
            fields.setdefault(match[2], match[3].strip('"'))
    def number(key: str) -> int | None:
        try:
            return int(fields[key])
        except (KeyError, ValueError):
            return None
    return Status(name, True, fields.get("state", "loaded"), number("pid"), number("last exit code"),
                  fields.get("path"))


def plist(profile: Profile, paths: Paths) -> dict:
    """Stable interpreter/working directory; launchd does not run an interactive shell."""
    return {
        "Label": profile.label,
        "ProgramArguments": [profile.python, "-u", "-m", "tensorfold.control.runner", "--profile",
                             str(paths.profile(profile.name)), "--log", str(paths.log(profile.name))],
        "WorkingDirectory": str(paths.working(profile.name)),
        "EnvironmentVariables": {"PATH": f"{Path(profile.python).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
                                 "HOME": str(paths.home), "PYTHONUNBUFFERED": "1",
                                 "TENSORFOLD_NO_LIVE": "1", "TENSORFOLD_NO_UPDATE_CHECK": "1"},
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ExitTimeOut": 20,
        "ProcessType": "Interactive",
        "Umask": 0o077,
        # The runner owns size-bounded logs, including config and child-launch errors.
        "StandardOutPath": "/dev/null",
        "StandardErrorPath": "/dev/null",
    }


class Manager:
    def __init__(self, paths: Paths | None = None, *, run: Callable = execute,
                 platform: str | None = None, uid: int | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable = time.sleep):
        self.paths = paths or Paths()
        self.store = Store(self.paths)
        self.run = run
        self.platform = sys.platform if platform is None else platform
        self.uid = (os.getuid() if hasattr(os, "getuid") else -1) if uid is None else uid
        self.clock, self.sleep = clock, sleep

    @property
    def domain(self) -> str:
        return f"gui/{self.uid}"

    def _guard(self) -> None:
        if self.platform != "darwin":
            raise ControlError("launchd control is macOS-only; use --url for read-only remote monitoring")
        if self.uid <= 0:
            raise ControlError("run as your logged-in user, not root or sudo")

    def _call(self, *arguments: str, check: bool = True) -> Result:
        result = self.run(["/bin/launchctl", *arguments], 25)
        if check and result.returncode:
            detail = redact(result.stderr or result.stdout, 1600)
            raise ControlError(f"launchctl {arguments[0]} failed ({result.returncode}): {detail}")
        return result

    def status(self, name: str) -> Status:
        return self._status(self.store.get(name))

    def _status(self, profile: Profile) -> Status:
        name = profile.name
        if self.platform != "darwin":
            return Status(name, False, "monitor-only", detail="launchd requires macOS")
        result = self._call("print", f"{self.domain}/{profile.label}", check=False)
        if result.returncode:
            # Never reinterpret permission errors, missing GUI domains or arbitrary failures as 'not loaded'.
            message = (result.stderr + result.stdout).lower()
            if result.returncode in (3, 113) and "could not find service" in message:
                return Status(name, False, "stopped")
            raise ControlError(f"cannot inspect {profile.label}: {redact(result.stderr or result.stdout, 1600)}")
        return parse_status(name, result.stdout)

    def _owned(self, profile: Profile) -> None:
        try:
            installed = plistlib.loads(private_read(self.paths.plist(profile.name)))
        except (plistlib.InvalidFileException, ValueError) as exc:
            raise ControlError("managed plist is malformed; refusing to modify it") from exc
        if installed != plist(profile, self.paths):
            raise ControlError("plist differs from the managed profile; refusing to overwrite or control it")
        state = self.status(profile.name)
        if state.loaded and state.path != str(self.paths.plist(profile.name)):
            raise ControlError("loaded label has a different or unknown plist path; refusing to control it")

    def _gui(self) -> None:
        self._call("print", self.domain)

    def preview(self, profile: Profile) -> bytes:
        return plistlib.dumps(plist(profile, self.paths), fmt=plistlib.FMT_XML, sort_keys=False)

    def install(self, profile: Profile, *, replace: bool = False, start: bool = False) -> Status:
        self._guard()
        self._gui()
        if not Path(profile.python).is_file() or not os.access(profile.python, os.X_OK):
            raise ControlError("the profile's Python interpreter does not exist or is not executable")
        read_environment(profile)   # validate secrets/permissions before writing anything
        with file_lock(self.paths.root / "operation.lock"):
            current = self.paths.profile(profile.name)
            target = self.paths.plist(profile.name)
            old_profile = old_plist = None
            if current.exists() or current.is_symlink() or target.exists() or target.is_symlink():
                if not replace:
                    raise ControlError("profile or plist already exists; stop it, then use install --replace")
                previous = self.store.get(profile.name)
                self._owned(previous)
                if self.status(profile.name).loaded:
                    raise ControlError("stop the service before replacing its profile")
                old_profile, old_plist = private_read(current), private_read(target)
            elif self._status(profile).loaded:
                raise ControlError("this label is already loaded without a managed profile; refusing installation")
            profiles, errors = self.store.list()
            if errors:
                raise ControlError("fix malformed profiles before installing: " + "; ".join(errors))
            # Conservative on purpose: even different bind addresses may overlap a wildcard listener.
            if any(p.name != profile.name and p.port == profile.port for p in profiles):
                raise ControlError(f"port {profile.port} is already reserved by another TensorFold profile")
            private_dir(self.paths.working(profile.name))
            private_dir(self.paths.log_dir(profile.name))
            try:
                self.store.put(profile)
                atomic_write(target, self.preview(profile))
            except BaseException:
                # Restore both files; never leave a half-updated profile/plist pair.
                for path, before in ((current, old_profile), (target, old_plist)):
                    if before is None:
                        if path.exists() and not path.is_symlink():
                            path.unlink()
                    else:
                        atomic_write(path, before)
                raise
            self._call("enable", f"{self.domain}/{profile.label}")
            if start:
                return self._start(profile)
            return Status(profile.name, False, "installed", detail="starts at next login, or with service start")

    def _start(self, profile: Profile) -> Status:
        self._owned(profile)
        self._gui()
        state = self.status(profile.name)
        self._call("enable", f"{self.domain}/{profile.label}")
        if not state.loaded:
            self._call("bootstrap", self.domain, str(self.paths.plist(profile.name)))
        # bootstrap + RunAtLoad may already have started it. kickstart without -k never kills a live process.
        self._call("kickstart", "-p", f"{self.domain}/{profile.label}")
        return self.status(profile.name)

    def start(self, name: str) -> Status:
        self._guard()
        with file_lock(self.paths.root / "operation.lock"):
            return self._start(self.store.get(name))

    def _stop(self, profile: Profile, timeout: float = 45) -> Status:
        self._owned(profile)
        target = f"{self.domain}/{profile.label}"
        self._call("disable", target)   # explicit stop persists over login and suppresses KeepAlive
        if self.status(profile.name).loaded:
            self._call("bootout", target)
        deadline = self.clock() + timeout
        while self.status(profile.name).loaded:
            if self.clock() >= deadline:
                raise ControlError("service is still unloading; restart was not attempted")
            self.sleep(0.1)
        return Status(profile.name, False, "stopped", detail="disabled until explicitly started")

    def stop(self, name: str) -> Status:
        self._guard()
        with file_lock(self.paths.root / "operation.lock"):
            return self._stop(self.store.get(name))

    def restart(self, name: str) -> Status:
        self._guard()
        with file_lock(self.paths.root / "operation.lock"):
            profile = self.store.get(name)
            self._stop(profile)          # graceful SIGTERM, bounded wait, no unconditional kickstart -k
            return self._start(profile)

    def uninstall(self, name: str) -> Status:
        self._guard()
        with file_lock(self.paths.root / "operation.lock"):
            profile = self.store.get(name)
            self._stop(profile)
            for path in (self.paths.plist(name), self.paths.profile(name)):
                no_symlinks(path)
                path.unlink()
            # Logs and working directory intentionally survive removal.
        return Status(name, False, "uninstalled", detail="logs retained; no model files were removed")
