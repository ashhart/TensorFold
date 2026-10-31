"""`tensorfold-web service`: keep TensorFold Web running, at login on macOS and with systemd on Linux.

macOS gets a LaunchAgent (TensorFold Web starts when the user's session opens, which an auto-login Mac does at boot);
Linux gets a systemd user unit, which starts at boot only with lingering on (`sudo loginctl enable-linger $USER`):
without it the user's systemd stops with the last session and takes TensorFold Web along. In both the engine TensorFold Web
starts outlives a restart of TensorFold Web, as when it runs by hand. Each machine's own variables (such as
HF_HUB_CACHE) go in `web.env` beside the settings, one KEY=value a line.
"""

from __future__ import annotations

import os
import plistlib
import re
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import settings as settings_mod

LABEL = "com.tensorfold.web"
UNIT = "tensorfold-web.service"
HAND_STARTED = "bin/tensorfold-web --no-browser"     # a TensorFold Web started by hand, stopped before the service starts


@dataclass
class Plan:
    """What `install` would do; `--check` prints it and does nothing."""

    platform: str
    path: Path
    content: str
    command: list[str]
    token_file: Path
    notes: list[str] = field(default_factory=list)


def platform_name() -> str:
    return "macos" if sys.platform == "darwin" else "linux" if sys.platform.startswith("linux") else sys.platform


def studio_command() -> list[str]:
    """This environment's `tensorfold-web`, or its module through this Python."""

    script = Path(sys.prefix) / "bin" / "tensorfold-web"
    return [str(script)] if script.is_file() else [sys.executable, "-m", "tensorfold_web.cli"]


def temporary_environment(prefix: str | None = None) -> bool:
    """`uvx` runs from an environment in uv's cache (`archive-v0/...`), which `uv cache clean` removes: a service
    recorded there would not start again."""

    parts = Path(prefix or sys.prefix).parts
    cache = os.environ.get("UV_CACHE_DIR")
    return any(re.fullmatch(r"archive-v\d+", x) for x in parts) or bool(
        cache and Path(prefix or sys.prefix).is_relative_to(Path(cache)))


def read_env(path: Path) -> dict[str, str]:
    """`KEY=value` lines; blank lines and # comments skipped."""

    out: dict[str, str] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _path_env() -> str:
    home = Path.home()
    brew = [Path("/opt/homebrew/bin")] if platform_name() == "macos" else []
    dirs = [home / ".tensorfold" / "bin", home / ".local" / "bin", *brew, Path("/usr/local/bin"), Path("/usr/bin"), Path("/bin")]
    return ":".join(str(d) for d in dirs)


def plan(*, port: int = 8090, host: str = "127.0.0.1", token_file: Path | None = None) -> Plan:
    conf = settings_mod.config_dir()
    token_file = token_file or conf / "web-token"
    cmd = [*studio_command(), "--no-browser", "--token-file", str(token_file)]
    if port != 8090:
        cmd += ["--port", str(port)]
    if host != "127.0.0.1":
        cmd += ["--host", host]
    notes: list[str] = []
    if not token_file.exists():
        notes.append(f"a new token will be written to {token_file} (readable by this user only)")
    env_file = conf / "web.env"
    name = platform_name()
    if name == "macos":
        logs = Path.home() / "Library" / "Logs" / "TensorFold Web"
        env = {"PATH": _path_env(), **read_env(env_file)}
        doc = {"Label": LABEL, "ProgramArguments": cmd, "RunAtLoad": True,
               "KeepAlive": {"SuccessfulExit": False}, "ThrottleInterval": 10,
               # the engine runs in its own session already; this keeps launchd from stopping it with TensorFold Web
               "AbandonProcessGroup": True, "WorkingDirectory": str(Path.home()), "EnvironmentVariables": env,
               "StandardOutPath": str(logs / "web.log"), "StandardErrorPath": str(logs / "web.log")}
        path = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
        if env_file.exists():
            notes.append(f"variables from {env_file} are copied into the agent: install again after changing them")
        return Plan(name, path, plistlib.dumps(doc).decode(), cmd, token_file, notes)
    if name == "linux":
        quoted = " ".join(_quote(a) for a in cmd)
        unit = "\n".join([
            "# Written by `tensorfold-web service install`; install again rather than editing it.",
            "[Unit]", "Description=TensorFold Web", "After=network-online.target",
            "# five failed starts in two minutes stop the unit instead of a restart every five seconds for ever",
            "StartLimitIntervalSec=120", "StartLimitBurst=5", "",
            "[Service]", f"WorkingDirectory={Path.home()}", f"Environment=PATH={_path_env()}",
            # a settings folder moved by XDG_CONFIG_HOME: the service reads the same settings, token and stores
            *([f"Environment={_quote('XDG_CONFIG_HOME=' + os.environ['XDG_CONFIG_HOME'])}"]
              if os.environ.get("XDG_CONFIG_HOME") else []),
            f"EnvironmentFile=-{env_file}", f"ExecStart={quoted}",
            f"StandardOutput=append:{conf / 'web.log'}", "StandardError=inherit",
            "# the engine TensorFold Web starts outlives a restart or a crash of TensorFold Web",
            "KillMode=process", "Restart=on-failure", "RestartSec=5",
            "# 4: another TensorFold Web of this user holds the instance lock; restarting would not help (3 is a failed start)",
            "RestartPreventExitStatus=4", "",
            "[Install]", "WantedBy=default.target", ""])
        path = Path.home() / ".config" / "systemd" / "user" / UNIT
        if not lingering():
            notes.append("lingering is off: the service stops when the last session ends and does not start at boot; "
                         "turn it on with `sudo loginctl enable-linger $USER`")
        return Plan(name, path, unit, cmd, token_file, notes)
    raise RuntimeError(f"no service for {name} yet: run TensorFold Web by hand")


def _quote(arg: str) -> str:
    """systemd's own quoting: a path with a space ("Application Support") in double quotes."""

    if not any(c in arg for c in ' "\\'):
        return arg
    return '"' + arg.replace("\\", "\\\\").replace('"', '\\"') + '"'     # a quote or backslash inside, escaped


def lingering() -> bool:
    user = os.environ.get("USER") or Path.home().name
    try:
        out = subprocess.run(["loginctl", "show-user", user, "-p", "Linger", "--value"], capture_output=True,
                             text=True, timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return out == "yes"


def _run(*cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), capture_output=True, text=True, timeout=30, check=False)


def _ensure_token(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_urlsafe(32) + "\n")


def _stop_hand_started() -> bool:
    """A TensorFold Web started by hand holds the port the service needs."""

    if shutil.which("pkill") is None:
        return False
    stopped = _run("pkill", "-f", HAND_STARTED).returncode == 0
    if stopped:
        time.sleep(3)
    return stopped


def install(p: Plan, *, start: bool = True) -> list[str]:
    """Write the agent or unit, and (re)start the service; the engine keeps running."""

    done: list[str] = []
    conf = settings_mod.config_dir()
    conf.mkdir(parents=True, exist_ok=True)
    conf.chmod(0o700)                             # the settings, keys and (on Linux) the log: this user's only
    _ensure_token(p.token_file)                   # a token kept elsewhere is 0600 itself; its folder is left alone
    p.path.parent.mkdir(parents=True, exist_ok=True)
    changed = not p.path.exists() or p.path.read_text() != p.content
    if changed:
        # readable by this user only: the agent carries web.env's variables (a Hugging Face token among them)
        settings_mod.write_private(p.path, p.content)
        done.append(f"wrote {p.path}")
    if p.platform == "macos":
        logs = Path.home() / "Library" / "Logs" / "TensorFold Web"
        logs.mkdir(parents=True, exist_ok=True)
        logs.chmod(0o700)
        domain = f"gui/{os.getuid()}"
        loaded = _run("launchctl", "print", f"{domain}/{LABEL}").returncode == 0
        if start:
            if loaded:
                _run("launchctl", "bootout", f"{domain}/{LABEL}")
                time.sleep(2)
            elif _stop_hand_started():
                done.append("stopped TensorFold Web started by hand")
            r = _run("launchctl", "bootstrap", domain, str(p.path))
            if r.returncode != 0:
                raise RuntimeError(f"launchctl bootstrap failed: {r.stderr.strip() or r.stdout.strip()}")
            done.append("started the agent")
    else:
        _run("systemctl", "--user", "daemon-reload")
        _run("systemctl", "--user", "enable", "-q", UNIT)
        if start:
            active = _run("systemctl", "--user", "is-active", "-q", UNIT).returncode == 0
            if not active and _stop_hand_started():
                done.append("stopped TensorFold Web started by hand")
            r = _run("systemctl", "--user", "restart", UNIT)
            if r.returncode != 0:
                raise RuntimeError(f"systemctl restart failed: {r.stderr.strip()}")
            done.append("enabled and (re)started the unit")
    return done


def uninstall(p: Plan) -> list[str]:
    """Stop the service and remove it; the engine keeps running, and a later TensorFold Web finds it."""

    done: list[str] = []
    if p.platform == "macos":
        if _run("launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}").returncode == 0:
            done.append("stopped the agent")
    else:
        _run("systemctl", "--user", "disable", "--now", "-q", UNIT)
        done.append("stopped and disabled the unit")
    if p.path.exists():
        p.path.unlink()
        done.append(f"removed {p.path}")
        if p.platform == "linux":
            _run("systemctl", "--user", "daemon-reload")
    return done


def status(p: Plan) -> dict:
    out = {"platform": p.platform, "file": str(p.path), "installed": p.path.exists(),
           "current": p.path.exists() and p.path.read_text() == p.content}
    if p.platform == "macos":
        r = _run("launchctl", "print", f"gui/{os.getuid()}/{LABEL}")
        out["running"] = r.returncode == 0 and "state = running" in r.stdout
    else:
        out["running"] = _run("systemctl", "--user", "is-active", "-q", UNIT).returncode == 0
        out["enabled"] = _run("systemctl", "--user", "is-enabled", "-q", UNIT).returncode == 0
        out["lingering"] = lingering()
    return out


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="tensorfold-web service",
                                 description="Keep TensorFold Web running: a LaunchAgent on macOS, a systemd user unit on Linux")
    ap.add_argument("action", choices=["install", "uninstall", "status"])
    ap.add_argument("--check", action="store_true", help="say what would be done, do nothing")
    ap.add_argument("--no-start", action="store_true", help="install without starting it now")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--token-file", type=Path, default=None)
    a = ap.parse_args(argv)
    try:
        p = plan(port=a.port, host=a.host, token_file=a.token_file)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    if a.action == "status":
        for k, v in status(p).items():
            print(f"{k}: {v}")
        return 0
    if a.check:
        verb = "install" if a.action == "install" else "remove"
        print(f"would {verb} {p.path} ({p.platform}), running: {' '.join(p.command)}")
        for n in p.notes:
            print(f"note: {n}")
        if a.action == "install":
            print("---\n" + p.content)
        return 0
    if a.action == "install" and temporary_environment():
        print("this tensorfold-web runs from uv's cache (uvx), which `uv cache clean` empties: the service could not "
              "start again. Install it for good first, then run this again:\n  uv tool install tensorfold-web\n"
              "  tensorfold-web service install", file=sys.stderr)
        return 2
    try:
        done = install(p, start=not a.no_start) if a.action == "install" else uninstall(p)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    for line in done or ["nothing to do"]:
        print(line)
    for n in p.notes:
        print(f"note: {n}")
    return 0
