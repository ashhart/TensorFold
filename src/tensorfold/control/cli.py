"""Lazy CLI registration. Service commands need only the standard library, not TUI/GPU packages."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ..i18n import t
from ..i18n.parser import Parser
from . import __version__
from .config import Profile, Store, install_name
from .launchd import Manager, plist
from .logs import Tail
from .safety import ControlError, absolute, redact
from .telemetry import Client


def register(commands) -> None:
    service = commands.add_parser("service", help=t("manage per-user macOS launchd services"))
    actions = service.add_subparsers(dest="action", required=True)
    install = actions.add_parser("install", help=t("install a private LaunchAgent (starts next login; --start for now)"))
    install.add_argument("model", help=t("cached Hugging Face model ID or absolute local model directory"))
    install.add_argument("--name", default="default", help=t("profile name; lowercase letters, digits, hyphens"))
    install.add_argument("--python", default=str(absolute(sys.executable)), help=t("absolute serving-venv Python path"))
    install.add_argument("--host", default="127.0.0.1")
    install.add_argument("--port", type=int, default=8080)
    install.add_argument("--backend", choices=("mlx", "auto"), default="mlx")
    install.add_argument("--context", type=int)
    install.add_argument("--parallel", default="auto",
                         help=t("serve --parallel value written on the service command (default auto)"))
    install.add_argument("--drafter")
    install.add_argument("--arg", action="append", default=[], help=t("extra literal serve argument, e.g. --arg=--vision"))
    install.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help=t("non-secret override only"))
    install.add_argument("--env-file", help=t("absolute private JSON file for credentials/overrides; mode 0600"))
    install.add_argument("--api-key-file", help=t("restricted API key file passed to tensorfold serve"))
    install.add_argument(
        "--allow-network", action="store_true",
        help=t("acknowledge unauthenticated non-loopback binding"))
    install.add_argument("--allow-download", action="store_true", help=t("allow model downloads when service starts"))
    install.add_argument("--replace", action="store_true", help=t("replace a stopped, owned profile"))
    install.add_argument("--start", action="store_true", help=t("start now as well as at login"))
    install.add_argument(
        "--dry-run", action="store_true",
        help=t("print plist without writing files or calling launchctl"))
    install.add_argument("--json", action="store_true")
    install.set_defaults(func=cmd_service)
    for verb in ("start", "stop", "restart", "uninstall", "status", "doctor", "logs"):
        command = actions.add_parser(verb)
        command.add_argument("name", nargs="?", default="default")
        command.add_argument("--json", action="store_true")
        if verb == "uninstall":
            command.add_argument("--yes", action="store_true", help=t("confirm removal; logs and models are retained"))
        if verb == "logs":
            command.add_argument("--lines", type=int, default=80)
            command.add_argument("--follow", action="store_true")
        command.set_defaults(func=cmd_service)
    listing = actions.add_parser("list", help=t("list owned profiles without loading any models"))
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_service)
    tui = commands.add_parser("tui", help=t("TensorFold terminal control room (install the tui extra)"))
    tui.add_argument("--profile", help=t("initial local profile"))
    tui.add_argument("--url", action="append", default=[], help=t("read-only HTTP(S) endpoint; repeat for more"))
    tui.add_argument("--token-env", help=t("environment variable with API token, never saved or put in URLs"))
    tui.add_argument("--interval", type=float, default=1, help=t("poll interval in seconds, 0.5–30"))
    tui.add_argument("--color", choices=("auto", "truecolor", "256", "mono"), default="auto")
    tui.add_argument("--demo", action="store_true", help=t("simulated preview; no network or service operations"))
    tui.add_argument("--snapshot", type=Path, help=t("write one .svg/.html/.txt frame instead of opening a terminal"))
    tui.add_argument("--width", type=int, default=144, help=t("snapshot width"))
    tui.add_argument("--height", type=int, default=42, help=t("snapshot height"))
    tui.set_defaults(func=cmd_tui)


def parser() -> argparse.ArgumentParser:
    result = Parser(
        prog="tensorfold-control",
        description=t("TensorFold service and terminal control plane"))
    result.add_argument("--version", action="version", version=__version__,
                        help=t("show program's version number and exit"))
    register(result.add_subparsers(dest="command", required=True))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args) or 0)


def _profile(args) -> Profile:
    install_name(args.name)
    model = str(absolute(args.model)) if Path(args.model).expanduser().is_dir() else args.model
    extra = ["--parallel", args.parallel]
    if args.context is not None:
        if args.context < 0:
            raise ControlError(t("context must be nonnegative"))
        extra += ["--context", str(args.context)]
    if args.drafter:
        extra += ["--drafter", args.drafter]
    extra += args.arg
    environment = {}
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep or key in environment:
            raise ControlError(t("--env needs unique KEY=VALUE assignments"))
        environment[key] = value
    return Profile(args.name, model, python=args.python, host=args.host, port=args.port,
                   backend=args.backend, args=tuple(extra), environment=environment,
                   environment_file=str(absolute(args.env_file)) if args.env_file else None,
                   api_key_file=str(absolute(args.api_key_file)) if getattr(args, "api_key_file", None) else None,
                   allow_network=args.allow_network, allow_download=args.allow_download)


def _print(value: object, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(value, indent=2, ensure_ascii=False))
    elif isinstance(value, list):
        if not value:
            print(t("No TensorFold services installed. Use tensorfold service install MODEL."))
        for item in value:
            print(redact(f"{item.get('name', ''):<22} {item.get('state', ''):<14} {item.get('endpoint', '')}"))
    elif isinstance(value, dict):
        for key, item in value.items():
            print(redact(f"{key}: {item}"))
    else:
        print(redact(value))


def doctor(manager: Manager, name: str) -> tuple[dict, bool]:
    profile = manager.store.get(name)
    checks = []
    checks.append({"name": t("macOS user session"), "ok": manager.platform == "darwin" and manager.uid > 0})
    try:
        state = manager.status(name)
        checks.append({"name": t("launchd query"), "ok": manager.platform == "darwin", "state": state.state})
    except ControlError as exc:
        checks.append({"name": t("launchd query"), "ok": False, "detail": redact(str(exc))})
    try:
        from .config import read_environment
        read_environment(profile)
        checks.append({"name": t("private environment"), "ok": True})
    except (ControlError, OSError) as exc:
        checks.append({"name": t("private environment"), "ok": False, "detail": redact(str(exc))})
    try:
        probe = subprocess.run(
            [profile.python, "-c",
             "import importlib.util as u; "
             "assert u.find_spec('tensorfold.cli'); "
             "assert u.find_spec('tensorfold.control.runner')"],
            capture_output=True, text=True, timeout=8, check=False,
            cwd=manager.paths.working(name),
            env=plist(profile, manager.paths)["EnvironmentVariables"])
        checks.append({"name": t("serving interpreter imports"), "ok": probe.returncode == 0,
                       "detail": t("ready") if probe.returncode == 0
                       else t("install TensorFold and control in this venv")})
    except (OSError, subprocess.TimeoutExpired):
        checks.append({"name": t("serving interpreter imports"), "ok": False,
                       "detail": t("interpreter unavailable")})
    sample = Client(profile.endpoint).sample()
    checks.append({"name": t("HTTP readiness"), "ok": sample.online and sample.phase == "ready",
                   "state": t(sample.phase)})
    return {"name": name, "checks": checks}, all(c["ok"] for c in checks)


def cmd_service(args) -> int:
    manager = Manager()
    try:
        if args.action == "install":
            profile = _profile(args)
            if args.dry_run:
                print(manager.preview(profile).decode())
                return 0
            result = manager.install(profile, replace=args.replace, start=args.start)
        elif args.action == "list":
            profiles, errors = manager.store.list()
            rows = []
            for profile in profiles:
                try:
                    state = manager.status(profile.name).state
                except ControlError as exc:
                    state = t("query error: {error}", error=redact(str(exc), 120))
                rows.append({"name": profile.name, "state": state, "endpoint": profile.endpoint})
            _print(rows if not args.json else {"profiles": rows, "errors": errors}, args.json)
            for message in errors:
                print(redact(message), file=sys.stderr)
            return int(bool(errors))
        elif args.action == "logs":
            manager.store.get(args.name)
            if not 1 <= args.lines <= 2000:
                raise ControlError(t("--lines must be between 1 and 2000"))
            if args.json and args.follow:
                raise ControlError(t("--json and --follow cannot be combined"))
            tail = Tail(manager.paths.log(args.name), limit=args.lines)
            while True:
                lines = tail.read_new() if args.follow else tail.read()
                if args.json:
                    _print({"lines": lines}, True)
                else:
                    for line in lines:
                        print(line, flush=True)
                if not args.follow:
                    break
                time.sleep(0.5)
            return 0
        elif args.action == "doctor":
            report, ok = doctor(manager, args.name)
            _print(report, args.json)
            return 0 if ok else 1
        else:
            if args.action == "uninstall" and not args.yes:
                raise ControlError(t("uninstall requires --yes; logs and model files will be retained"))
            result = getattr(manager, args.action)(args.name)
        _print(asdict(result), args.json)
        return 0
    except (ControlError, OSError, ValueError) as exc:
        if args.json:
            _print({"error": redact(str(exc))}, True)
        else:
            print(t("tensorfold service: {error}", error=redact(str(exc))), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _tui_install(missing: str) -> str:
    spec = {"prompt_toolkit": "prompt-toolkit>=3.0.51,<4", "rich": "rich>=14,<16"}.get(
        missing, "prompt-toolkit>=3.0.51,<4")
    return f"{sys.executable} -m pip install '{spec}'"


def cmd_tui(args) -> int:
    try:
        from .app import ControlApp
        from .view import console_frame, use_truecolor
    except ImportError as exc:
        missing = (exc.name or "prompt_toolkit").split(".", 1)[0]
        print(t("tensorfold tui needs {missing}.", missing=missing), file=sys.stderr)
        print(_tui_install(missing), file=sys.stderr)
        return 1
    try:
        token = None
        if args.token_env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.token_env):
                raise ControlError(t("--token-env must name an environment variable"))
            token = os.environ.get(args.token_env)
            if not token:
                raise ControlError(t("the requested token environment variable is unset or empty"))
        terminal_options = {}
        if args.snapshot:
            from prompt_toolkit.input import DummyInput
            from prompt_toolkit.output import DummyOutput
            terminal_options = {"input": DummyInput(), "output": DummyOutput()}
        app = ControlApp(urls=args.url, profile=args.profile, demo=args.demo, interval=args.interval,
                         token=token, color=args.color, **terminal_options)
        if args.snapshot:
            if not 72 <= args.width <= 240 or not 23 <= args.height <= 100:
                raise ControlError(t("snapshot dimensions must be 72–240 by 23–100"))
            if not args.demo:
                import asyncio
                asyncio.run(app.refresh())
            _, console = console_frame(app.view, args.width, args.height, record=True,
                                       color=args.color != "mono", truecolor=use_truecolor(args.color))
            suffix = args.snapshot.suffix.lower()
            if suffix == ".svg":
                from rich.terminal_theme import TerminalTheme
                theme = TerminalTheme((10, 13, 23), (228, 236, 250), [(10, 13, 23)] * 8)
                data = console.export_svg(
                    title="TensorFold / Control Room" + (" · DEMO" if args.demo else ""),
                    theme=theme)
                # No network fonts/assets: previews should open entirely offline.
                data = re.sub(r"@font-face\s*\{.*?\}", "", data, flags=re.S)
                data = data.replace("Fira Code, monospace", "DejaVu Sans Mono, Menlo, Consolas, monospace")
            elif suffix == ".html":
                data = console.export_html()
            elif suffix == ".txt":
                data = console.export_text()
            else:
                raise ControlError(t("snapshot must end in .svg, .html or .txt"))
            from .safety import atomic_write
            atomic_write(absolute(args.snapshot), data.encode())
            print(t("Saved {path}", path=args.snapshot))
            return 0
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise ControlError(
                t("interactive TUI needs a terminal; use --demo --snapshot preview.svg for an offline preview"))
        app.run()
        return 0
    except (ControlError, OSError, ValueError) as exc:
        print(t("tensorfold tui: {error}", error=redact(str(exc))), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
