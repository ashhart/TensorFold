"""`tensorfold serve --engine auto|zig|python`: exec the native engine where the release gate passed it, else Python."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
from typing import Any, Callable, NoReturn

from tensorfold import native
from tensorfold.native import contract

ENV = "TENSORFOLD_ENGINE"
KNOBS = ("TENSORFOLD_", "TF_")       # a set variable with these prefixes must be one the native engine honours


class _Python(Exception):
    """Engine auto keeps the Python engine for this command."""


def requested(args: argparse.Namespace) -> tuple[str, str]:
    """(engine, how it was asked for): --engine, else TENSORFOLD_ENGINE, else auto (asked for by nobody)."""

    if getattr(args, "engine", None):
        return args.engine, f"--engine {args.engine}"
    value = os.environ.get(ENV, "").strip().lower()
    if value and value not in native.ENGINES:
        raise ValueError(f"{ENV} is auto, zig or python, not {value!r}")
    return (value, f"{ENV}={value}") if value else ("auto", "")


def backend(choice: str, platform: str = sys.platform) -> str:
    """The native backend for ``--backend``: auto is Metal on macOS and CUDA elsewhere, as for the Python engine."""

    return "metal" if choice == "mlx" or (choice == "auto" and platform == "darwin") else "cuda"


def hand_off(parser: argparse.ArgumentParser, args: argparse.Namespace, argv: list[str],
             config_dir: Callable[[], Path]) -> None:
    """Exec the native engine with this serve command's argv where it may serve it; else return to the Python engine."""

    engine, how = requested(args)
    if engine == "python":
        if how:
            print(f"[tensorfold] engine: python ({how})", flush=True)
        return

    def missing(reason: str) -> NoReturn:
        raise ValueError(f"{how}: {reason}") if engine == "zig" else _Python()

    try:
        binary, native_argv, line = _choose(parser, args, argv, config_dir, engine == "zig", missing, how)
    except _Python:
        return
    except Exception:
        if engine == "zig":
            raise
        return                                  # anything auto can't read: the Python engine serves, as it always did
    _exec(binary, native_argv, line, missing)


def _choose(parser: argparse.ArgumentParser, args: argparse.Namespace, argv: list[str],
            config_dir: Callable[[], Path], forced: bool, missing: Callable[[str], NoReturn],
            how: str) -> tuple[Path, list[str], str]:
    entries: list[dict[str, Any]] = []
    if not forced:
        entries = contract.manifest()["entries"]
        if not entries:                         # the empty manifest stops here, before any other look
            missing("no gate entries")
    binary = native.binary()
    if binary is None:
        missing(f"this install has no native engine ({native.ROOT / native.BINARY} is missing)")
    flags, native_argv, odd = scan(parser, argv)
    if odd:
        missing(f"the native engine takes full flag names and no '--' ({', '.join(odd)})")
    from tensorfold import families

    directory = config_dir()                    # forced: the Python engine's own error for a model it can't find
    config = families.read_config(directory)
    cell = {"family": families.model_type(directory), "format": contract.weight_format(config),
            "backend": backend(args.backend)}
    entries = [entry for entry in entries if all(entry[key] == value for key, value in cell.items())]
    if not forced and not entries:
        missing("no gate entry for this checkpoint and backend")
    try:
        found = contract.capabilities(binary)
    except ValueError as exc:
        missing(f"the native engine did not report its capabilities: {exc}")
    gaps = _gaps(found, flags, cell)
    if gaps:
        missing(f"the native engine {found['version']} lacks {', '.join(gaps)}")
    if forced:
        return binary, native_argv, f"[tensorfold] engine: zig {found['version']} ({how})"
    entries = [entry for entry in entries if entry["chip"] == found["chip"]]
    digest = native.bundle_digest() if entries else ""
    for entry in entries:
        if entry["bundle"] == digest:
            return binary, native_argv, (f"[tensorfold] engine: zig {found['version']} (release gate {entry['run']}: "
                                         f"{cell['family']} {cell['format']} on {found['chip']})")
    missing("no gate entry for this chip and bundle")


def scan(parser: argparse.ArgumentParser, argv: list[str]) -> tuple[list[tuple[str, str | None]], list[str], list[str]]:
    """(each serve flag given, with its value; argv without --engine; spellings the native engine won't take)."""

    # argparse keeps no public index of a subcommand's options; this reads argv argparse has already accepted
    commands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    options = commands.choices["serve"]._option_string_actions
    flags: list[tuple[str, str | None]] = []
    kept, odd = argv[:1], [] if argv[:1] == ["serve"] else argv[:1]
    tokens = iter(argv[1:])
    for token in tokens:
        if token == "--":
            odd.append(token)
            kept += [token, *tokens]
            break
        if not token.startswith("--"):
            kept.append(token)                  # the model, or a value such as -1
            continue
        name, sep, value = token.partition("=")
        known = [name] if name in options else [option for option in options if option.startswith(name)]
        if not known:
            kept.append(token)                  # argparse reads an unknown --word with a space in it as a value
            continue
        action = options[known[0]]
        if len(known) > 1 or action.nargs not in (None, 0):
            odd.append(token)
            continue
        if action.nargs is None and not sep:
            value = next(tokens, "")
        if action.dest == "engine":
            continue
        if known[0] != name:
            odd.append(name)                    # an abbreviation argparse accepted
        kept += [token] if sep or action.nargs == 0 else [token, value]
        flags.append((known[0], None if action.nargs == 0 else value))
    return flags, kept, odd


def _gaps(found: dict[str, Any], flags: list[tuple[str, str | None]], cell: dict[str, str]) -> list[str]:
    """What this command needs that the native engine doesn't report: flags, values, variables, checkpoint, backend."""

    gaps = []
    for name, value in flags:
        rule = found["serve"]["flags"].get(name)
        if rule is None or (value is not None and value not in rule.get("values", [value])):
            gaps.append(name if rule is None else f"{name} {value}")
    gaps += sorted(key for key in os.environ if key.startswith(KNOBS) and key != ENV and key not in found["env"])
    formats = found["families"].get(cell["family"])
    if formats is None:
        gaps.append(f"{cell['family']} checkpoints")
    elif cell["format"] not in formats:
        gaps.append(f"{cell['format']} weights for {cell['family']}")
    if cell["backend"] not in found["backends"]:
        gaps.append(f"the {cell['backend']} backend")
    return list(dict.fromkeys(gaps))


def _exec(binary: Path, argv: list[str], line: str, missing: Callable[[str], NoReturn]) -> None:
    """Replace this process: the same pid, stdio, signals and exit status, as if one program ran."""

    print(line, flush=True)
    sys.stderr.flush()
    restore = {}
    for name in ("SIGPIPE", "SIGXFSZ"):      # Python ignores these; a program started from a shell does not
        number = getattr(signal, name, None)
        if number is not None:
            restore[number] = signal.signal(number, signal.SIG_DFL)
    try:
        os.execv(binary, ["tensorfold", *argv])
    except OSError as exc:
        for number, handler in restore.items():
            signal.signal(number, handler)
        try:
            missing(f"the native engine did not start ({exc.strerror})")
        except _Python:
            print(f"[tensorfold] engine: python (the native engine did not start: {exc.strerror})", flush=True)
