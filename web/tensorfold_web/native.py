"""TensorFold's native engine (1.0 and later): find the binary, ask what it can do, and build its `serve` command.

The binary is `tensorfold-native`; Homebrew installs it as `tensorfold`, which is also the name of the frozen Python
line's command, so a `tensorfold` on PATH counts only when its `--version` says it is the native one. The official
installer (`curl -fsSL https://tensorfold.dev/install.sh | sh`) links it as `~/.tensorfold/bin/tensorfold`. A TensorFold Web
started without these folders on its PATH (over SSH, from launchd) still finds it in their usual places.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .settings import Settings

TIMEOUT = 15.0
# where the official installer, Homebrew and release archives put the binary, for a TensorFold Web whose PATH lacks them
CANDIDATES = ("~/.tensorfold/bin/tensorfold", "/opt/homebrew/bin/tensorfold", "/usr/local/bin/tensorfold", "~/.local/bin/tensorfold-native",
              "/opt/tensorfold/bin/tensorfold-native")
# `serving NAME at URL (sampling: …; drafts: on; context: 262144; loaded in 13.9s)`
_SERVING = re.compile(r"serving (?P<name>\S+) at (?P<url>\S+) \(.*?drafts: (?P<drafts>\w+); "
                      r"context: (?P<context>\d+); loaded in (?P<loaded>[\d.]+)s\)")
# the CUDA engine's sizing line: `model 17.99 GiB; 8 streams at once, 0.18 GiB each at a 16384-token window, of
# 83.8 GiB left after a 12.2 GiB reserve`
_CUDA_SIZING = re.compile(r"model (?P<model>[\d.]+) GiB; (?P<streams>\d+) streams at once, (?P<each>[\d.]+) GiB each at a "
                          r"(?P<window>[\d,]+)-token window, of (?P<left>[\d.]+) GiB left after a (?P<reserve>[\d.]+) GiB reserve")
_PROMPT_CACHE = re.compile(r"a (?P<gib>[\d.]+) GiB prompt cache")      # 1.0.5's CUDA sizing line
# 1.0.5 on Metal: "prompt cache: 1.8 GiB from 1.8 GiB free under the 70% cap (18.6 GiB in use once loaded, …)"
_METAL_CACHE = re.compile(r"prompt cache: (?P<gib>[\d.]+) GiB from (?P<free>[\d.]+) GiB free(?:.*?\((?P<used>[\d.]+) GiB in use)?")
_INFO_BYTES = re.compile(r"\((\d+) bytes\)")
_INFO_GIB = re.compile(r"([\d.]+) GiB")

_found: dict[str, str | None] = {}            # binary path by PATH seen, so a poll does not spawn processes
# what a binary says of itself, by the file it is (its link's target and its time): an upgrade behind the same link,
# as `brew upgrade` or a new archive linked in place, is read afresh without restarting TensorFold Web
_caps: dict[tuple, dict] = {}
_flags: dict[tuple, dict[str, dict]] = {}
_versions: dict[tuple, str | None] = {}


def _stamp(path: str) -> tuple:
    try:
        real = os.path.realpath(path)
        return real, os.stat(real).st_mtime_ns
    except OSError:
        return path, None
# flags TensorFold Web sets itself from Settings, and flags that would put a secret on the command line
MANAGED = {"--host": "TensorFold Web sets it (Settings, Share on the network)", "--port": "the engine address in Settings",
           "--context": "Longest conversation in Settings", "--thinking": "Thinking in Settings",
           "--no-thinking": "Thinking in Settings", "--dashboard": "TensorFold Web turns it on",
           "--cortheon": "Web research in Settings, which says what leaves the machine"}
# Living Weights: learning is turned on by TensorFold Web for a copy that learns, bound to loopback; never by hand
LEARNING = ("--slide",)


def managed(flag: str) -> str | None:
    """Why TensorFold Web keeps ``flag`` to itself, or None: Settings' own flags, and every learning flag (--slide…)."""

    if flag in MANAGED:
        return MANAGED[flag]
    if flag.startswith(LEARNING):
        return "Teach turns learning on for a copy, kept on this machine; not an extra flag"
    return None
SECRET = {"--api-key": "the engine key goes in Settings and reaches the engine through its environment",
          "--api-key-file": "the engine key goes in Settings and reaches the engine through its environment"}
# `[--flag VALUE]`, `[--flag {a,b}]`, `[--flag]` and `[--a | --b]` in argparse's usage line
_USAGE = re.compile(r"\[(--[\w-]+)(?: ([A-Z_]+|\{[^}]*\}))?(?: \| (--[\w-]+))?\]")


def version_of(path: str) -> str | None:
    """`1.0.2` when ``path`` is the native engine, else None (the Python line's `tensorfold` answers otherwise)."""

    key = _stamp(path)
    if key not in _versions:
        try:
            out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=TIMEOUT,
                                 check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        m = re.match(r"tensorfold-native (\S+)", out)
        _versions[key] = m.group(1) if m else None
    return _versions[key]


def version_tuple(version: str | None) -> tuple[int, ...]:
    """`1.0.11` as (1, 0, 11), to compare releases; () when there is no version."""

    return tuple(int(x) for x in re.findall(r"\d+", version or "")[:3])


def find_binary() -> str | None:
    """The native binary: `TENSORFOLD_BIN` when it is one, else the newest release among `tensorfold-native` and a
    native `tensorfold` on PATH and the installer's, Homebrew's and the archives' usual places (the first found on a
    tie). Two installs side by side, such as the installer's 1.0.11 and an older Homebrew one, give the newer."""

    key = os.environ.get("TENSORFOLD_BIN", "") + "|" + os.environ.get("PATH", "")
    if key in _found and os.access(_found[key], os.X_OK):
        return _found[key]
    chosen = os.environ.get("TENSORFOLD_BIN")
    found = chosen if chosen and os.access(chosen, os.X_OK) and version_of(chosen) else None
    if found is None:
        names = [shutil.which("tensorfold-native"), shutil.which("tensorfold"),
                 *(os.path.expanduser(c) for c in CANDIDATES)]
        natives = [p for p in dict.fromkeys(names) if p and os.access(p, os.X_OK) and version_of(p)]
        found = max(natives, key=lambda p: version_tuple(version_of(p)), default=None)
    if found:                                     # a miss is not kept: an engine installed later is found then
        _found[key] = found
    return found


def forget() -> None:
    """Look for the binary again at the next call (after an install or an upgrade)."""

    _found.clear()
    _caps.clear()
    _flags.clear()
    _versions.clear()


def capabilities(binary: str) -> dict:
    """`capabilities --json`: version, chip, backends, families and their formats, `serve` flags, environment."""

    key = _stamp(binary)
    if key not in _caps:
        out = subprocess.run([binary, "capabilities", "--json"], capture_output=True, text=True, timeout=TIMEOUT, check=False)
        if out.returncode != 0:
            raise RuntimeError(f"`{binary} capabilities --json` failed: {out.stderr.strip()[:200]}")
        _caps[key] = json.loads(out.stdout)
    return _caps[key]


def parse_usage(text: str) -> dict[str, dict]:
    """Each flag of a usage line, whether it takes a value, and its choices when argparse lists them."""

    out: dict[str, dict] = {}
    for flag, value, alt in _USAGE.findall(text):
        choices = value[1:-1].split(",") if value.startswith("{") else None
        out[flag] = {"value": bool(value), "choices": choices}
        if alt:
            out[alt] = {"value": False, "choices": None}
    return out


def serve_flags(binary: str) -> dict[str, dict]:
    """The `serve` flags this binary offers (`capabilities --json`), each with its arity read from `serve --help`.

    One list drives the checks of the extra flags in Settings, the command TensorFold Web builds and the page's hints.
    A flag the help does not describe keeps ``value`` None: the check then accepts it with or without a value.
    """

    key = _stamp(binary)
    if key not in _flags:
        names = list(capabilities(binary).get("serve", {}).get("flags", {}))
        try:
            usage = subprocess.run([binary, "serve", "--help"], capture_output=True, text=True, timeout=TIMEOUT,
                                   check=False).stdout
        except (OSError, subprocess.SubprocessError):
            usage = ""
        arity = parse_usage(usage)
        _flags[key] = {n: {**arity.get(n, {"value": None, "choices": None}),
                              "managed": managed(n) or SECRET.get(n)} for n in names}
    return _flags[key]


def safe_extra(args: list[str]) -> tuple[list[str], list[str]]:
    """The extra flags with the managed and secret ones taken out (and their values): what reaches a command
    line, whatever was saved before the check existed. Returns what is kept and what was dropped."""

    kept, dropped = [], []
    skip_value = False
    for a in args:
        if skip_value and not a.startswith("--"):
            skip_value = False
            continue
        skip_value = False
        flag, eq, _ = a.partition("=")
        if managed(flag) or flag in SECRET:
            dropped.append(flag)
            skip_value = not eq
            continue
        kept.append(a)
    return kept, dropped


def check_extra(args: list[str], flags: dict[str, dict]) -> list[str]:
    """The extra `serve` flags as the engine will read them; raises ValueError on the first one it would not.

    Refused: a flag this engine does not offer, an abbreviation (argparse would take it, the next version may not),
    a flag TensorFold Web sets from Settings, and a secret on the command line.
    """

    out: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        flag, eq, inline = token.partition("=")
        if not flag.startswith("--"):
            raise ValueError(f"{token!r} is not a flag; flags start with --")
        if flag not in flags:
            longer = [f for f in flags if f.startswith(flag)]
            if longer:
                raise ValueError(f"{flag} is short for {', '.join(longer)}: write the flag in full")
            raise ValueError(f"{flag} is not a flag this engine offers")
        if flag in SECRET:
            raise ValueError(f"{flag} would put a secret on the command line: {SECRET[flag]}")
        if managed(flag):
            raise ValueError(f"{flag} is set by TensorFold Web: {managed(flag)}")
        spec = flags[flag]
        if eq:
            value = inline
        elif spec.get("value") is not False and i + 1 < len(args) and not args[i + 1].startswith("--"):
            value, i = args[i + 1], i + 1
        else:
            value = None
        if spec.get("value") is True and value is None:
            raise ValueError(f"{flag} needs a value")
        if spec.get("value") is False and value is not None:
            raise ValueError(f"{flag} takes no value")
        if value is not None and spec.get("choices") and value not in spec["choices"]:
            raise ValueError(f"{flag} takes one of {', '.join(spec['choices'])}")
        out += [flag] if value is None else [flag, value]
        i += 1
    return out


def parse_info(text: str) -> dict:
    """`tensorfold info` as numbers: family, format, context, weights and memory floor in bytes."""

    out: dict = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key, value = key.strip().lower(), value.strip()
        if key == "family":
            out["family"] = value
        elif key == "format":
            out["format"] = value
        elif key == "context":
            out["context"] = int(value.replace(",", ""))
        elif key == "weights":
            m = _INFO_BYTES.search(value)
            out["weights_bytes"] = int(m.group(1)) if m else None
        elif key == "memory floor":
            m = _INFO_GIB.search(value)
            out["floor_bytes"] = int(float(m.group(1)) * 1024**3) if m else None
    return out


def info(binary: str, model: str) -> dict | None:
    """What the engine knows of a checkpoint in the cache, or None when it is not there."""

    out = subprocess.run([binary, "info", model], capture_output=True, text=True, timeout=TIMEOUT, check=False)
    return parse_info(out.stdout) if out.returncode == 0 else None


def parse_startup(line: str) -> dict[str, float]:
    """The numbers of the engine's `serving …` line: the context it admitted and how long loading took."""

    mc = _METAL_CACHE.search(line)
    if mc:
        return {"prompt_cache_gib": float(mc.group("gib")), "prompt_cache_free_gib": float(mc.group("free")),
                **({"in_use_gib": float(mc.group("used"))} if mc.group("used") else {})}
    c = _CUDA_SIZING.search(line)
    if c:
        pc = _PROMPT_CACHE.search(line)
        return {"resident_gib": float(c.group("model")), "streams": float(c.group("streams")),
                "stream_gib": float(c.group("each")), "left_gib": float(c.group("left")),
                "reserve_gib": float(c.group("reserve")), **({"prompt_cache_gib": float(pc.group("gib"))} if pc else {})}
    m = _SERVING.search(line)
    if not m:
        return {}
    return {"context_tokens": float(m.group("context")), "loaded_s": float(m.group("loaded")),
            "drafts": 1.0 if m.group("drafts") == "on" else 0.0}


# a model the binary lists by family but will not load on this chip, such as the 27B on an M6 under 1.0.6:
# "cannot load <path> (QwenMetalQualificationRequired)", said before any weight is read
_REFUSED = re.compile(r"cannot load .*\((?P<why>\w*QualificationRequired)\)")


def refusal_in(lines: list[str]) -> str | None:
    """The engine's qualification refusal among its log lines, if it gave one."""

    for line in lines:
        m = _REFUSED.search(line)
        if m:
            return m.group("why")
    return None


def _refusals_path() -> Path:
    from .settings import config_dir

    return config_dir() / "native-refusals.json"


def refused(binary: str, model: str) -> str | None:
    """Why this binary's version refused ``model`` on this machine before, if it did: kept per version, so an
    upgrade asks again."""

    try:
        doc = json.loads(_refusals_path().read_text())
    except (OSError, ValueError):
        return None
    return (doc.get(version_of(binary) or "") or {}).get(model)


def remember_refusal(binary: str, model: str, why: str) -> None:
    from .settings import write_private

    path = _refusals_path()
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        doc = {}
    doc.setdefault(version_of(binary) or "", {})[model] = why
    write_private(path, json.dumps(doc, indent=1))


def build_command(binary: str, model: str, s: Settings, *, host: str = "127.0.0.1",
                  drafter: str | None = None, slide: str | None = None) -> tuple[list[str], dict, list[str]]:
    """The `serve` command for the settings, with only the flags this binary lists; the notes say what was left out.

    The native engine sizes its memory itself, so TensorFold Web's budget is not passed. The API key goes through the
    environment, never the command line.
    """

    flags = capabilities(binary).get("serve", {}).get("flags", {})
    notes: list[str] = []
    # a copy that learns stays on this machine whatever sharing says: anyone reaching the engine could teach it
    cmd = [binary, "serve", model, "--host", "0.0.0.0" if s.share and not slide else host, "--port", str(s.engine_port())]
    if s.share and slide:
        notes.append("a copy that learns is served on this machine only; reach it through TensorFold Web's relay")

    def add(flag: str, *values: str) -> None:
        if flag in flags:
            cmd.extend([flag, *values])
        else:
            notes.append(f"{flag} is not offered by this engine; left out")

    if drafter:                                   # the 27B drafts with a separate DFlash2 model, named here
        add("--drafter", drafter)
    if slide:                                     # a living copy: it learns, and keeps its fact graph in this file
        add("--slide")
        add("--slide-graph", slide)
    if s.context:
        add("--context", str(int(s.context)))
    if s.thinking == "on":
        add("--thinking")
    elif s.thinking == "off":
        add("--no-thinking")
    if s.cortheon:                                # searches and page reads leave the machine: Settings says so
        add("--cortheon")
    if "--dashboard" in flags:
        cmd.append("--dashboard")                # /stats for TensorFold Web, /dashboard for anyone curious
    try:
        cmd += check_extra(list(s.extra_args), serve_flags(binary))
    except ValueError as exc:                     # flags saved for another version: the engine starts without them
        notes.append(f"extra flags left out: {exc}")
    if s.budget_fraction:
        notes.append("the native engine sizes its memory itself; the memory budget does not apply")
    env = dict(os.environ)
    if s.api_key:
        env["TENSORFOLD_API_KEY"] = s.api_key
    return cmd, env, notes

