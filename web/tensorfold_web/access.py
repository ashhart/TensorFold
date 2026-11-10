
from __future__ import annotations

import argparse
import hmac
import os
import platform
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path

from . import settings as settings_mod

LINK_SECONDS = 300


def token_path() -> Path:
    """Where `tensorfold-web service install` keeps the token (and where a TensorFold Web started by hand can keep it)."""

    return settings_mod.config_dir() / "web-token"


class Links:
    """One-time codes, in memory: a TensorFold Web restart forgets them, which only makes them shorter-lived."""

    def __init__(self) -> None:
        self._codes: dict[str, float] = {}
        self._lock = threading.Lock()

    def make(self) -> dict:
        code = secrets.token_urlsafe(24)
        with self._lock:
            now = time.time()
            self._codes = {c: t for c, t in self._codes.items() if t > now}
            self._codes[code] = now + LINK_SECONDS
        return {"code": code, "expires_in": LINK_SECONDS}

    def spend(self, code: str) -> bool:
        """The code is good once, before its time is up."""

        with self._lock:
            now = time.time()
            for known, until in list(self._codes.items()):
                if until <= now:
                    del self._codes[known]
                elif code.isascii() and hmac.compare_digest(known, code):
                    del self._codes[known]
                    return True
        return False


def _copy(text: str) -> bool:
    """To this machine's clipboard, with what it has: pbcopy, wl-copy, xclip, or Windows' clip from WSL."""

    tools = (["pbcopy"],) if platform.system() == "Darwin" else (
        ["wl-copy"], ["xclip", "-selection", "clipboard"], ["clip.exe"])
    for cmd in tools:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode(), check=True, timeout=5)
                return True
            except (OSError, subprocess.SubprocessError):
                continue
    return False


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="tensorfold-web token",
                                description="Show this machine's TensorFold Web token, copy it, or make a new one.")
    p.add_argument("--file", default=None, help=f"the token file (default {token_path()})")
    p.add_argument("--copy", action="store_true", help="copy it to the clipboard instead of printing it")
    p.add_argument("--new", action="store_true", help="make a new token; pages holding the old one ask again")
    args = p.parse_args(argv)
    path = Path(args.file).expanduser() if args.file else token_path()
    if args.new:
        settings_mod.write_private(path, secrets.token_urlsafe(32) + "\n")
        print(f"A new token is in {path}. Restart TensorFold Web for it to apply "
              "(`tensorfold-web service install`, or start it again with --token-file).")
    if not path.is_file():
        print(f"No token file at {path}. A TensorFold Web started without --token needs none and answers this machine only; "
              "`tensorfold-web service install` makes one, or start it with --token-file.")
        return 1
    token = path.read_text().strip()
    if args.copy:
        if _copy(token):
            print(f"The token from {path} is on the clipboard.")
            return 0
        print("No clipboard tool here (pbcopy, wl-copy, xclip); printing it instead.")
    print(token)
    if os.isatty(1):
        print(f"(from {path}; paste it when a TensorFold Web page asks, once per browser)")
    return 0
