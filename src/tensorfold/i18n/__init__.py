"""Simplified Chinese for the command line and the control room; every other locale stays English."""
from __future__ import annotations

import os
import subprocess
import sys
import unicodedata
from functools import lru_cache
from typing import Any

#: environment variable that forces a locale ("zh", "en" or any locale tag)
LANG_ENV = "TENSORFOLD_LANG"


def locale() -> str:
    """The active locale: "zh" for Simplified Chinese, "en" for English. Re-resolved on every call."""

    # TENSORFOLD_LANG, pytest, LC_ALL, LC_MESSAGES, LANGUAGE, LANG, the macOS system language, English.
    # zh, zh_CN, zh-Hans and zh_SG select Chinese; zh_TW, zh_HK and zh-Hant select English.
    if value := os.environ.get(LANG_ENV, ""):
        return _locale_of(value)
    if _under_test():
        return "en"
    for key in ("LC_ALL", "LC_MESSAGES", "LANGUAGE", "LANG"):
        value = os.environ.get(key, "")
        if value and not _posix(value):
            return _locale_of(value)
    return _system_locale()


def t(message: str, **fields: Any) -> str:
    """The Simplified Chinese form of ``message``: the catalog entry, formatted, or the message itself."""

    text = _catalog().get(message, message) if locale() == "zh" else message
    return text.format(**fields) if fields else text


def pad(text: str, cells: int) -> str:
    """Left-justify ``text`` to ``cells`` terminal cells, so CJK and Latin label columns line up."""

    width = sum(2 if unicodedata.east_asian_width(character) in "WF" else 1 for character in text)
    return text + " " * max(0, cells - width)


def _locale_of(value: str) -> str:
    return "zh" if _is_zh(value) else "en"


def _posix(value: str) -> bool:
    """C, POSIX and their UTF-8 spellings carry no language preference."""

    return _tag(value) in ("", "c", "posix")


def _tag(value: str) -> str:
    """Lowercase ``value`` down to its language tag: before the first "." or "@", "_" as "-"."""

    for separator in (".", "@"):
        value = value.split(separator, 1)[0]
    return value.strip().lower().replace("_", "-")


def _is_zh(value: str) -> bool:
    """Whether a locale selects Simplified Chinese; Traditional Chinese selects English."""

    language, _, rest = _tag(value).partition("-")
    if language != "zh":
        return False
    return not any(part in ("tw", "hk", "mo", "hant") for part in rest.split("-"))


def _under_test() -> bool:
    return "pytest" in sys.modules


@lru_cache(maxsize=1)
def _system_locale() -> str:
    """macOS's preferred language; only reached when the environment carries no locale at all."""

    if sys.platform != "darwin":
        return "en"
    try:
        output = subprocess.run(["defaults", "read", "-g", "AppleLanguages"], capture_output=True, text=True,
                                timeout=2, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return "en"
    for line in output.splitlines():
        if value := line.strip().strip(",").strip('"').strip():
            return _locale_of(value)
    return "en"


@lru_cache(maxsize=1)
def _catalog() -> dict[str, str]:
    """Every Simplified Chinese catalog file merged into one lookup table."""

    from . import zh_cli, zh_control, zh_help, zh_parser, zh_tui

    merged: dict[str, str] = {}
    for source in (zh_help, zh_parser, zh_cli, zh_control, zh_tui):
        merged.update(source.MESSAGES)
    return merged


__all__ = ["LANG_ENV", "locale", "pad", "t"]
