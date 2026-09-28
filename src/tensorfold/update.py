"""Check releases without blocking the server; update with pip or fast-forward a clean editable clone."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from tensorfold import __version__

REPO = "ashhart/TensorFold"
RELEASES_API = f"https://api.github.com/repos/{REPO}/releases/latest"
REPO_URL = f"https://github.com/{REPO}.git"
CACHE = Path.home() / ".cache" / "tensorfold" / "update-check.json"
CACHE_SECONDS = 24 * 3600
CHANGELOG = "CHANGELOG.md"
SEEN = Path.home() / ".cache" / "tensorfold" / "version-seen"    # the last version whose news was shown


def parse_version(text: str) -> tuple[int, ...]:
    """``v0.3.1`` or ``0.3.1`` -> (0, 3, 1); anything after the numbers is ignored."""

    match = re.match(r"v?(\d+(?:\.\d+)*)", text.strip())
    if not match:
        raise ValueError(f"not a version: {text!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def newer(latest: str, current: str = __version__) -> bool:
    try:
        return parse_version(latest) > parse_version(current)
    except ValueError:
        return False


def latest_release(timeout: float = 3.0, *, use_cache: bool = True) -> str | None:
    """The newest release tag on GitHub (``v0.3.1``), from the day's cache when there is one; None offline."""

    if use_cache:
        try:
            cached = json.loads(CACHE.read_text())
            if time.time() - float(cached["checked"]) < CACHE_SECONDS and cached.get("tag"):
                return str(cached["tag"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    import urllib.request

    request = urllib.request.Request(RELEASES_API, headers={"Accept": "application/vnd.github+json",
                                                            "User-Agent": f"tensorfold/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            tag = str(json.loads(response.read())["tag_name"])
    except Exception:                  # offline, rate-limited or blocked: no answer is fine
        return None
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"tag": tag, "checked": time.time()}))
    except OSError:
        pass
    return tag


def notice(tag: str | None) -> str | None:
    if tag and newer(tag):
        return (f"[tensorfold] TensorFold {tag.lstrip('v')} is available (this is {__version__}): run "
                f"`tensorfold update`, then restart the server")
    return None


def _checks_off() -> bool:
    return os.environ.get("TENSORFOLD_NO_UPDATE_CHECK", "").strip().lower() in ("1", "true", "yes", "on")


def whats_new(text: str, since: str, upto: str) -> str:
    """The changelog's sections for versions after ``since`` up to ``upto``, as written (newest first)."""

    lines, keep = [], False
    for line in text.splitlines():
        heading = re.match(r"##\s+v?(\d+(?:\.\d+)+)", line)
        if heading:
            keep = parse_version(since) < parse_version(heading.group(1)) <= parse_version(upto)
        elif line.startswith("# "):
            keep = False
        if keep:
            lines.append(line)
    return "\n".join(lines).strip()


def changelog(tag: str, clone: Path | None = None) -> str | None:
    """CHANGELOG.md at ``tag``: the clone's copy when there is one, else GitHub's; None when unreachable."""

    if clone is not None:
        try:
            return (clone / CHANGELOG).read_text()
        except OSError:
            return None
    import urllib.request

    request = urllib.request.Request(f"https://raw.githubusercontent.com/{REPO}/{tag}/{CHANGELOG}",
                                     headers={"User-Agent": f"tensorfold/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read().decode("utf-8", "replace")
    except Exception:                  # offline or blocked: the link printed after it still works
        return None


def _remember(version: str) -> None:
    try:
        SEEN.parent.mkdir(parents=True, exist_ok=True)
        SEEN.write_text(version)
    except OSError:
        pass


def show_whats_new(since: str, tag: str, clone: Path | None = None) -> None:
    """Print what changed between ``since`` and ``tag``, then where every release's notes live."""

    text = changelog(tag, clone)
    notes = whats_new(text, since, tag) if text else ""
    if notes:
        print(f"\nWhat's new since {since}:\n\n{notes}\n", flush=True)
    print(f"[tensorfold] every release's notes: https://github.com/{REPO}/blob/{tag}/{CHANGELOG}", flush=True)
    _remember(tag.lstrip("v"))


def first_run_notice() -> str | None:
    """One line the first time a version runs, so installs updated some other way still hear what's new."""

    if _checks_off():
        return None
    try:
        seen = SEEN.read_text().strip()
    except OSError:
        seen = ""
    if seen == __version__:
        return None
    _remember(__version__)
    if seen and not newer(__version__, seen):
        return None
    return (f"[tensorfold] this is TensorFold {__version__}; what's new: "
            f"https://github.com/{REPO}/blob/v{__version__}/{CHANGELOG}")


def check_in_background() -> threading.Thread | None:
    """Look for a newer release without delaying anything; print one line if there is one."""

    if _checks_off():
        return None

    def run() -> None:
        line = notice(latest_release())
        if line:
            print(line, flush=True)

    thread = threading.Thread(target=run, name="tensorfold-update-check", daemon=True)
    thread.start()
    return thread


def _editable_clone() -> Path | None:
    """The git clone this package runs from when it was installed with ``pip install -e``, else None."""

    try:
        from importlib.metadata import distribution

        direct = distribution("tensorfold").read_text("direct_url.json")
        info: dict[str, Any] = json.loads(direct) if direct else {}
    except Exception:
        info = {}
    if not (info.get("dir_info") or {}).get("editable"):
        return None
    source = Path(__file__).resolve()
    for folder in source.parents:
        if (folder / ".git").exists():
            return folder
    return None


def _run(command: list[str], **kwargs: Any) -> int:
    print("[tensorfold] " + " ".join(command), flush=True)
    return subprocess.call(command, **kwargs)


def update(*, check_only: bool = False, force: bool = False) -> int:
    """``tensorfold update``: install the newest release (or say this one is current)."""

    tag = latest_release(timeout=10.0, use_cache=False)
    if tag is None:
        print(f"[tensorfold] could not reach GitHub to look for releases ({RELEASES_API})", file=sys.stderr)
        return 1
    if not newer(tag) and not force:
        print(f"[tensorfold] TensorFold {__version__} is the latest release")
        return 0
    print(f"[tensorfold] TensorFold {tag.lstrip('v')} is available (this is {__version__})")
    if check_only:
        return 0
    clone = _editable_clone()
    if clone is not None:
        dirty = subprocess.run(["git", "-C", str(clone), "status", "--porcelain"], capture_output=True, text=True)
        if dirty.returncode != 0 or dirty.stdout.strip():
            print(f"[tensorfold] this is an editable install from {clone}, which has local changes: update it "
                  f"yourself (git fetch --tags, then check out {tag})", file=sys.stderr)
            return 1
        if _run(["git", "-C", str(clone), "fetch", "--tags", "origin"]) != 0:
            return 1
        code = _run(["git", "-C", str(clone), "merge", "--ff-only", tag])
        if code != 0:
            print(f"[tensorfold] {clone} could not fast-forward to {tag}: update it yourself", file=sys.stderr)
        else:
            show_whats_new(__version__, tag, clone)
        return code
    # Upgrade dependencies only when the new release requires different versions.
    code = _run([sys.executable, "-m", "pip", "install", "--upgrade", f"git+{REPO_URL}@{tag}"])
    if code == 0:
        installed = subprocess.run([sys.executable, "-c", "import tensorfold; print(tensorfold.__version__)"],
                                   capture_output=True, text=True).stdout.strip()
        print(f"[tensorfold] installed TensorFold {installed or tag.lstrip('v')}; restart any running server")
        try:
            CACHE.unlink()
        except OSError:
            pass
        show_whats_new(__version__, tag)
    return code
