"""Another machine's Studio app read over its API (a job, a clip it made); no login, so trusted network only."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SUFFIXES = (".mp4", ".mov", ".png", ".jpg", ".jpeg", ".webp")
LIMIT = 4 << 30  # the largest file taken


def address(given: str) -> str:
    """`http://host:port` of an app from what a person types (`spark-5:7870`); ValueError with the reason otherwise."""

    text = (given or "").strip().rstrip("/")
    if text and not re.match(r"^[a-z]+://", text):
        text = "http://" + text
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.path not in ("", "/") or parts.query or parts.username:
        raise ValueError("another app's address is http://host:port, as its own page is opened")
    return f"{parts.scheme}://{parts.netloc}"


def job(base: str, job_id: str, timeout: float = 20) -> dict:
    """A job as that app reports it; LookupError when it has none, OSError when the app does not answer."""

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id or ""):
        raise ValueError("a job id is letters, digits and dashes")
    try:
        with urllib.request.urlopen(f"{address(base)}/api/jobs/{job_id}", timeout=timeout) as reply:
            return json.loads(reply.read())
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise LookupError(f"{address(base)} has no job {job_id} (its app forgets its jobs when it restarts: give the clip's path)") from None
        raise OSError(f"{address(base)} answered {error.code}") from None


def fetch(base: str, path: str, target: Path, timeout: float = 60, limit: int = LIMIT) -> Path:
    """That app's file `path` to ``target`` via .part; ValueError if not a clip or picture, LookupError if absent."""

    path = (path or "").strip().lstrip("/")
    if not path or ".." in Path(path).parts or Path(path).suffix.lower() not in SUFFIXES:
        raise ValueError("a clip's path is one inside the other app's studio folder, like renders/1013-131100-5eb633.mp4")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(f"{address(base)}/files/{urllib.parse.quote(path)}", timeout=timeout) as reply, open(part, "wb") as out:
            done = 0
            while chunk := reply.read(1 << 20):
                done += len(chunk)
                if done > limit:
                    raise ValueError(f"the file is over {limit >> 30} GB")
                out.write(chunk)
        if not done:
            raise LookupError(f"{address(base)} sent an empty file for {path}")
        part.replace(target)
    except urllib.error.HTTPError as error:
        if error.code in (400, 404):
            raise LookupError(f"{address(base)} has no file {path}") from None
        raise OSError(f"{address(base)} answered {error.code}") from None
    finally:
        part.unlink(missing_ok=True)
    return target
