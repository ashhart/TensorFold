"""HTTP boundaries, download names and relay policy."""

from __future__ import annotations

import hmac
import mimetypes
import socket
from pathlib import Path
from urllib.parse import quote, urlsplit

from . import (
    recipes,
)

UI_DIST = (Path(__file__).parent / "ui_dist").resolve()

mimetypes.add_type("application/manifest+json", ".webmanifest")    # not in every system's table

TOKEN_HEADER = "x-studio-token"

def lan_hostname() -> str:
    name = socket.gethostname()
    return name if "." in name else name + ".local"

def connect_ctx(engine_url: str, model: str | None, *, share: bool, api_key: str | None) -> recipes.Ctx:

    parts = urlsplit(engine_url)
    port = parts.port or 8080
    base = f"http://{lan_hostname()}:{port}" if share and parts.hostname in ("127.0.0.1", "localhost") else engine_url.rstrip("/")
    # an engine started with a key wants it from every client, shared or not
    return recipes.Ctx(base=base, key=api_key or "local", model=model or "the served model",
                       docker_base=f"http://host.docker.internal:{port}")

LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1", "[::1]"}

UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}

def _site(netloc: str, scheme: str) -> tuple[str, int] | None:
    """`host[:port]` as (host, port), the port the scheme's default when absent; None for nothing."""

    netloc = netloc.strip().lower()
    if not netloc:
        return None
    parts = urlsplit(f"//{netloc}")
    try:
        port = parts.port
    except ValueError:
        return None
    return (parts.hostname or "", port or (443 if scheme == "https" else 80))

def attachment(name: str, fallback: str, ext: str, prefix: str = "") -> dict[str, str]:
    """A download's header: an ASCII file name every browser takes, and the name as written (any script) beside it."""

    slug = "".join(c if c.isascii() and c.isalnum() else "-" for c in name.lower()).strip("-")[:40] or fallback
    slug = "-".join(p for p in slug.split("-") if p) or fallback
    full = quote(f"{prefix}{name.strip()[:80] or fallback}{ext}", safe="")
    return {"Content-Disposition": f"attachment; filename=\"{prefix}{slug}{ext}\"; filename*=UTF-8''{full}"}

def same_secret(given: str, known: str) -> bool:

    return given.isascii() and known.isascii() and hmac.compare_digest(given, known)

def frame_policy(csp: str | None) -> str:
    """``csp`` with ``frame-ancestors 'none'``: an engine page's own policy is kept, its frame-ancestors replaced."""

    kept = [d.strip() for d in (csp or "").split(";") if d.strip() and not d.strip().lower().startswith("frame-ancestors")]
    return "; ".join([*kept, "frame-ancestors 'none'"])

def _host_of(value: str) -> str:
    """`host` from `host:port`, `[v6]:port` or a URL's netloc, lower-cased."""

    value = value.strip().lower()
    if value.startswith("["):
        return value[: value.find("]") + 1]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value

def relay_refuses(raw: str, path: str) -> str | None:

    from urllib.parse import unquote

    decoded = unquote(path)
    if any(c in decoded for c in "?#\\") or "%" in decoded or any(seg in ("..", ".") for seg in decoded.split("/")):
        return "this path is not forwarded to the engine"
    segments = [x.lower() for x in decoded.split("/") if x]
    if segments[:1] == ["slide"] or segments[:2] == ["v1", "slide"] or any(x.startswith("slide") for x in segments[:2]):
        return "teach and read what a copy learned through TensorFold Web (Teach screen or /api/living), not the relay"
    if raw and unquote(raw).strip("/") != decoded.strip("/"):
        return "this path is not forwarded to the engine"
    return None

REDEEM_BYTES = 1024                         # the body of the one route open without the token

VIEW, VIEW_COOKIE = "/living-view", "tfs_living_view"

LOGS_STREAM, LOGS_COOKIE = "/api/engine/logs/stream", "tfs_logs"      # an EventSource sends no header: a cookie

ENGINE_SETTINGS = ("engine_url", "remote", "model", "budget_fraction", "context", "thinking", "extra_args", "share",
                   "api_key", "cortheon")
