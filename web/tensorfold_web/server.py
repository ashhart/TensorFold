"""TensorFold Web's HTTP app: machine, catalog, engine control, downloads, settings, the relay, and the page."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hmac
import json
import mimetypes
import queue
import re
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)

from . import (
    __version__,
    bench,
    brands,
    compare,
    conversations,
    diagnostic,
    engines,
    fit,
    fleet,
    health,
    history,
    living,
    living_checks,
    living_versions,
    quality,
    recipes,
    replay,
    report,
    series,
    storage,
    tooleval,
    usage,
)
from . import access as access_mod
from . import settings as settings_mod
from . import suites as custom_suites
from .catalog import Catalog, capacity_of
from .engine import Engine
from .gate import Busy, Gate
from .machine import detect
from .proxy import RELAYED, Proxy
from .pull import Downloads, NoRoom

UI_DIST = (Path(__file__).parent / "ui_dist").resolve()
mimetypes.add_type("application/manifest+json", ".webmanifest")    # not in every system's table
TOKEN_HEADER = "x-studio-token"


def lan_hostname() -> str:
    name = socket.gethostname()
    return name if "." in name else name + ".local"


def connect_ctx(engine_url: str, model: str | None, *, share: bool, api_key: str | None) -> recipes.Ctx:
    """The recipes' address, key and model, from the client's point of view: the engine's own address, or its
    network name when it is shared. A container on its own network reaches a shared engine through the host."""

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
    """A constant-time comparison that answers False, rather than raising, on a credential with non-ASCII text
    (`hmac.compare_digest` refuses those on str)."""

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
    """Why the relay will not forward this path, if it will not: an encoded character that would change what the
    engine reads (a ?, #, backslash, a dot segment), or any route of the learner (`/v1/slide/…`, `/slide…`)."""

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


# TensorFold's 3D view of what a copy learned, served through TensorFold Web, and the cookie that opens it
REDEEM_BYTES = 1024                         # the body of the one route open without the token
VIEW, VIEW_COOKIE = "/living-view", "tfs_living_view"
LOGS_STREAM, LOGS_COOKIE = "/api/engine/logs/stream", "tfs_logs"      # an EventSource sends no header: a cookie

# settings that change what the engine is or how it starts: not changed while something drives it
ENGINE_SETTINGS = ("engine_url", "remote", "model", "budget_fraction", "context", "thinking", "extra_args", "share",
                   "api_key", "cortheon")


def create_app(*, settings: settings_mod.Settings | None = None, engine: Engine | None = None,
               catalog: Catalog | None = None, token: str | None = None, bench_store: bench.Store | None = None,
               bind_host: str = "127.0.0.1", allowed_hosts: list[str] | None = None) -> FastAPI:
    s = settings or settings_mod.load()
    settings_mod.secure_config_dir()                 # conversations and the engine's log are this user's alone
    eng = engine or Engine(s, state_dir=settings_mod.config_dir())
    eng.adopt()                                  # an engine a previous TensorFold Web started may still run
    cat = catalog or Catalog()
    dl = Downloads()
    proxy = Proxy(lambda: s.engine_url, eng.key_in_use, target=lambda: eng.target()[1:])
    def bench_target() -> tuple[str, str | None, str | None]:
        h = eng.status()
        model = None
        if h.status == "running":
            # a bare /health behind a key names no model: /v1/models does
            model = h.model or next(iter(engines.identify(s.engine_url, key=eng.key_in_use()).models), None)
        return s.engine_url.rstrip("/").removesuffix("/v1"), eng.key_in_use(), model

    runner = bench.Bench(bench_store or bench.Store(settings_mod.config_dir() / "bench.sqlite"), bench_target)
    runner.price = lambda: (s.energy_price, s.currency)
    settings_lock = threading.Lock()             # a save's snapshot and a comparison's temporary values, one at a time

    def save_settings() -> None:
        """The settings to disk: the user's own, never a comparison's temporary configuration."""

        with settings_lock:
            settings_mod.save(dataclasses.replace(s, **committed))

    def user_settings() -> dict:
        """The settings as Settings shows them: the user's own while a comparison runs other configurations."""

        with settings_lock:
            return dataclasses.replace(s, **committed).to_dict(mask=True)

    start_lock = asyncio.Lock()                  # one start transaction at a time: settings, launch, save
    gate = Gate()                                # one change of the engine or its files at a time (see gate.py)
    app = FastAPI(title="TensorFold Web", version=__version__,
                  docs_url=None, redoc_url=None, openapi_url=None)   # no route map, and no CDN script, outside the token
    app.state.settings, app.state.engine, app.state.catalog, app.state.token = s, eng, cat, token
    app.state.bench, app.state.gate = runner, gate

    def _engine_key_given(request: Request) -> bool:
        # the key the running engine uses, and a new one waiting for its restart: tools holding either still work
        keys = {k for k in (eng.key_in_use(), s.api_key) if k}
        if not keys:
            return False
        auth = request.headers.get("authorization") or ""
        given = auth[7:] if auth.lower().startswith("bearer ") else request.headers.get("x-api-key") or ""
        return any(same_secret(given, k) for k in keys)

    def view_cookie() -> str:
        return hmac.new((token or "").encode(), b"living-view", "sha256").hexdigest()

    def logs_cookie() -> str:
        return hmac.new((token or "").encode(), b"logs-stream", "sha256").hexdigest()

    @app.middleware("http")
    async def _gate(request: Request, call_next):
        """With a TensorFold Web token, TensorFold Web's own API needs that token; the relay takes it or the engine's key.

        Administration and inference are two rights: a tool holding the engine's API key may talk to the model
        through `/engine/` without being able to stop it, read the settings or the secrets. The page is open.
        """

        p = request.url.path
        host = _host_of(request.headers.get("host", ""))
        # Without a token TensorFold Web trusts whoever reaches it, so only through the names of this machine: a page
        # from elsewhere that points its own name at 127.0.0.1 (DNS rebinding) is refused.
        if not token and host and host not in LOOPBACK_NAMES | {_host_of(h) for h in allowed_hosts or []}:
            return JSONResponse({"detail": f"this TensorFold Web answers on localhost only, not as {host!r}; start it with "
                                           "--token to reach it under another name"}, status_code=421)
        # A change asked by a page of another site (a form posted cross-site) is refused, token or not. The origin
        # is compared with the address asked, port included: another local server (localhost:3000) is another site.
        origin = request.headers.get("origin")
        # A tool that sends the engine's key in its own header is no cross-site form: a page of another site cannot
        # add that header for someone, and an app built on a web view (Jan) sends its own origin with every call.
        if p.startswith("/engine/") and _engine_key_given(request):
            return await call_next(request)
        if request.method in UNSAFE and origin == "null" and not token:
            return JSONResponse({"detail": "a change from a page without an origin is refused"}, status_code=403)
        if request.method in UNSAFE and origin and origin != "null":
            o = urlsplit(origin)
            mine = {_site(request.headers.get(h, ""), o.scheme) for h in ("host", "x-forwarded-host")} - {None}
            allowed = {_site(h, o.scheme) for h in allowed_hosts or []}
            if _site(o.netloc, o.scheme) not in mine | allowed:
                return JSONResponse({"detail": "a change from another site's page is refused"}, status_code=403)
        if request.method in UNSAFE and request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "a change from another site's page is refused"}, status_code=403)
        if token and p.startswith(VIEW):
            # TensorFold's fact graph page loads its own scripts and data: the token, once, becomes a cookie for
            # that path only, so neither the token nor the facts are open to whoever reaches TensorFold Web
            given = request.headers.get(TOKEN_HEADER) or ""      # never in the address: logs and history keep it
            if not (same_secret(given, token) or
                    same_secret(request.cookies.get(VIEW_COOKIE, ""), view_cookie())):
                return HTMLResponse("<p>This view needs TensorFold Web's token: open it from Teach.</p>", status_code=401)
        if token and p.startswith(("/api/", "/engine/")) and p != "/api/access/redeem":   # a one-time code is its own key
            # the header only: a token in an address ends up in logs and histories (the page never sends it there)
            given = request.headers.get(TOKEN_HEADER) or ""
            by_cookie = p == LOGS_STREAM and same_secret(request.cookies.get(LOGS_COOKIE, ""), logs_cookie())
            if not same_secret(given, token) and not by_cookie and \
                    not (p.startswith("/engine/") and _engine_key_given(request)):
                detail = ("a TensorFold Web token is required" if p.startswith("/api/") else
                          "a TensorFold Web token or the engine's API key is required")
                return JSONResponse({"detail": detail}, status_code=401)
        return await call_next(request)

    @app.middleware("http")
    async def _no_framing(request: Request, call_next):
        """No page of another site may frame these pages: every response, the gate's refusals and the relay's
        included. Registered after the gate, so it wraps it."""

        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = frame_policy(response.headers.get("content-security-policy"))
        return response

    # figures kept by the server, so Monitor's charts draw as soon as a page opens
    used = usage.Usage(settings_mod.config_dir() / "usage.sqlite")
    kept = history.History(settings_mod.config_dir() / "history.sqlite", lambda: eng.info(reader="history"),
                           then=used.record)
    app.state.history = kept

    @app.on_event("startup")
    async def _startup():
        cat.refresh_sizes_async()
        kept.start()

    @app.on_event("shutdown")
    async def _shutdown():
        kept.stop()

    @app.get("/api/history")
    def figures_history(window: int = 1800):
        if not 60 <= window <= int(history.KEEP_S):
            raise HTTPException(400, f"window must be 60 to {int(history.KEEP_S)} seconds")
        return kept.since(window)

    @app.get("/api/usage")
    def usage_view(days: int = 30):
        """What the engine served by day and by hour, by model, kept across engine and TensorFold Web restarts."""

        if not 1 <= days <= usage.KEEP_DAYS:
            raise HTTPException(400, f"days must be 1 to {usage.KEEP_DAYS}")
        return used.summary(days)

    # -- machine, catalog ----------------------------------------------------------------
    @app.get("/api/machine")
    def machine():
        m = detect()
        cap = capacity_of(m)
        return {**m.to_dict(), "studio_version": __version__,
                "budget": {"default_fraction": cap.fraction if cap else fit.DEFAULT_FRACTION,
                           "raised_fraction": cap.raised if cap else fit.RAISED_FRACTION,
                           "current_fraction": s.budget_fraction or (cap.fraction if cap else fit.DEFAULT_FRACTION),
                           "reserve_bytes": cap.reserve_bytes if cap else fit.PROCESS_RESERVE,
                           "ceiling_bytes": cap.ceiling_bytes if cap else None,
                           "capacity_bytes": cap.bytes if cap else None, "basis": cap.basis if cap else "none"}}

    @app.post("/api/catalog/refresh")
    def catalog_refresh():
        """Sizes read again from Hugging Face: a POST, so a page elsewhere cannot make TensorFold Web do it with a GET."""

        cat.refresh_sizes(force=True)
        return catalog_view()

    @app.get("/api/catalog")
    def catalog_view():
        m = detect()
        return {"models": cat.view(m, fraction=s.budget_fraction), "sizes_known": bool(cat._sizes),
                "refreshing": cat.refreshing, "refreshed_at": cat.refreshed_at}

    @app.get("/api/catalog/{ref}/verdicts")
    def copy_verdicts(ref: str):
        """A Living Weights copy (``living:<name>``) takes the memory of the model it was copied from."""

        if not ref.startswith(living.PREFIX):
            raise HTTPException(404, "unknown model")
        try:
            base = living.base_of(ref.removeprefix(living.PREFIX), [])
        except ValueError as exc:
            raise HTTPException(404, "unknown copy") from exc
        if not base or "/" not in base:
            raise HTTPException(404, "the model this copy came from is not known")
        return verdicts(*base.split("/", 1))

    @app.get("/api/catalog/{owner}/{name}/verdicts")
    def verdicts(owner: str, name: str):
        """The verdict at every budget from the default to the raised one, each one for exactly that budget."""

        m = detect()
        cap = capacity_of(m)
        repo = f"{owner}/{name}"
        entry = next((e for e in cat.entries if e.id == repo), None)
        if entry is None:
            raise HTTPException(404, "unknown model")
        if cap is None:
            raise HTTPException(409, "no supported GPU on this machine")
        weights, workspace, _ = cat.fit_inputs(entry)
        if not weights:
            raise HTTPException(409, "size unknown yet")
        out = []
        lo, hi = round(cap.fraction * 100), round(cap.raised * 100)
        for pct in range(lo, hi + 1, 5):
            v = cat.verdict_of(entry, m, weights, workspace, pct / 100, strict=True)
            out.append({"fraction": pct / 100, **v.to_dict()})
        return {"model": repo, "weights_bytes": weights, "verdicts": out}

    # -- engine ---------------------------------------------------------------------------
    signals, checker = health.Signals(), health.Checker()

    def friendly_name(served: str | None) -> str | None:
        """The served model as a person knows it: the catalog's name ("Nemotron 3.5 Lightning 30B-A3B"), a living
        copy's own name, or what the engine says."""

        if not served:
            return None
        copy = living.name_of(served)
        if copy:
            return copy
        low = served.lower().rstrip("/")
        for e in cat.entries:
            if low in (e.id.lower(), e.id.split("/")[-1].lower()) or low.endswith("/" + e.id.lower()):
                return e.name
        return served

    @app.get("/api/engine")
    def engine_info():
        info = eng.info()
        # this machine's own signals say nothing about a remote engine's
        info["findings"] = [f.to_dict() for f in checker.check(info, {} if s.remote else signals.read())]
        info["last_served"] = used.last_served()
        info["model_name"] = friendly_name(info.get("model") or (eng.model if eng.owned else None))
        return info

    @app.post("/api/engine/start")
    async def engine_start(body: dict[str, Any]):
        """Start the engine with the body's settings: validated first, applied for the launch, saved only once it ran.

        One transaction at a time; any failure, a refusal or a system error alike, puts the settings back.
        """

        if s.remote:
            raise HTTPException(409, "the engine is remote; start it on its own machine")
        try:
            patch = settings_mod.validate({k: body[k] for k in ("model", "context", "budget_fraction", "thinking")
                                           if k in body})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        async with start_lock:
            return await held_in_thread("an engine start", lambda: _start_held(patch))

    def key_for_learning() -> None:
        """TensorFold refuses `--slide` without an API key (zig/src/server/slide_key.zig). A copy that learns gets one,
        kept in Settings as a shared engine's is, so the relay and the lessons present it."""

        if s.api_key:
            return
        with eng.target_lock:
            s.api_key = settings_mod.new_api_key()
            eng.retargeted()
        save_settings()

    def launch(model: str, entry: Any) -> dict:
        """Start ``model`` on the native engine when it serves it here, else on the Python line. A native engine that
        lists the model's family but refuses this chip (it says so before reading any weight) is remembered for its
        version, and the model starts on the Python line at once."""

        from . import native

        copy = living.name_of(model)
        if copy:                                            # a living copy: its folder, learning where the machine can
            path = living.folder(copy)
            if not path.is_dir() or living.is_unfinished(copy):
                raise RuntimeError(f"there is no living copy named {copy} on this machine")
            why = living.family_refusal(copy) or living.teachable_here(native.find_binary())
            if not why:
                key_for_learning()
            result = eng.start(model, native_model=True, serve=str(path),
                               slide=None if why else str(living.graph_path(copy)))
            if why:
                eng.note(f"{copy} serves without learning: {why}")
            return result
        here = bool(entry and cat.native_here(entry))
        result = eng.start(model, native_model=here, drafter=entry.drafter if here and entry else None)
        if not here:
            return result
        end = time.time() + 8
        while time.time() < end:                            # a refusal comes before any weight is read
            if native.refusal_in(list(eng.log)[-20:]) or eng.exit_code is not None or eng.health().ok:
                break
            time.sleep(0.25)
        if eng.owned and not native.refusal_in(list(eng.log)[-20:]):
            return result                                   # loading or serving: nothing was refused
        if not eng.wait_stopped(5):                         # gone, and its log read to the last line
            return result                                   # still there: nothing is started over it
        why = native.refusal_in(list(eng.log)[-40:])
        binary = native.find_binary()
        if not why or binary is None:
            return result
        native.remember_refusal(binary, model, why)
        result = eng.start(model, native_model=False)      # a start clears the log: the reason goes after it
        eng.note(f"the native engine does not load {model} on this machine ({why}); started on the Python line")
        return result

    async def held_in_thread(who: str, work: Any) -> Any:
        """``work`` in a worker thread that takes the gate itself and gives it back when the work ends. Nothing is held
        on the request's side: cancelled before the thread runs, nothing was taken; cancelled after, the thread keeps
        the gate until the engine has really started or stopped."""

        def run() -> Any:
            try:
                gate.take(who)
            except Busy as exc:
                raise HTTPException(409, str(exc)) from exc
            try:
                return work()
            finally:
                gate.give()
        return await asyncio.to_thread(run)

    def _start_held(patch: dict[str, Any]) -> dict:
        """The start itself, the gate held: settings applied for the launch, put back if it does not happen."""

        model = patch.get("model") or s.model
        if not model:
            raise HTTPException(400, "no model chosen")
        patch["model"] = model
        previous = {k: getattr(s, k) for k in patch}
        for k, v in patch.items():
            setattr(s, k, v)
        try:
            entry = next((e for e in cat.entries if e.id == model), None)
            result = launch(model, entry)
        except Exception as exc:
            for k, v in previous.items():                   # a start that did not happen changes nothing
                setattr(s, k, v)
            if isinstance(exc, RuntimeError):
                raise HTTPException(409, str(exc)) from exc
            raise HTTPException(500, f"could not launch the engine: {exc}") from exc
        save_settings()
        return {**result, **eng.info()}

    links = access_mod.Links()

    @app.post("/api/access/link")
    def access_link():
        """A one-time code, five minutes, for a page on another device: made only for someone already in."""

        if not token:
            raise HTTPException(409, "this TensorFold Web has no token: it answers this machine only, and needs no link")
        return links.make()

    @app.post("/api/access/redeem")
    async def access_redeem(request: Request):
        """A one-time code traded for the token, once: the page keeps the token as if it had been typed. The one route
        open without the token reads at most a kilobyte: a larger body is refused before it is held in memory."""

        raw = b""
        async for part in request.stream():
            raw += part
            if len(raw) > REDEEM_BYTES:
                raise HTTPException(413, "a one-time code is a few dozen characters")
        try:
            body = json.loads(raw or b"{}")
        except ValueError as exc:
            raise HTTPException(400, "expected a JSON object with the code") from exc
        code = body.get("code") if isinstance(body, dict) else None
        if not token or not isinstance(code, str) or not links.spend(code):
            raise HTTPException(403, "this link was used already, has expired, or is not this TensorFold Web's")
        return {"token": token}

    @app.get("/api/access")
    def access(request: Request):
        """How TensorFold Web and its engine can be reached, for the header and the security notes."""

        # bound to loopback but asked under another name: a proxy (tailscale serve, a tunnel) carries it further
        asked = _host_of(request.headers.get("x-forwarded-host") or request.headers.get("host", ""))
        via = asked if asked and asked not in LOOPBACK_NAMES else None
        studio_local = _host_of(bind_host) in LOOPBACK_NAMES and via is None
        eng_host = _host_of(urlsplit(s.engine_url).netloc)
        engine_local = eng_host in LOOPBACK_NAMES and not (s.share and not s.remote)
        open_without_key = None
        if not engine_local:                      # does the engine answer anyone, without its key?
            try:
                r = httpx.get(s.engine_url.rstrip("/") + "/v1/models", timeout=2.0)
                open_without_key = r.status_code == 200
            except httpx.HTTPError:
                open_without_key = None
        posture = ("danger" if open_without_key else "warn" if not engine_local or not studio_local else "ok")
        return {"studio": {"bind": bind_host, "local_only": studio_local, "token": bool(token), "via": via},
                "engine": {"url": s.engine_url, "local_only": engine_local, "key_set": bool(s.api_key),
                           "answers_without_key": open_without_key},
                "posture": posture}

    @app.get("/api/engine/flags")
    def engine_flags():
        """The native engine's `serve` flags, for the extra flags field in Settings."""

        from . import native
        binary = native.find_binary()
        if not binary:
            return {"binary": None, "version": None, "flags": {}}
        try:
            return {"binary": binary, "version": native.version_of(binary), "flags": native.serve_flags(binary)}
        except (RuntimeError, ValueError, OSError) as exc:
            raise HTTPException(502, f"the engine did not describe its flags: {exc}") from exc

    @app.post("/api/engine/stop")
    async def engine_stop():
        await held_in_thread("an engine stop", eng.stop)
        return eng.info()

    @app.get("/api/engine/logs")
    def engine_logs(lines: int = 200):
        return {"lines": list(eng.log)[-lines:]}

    @app.post("/api/engine/logs/session")
    def engine_logs_session(request: Request):
        """The page asks with its token in a header; the answer is a cookie for the stream's path only, so the token
        never goes in the stream's address (which proxies and access logs keep)."""

        resp = JSONResponse({"ok": True})
        resp.set_cookie(LOGS_COOKIE, logs_cookie(), httponly=True, samesite="strict", path=LOGS_STREAM,
                        secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https")
        return resp

    @app.get("/api/engine/logs/stream")
    async def engine_logs_stream():
        q: queue.Queue = queue.Queue(maxsize=1000)
        eng.subscribe(q)

        async def gen():
            try:
                for line in list(eng.log)[-50:]:
                    yield f"data: {json.dumps(line)}\n\n"
                while True:
                    try:
                        line = q.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.25)
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(line)}\n\n"
            finally:
                eng.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # -- downloads --------------------------------------------------------------------------
    @app.get("/api/pull")
    def pull_list():
        return {"jobs": dl.list()}

    @app.post("/api/pull")
    def pull_start(body: dict[str, Any]):
        repo = body.get("repo")
        if not isinstance(repo, str) or "/" not in repo:
            raise HTTPException(400, "repo must be owner/name")
        try:
            with gate.hold("a download's start"):
                return dl.start(repo, total_bytes=cat.size_of(repo)).to_dict()
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        except NoRoom as exc:
            raise HTTPException(507, str(exc)) from exc

    @app.get("/api/pull/{owner}/{name}")
    def pull_status(owner: str, name: str):
        return dl.get(f"{owner}/{name}").to_dict()

    @app.delete("/api/pull/{owner}/{name}")
    def pull_cancel(owner: str, name: str):
        """Stops a download; what reached the disk stays and the next Download resumes it."""
        if not dl.cancel(f"{owner}/{name}"):
            raise HTTPException(404, "no download of that model is going")
        return dl.get(f"{owner}/{name}").to_dict()

    # -- bench --------------------------------------------------------------------------------
    qd = quality.load()
    quality_info = {"about": qd["about"], "version": qd["version"], "digest": qd["digest"],
                    "categories": [{"id": k, "name": v["name"], "items": len(v["items"])} for k, v in qd["categories"].items()]}

    @app.get("/api/bench/suites")
    def bench_suites():
        return {"modes": bench.MODES, "protocol": bench.PROTOCOL, "sizes": bench.SIZES, "about": runner.suites.get("about"),
                "version": runner.suites.get("version"), "digest": runner.suites.get("digest"),
                "suites": [{"id": x["id"], "name": x["name"], "prompts": len(x["prompts"]),
                            "quick": sum(1 for p in x["prompts"] if p.get("quick"))} for x in runner.suites["suites"]],
                "custom": [{**c, "digest": custom_suites.digest(c)} for c in runner.custom.list()],
                "quality": quality_info,
                "tools": {"installed": tooleval.find() is not None}}

    # -- custom suites: written in the page or imported from a file, exported as one JSON file -------------------
    @app.post("/api/bench/tools/install")
    def tool_eval_install():
        """Installs tool-eval-bench for this user with uv, so the Tool calls run needs no command typed."""
        try:
            return tooleval.install()
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/bench/suites")
    def custom_suite_create(body: dict[str, Any]):
        try:
            return runner.custom.save(body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/bench/suites/copy/{builtin_id}")
    def custom_suite_copy(builtin_id: str):
        """A built-in suite copied into the user's, to change its prompts; the built-in one stays as it is."""
        suite = next((x for x in runner.suites["suites"] if x["id"] == builtin_id), None)
        if suite is None:
            raise HTTPException(404, "unknown built-in suite")
        return runner.custom.save(custom_suites.from_builtin(suite))

    @app.put("/api/bench/suites/{suite_id}")
    def custom_suite_update(suite_id: str, body: dict[str, Any]):
        try:
            return runner.custom.save(body, suite_id)
        except KeyError as exc:
            raise HTTPException(404, "unknown suite") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/bench/suites/{suite_id}")
    def custom_suite_delete(suite_id: str):
        if not runner.custom.delete(suite_id):
            raise HTTPException(404, "unknown suite")
        return {"deleted": suite_id}

    @app.get("/api/bench/suites/{suite_id}/export")
    def custom_suite_export(suite_id: str):
        suite = runner.custom.get(suite_id)
        if suite is None:
            raise HTTPException(404, "unknown suite")
        return JSONResponse(custom_suites.export(suite), headers=attachment(
            suite["name"], "suite", ".json", prefix="suite-"))

    @app.get("/api/bench/runs")
    def bench_runs(limit: int = 50):
        return {"runs": runner.store.runs(limit), "busy": runner.busy(),
                "current": runner.current["id"] if runner.busy() and runner.current else None}

    def served_name() -> str | None:
        """The served engine as receipts name it: TensorFold's native version when this TensorFold Web runs it."""

        ident = engines.identify(s.engine_url, key=eng.key_in_use())
        name = ident.engine + (f" {ident.version}" if ident.version else "") if ident.engine else None
        if ident.engine == "TensorFold" and eng.owned and eng.kind == "native":
            from . import native
            binary = native.find_binary()
            version = native.version_of(binary) if binary else None
            name = f"TensorFold {version} (native)" if version else name
        return name

    # -- compare mode: configurations of the served engine, one after the other --------------------------------
    def engine_now() -> dict:
        h = eng.status()
        return {"model": eng.model if eng.owned and h.status in ("running", "starting") else None,
                "context": s.context, "budget_fraction": s.budget_fraction, "settings_model": s.model}

    # what a comparison sets for one of its configurations is not the user's: kept here, the values it replaced are
    # what a settings save writes, and they come back when the comparison ends
    committed: dict[str, Any] = {}

    def temporarily(**values: Any) -> None:
        with settings_lock:
            for k, v in values.items():
                committed.setdefault(k, getattr(s, k))
                setattr(s, k, v)

    def comparison_ended() -> None:
        with settings_lock:
            for k, v in committed.items():
                setattr(s, k, v)
            committed.clear()

    def switch(c: dict, cancelled: Callable[[], bool] = lambda: False) -> dict:
        """Puts the engine in configuration ``c`` and waits until it serves; restarts it only when needed. ``cancelled``
        is asked while the engine loads: a comparison cancelled meanwhile stops the load instead of waiting it out."""

        if s.remote:
            raise RuntimeError("the engine is remote: TensorFold Web cannot restart it")
        model, ctx, frac = c.get("model"), c.get("context"), c.get("budget_fraction")
        entry = next((e for e in cat.entries if e.id == model), None)
        if model and frac is None and entry is not None:
            # a model that fits only tightly gets the budget its verdict names, as Models starts it
            v = next((x.get("verdict") for x in cat.view(detect(), fraction=None) if x["id"] == model), None)
            if v and v.get("level") == "tight":
                frac = v.get("budget_fraction")
        if not model:                                   # the engine was not serving before the comparison
            eng.stop()
            temporarily(context=c.get("context"), budget_fraction=c.get("budget_fraction"), model=c.get("settings_model"))
            return {"model": None}
        same = eng.owned and eng.model == model and s.context == ctx and s.budget_fraction == frac
        if not (same and eng.status().status == "running"):
            eng.stop()
            temporarily(model=model, context=ctx, budget_fraction=frac)
            launch(model, entry)
            end = time.time() + 900
            while time.time() < end:
                if cancelled():
                    eng.stop()
                    raise RuntimeError("cancelled while the engine was loading")
                st = eng.status()
                if st.status == "running":
                    break
                if st.status == "failed" or not eng.owned:
                    raise RuntimeError(st.error or "the engine stopped while loading; its log says why")
                time.sleep(2)
            else:
                raise RuntimeError("the engine did not serve within 15 minutes")
            # the admitted context comes in a startup line that may follow the first answer: give it a moment
            settle = time.time() + 15
            while "context_tokens" not in eng.startup and time.time() < settle:
                time.sleep(0.5)
        if "settings_model" in c:
            temporarily(model=c["settings_model"])
        return {"model": model, "context_tokens": eng.startup.get("context_tokens"), "budget_fraction": frac,
                "loaded_s": eng.startup.get("loaded_s"), "kind": eng.kind}

    comparisons = series.Series(runner.store.path, runner, switch, engine_now, served_name, gate=gate,
                                on_end=comparison_ended)
    app.state.series = comparisons
    app.state.temporarily, app.state.comparison_ended = temporarily, comparison_ended

    @app.post("/api/bench/series")
    def series_start(body: dict[str, Any]):
        if s.remote:
            raise HTTPException(409, "the engine is remote: a comparison restarts it, which only its own TensorFold Web can do")
        try:
            return comparisons.start(body.get("configs"), mode=str(body.get("mode") or "profile"),
                                     suites=body.get("suites") or None, exactness=bool(body.get("exactness", True)))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/bench/series")
    def series_list():
        return {"series": [series.matrix(d, runner.store.run) for d in comparisons.list()],
                "current": comparisons.doc["id"] if comparisons.busy() and comparisons.doc else None}

    @app.get("/api/bench/series/{sid}")
    def series_get(sid: str):
        doc = comparisons.doc if comparisons.busy() and comparisons.doc and comparisons.doc["id"] == sid \
            else comparisons.get(sid)
        if doc is None:
            raise HTTPException(404, "unknown comparison")
        return series.matrix(doc, runner.store.run)

    @app.post("/api/bench/series/{sid}/cancel")
    def series_cancel(sid: str):
        if not (comparisons.busy() and comparisons.doc and comparisons.doc["id"] == sid):
            raise HTTPException(409, "this comparison is not going")
        comparisons.cancel()
        return {"cancelling": True}

    def concurrency(body: dict) -> int | None:
        n = body.get("max_concurrency")
        if n is not None and (isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= max(bench.LEVELS)):
            raise ValueError(f"max_concurrency must be 1 to {max(bench.LEVELS)}")
        return n

    def bench_go(mode: str, **kw: Any) -> dict:
        # held for the whole run, not only its start: no lesson, restart or settings change lands in a measurement
        gate.take("a Bench run")
        try:
            if kw.get("target") is None:                 # the copy and its facts, read under the gate: they cannot move
                tag = living_tag()
                if tag:
                    kw["tags"] = {**(kw.get("tags") or {}), "living": tag}
            run = runner.start(mode, **kw)
        except BaseException:
            gate.give()
            raise

        def until_done() -> None:
            try:
                while runner.busy():
                    time.sleep(0.5)
            finally:
                gate.give()
        threading.Thread(target=until_done, name="bench-gate", daemon=True).start()
        return run

    @app.post("/api/bench/runs")
    def bench_start(body: dict[str, Any]):
        if comparisons.busy():
            raise HTTPException(409, "a comparison of configurations is going; cancel it or wait")
        if runner.busy():
            raise HTTPException(409, "a run is already going; cancel it or wait")
        suites = body.get("suites")
        if suites is not None and not (isinstance(suites, list) and all(isinstance(x, str) for x in suites)):
            raise HTTPException(400, "suites must be a list of suite ids")
        target = body.get("target")
        try:
            # a KL run's reference: another engine (an address, or a Fleet machine), or (none given) the same engine
            # with drafts off; and what both read: their replies to the suites, or a fixed text scored at every position
            reference, kl_load, kl_text, tags = None, 0, None, {}
            if body.get("mode") == "kl":
                ref = body.get("reference")
                if ref is not None:
                    r_ident, r_key, r_headers, r_peer = engine_named(ref, "reference")
                    if r_ident.error:
                        raise ValueError(f"{r_peer or r_ident.url}: {r_ident.error}")
                    r_model = ref.get("model") or next(iter(r_ident.models), None)
                    if r_model not in r_ident.models:
                        raise ValueError(f"{r_peer or r_ident.url} does not serve {r_model!r}")
                    reference = (r_ident.url, r_key, r_model, r_headers)
                    if r_peer:
                        tags["reference_peer"] = r_peer
                kl_load = body.get("kl_load") or 0
                if isinstance(kl_load, bool) or not isinstance(kl_load, int) or not 0 <= kl_load <= 32:
                    raise ValueError("kl_load must be 0 to 32 requests")
                kl_text = body.get("kl_text")
            if target is not None:
                # another engine: identified, its model the one asked or the first it lists
                ident, key, headers, peer = engine_named(target, "target")
                at = peer or ident.url
                if ident.error:
                    raise ValueError(f"{at}: {ident.error}")
                model = target.get("model") or next(iter(ident.models), None)
                if model not in ident.models:
                    raise ValueError(f"{at} does not serve {model!r}; it serves {', '.join(ident.models) or 'nothing'}")
                name = ident.engine + (f" {ident.version}" if ident.version else "")
                label = target.get("label")
                if isinstance(label, str) and label.strip() and ident.engine == "OpenAI-compatible":
                    name = label.strip()[:60]           # a server that does not say what it is, named by the user
                if peer:
                    tags["peer"] = peer
                max_context = body.get("max_context")
                if max_context is not None and not (isinstance(max_context, int) and max_context > 0):
                    raise ValueError("max_context must be a number of tokens")
                return bench_go(str(body.get("mode") or "quick"), suites=suites or None,
                                    exactness=bool(body.get("exactness", True)), engine_name=name,
                                    max_context=max_context, target=(ident.url, key, model, headers),
                                    max_concurrency=concurrency(body), full=bool(body.get("full")),
                                    reference=reference, kl_load=kl_load, kl_text=kl_text, tags=tags or None)
            name = served_name()
            max_context = body.get("max_context")
            if max_context is not None and not (isinstance(max_context, int) and max_context > 0):
                raise ValueError("max_context must be a number of tokens")
            return bench_go(str(body.get("mode") or "quick"), suites=suites or None,
                                exactness=bool(body.get("exactness", True)), engine_name=name, max_context=max_context,
                                max_concurrency=concurrency(body), full=bool(body.get("full")),
                                reference=reference, kl_load=kl_load, kl_text=kl_text, tags=tags or None)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    def fleet_engine(pid: object) -> tuple[engines.Endpoint, str | None, dict, str]:
        """A Fleet machine's engine, reached through its TensorFold Web's relay with the key or token kept here: the page names
        the machine, never holds its secrets. Unknown machine: 404; one that does not answer: 502."""

        if not isinstance(pid, str) or not pid:
            raise HTTPException(400, "peer must be the id of a machine in Fleet")
        peer = next((p for p in peers.all() if p["id"] == pid), None)
        if peer is None:
            raise HTTPException(404, "no such machine in the fleet")
        try:
            base, key, headers = fleet.engine_of(peer)
            with fleet.client() as c:
                c.headers.update(headers)
                ident = engines.identify(base, key=key, client=c)
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(502, f"{peer['name']} did not answer: {exc}") from exc
        return ident, key, headers, peer["name"]

    def engine_named(spec: object, what: str) -> tuple[engines.Endpoint, str | None, dict, str | None]:
        """The engine a Bench request names: a Fleet machine (``peer``), or an address with the key typed. Returns
        it identified, its key, the headers to send beside it, and the machine's name for a Fleet machine."""

        if isinstance(spec, dict) and "peer" in spec:
            return fleet_engine(spec["peer"])
        if not isinstance(spec, dict) or not isinstance(spec.get("url"), str):
            raise ValueError(f"{what} must be an object with a url, or a Fleet machine's id as peer")  # noqa: TRY004 - one error type for the page
        key = spec.get("key") or None
        return engines.identify(settings_mod._engine_url(spec["url"]), key=key), key, {}, None

    @app.post("/api/bench/identify")
    def bench_identify(body: dict[str, Any]):
        """Which engine answers at an address, or serves a Fleet machine (``peer``), and the models it serves: for
        the Bench's other-engine target and the KL reference."""

        try:
            ep, *_ = engine_named(body, "the engine")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"url": ep.url, "engine": ep.engine, "version": ep.version, "models": ep.models, "error": ep.error}

    @app.get("/api/bench/runs/{run_id}")
    def bench_run(run_id: str):
        # the server's clock goes with the run, so the page counts elapsed time from it and not its own
        if runner.busy() and runner.current and runner.current["id"] == run_id:
            return {**runner.current, "results": runner.store.run(run_id)["results"], "server_time": time.time()}
        run = runner.store.run(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        return {**run, "server_time": time.time(), "findings": report.findings(run), "share_text": report.share_text(run)}

    @app.post("/api/bench/runs/{run_id}/cancel")
    def bench_cancel(run_id: str):
        if not (runner.busy() and runner.current and runner.current["id"] == run_id):
            raise HTTPException(409, "this run is not going")
        runner.cancel()
        return {"cancelling": True}

    @app.delete("/api/bench/runs/{run_id}")
    def bench_delete(run_id: str):
        if runner.busy() and runner.current and runner.current["id"] == run_id:
            raise HTTPException(409, "cancel the run before deleting it")
        if not runner.store.delete(run_id):
            raise HTTPException(404, "unknown run")
        return {"deleted": run_id}

    @app.get("/api/bench/runs/{run_id}/receipt")
    def bench_receipt(run_id: str):
        run = runner.store.run(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        return JSONResponse(bench.receipt(run), headers={
            "Content-Disposition": f'attachment; filename="bench-{run_id}.json"'})

    def _two(a: str, b: str) -> tuple[dict, dict]:
        ra, rb = runner.store.run(a), runner.store.run(b)
        if ra is None or rb is None:
            raise HTTPException(404, "unknown run")
        return ra, rb

    @app.get("/api/bench/compare")
    def bench_compare(a: str, b: str):
        ra, rb = _two(a, b)
        try:
            return compare.compare(ra, rb)
        except compare.Refused as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/bench/compare/report")
    def bench_compare_report(a: str, b: str):
        ra, rb = _two(a, b)
        try:
            c = compare.compare(ra, rb)
        except compare.Refused as exc:
            raise HTTPException(409, str(exc)) from exc
        return HTMLResponse(compare.html_report(ra, rb, c), headers={
            "Content-Disposition": f'attachment; filename="bench-{a}-vs-{b}.html"'})

    @app.get("/api/bench/runs/{run_id}/report")
    def bench_report(run_id: str):
        run = runner.store.run(run_id)
        if run is None:
            raise HTTPException(404, "unknown run")
        return HTMLResponse(report.html_report(run), headers={
            "Content-Disposition": f'attachment; filename="bench-{run_id}.html"'})

    # -- settings, connect ------------------------------------------------------------------
    @app.get("/api/settings")
    def get_settings():
        return user_settings()

    @app.put("/api/settings")
    def put_settings(body: dict[str, Any]):
        try:
            patch = settings_mod.validate(body)
            if patch.get("extra_args"):
                from . import native
                binary = native.find_binary()
                if binary:                               # checked against the flags this engine offers
                    patch["extra_args"] = native.check_extra(patch["extra_args"], native.serve_flags(binary))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        held = False
        if any(k in patch for k in ENGINE_SETTINGS):
            try:
                gate.take("a settings change")              # compared and applied under the gate, never before it
                held = True
            except Busy as exc:
                # something drives the engine: its settings stay as that leaves them; the rest of the patch applies
                if any(patch[k] != getattr(s, k) for k in ENGINE_SETTINGS if k in patch):
                    raise HTTPException(409, f"the engine's settings are not changed now: {exc}") from exc
                patch = {k: v for k, v in patch.items() if k not in ENGINE_SETTINGS}
        try:
            target_changed = any(k in patch and patch[k] != getattr(s, k) for k in ("engine_url", "remote"))
            if target_changed and eng.owned:            # its key and its process belong to the address it was given
                raise HTTPException(409, f"this TensorFold Web runs {eng.model} at {s.engine_url}: stop it before pointing "
                                         "TensorFold Web at another engine")
            with eng.target_lock:                            # address and key change together, as one new target
                before = (s.engine_url, s.api_key, s.remote)
                for key, value in patch.items():
                    setattr(s, key, value)
                if held and s.share and not s.api_key:
                    s.api_key = settings_mod.new_api_key()      # sharing without a key is never offered
                if (s.engine_url, s.api_key, s.remote) != before:
                    eng.retargeted()
                if target_changed:
                    eng.reset_target()                       # nothing read from the previous engine applies here
            save_settings()
        finally:
            if held:
                gate.give()
        return {**user_settings(), "key_pending_restart": eng.key_pending_restart()}

    @app.post("/api/settings/api-key")
    def rotate_api_key():
        """A new key for the engine's next start; an engine already running keeps the one it was started with."""

        try:
            with gate.hold("a new engine key"):              # a comparison restarting the engine would take it midway
                with eng.target_lock:
                    s.api_key = settings_mod.new_api_key()
                    eng.retargeted()
                save_settings()
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        pending = eng.key_pending_restart()
        return {"api_key_set": True, "key_pending_restart": pending,
                "note": "the running engine keeps its previous key until it restarts" if pending else
                        "applies to the engine at its next start"}

    # -- chat conversations, kept by TensorFold Web ---------------------------------------------------------------------
    chats = conversations.Conversations(settings_mod.config_dir() / "chat.sqlite")
    app.state.chats = chats

    @app.get("/api/conversations")
    def chat_list(q: str = "", offset: int = 0):
        rows, more = chats.page(q, offset=max(0, offset))
        return {"conversations": rows, "more": more}

    @app.post("/api/conversations")
    def chat_create(body: dict[str, Any]):
        try:
            return chats.create(body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def served_target() -> tuple[str, str | None, str | None, str | None]:
        base, key, model = bench_target()
        return base, key, model, served_name()

    replays = replay.Replays(chats, served_target)

    @app.get("/api/conversations/compare")
    def chat_compare(a: str, b: str):
        ca, cb = chats.get(a), chats.get(b)
        if ca is None or cb is None:
            raise HTTPException(404, "unknown conversation")
        return replay.compare_replies(ca, cb)

    @app.post("/api/conversations/{cid}/replay")
    def chat_replay(cid: str, body: dict[str, Any]):
        target = body.get("target") or {"kind": "served", "drafts": True}
        if not isinstance(target, dict):
            raise HTTPException(400, "target must be an object")
        try:
            return replays.start(cid, target)
        except KeyError as exc:
            raise HTTPException(404, "unknown conversation") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/replay")
    def replay_progress():
        return {"busy": replays.busy(), "current": replays.current}

    @app.post("/api/replay/cancel")
    def replay_cancel():
        return {"cancelling": replays.cancel()}

    @app.get("/api/conversations/{cid}")
    def chat_get(cid: str):
        doc = chats.get(cid)
        if doc is None:
            raise HTTPException(404, "unknown conversation")
        # a page saves every few seconds while a reply streams: a reply without an ending in a conversation saved a
        # minute ago or more has lost its page
        return conversations.settle(doc, live={cid} if time.time() - doc["updated"] < 60 else set())

    @app.put("/api/conversations/{cid}")
    def chat_update(cid: str, body: dict[str, Any]):
        rev = body.pop("rev", None)
        if rev is not None and (isinstance(rev, bool) or not isinstance(rev, int)):
            raise HTTPException(400, "rev must be the revision number the page read")
        if rev is None and ("messages" in body or "settings" in body):
            # a write of content says which revision it was made on, so it never overwrites one it has not seen
            raise HTTPException(400, "rev is required to change messages or settings: send the revision you read")
        try:
            return chats.update(cid, body, rev=rev)
        except conversations.Conflict as exc:
            return JSONResponse({"detail": str(exc), "rev": exc.current.get("rev", 0)}, status_code=409)
        except KeyError as exc:
            raise HTTPException(404, "unknown conversation") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/conversation-folders")
    def chat_folders():
        return {"folders": chats.folders()}

    @app.post("/api/conversation-folders")
    def chat_folder_make(body: dict[str, Any]):
        try:
            return chats.make_folder(body.get("name"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put("/api/conversation-folders/{name:path}")
    def chat_folder_rename(name: str, body: dict[str, Any]):
        try:
            return chats.rename_folder(name, body.get("name"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/conversation-folders/{name:path}")
    def chat_folder_remove(name: str):
        return {"removed": name, "moved_out": chats.remove_folder(name)}

    @app.get("/api/chat-presets")
    def chat_presets():
        return {"presets": chats.presets()}

    @app.post("/api/chat-presets")
    def chat_preset_save(body: dict[str, Any]):
        """Saves a preset; one of the same name is replaced only when `replace` says so (409 otherwise)."""
        try:
            return chats.save_preset(body.get("preset"), replace=bool(body.get("replace")))
        except conversations.Taken as exc:
            raise HTTPException(409, f"a preset named {exc} is kept already") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put("/api/chat-presets/{name:path}")
    def chat_preset_rename(name: str, body: dict[str, Any]):
        try:
            return chats.rename_preset(name, body.get("name"))
        except KeyError as exc:
            raise HTTPException(404, "unknown preset") from exc
        except conversations.Taken as exc:
            raise HTTPException(409, f"a preset named {exc} is kept already") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/chat-presets/{name:path}")
    def chat_preset_remove(name: str):
        if not chats.remove_preset(name):
            raise HTTPException(404, "unknown preset")
        return {"removed": name}

    # makers added or completed from Settings (brands.py): the page puts them before its own list
    makers = brands.Brands(settings_mod.config_dir() / "brands.json")

    @app.get("/api/brands")
    def brand_list(request: Request):
        # up to a hundred logos inline: sent again only when the file changed (the browser revalidates each time)
        tag = f'"{makers.version()}"'
        if request.headers.get("if-none-match") == tag:
            return Response(status_code=304, headers={"ETag": tag, "Cache-Control": "no-cache"})
        return JSONResponse({"brands": makers.all()}, headers={"ETag": tag, "Cache-Control": "no-cache"})

    @app.put("/api/brands/{brand_id}")
    def brand_save(brand_id: str, body: dict[str, Any]):
        try:
            return makers.save(brand_id, body)
        except brands.Invalid as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/api/brands/{brand_id}")
    def brand_remove(brand_id: str, kind: str | None = None):
        if not makers.remove(brand_id, kind):
            raise HTTPException(404, "unknown maker")
        return {"removed": brand_id}

    @app.delete("/api/conversations/{cid}")
    def chat_delete(cid: str):
        if not chats.delete(cid):
            raise HTTPException(404, "unknown conversation")
        return {"deleted": cid}

    @app.get("/api/conversations/{cid}/export")
    def chat_export(cid: str, format: str = "md"):
        doc = chats.get(cid)
        if doc is None:
            raise HTTPException(404, "unknown conversation")
        if format == "json":
            return JSONResponse({"kind": "tensorfold-studio-conversation", **doc},
                                headers=attachment(doc["title"], "conversation", ".json"))
        return Response(conversations.markdown(doc), media_type="text/markdown",
                        headers=attachment(doc["title"], "conversation", ".md"))

    @app.get("/api/connect")
    def connect(where: str = "here", relay: str = ""):
        """The recipes for a tool on this machine, or on another computer through TensorFold Web's own address
        (``relay``, the address the page was opened at), with the key that opens it."""

        h = eng.last_health or eng.health()
        c = connect_ctx(s.engine_url, h.model if h else None, share=s.share, api_key=eng.key_in_use())
        if where == "away":
            u = urlsplit(relay)
            # a name or address and a port, nothing else: the recipes paste it into a shell
            if u.scheme not in ("http", "https") or len(relay) > 200 or u.path not in ("", "/") or u.query \
                    or u.fragment or not re.fullmatch(r"(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9.-]+)(:[0-9]{1,5})?", u.netloc):
                raise HTTPException(400, "relay must be TensorFold Web's own address, http(s)://name[:port]")
            base = f"{u.scheme}://{u.netloc}/engine"
            c = recipes.Ctx(base=base, key=s.api_key or eng.key_in_use() or "YOUR-KEY", model=c.model,
                            docker_base=base, away=True)
        # asked of the engine itself (on its own address): a coding agent needs a long window, and its recipe says so
        window = recipes.served_window(recipes.Ctx(base=s.engine_url.rstrip("/").removesuffix("/v1"),
                                                   key=eng.key_in_use() or "local", model=c.model)) \
            if h and h.status == "running" else None
        return {"engine_url": s.engine_url, "share": s.share,
                "lan_url": f"http://{lan_hostname()}:{s.engine_port()}" if s.share else None,
                # the key that opens the engine now; a new one waiting for the restart is not copied before it works
                "api_key": eng.key_in_use(), "key_pending_restart": eng.key_pending_restart(),
                # the relay takes the new key at once (and the old one until the restart): the one to give a tool
                "relay_key": s.api_key or eng.key_in_use(),
                "studio_token_set": bool(token),
                "kinds": recipes.KINDS, "window": window, "where": "away" if c.away else "here",
                "snippets": [r.to_dict(dataclasses.replace(c, window=window)) for r in recipes.RECIPES]}

    @app.post("/api/connect/test")
    def connect_test(body: dict[str, Any]):
        """A recipe's checks, made from this machine against the engine, the calls the tool makes."""

        rec = next((r for r in recipes.RECIPES if r.id == body.get("recipe")), None)
        if rec is None:
            raise HTTPException(404, "unknown recipe")
        h = eng.status()
        if h.status != "running":
            raise HTTPException(409, "no model is being served; start one from Models")
        model = h.model or next(iter(engines.identify(s.engine_url, key=eng.key_in_use()).models), None)
        c = recipes.Ctx(base=s.engine_url.rstrip("/").removesuffix("/v1"), key=eng.key_in_use() or "local",
                        model=model or "")
        return recipes.test(rec, c)

    @app.get("/api/connect/compose/open-webui")
    def connect_compose():
        h = eng.last_health or eng.health()
        c = connect_ctx(s.engine_url, h.model if h else None, share=s.share, api_key=eng.key_in_use())
        return Response(recipes.compose(c), media_type="text/yaml",
                        headers={"Content-Disposition": 'attachment; filename="docker-compose.yml"'})

    # -- fleet: other machines' TensorFold Web ----------------------------------------------------
    peers = fleet.Peers(settings_mod.config_dir() / "fleet.json")

    def fleet_body(body: object) -> tuple[str, str | None]:
        if not isinstance(body, dict):
            raise HTTPException(400, "a JSON object is needed")
        try:
            url = fleet.normalize_url(body.get("url"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        token = body.get("token")
        if token is not None and (not isinstance(token, str) or len(token) > 400):
            raise HTTPException(400, "the token is text")
        return url, (token or "").strip() or None

    @app.get("/api/fleet")
    def fleet_view():
        """This machine and the others, each read now; a machine that does not answer says why."""

        here = {"id": "self", "name": "This machine", "url": None, "token_set": True, "reachable": True,
                "error": None, "ms": 0, "self": True, **fleet.summary(machine(), engine_info())}
        return {"machines": [here, *fleet.probe_all(peers.all())], "at": time.time()}

    @app.post("/api/fleet/test")
    def fleet_test(body: dict[str, Any]):
        url, token = fleet_body(body)
        return fleet.test(url, token, own_hostname=detect().hostname, own_version=__version__,
                          own_engine_version=detect().tensorfold_version)

    @app.post("/api/fleet")
    def fleet_add(body: dict[str, Any]):
        """Added only once its checks pass, and seen at once: no restart."""

        url, token = fleet_body(body)
        result = fleet.test(url, token, own_hostname=detect().hostname, own_version=__version__,
                            own_engine_version=detect().tensorfold_version)
        if not result["ok"]:
            failed = next(c for c in result["checks"] if c["status"] == "fail")
            raise HTTPException(409, f"{failed['label']}: {failed['detail']}")
        host = (result.get("machine") or {}).get("hostname") or urlsplit(url).hostname or url
        try:
            name = fleet.validate_name(body.get("name"), host)
            peer = peers.add(name, url, token)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"id": peer["id"], "name": peer["name"], "url": peer["url"], "token_set": bool(peer["token"])}

    @app.put("/api/fleet/{pid}")
    def fleet_update(pid: str, body: dict[str, Any]):
        try:
            name = fleet.validate_name(body["name"], "") or None if "name" in body else None
            token = body.get("token")
            if token is not None and (not isinstance(token, str) or len(token) > 400):
                raise ValueError("the token is text")
            peer = peers.update(pid, name=name, token=token.strip() if isinstance(token, str) else None)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, "no such machine in the fleet") from exc
        return {"id": peer["id"], "name": peer["name"], "url": peer["url"], "token_set": bool(peer["token"])}

    @app.post("/api/fleet/{pid}/open")
    def fleet_open(pid: str):
        """A one-time link into another machine's TensorFold Web, asked of it with the token this TensorFold Web keeps for it: the
        browser opens it signed in, and the token never reaches the page."""

        peer = next((x for x in peers.all() if x["id"] == pid), None)
        if peer is None:
            raise HTTPException(404, "no such machine in the fleet")
        if not peer.get("token"):
            return {"url": peer["url"] + "/", "signed_in": False}      # a TensorFold Web without a token opens as it is
        try:
            with fleet.client() as c:
                r = c.post(peer["url"] + "/api/access/link", headers={fleet.TOKEN_HEADER: peer["token"],
                                                                      "origin": peer["url"]})
                if r.status_code == 404:                 # an older TensorFold Web, without links
                    raise HTTPException(409, f"{peer['name']} runs an older TensorFold Web without one-time links")
                if r.status_code == 403:                 # it refused this TensorFold Web's origin: served under another name
                    raise HTTPException(409, f"{peer['name']} refused the request's origin ({peer['url']}): it is "
                                             "served under another name; add this machine with the address its page "
                                             "opens at, or start it with --allow-host")
                r.raise_for_status()
                code = r.json()["code"]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise HTTPException(502, f"{peer['name']} did not make a link: {exc}") from exc
        return {"url": f"{peer['url']}/?code={code}", "signed_in": True}

    @app.delete("/api/fleet/{pid}")
    def fleet_remove(pid: str):
        try:
            peers.remove(pid)
        except KeyError as exc:
            raise HTTPException(404, "no such machine in the fleet") from exc
        return {"removed": pid}

    # -- diagnostic: the checks of the first-run list, and a report to send -----------------
    def run_checks() -> tuple[list, dict, dict]:
        from . import pull, service

        m, info = machine(), engine_info()
        try:
            svc = service.status(service.plan())
        except (RuntimeError, OSError, subprocess.SubprocessError):   # no systemctl or launchctl here, or one that hangs
            svc = None
        installed = [] if s.remote else [v["name"] for v in cat.view(detect(), fraction=s.budget_fraction)
                                         if v.get("installed") and v.get("runs_here")]
        checks = diagnostic.run(machine=m, info=info, settings=s.to_dict(mask=True), cache_root=pull.cache_root(),
                                installed=installed, bind=bind_host, token=bool(token), service_status=svc)
        return checks, m, info

    @app.get("/api/diagnostic")
    def diagnostic_checks():
        checks, _, _ = run_checks()
        return {"checks": [c.to_dict() for c in checks], "at": time.time()}

    @app.get("/api/diagnostic/report")
    def diagnostic_report():
        checks, m, info = run_checks()
        doc = diagnostic.report(checks=checks, machine=m, info=info, settings=s.to_dict(mask=True),
                                studio_version=__version__, log=list(eng.log), peers=peers.public(),
                                usage=used.summary(30), secrets=[x for x in (token, s.api_key, eng.key_in_use()) if x])
        name = f"tensorfold-web-diagnostic-{m.get('hostname') or 'machine'}-{time.strftime('%Y%m%d-%H%M')}.json"
        return Response(json.dumps(doc, indent=1), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    # -- storage: what TensorFold Web keeps, and clean-ups scoped by kind and age --------------------------------
    def catalog_repos() -> set[str]:
        out = set()
        for e in cat.entries:
            out.add(e.id)
            out.update(r for r in (e.drafter, e.drafter_optional) if r)
        return out

    def local_served() -> str | None:
        """The repo an engine on this machine serves, ours or not, as the catalog names it when it can."""

        if eng.owned and eng.model:
            return eng.model
        h = eng.last_health
        # "elsewhere" in Settings may still be this machine (an engine started by hand on 127.0.0.1): its files too
        here = not s.remote or bench.on_this_machine(urlsplit(s.engine_url).hostname or "")
        if not here or not h or h.status not in ("running", "starting") or not h.model:
            return None
        name = h.model.lower()
        if "models--" in name:                           # a path into the Hugging Face cache names its repo
            name = name.split("models--", 1)[1].split("/", 1)[0].replace("--", "/", 1)
        name = name.rstrip("/")
        for e in cat.entries:
            if name in (e.id.lower(), e.id.split("/")[-1].lower()) or name.endswith("/" + e.id.lower()):
                return e.id
        return h.model

    def in_use(repo: str) -> str | None:
        served = local_served()
        if served and served.lower() in (repo.lower(), repo.split("/")[-1].lower()):
            return "it is being served: stop it first" if eng.owned else \
                "an engine on this machine serves it: stop that engine first"
        if comparisons.busy():
            return "a comparison of configurations may start it"
        heirs = [c["name"] for c in living.copies() if c.get("base") == repo]
        if heirs:                                        # their versions and Forget restore files from it
            return f"the living copy {heirs[0]} came from it; its versions need it"
        entry = next((e for e in cat.entries if e.id == served), None)
        if entry and repo in (entry.drafter, entry.drafter_optional):
            return f"it drafts for {entry.name}, being served: stop it first"
        if (job := dl.jobs.get(repo)) is not None and job.state == "running":
            return "it is downloading"
        if creating.get("state") == "copying" and creating.get("base") == repo:
            return f"the copy {creating.get('name')} is being made from it"
        if chore.get("state") == "running":           # a fetch from a peer, a restore: they may read its files
            return f"a {chore.get('kind', 'task')} of {chore.get('copy', 'a copy')} is going"
        return None

    @app.get("/api/storage")
    def storage_view():
        from . import pull

        cache = pull.cache_root()
        rows = storage.models(cache, catalog_repos())
        for m in rows:
            m["busy"] = in_use(m["repo"])
        return {"stores": storage.summary(settings_mod.config_dir()), "models": rows, "cache": str(cache)}

    @app.post("/api/storage/clean")
    def storage_clean(body: dict[str, Any]):
        days = body.get("older_than_days")
        if days is not None and (isinstance(days, bool) or not isinstance(days, (int, float))):
            raise HTTPException(400, "older_than_days must be a number of days, or null for everything")
        try:
            return storage.clean(settings_mod.config_dir(), str(body.get("kind")), days,
                                 dry_run=bool(body.get("dry_run", True)))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/storage/models/remove")
    def storage_remove_model(body: dict[str, Any]):
        from . import pull

        repo = body.get("repo")
        if not isinstance(repo, str) or repo not in catalog_repos():
            raise HTTPException(400, "only the catalog's models are removed from here")
        try:
            with gate.hold("a model's removal"):          # nothing starts serving or downloading it meanwhile
                busy = in_use(repo)
                if busy:
                    raise HTTPException(409, f"{repo} is not removed: {busy}")
                return storage.remove_model(pull.cache_root(), repo, dry_run=bool(body.get("dry_run", True)))
        except KeyError as exc:
            raise HTTPException(404, f"{repo} is not in the cache") from exc
        except RuntimeError as exc:                      # Busy too
            raise HTTPException(409, str(exc)) from exc

    # -- relay to the engine --------------------------------------------------------------
    @app.api_route("/engine/{path:path}", methods=["GET", "POST", "DELETE"])
    async def relay(request: Request, path: str):
        # teaching changes the model for everyone, and its facts may be private: every route of TensorFold's learner
        # goes through TensorFold Web's own API (its token), never through the relay with the engine key. The path is
        # judged as the engine will read it: decoded, without what an encoded "?" or "#" would cut off
        if request.headers.get(RELAYED):            # the engine's address is this relay itself: answering would loop
            return JSONResponse({"detail": "the engine's address in Settings points back at this TensorFold Web's relay "
                                           "(/engine): give the engine's own address"}, status_code=508)
        refused = relay_refuses(request.url.path.removeprefix("/engine"), path)
        if refused:
            return JSONResponse({"detail": refused}, status_code=403 if "teach" in refused else 400)
        return await proxy.handle(request, path)

    # -- Living Weights: teachable copies, a preview of the facts, lessons --------------------------------------------
    creating: dict[str, Any] = {}

    def served_copy() -> str | None:
        name = living.name_of(eng.model) if eng.owned else None
        return name if name and eng.status().status == "running" else None

    def lesson_target() -> living.Target:
        h = eng.status()
        if h.status != "running" or not served_copy():
            raise RuntimeError("serve a living copy first (Teach, then Serve)")
        base = s.engine_url.rstrip("/").removesuffix("/v1")
        return living.Target(base, eng.key_in_use(), h.model or eng.model or "")

    for said in living_versions.recover() + living_versions.recover_unfinished():   # left halfway by a stopped TensorFold Web
        eng.note(said)
    def undo_lesson(copy: str, version: str) -> None:
        """A lesson's own rollback, the gate already held by the lesson: stop, restore the version, serve again."""

        eng.stop()
        before = living.lessons(copy)
        living_versions.restore(copy, version)
        living_checks.undone(copy, before, living.lessons(copy))
        launch(living.PREFIX + copy, None)
        wait_running()

    teacher = living.Teacher(lesson_target, lambda: gate.take("a lesson"), gate.give, lambda: list(eng.log),
                             served=lambda: served_copy(), graph=lambda: engine_graph(), undo=undo_lesson)

    def engine_graph() -> dict:
        r = engine_get("/v1/slide/graph")
        r.raise_for_status()                              # an error page is not a graph
        return r.json()

    said = teacher.recover(engine_graph, alive=lambda: eng.owned)
    if said:
        eng.note(said)
    app.state.teacher = teacher

    @app.get("/api/living")
    def living_view():
        from . import native

        bases = [e.id for e in cat.entries if e.family in living.LEARNABLE]
        copies = living.copies()
        for c in copies:
            if c["base"] is None and c["name"] not in looked_for_base:
                looked_for_base.add(c["name"])                # a copy made by hand: its base found once
                c["base"] = living.base_of(c["name"], bases)
        return {"root": str(living.root()), "copies": copies, "served": served_copy(),
                "teachable": living.teachable_here(native.find_binary()), "creating": dict(creating),
                "job": teacher.job.to_dict() if teacher.job else None, "busy": teacher.busy(),
                "recovering": teacher.recovering,
                "chore": dict(chore), "bases": bases,
                "base_names": {b: next((e.name for e in cat.entries if e.id == b), b) for b in bases},
                "peers": [{"id": p["id"], "name": p["name"]} for p in peers.all()]}

    def living_tag() -> dict | None:
        name = served_copy()
        if not name:
            return None
        record = living.lessons(name)
        versions = living_versions.manifests(name)
        try:                                              # the engine's graph counts facts taught outside TensorFold Web too
            graph = json.loads(living.graph_path(name).read_text())
            kept = sum(1 for f in graph.get("facts") or [] if f.get("state") == "learned")
        except (OSError, ValueError):
            kept = sum(1 for x in record if x.get("outcome") in living.KEPT)
        return {"copy": name, "kept": kept, "lessons": len(record),
                "last_lesson": record[-1]["job"] if record else None, "last_version": versions[-1]["id"] if versions
                else None}

    @app.get("/api/living/copies/{name}/runs")
    def living_runs(name: str):
        """The Bench runs made on this copy, newest first, with the facts it held then."""

        copy_name(name)
        out = []
        for r in runner.store.runs(200):
            doc = runner.store.run(r["id"]) or {}
            tag = (doc.get("tags") or {}).get("living") or {}
            if tag.get("copy") == name:
                o = (r.get("summary") or {}).get("overall") or {}
                q = ((r.get("summary") or {}).get("quality") or {}).get("overall") or {}
                out.append({"id": r["id"], "created": r["created"], "mode": r["mode"], "status": r["status"],
                            "kept": tag.get("kept"), "tps": o.get("tps"), "exact": o.get("exact"),
                            "checked": o.get("checked"), "right": q.get("passed"), "items": q.get("items"),
                            "mean_kl": (r.get("kl") or {}).get("mean_kl"),
                            "reference": (doc.get("kl_reference") or {}).get("url")})
        return {"runs": out}

    @app.post("/api/living/kl")
    def living_kl(body: dict[str, Any]):
        """KL of the served copy against the untaught model served by another machine of the fleet, on prompts that
        have nothing to do with what it learned: how far learning moved it where it should not."""

        name = served_copy()
        if not name:
            raise HTTPException(409, "serve a living copy first")
        peer = next((p for p in peers.all() if p["id"] == body.get("peer")), None)
        if peer is None:
            raise HTTPException(404, "no such machine in the fleet")
        try:
            with fleet.client() as c:
                h = {fleet.TOKEN_HEADER: peer["token"]} if peer.get("token") else {}
                info = c.get(peer["url"] + "/api/engine", headers=h).raise_for_status().json()
                key = c.get(peer["url"] + "/api/connect", headers=h).raise_for_status().json().get("api_key")
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(502, f"{peer['name']} did not answer: {exc}") from exc
        model = info.get("model")
        if info.get("status") != "running" or not model:
            raise HTTPException(409, f"{peer['name']} serves nothing now; serve the untaught model there")
        base = (living.base_of(name, []) or "").split("/")[-1]
        if base and base != model and not str(info.get("model_repo") or "").endswith(base):
            raise HTTPException(409, f"{peer['name']} serves {model}, not {base}, the model {name} came from")
        try:
            return bench_go("kl", suites=None, exactness=False, engine_name=served_name(),
                            reference=(peer["url"].rstrip("/") + "/engine", key, model,
                                       {} if key else ({fleet.TOKEN_HEADER: peer["token"]} if peer.get("token") else {})),
                            kl_load=0)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/living/copies/{name}/forget")
    def living_forget(name: str, body: dict[str, Any]):
        """One fact out: a version of now kept (to undo the forgetting), the version from before the fact's lesson
        restored, and every fact kept since taught again, as a lesson of its own."""

        copy_name(name)
        if served_copy() != name:
            raise HTTPException(409, f"serve {name} to make it forget")
        fact = str(body.get("fact") or "")
        try:
            plan = living.forget_plan(name, fact, living_versions.manifests(name))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        fact = plan["fact"]                                     # the words it was taught in, whichever were given
        from . import native

        why = living.teachable_here(native.find_binary())      # checked before anything is taken out
        if why and plan["again"]:
            raise HTTPException(409, f"the facts kept since could not be taught again here: {why}")
        if len(plan["again"]) > 24:
            raise HTTPException(409, f"{len(plan['again'])} facts were kept since; teaching them again takes more than "
                                     "one lesson (24): restore an earlier version and teach them in parts")

        def work() -> dict:
            # one operation under one reservation, from the version kept to undo it to the replay and any rollback:
            # nothing else (a Bench run, a lesson) can slip in between and keep it from being undone
            gate.take("a fact being forgotten")
            try:
                undo = living_versions.save(name, f"before forgetting: {fact[:80]}")
                lost: list[str] = []
                try:
                    eng.stop()
                    before = living.lessons(name)
                    living_versions.restore(name, plan["version"])
                    living_checks.undone(name, before, living.lessons(name))
                    launch(living.PREFIX + name, None)
                    wait_running()
                    if not plan["again"]:
                        return {"undo": undo["id"], "restored": plan["version"], "taught_again": 0}
                    # a version before the replay keeps the lineage: each fact taught again can be forgotten later
                    job = teacher.start(name, plan["again"], "replay after forgetting", check=True, gate_held=True)
                    while teacher.busy():
                        time.sleep(2)
                    lost = [i.fact for i in job.items if i.state != "kept"]
                    if job.status == "done" and not lost:
                        return {"undo": undo["id"], "restored": plan["version"], "taught_again": len(plan["again"]),
                                "job": job.id}
                    why = f"{len(lost)} of the facts kept since did not come back ({'; '.join(lost)[:200]})"
                except Exception as exc:  # noqa: BLE001 - any failure undoes the whole forgetting
                    why = f"it failed: {exc}"
                # the forgetting is undone as a whole: the copy goes back to the version kept just before it
                with contextlib.suppress(Exception):
                    eng.stop()
                living_versions.restore(name, undo["id"])
                launch(living.PREFIX + name, None)
                wait_running()
                raise RuntimeError(f"forgetting was undone, {why}; the copy is as it was before")
            finally:
                gate.give()
        return run_chore("forgetting a fact", name, work)

    @app.get("/api/living/copies/{name}/card")
    def living_card(name: str):
        copy_name(name)
        if not living.folder(name).is_dir():
            raise HTTPException(404, f"no living copy named {name}")
        try:
            graph = json.loads(living.graph_path(name).read_text())
        except (OSError, ValueError):
            graph = {}
        text = living.card(name, graph=graph, versions=[living_versions.public(m) for m in
                                                        living_versions.manifests(name)],
                           runs=living_runs(name)["runs"], engine=served_name() if served_copy() == name else None)
        return Response(text, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}-model-card.md"'})

    @app.get("/api/living/copies/{name}/manifest")
    def living_manifest(name: str):
        """For another machine of the fleet fetching this copy: each file's hash and size, and its record."""

        copy_name(name)
        settled(name)
        try:
            return living_versions.current(name)
        except KeyError as exc:
            raise HTTPException(404, f"no living copy named {name}") from exc

    def settled(name: str) -> None:
        """A copy's files are given to another machine only while nothing here writes them."""

        if (teacher.busy() and ((teacher.job and teacher.job.copy == name) or teacher.recovering == name)) or \
                (chore.get("state") == "running" and chore.get("copy") == name) or \
                (creating.get("state") == "copying" and creating.get("name") == name):
            raise HTTPException(409, f"{name} is being written here (a lesson or a long task); fetch it later")
        if living.is_unfinished(name):
            raise HTTPException(409, f"{name} is not complete here; fetch it later")

    @app.get("/api/living/copies/{name}/file")
    def living_file(name: str, rel: str):
        copy_name(name)
        settled(name)
        if not living_versions.safe_rel(rel):
            raise HTTPException(404, "no such file in the copy")
        folder = living.folder(name).resolve()
        path = (folder / rel).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise HTTPException(404, "no such file in the copy")
        return FileResponse(path, media_type="application/octet-stream")

    @app.post("/api/living/fetch")
    def living_fetch(body: dict[str, Any]):
        """A copy of another machine's living copy made here: only the files that differ from the base travel."""

        peer = next((p for p in peers.all() if p["id"] == body.get("peer")), None)
        if peer is None:
            raise HTTPException(404, "no such machine in the fleet")
        src, new = copy_name(str(body.get("name") or "")), copy_name(str(body.get("as") or body.get("name") or ""))
        h = {fleet.TOKEN_HEADER: peer["token"]} if peer.get("token") else {}

        def manifest() -> dict:
            with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0)) as c:
                return c.get(f"{peer['url']}/api/living/copies/{src}/manifest", headers=h).raise_for_status().json()

        def file(rel: str):
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as c, \
                    c.stream("GET", f"{peer['url']}/api/living/copies/{src}/file", params={"rel": rel}, headers=h) as r:
                r.raise_for_status()
                yield from r.iter_bytes(8 * 1024 * 1024)
        return run_chore("a copy from " + peer["name"], new,
                         lambda: living_versions.fetch(manifest, file, new, f"{peer['name']}:{src}"))

    @app.get("/api/living/peer/{pid}")
    def living_peer(pid: str):
        """Another machine's living copies, to fetch one."""

        peer = next((p for p in peers.all() if p["id"] == pid), None)
        if peer is None:
            raise HTTPException(404, "no such machine in the fleet")
        try:
            with fleet.client() as c:
                h = {fleet.TOKEN_HEADER: peer["token"]} if peer.get("token") else {}
                v = c.get(peer["url"] + "/api/living", headers=h).raise_for_status().json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(502, f"{peer['name']} did not answer: {exc}") from exc
        return {"copies": [{"name": x.get("name"), "base": x.get("base"), "bytes": x.get("bytes") or 0,
                            "taught": x.get("taught") or 0} for x in v.get("copies") or [] if x.get("name")]}

    def engine_get(path: str) -> httpx.Response:
        key = eng.key_in_use()
        return httpx.get(s.engine_url.rstrip("/").removesuffix("/v1") + path, timeout=10.0,
                         headers={"Authorization": f"Bearer {key}"} if key else {})

    def no_store(resp: Response) -> Response:
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.post(VIEW + "/session")
    def living_view_session(request: Request):
        """The page asks with its token in a header; the answer is a cookie for the view's path only."""

        resp = JSONResponse({"ok": True})
        resp.set_cookie(VIEW_COOKIE, view_cookie(), httponly=True, samesite="strict", path=VIEW,
                        secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https")
        return resp

    @app.get(VIEW)
    @app.get(VIEW + "/")
    def living_view_page():
        """TensorFold's `/slide` page, its paths turned to TensorFold Web's: the copy's facts in 3D, live."""

        if not served_copy():
            return HTMLResponse("<p>Serve a living copy from Teach to see what it learned.</p>", status_code=409)
        try:
            r = engine_get("/slide")
        except httpx.HTTPError as exc:
            return HTMLResponse(f"<p>The engine did not answer: {exc}</p>", status_code=502)
        if r.status_code != 200:
            return HTMLResponse("<p>This engine has no fact view (TensorFold 1.0.7 with --slide).</p>", status_code=404)
        return no_store(HTMLResponse(r.text.replace('"/slide/', f'"{VIEW}/')))

    @app.get(VIEW + "/graph")
    def living_view_graph():
        try:
            r = engine_get("/v1/slide/graph")
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"the engine did not answer: {exc}") from exc
        return no_store(Response(r.content, status_code=r.status_code, media_type="application/json"))

    @app.get(VIEW + "/{script}")
    def living_view_script(script: str):
        if not re.fullmatch(r"[a-z]+\.js", script):
            raise HTTPException(404, "unknown")
        try:
            r = engine_get(f"/slide/{script}")
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"the engine did not answer: {exc}") from exc
        if r.status_code != 200:
            raise HTTPException(404, "unknown")
        body = r.text.replace('"/v1/slide/graph"', f'"{VIEW}/graph"').replace('"/slide/', f'"{VIEW}/')
        return no_store(Response(body, media_type="text/javascript; charset=utf-8"))

    looked_for_base: set[str] = set()
    chore: dict[str, Any] = {}                             # one long task at a time: a version, a restore, checks

    chore_lock = threading.Lock()                           # a check and a reservation, as one step
    chore_cancel = threading.Event()                        # set by Teach's cancel; a chore that asks questions reads it
    app.state.chore, app.state.chore_cancel = chore, chore_cancel

    def run_chore(kind: str, copy: str, work: Any) -> dict:
        with chore_lock:
            if chore.get("state") == "running":
                raise HTTPException(409, f"{chore['kind']} of {chore['copy']} is going; wait for it")
            if creating.get("state") == "copying":
                raise HTTPException(409, f"{creating.get('name')} is being copied; wait for it")
            if teacher.busy():
                raise HTTPException(409, "a lesson is going; wait for it")
            chore.clear()
            chore_cancel.clear()
            chore.update(kind=kind, copy=copy, state="running", started=time.time())

        def go() -> None:
            try:
                chore.update(result=work(), state="done")
            except Exception as exc:  # noqa: BLE001 - said on the page
                chore.update(state="failed", error=str(exc)[:300])
            chore["finished"] = time.time()
        threading.Thread(target=go, name="living-chore", daemon=True).start()
        return dict(chore)

    def wait_running(seconds: float = 300) -> None:
        end = time.time() + seconds
        while time.time() < end:
            st = eng.status().status
            if st == "running":
                return
            if st in ("failed", "stopped"):
                raise RuntimeError(f"the copy did not start ({st}); its log says why")
            time.sleep(2)
        raise RuntimeError("the copy did not serve in time")

    def copy_name(name: str) -> str:
        try:
            living.folder(name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return name

    @app.get("/api/living/copies/{name}/versions")
    def living_versions_list(name: str):
        copy_name(name)
        return {"versions": [living_versions.public(m) for m in living_versions.manifests(name)],
                "store_bytes": living_versions.store_bytes(name)}

    @app.post("/api/living/copies/{name}/versions")
    def living_version_save(name: str, body: dict[str, Any]):
        copy_name(name)
        label = str(body.get("label") or "kept by hand")
        return run_chore("a version", name, lambda: living_versions.save(name, label))

    @app.post("/api/living/copies/{name}/versions/{vid}/restore")
    def living_version_restore(name: str, vid: str):
        copy_name(name)

        def work() -> dict:
            gate.take("a version being restored")
            try:
                was_served = eng.owned and living.name_of(eng.model) == name
                if was_served:
                    eng.stop()
                before = living.lessons(name)
                out = living_versions.restore(name, vid)
                living_checks.undone(name, before, living.lessons(name))
                if was_served:                             # it goes back to serving, as the version left it
                    launch(living.PREFIX + name, None)
                    wait_running()
                return out
            finally:
                gate.give()
        return run_chore("a restore", name, work)

    @app.post("/api/living/copies/{name}/versions/{vid}/branch")
    def living_version_branch(name: str, vid: str, body: dict[str, Any]):
        copy_name(name)
        new = copy_name(str(body.get("name") or ""))
        base = living.base_of(name, [e.id for e in cat.entries if e.family in living.LEARNABLE])
        if not base:
            raise HTTPException(409, f"the model {name} came from is not known or not downloaded here")

        def work() -> dict:
            living.create(new, base, finish=False)       # not a copy until the version is in it
            try:
                out = living_versions.restore(name, vid, into=new)
                living.unfinished(new).unlink()
            except Exception:
                living.remove(new)
                raise
            return out
        return run_chore("a branch", name, work)

    @app.delete("/api/living/copies/{name}/versions/{vid}")
    def living_version_remove(name: str, vid: str):
        copy_name(name)
        with chore_lock:                                 # never while a version is being kept or restored
            if chore.get("state") == "running" or teacher.busy():
                raise HTTPException(409, "a lesson or a long task is going; wait for it")
            try:
                return living_versions.remove(name, vid)
            except KeyError as exc:
                raise HTTPException(404, "no such version") from exc

    @app.get("/api/living/copies/{name}/checks")
    def living_checks_view(name: str):
        copy_name(name)
        return {"set": living_checks.check_set(name), "runs": living_checks.runs(name)}

    @app.put("/api/living/copies/{name}/checks")
    def living_checks_set(name: str, body: dict[str, Any]):
        copy_name(name)
        qs = body.get("questions")
        if not isinstance(qs, list):
            raise HTTPException(400, "questions must be a list of {q, kind}")
        un = body.get("unrelated")
        return living_checks.set_own(name, qs, un if isinstance(un, list) else None)

    @app.post("/api/living/copies/{name}/checks/run")
    def living_checks_run(name: str, body: dict[str, Any]):
        copy_name(name)
        if served_copy() != name:
            raise HTTPException(409, f"serve {name} to check it")
        restart, thinking = bool(body.get("restart")), body.get("thinking", True) is not False

        def work() -> dict:
            label = "after a restart" if restart else str(body.get("label") or "by hand")
            gate.take("checks of a copy")                    # one model state from the first question to the last
            try:
                if served_copy() != name:                    # checked again under the gate
                    raise RuntimeError(f"{name} is no longer the copy served")
                if restart:                                  # what it learned is in its files, not in memory
                    eng.stop()
                    launch(living.PREFIX + name, None)
                    wait_running()
                doc = living_checks.run(lesson_target(), name, label, thinking=thinking, cancel=chore_cancel.is_set)
            finally:
                gate.give()
            return {k: v for k, v in doc.items() if k != "rows"}
        return run_chore("checks", name, work)

    @app.get("/api/living/copies/{name}/checks/{rid}")
    def living_checks_get(name: str, rid: str, against: str | None = None):
        copy_name(name)
        try:
            doc = living_checks.get(name, rid)
            if against:
                return {**{k: v for k, v in doc.items() if k != "rows"},
                        "compare": living_checks.compare(living_checks.get(name, against), doc), "against": against}
            return doc
        except KeyError as exc:
            raise HTTPException(404, "no such check run") from exc

    @app.post("/api/living/copies")
    def living_create(body: dict[str, Any]):
        name, base = body.get("name"), body.get("base")
        if not isinstance(name, str) or not isinstance(base, str):
            raise HTTPException(400, "name and base are required")
        try:
            living.folder(name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        with chore_lock:                                 # the check and the reservation as one step
            if creating.get("state") == "copying":
                raise HTTPException(409, f"{creating.get('name')} is being copied; wait for it")
            if chore.get("state") == "running":
                raise HTTPException(409, f"{chore['kind']} of {chore['copy']} is going; wait for it")
            if teacher.busy():                           # a lesson writes its version while the copy fills the disk
                raise HTTPException(409, "a lesson is going; wait for it")
            creating.clear()
            creating.update(name=name, base=base, state="copying", started=time.time())

        def go() -> None:
            try:
                living.create(name, base)
                creating.update(state="done", finished=time.time())
            except Exception as exc:  # noqa: BLE001 - said on the page
                creating.update(state="failed", error=str(exc)[:300], finished=time.time())
        threading.Thread(target=go, name="living-copy", daemon=True).start()
        return dict(creating)

    @app.delete("/api/living/copies/{name}")
    def living_remove(name: str):
        try:
            # the checks and the deletion as one step: nothing starts serving, teaching or keeping it meanwhile
            with gate.hold(f"the deletion of {name}"), chore_lock:
                if eng.owned and living.name_of(eng.model) == name:
                    raise HTTPException(409, f"{name} is being served: stop it first")
                if teacher.busy() and teacher.job and teacher.job.copy == name:
                    raise HTTPException(409, f"{name} is learning: cancel the lesson first")
                if chore.get("state") == "running" and chore.get("copy") == name:
                    raise HTTPException(409, f"{chore['kind']} of {name} is going; wait for it")
                if creating.get("state") == "copying" and creating.get("name") == name:
                    raise HTTPException(409, f"{name} is being copied; wait for it")
                living.remove(name)
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, f"no living copy named {name}") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"removed": name}

    @app.post("/api/living/preview")
    def living_preview(body: dict[str, Any]):
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(400, "text is required")
        if len(text) > 20000:
            raise HTTPException(400, "20,000 characters at most: a long document is the weak case; split it")
        if teacher.busy() or chore.get("state") == "running":
            raise HTTPException(409, "a lesson or a long task is going; wait for it")
        try:
            return {"copy": served_copy(), "facts": living.preview(lesson_target(), text.strip())}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/living/teach")
    def living_teach(body: dict[str, Any]):
        facts = body.get("facts")
        if not isinstance(facts, list):
            raise HTTPException(400, "facts must be a list of {fact, question}")
        copy = served_copy()
        if not copy:
            raise HTTPException(409, "serve a living copy first")
        if body.get("copy") and body["copy"] != copy:
            raise HTTPException(409, f"the facts were previewed on {body['copy']}, and {copy} is served now; preview again")
        from . import native

        why = living.family_refusal(copy) or living.teachable_here(native.find_binary())
        if why:
            raise HTTPException(409, why)
        try:
            with chore_lock:                             # the same reservation as versions, restores and deletions
                if chore.get("state") == "running":
                    raise HTTPException(409, f"{chore['kind']} of {chore['copy']} is going; wait for it")
                if creating.get("state") == "copying":
                    raise HTTPException(409, f"{creating.get('name')} is being copied; wait for it")
                return teacher.start(copy, facts, str(body.get("source") or "studio")[:40],
                                     version_before=body.get("version_before", True) is not False,
                                     check=body.get("check", True) is not False).to_dict()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:                         # Busy too
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/living/cancel")
    def living_cancel(body: dict[str, Any] | None = None):
        """A lesson, or a run of checks. By default a lesson stops after the fact being learned (or before any, during
        the checks); `{"now": true}` closes the learn request, so the engine drops the fact at its next check and
        whatever it kept of it is rolled back. A run of checks stops after the question it is on."""

        chore_going = chore.get("state") == "running" and chore.get("kind") == "checks"
        if chore_going:
            chore_cancel.set()
        return {"cancelling": teacher.cancel(now=bool((body or {}).get("now"))) or chore_going}

    @app.get("/api/living/copies/{name}/facts")
    def living_facts(name: str):
        try:
            living.folder(name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        graph: dict = {}
        if served_copy() == name:
            try:
                r = httpx.get(s.engine_url.rstrip("/").removesuffix("/v1") + "/v1/slide/graph", timeout=5.0,
                              headers={"Authorization": f"Bearer {eng.key_in_use()}"} if eng.key_in_use() else {})
                graph = r.json() if r.status_code == 200 else {}
            except (httpx.HTTPError, ValueError):
                graph = {}
        if not graph:
            try:
                graph = json.loads(living.graph_path(name).read_text())
            except (OSError, ValueError):
                graph = {}
        return {"facts": living.graph_summary(graph), "lessons": living.lessons(name)}

    # -- the page ---------------------------------------------------------------------------
    @app.get("/{path:path}", include_in_schema=False)
    def page(path: str):
        if path.startswith(("api/", "engine/")):
            return JSONResponse({"detail": "unknown route"}, status_code=404)
        index = UI_DIST / "index.html"
        if not index.is_file():
            return JSONResponse({"studio": __version__, "note": "the web UI is not built; run `npm run build` in ui/"})
        if path:
            candidate = (UI_DIST / path).resolve()
            if candidate.is_file() and candidate.is_relative_to(UI_DIST):      # never a file outside the page's folder
                # hashed assets may be cached for good; the manifest names them and changes with each build
                headers = {"Cache-Control": "no-cache"} if path.endswith(".webmanifest") else {}
                return FileResponse(candidate, headers=headers)
            if "." in path.rsplit("/", 1)[-1]:                                   # an asset that does not exist
                return JSONResponse({"detail": "not found"}, status_code=404)
        # always revalidated: a page cached across a deploy would ask for assets that no longer exist
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app
