"""Engine HTTP routes."""

from __future__ import annotations

import asyncio
import json
import queue
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)

from . import settings as settings_mod
from .gate import Busy
from .pull import NoRoom
from .server_http import (
    LOGS_COOKIE,
    LOGS_STREAM,
    REDEEM_BYTES,
    _host_of,
)


def install(_start_held, app, bind_host, cat, checker, dl, eng, friendly_name, gate, held_in_thread, links, logs_cookie, s, signals, start_lock, token, used, LOOPBACK_NAMES):
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

        if s.remote:
            raise HTTPException(409, "the engine is remote; start it on its own machine")
        try:
            patch = settings_mod.validate({k: body[k] for k in ("model", "context", "budget_fraction", "thinking")
                                           if k in body})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        async with start_lock:
            return await held_in_thread("an engine start", lambda: _start_held(patch))

    @app.post("/api/access/link")
    def access_link():
        """A one-time code, five minutes, for a page on another device: made only for someone already in."""

        if not token:
            raise HTTPException(409, "this TensorFold Web has no token: it answers this machine only, and needs no link")
        return links.make()

    @app.post("/api/access/redeem")
    async def access_redeem(request: Request):

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
                r = httpx.get(trust_env=False, url=s.engine_url.rstrip("/") + "/v1/models", timeout=2.0)
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
    return engine_info
