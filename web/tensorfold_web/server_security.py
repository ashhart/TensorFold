"""Security HTTP routes."""

from __future__ import annotations

from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)

from .server_http import (
    LOGS_COOKIE,
    LOGS_STREAM,
    TOKEN_HEADER,
    UNSAFE,
    VIEW,
    VIEW_COOKIE,
    _host_of,
    _site,
    frame_policy,
    same_secret,
)


def install(_engine_key_given, allowed_hosts, app, logs_cookie, token, view_cookie, LOOPBACK_NAMES) -> None:
    @app.middleware("http")
    async def _gate(request: Request, call_next):

        p = request.url.path
        host = _host_of(request.headers.get("host", ""))
        # Without a token, accept only local host names to prevent a hostile site's DNS rebinding.
        if not token and host and host not in LOOPBACK_NAMES | {_host_of(h) for h in allowed_hosts or []}:
            return JSONResponse({"detail": f"this TensorFold Web answers on localhost only, not as {host!r}; start it with "
                                           "--token to reach it under another name"}, status_code=421)
        # Reject cross-site changes by comparing Origin and the requested host, including the port.
        origin = request.headers.get("origin")
        # An explicit engine-key header distinguishes an API tool from a cross-site browser form.
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
            # The fact graph uses a path-scoped cookie to protect its private data and scripts.
            given = request.headers.get(TOKEN_HEADER) or ""      # never in the address: logs and history keep it
            if not (same_secret(given, token) or
                    same_secret(request.cookies.get(VIEW_COOKIE, ""), view_cookie())):
                return HTMLResponse("<p>This view needs TensorFold Web's token: open it from Teach.</p>", status_code=401)
        # a one-time code is its own key
        if token and p.startswith(("/api/", "/engine/")) and p != "/api/access/redeem":
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

        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = frame_policy(response.headers.get("content-security-policy"))
        return response
