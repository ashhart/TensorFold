"""Authenticated lifecycle routes and leases covering each HTTP request's runtime references."""

from __future__ import annotations

from contextlib import nullcontext
import hmac
import json
import threading
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from tensorfold.server.errors import RequestError
from tensorfold.server.lifecycle import LifecycleError
from tensorfold.server.request_body import read_body


def _send(handler: Any, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode()
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        if handler.close_connection:
            handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(body)
        handler.wfile.flush()
    except OSError:
        handler.close_connection = True


def _error(handler: Any, exc: LifecycleError) -> None:
    _send(handler, exc.status, {"error": {"message": str(exc), "code": exc.code,
                                         "type": "invalid_request_error" if exc.status < 500 else "server_error"}})


def control_route(handler: Any, app: Any) -> str | None:
    """Enabled lifecycle routes use their own credential, with or without the API prefix."""

    route = handler.path.split("?", 1)[0].rstrip("/")
    route = route.removeprefix("/v1")
    if getattr(app, "lifecycle", None) is not None and (handler.command, route) in {
            ("POST", "/sleep"), ("POST", "/wake_up"), ("GET", "/is_sleeping")}:
        return route
    return None


def admin(handler: Any, app: Any) -> bool:
    """Handle only enabled control routes; consume valid framing before changing lifecycle state."""

    route = control_route(handler, app)
    if route is None:
        return False
    lifecycle = app.lifecycle
    try:
        if "Origin" in handler.headers:
            handler.close_connection = True
            raise LifecycleError("browser Origin requests cannot control model sleep", status=403,
                                 code="sleep_origin_forbidden")
        secrets = handler.headers.get_all("Authorization", [])
        parts = secrets[0].split() if len(secrets) == 1 else []
        secret = getattr(app, "sleep_token", "")
        if (not isinstance(secret, str) or not secret or len(parts) != 2 or parts[0].lower() != "bearer"
                or not hmac.compare_digest(parts[1].encode(), secret.encode())):
            handler.close_connection = True
            raise LifecycleError("a valid sleep bearer token is required", status=401, code="invalid_sleep_token")
        read_body(handler)
        query = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
        if route == "/sleep":
            levels = query.pop("level", ["2"])
            if query or len(levels) != 1 or levels[0] != "2":
                raise LifecycleError("this runtime supports sleep level 2 only", status=400,
                                     code="unsupported_sleep_level")
            result = lifecycle.sleep(level=2)
        else:
            if query:
                raise LifecycleError("this lifecycle route does not accept query parameters", status=400,
                                     code="invalid_lifecycle_query")
            result = lifecycle.wake_up() if route == "/wake_up" else lifecycle.snapshot()
        memory = getattr(app, "sleep_memory", None)
        if memory is not None:
            result = {**result, "memory": memory()}
        cache = getattr(app, "sleep_cache", None)
        if cache is not None:
            result = {**result, "cache": cache()}
        _send(handler, 200, result)
    except RequestError as exc:
        _error(handler, LifecycleError(str(exc), status=400, code="invalid_request_body"))
    except LifecycleError as exc:
        fatal = lifecycle.snapshot()["state"] == "error"
        if fatal:
            handler.close_connection = True
        try:
            _error(handler, exc)
        finally:
            if fatal:
                handler.server.lifecycle_failed = True
                threading.Thread(target=handler.server.shutdown, daemon=True).start()
    return True


def post(handler: Any, app: Any, handle: Callable[[], None]) -> None:
    """The outer lease includes translation, preparation, streaming and same-thread nested handlers."""

    if admin(handler, app):
        return
    lifecycle = getattr(app, "lifecycle", None)
    if lifecycle is not None:
        from tensorfold.server import anthropic, responses, token_routes

        path = handler.path.split("?", 1)[0].rstrip("/")
        if not (path.endswith(("/completions", "/decisions")) or path in token_routes.ROUTES
                or anthropic.route(path) or responses.route(path) == ""):
            handler.close_connection = True
            _send(handler, 404, {"error": {"message": "unknown path"}})
            return
    try:
        with lifecycle.admit() if lifecycle is not None else nullcontext():
            handle()
    except LifecycleError as exc:
        handler.close_connection = True                 # refused bodies cannot become another request
        _error(handler, exc)


def observe(app: Any, read: Callable[[], Any], fallback: Callable[[], Any]) -> Any:
    """Copy runtime readings to CPU values while admitted, releasing the lease before socket writes."""

    lifecycle = getattr(app, "lifecycle", None)
    if lifecycle is None:
        return read()
    lease = lifecycle.admit()
    try:
        lease.__enter__()
    except LifecycleError:
        return fallback()
    try:
        return read()
    finally:
        lease.__exit__(None, None, None)


def health(app: Any, body: dict) -> dict:
    lifecycle = getattr(app, "lifecycle", None)
    if lifecycle is not None:
        state = lifecycle.snapshot()
        body.update(lifecycle=state, ready=state["ready"] and not body.get("warming", False))
    return body
