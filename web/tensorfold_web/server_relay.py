"""Relay HTTP routes."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import (
    JSONResponse,
)

from .proxy import RELAYED
from .server_http import (
    relay_refuses,
)


def install(app, proxy) -> None:
    @app.api_route("/engine/{path:path}", methods=["GET", "POST", "DELETE"])
    async def relay(request: Request, path: str):
        # Learner routes need Web's token; decode paths as the engine does, including query delimiters.
        if request.headers.get(RELAYED):            # the engine's address is this relay itself: answering would loop
            return JSONResponse({"detail": "the engine's address in Settings points back at this TensorFold Web's relay "
                                           "(/engine): give the engine's own address"}, status_code=508)
        refused = relay_refuses(request.url.path.removeprefix("/engine"), path)
        if refused:
            return JSONResponse({"detail": refused}, status_code=403 if "teach" in refused else 400)
        return await proxy.handle(request, path)
