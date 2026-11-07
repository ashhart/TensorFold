"""Connect HTTP routes."""

from __future__ import annotations

import dataclasses
import re
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException
from fastapi.responses import (
    Response,
)

from . import (
    engines,
    recipes,
)
from .server_http import (
    connect_ctx,
    lan_hostname,
)


def install(app, eng, s, token) -> None:
    @app.get("/api/connect")
    def connect(where: str = "here", relay: str = ""):

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
