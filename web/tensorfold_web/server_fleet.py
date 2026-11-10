"""Fleet HTTP routes."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException
from fastapi.responses import (
    Response,
)

from . import (
    __version__,
    diagnostic,
    fleet,
)
from .machine import detect


def install(app, eng, engine_info, fleet_body, machine, peers, run_checks, s, token, used) -> None:
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
                # it refused this TensorFold Web's origin: served under another name
                if r.status_code == 403:
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
