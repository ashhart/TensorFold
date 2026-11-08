"""Living HTTP routes."""

from __future__ import annotations

import contextlib
import json
import re
import time
from typing import Any

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
)

from . import (
    fleet,
    living,
    living_checks,
    living_versions,
)
from .server_http import (
    VIEW,
    VIEW_COOKIE,
)


def install(app, bench_go, cat, chore, copy_name, creating, eng, engine_get, gate, launch, looked_for_base, no_store, peers, run_chore, runner, served_copy, served_name, settled, teacher, view_cookie, wait_running) -> None:
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
            # Hold one reservation across versioning, replay and rollback so no other job can interrupt recovery.
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
            with httpx.Client(trust_env=False, timeout=httpx.Timeout(1800.0, connect=10.0)) as c:
                return c.get(f"{peer['url']}/api/living/copies/{src}/manifest", headers=h).raise_for_status().json()

        def file(rel: str):
            with httpx.Client(trust_env=False, timeout=httpx.Timeout(600.0, connect=10.0)) as c, \
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
