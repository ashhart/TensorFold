"""Lessons HTTP routes."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx
from fastapi import HTTPException

from . import (
    living,
    living_checks,
    living_versions,
)
from .gate import Busy


def install(app, cat, chore, chore_cancel, chore_lock, copy_name, creating, eng, gate, launch, lesson_target, run_chore, s, served_copy, teacher, wait_running) -> None:
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
                r = httpx.get(trust_env=False, url=s.engine_url.rstrip("/").removesuffix("/v1") + "/v1/slide/graph", timeout=5.0,
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
