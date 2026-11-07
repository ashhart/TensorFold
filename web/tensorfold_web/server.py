"""Build the Web services and register their HTTP boundaries."""

from __future__ import annotations

import asyncio
import dataclasses
import hmac
import json
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    Response,
)

from . import (
    __version__,
    bench,
    brands,
    conversations,
    diagnostic,
    engines,
    fleet,
    health,
    history,
    living,
    living_checks,
    living_versions,
    quality,
    replay,
    series,
    server_bench,
    server_chat,
    server_connect,
    server_engine,
    server_engine_control,
    server_fleet,
    server_lessons,
    server_living,
    server_page,
    server_relay,
    server_security,
    server_settings,
    server_storage,
    server_system,
    usage,
)
from . import access as access_mod
from . import settings as settings_mod
from .catalog import Catalog
from .engine import Engine
from .gate import Gate
from .machine import detect
from .proxy import Proxy
from .pull import Downloads
from .server_http import (
    ENGINE_SETTINGS,
    LOGS_COOKIE,
    LOGS_STREAM,
    LOOPBACK_NAMES,
    REDEEM_BYTES,
    TOKEN_HEADER,
    UI_DIST,
    UNSAFE,
    VIEW,
    VIEW_COOKIE,
    _host_of,
    _site,
    attachment,
    connect_ctx,
    frame_policy,
    lan_hostname,
    relay_refuses,
    same_secret,
)


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
                  # no route map, and no CDN script, outside the token
                  docs_url=None, redoc_url=None, openapi_url=None)

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

    used = usage.Usage(settings_mod.config_dir() / "usage.sqlite")

    kept = history.History(settings_mod.config_dir() / "history.sqlite", lambda: eng.info(reader="history"),
                           then=used.record)

    app.state.history = kept

    signals, checker = health.Signals(), health.Checker()

    friendly_name, launch, held_in_thread, _start_held = server_engine_control.build(cat, eng, gate, s, save_settings)





    links = access_mod.Links()

    qd = quality.load()

    quality_info = {"about": qd["about"], "version": qd["version"], "digest": qd["digest"],
                    "categories": [{"id": k, "name": v["name"], "items": len(v["items"])} for k, v in qd["categories"].items()]}

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

    def engine_now() -> dict:
        h = eng.status()
        return {"model": eng.model if eng.owned and h.status in ("running", "starting") else None,
                "context": s.context, "budget_fraction": s.budget_fraction, "settings_model": s.model}

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

    def fleet_engine(pid: object) -> tuple[engines.Endpoint, str | None, dict, str]:

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

        if isinstance(spec, dict) and "peer" in spec:
            return fleet_engine(spec["peer"])
        if not isinstance(spec, dict) or not isinstance(spec.get("url"), str):
            raise ValueError(f"{what} must be an object with a url, or a Fleet machine's id as peer")  # noqa: TRY004
        key = spec.get("key") or None
        return engines.identify(settings_mod._engine_url(spec["url"]), key=key), key, {}, None

    def _two(a: str, b: str) -> tuple[dict, dict]:
        ra, rb = runner.store.run(a), runner.store.run(b)
        if ra is None or rb is None:
            raise HTTPException(404, "unknown run")
        return ra, rb

    chats = conversations.Conversations(settings_mod.config_dir() / "chat.sqlite")

    app.state.chats = chats

    def served_target() -> tuple[str, str | None, str | None, str | None]:
        base, key, model = bench_target()
        return base, key, model, served_name()

    replays = replay.Replays(chats, served_target)

    makers = brands.Brands(settings_mod.config_dir() / "brands.json")

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

    def run_checks() -> tuple[list, dict, dict]:
        from . import pull, service

        m, info = machine(), engine_info()
        try:
            svc = service.status(service.plan())
        # no systemctl or launchctl here, or one that hangs
        except (RuntimeError, OSError, subprocess.SubprocessError):
            svc = None
        installed = [] if s.remote else [v["name"] for v in cat.view(detect(), fraction=s.budget_fraction)
                                         if v.get("installed") and v.get("runs_here")]
        checks = diagnostic.run(machine=m, info=info, settings=s.to_dict(mask=True), cache_root=pull.cache_root(),
                                installed=installed, bind=bind_host, token=bool(token), service_status=svc)
        return checks, m, info

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

    # left halfway by a stopped TensorFold Web
    for said in living_versions.recover() + living_versions.recover_unfinished():
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

    def living_tag() -> dict | None:
        name = served_copy()
        if not name:
            return None
        record = living.lessons(name)
        versions = living_versions.manifests(name)
        # the engine's graph counts facts taught outside TensorFold Web too
        try:
            graph = json.loads(living.graph_path(name).read_text())
            kept = sum(1 for f in graph.get("facts") or [] if f.get("state") == "learned")
        except (OSError, ValueError):
            kept = sum(1 for x in record if x.get("outcome") in living.KEPT)
        return {"copy": name, "kept": kept, "lessons": len(record),
                "last_lesson": record[-1]["job"] if record else None, "last_version": versions[-1]["id"] if versions
                else None}

    def settled(name: str) -> None:
        """A copy's files are given to another machine only while nothing here writes them."""

        if (teacher.busy() and ((teacher.job and teacher.job.copy == name) or teacher.recovering == name)) or \
                (chore.get("state") == "running" and chore.get("copy") == name) or \
                (creating.get("state") == "copying" and creating.get("name") == name):
            raise HTTPException(409, f"{name} is being written here (a lesson or a long task); fetch it later")
        if living.is_unfinished(name):
            raise HTTPException(409, f"{name} is not complete here; fetch it later")

    def engine_get(path: str) -> httpx.Response:
        key = eng.key_in_use()
        return httpx.get(trust_env=False, url=s.engine_url.rstrip("/").removesuffix("/v1") + path, timeout=10.0,
                         headers={"Authorization": f"Bearer {key}"} if key else {})

    def no_store(resp: Response) -> Response:
        resp.headers["Cache-Control"] = "no-store"
        return resp

    looked_for_base: set[str] = set()

    chore: dict[str, Any] = {}                             # one long task at a time: a version, a restore, checks

    chore_lock = threading.Lock()                           # a check and a reservation, as one step

    # set by Teach's cancel; a chore that asks questions reads it
    chore_cancel = threading.Event()

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

    server_security.install(_engine_key_given, allowed_hosts, app, logs_cookie, token, view_cookie, LOOPBACK_NAMES)
    machine = server_system.install(app, cat, kept, s, used, eng)
    engine_info = server_engine.install(_start_held, app, bind_host, cat, checker, dl, eng, friendly_name, gate, held_in_thread, links, logs_cookie, s, signals, start_lock, token, used, LOOPBACK_NAMES)
    server_bench.install(_two, app, bench_go, comparisons, concurrency, engine_named, quality_info, runner, s, served_name)
    server_settings.install(app, eng, gate, s, save_settings, user_settings)
    server_chat.install(app, chats, makers, replays)
    server_connect.install(app, eng, s, token)
    server_fleet.install(app, eng, engine_info, fleet_body, machine, peers, run_checks, s, token, used)
    server_storage.install(app, catalog_repos, gate, in_use)
    server_relay.install(app, proxy)
    server_living.install(app, bench_go, cat, chore, copy_name, creating, eng, engine_get, gate, launch, looked_for_base, no_store, peers, run_chore, runner, served_copy, served_name, settled, teacher, view_cookie, wait_running)
    server_lessons.install(app, cat, chore, chore_cancel, chore_lock, copy_name, creating, eng, gate, launch, lesson_target, run_chore, s, served_copy, teacher, wait_running)
    server_page.install(app)
    return app


__all__ = ['ENGINE_SETTINGS', 'LOGS_COOKIE', 'LOGS_STREAM', 'LOOPBACK_NAMES', 'REDEEM_BYTES', 'TOKEN_HEADER', 'UI_DIST', 'UNSAFE', 'VIEW', 'VIEW_COOKIE', '_host_of', '_site', 'attachment', 'connect_ctx', 'create_app', 'frame_policy', 'lan_hostname', 'relay_refuses', 'same_secret']
