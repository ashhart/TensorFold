"""TensorFold Web against TensorFold main's server, mocked from its source (d377fcc6a): no GPU, no engine binary."""

import asyncio
from urllib.parse import urlsplit

import httpx

from tensorfold_web.proxy import Proxy


def main_origin_rule(request: httpx.Request) -> bool:
    """zig/src/server/request_guard.zig's browser rule: no Origin passes; an Origin must be http and name the
    request's own Host and port (no --allowed-origin here)."""

    origins = [v for k, v in request.headers.multi_items() if k.lower() == "origin"]
    if not origins:
        return True
    if len(origins) != 1:
        return False
    o = urlsplit(origins[0])
    host = request.headers.get("host", "")
    name, _, port = host.partition(":")
    return o.scheme == "http" and (o.hostname or "") == name.lower() and (o.port or 80) == int(port or 80)


def guarded_engine(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if not main_origin_rule(request):
            msg = "Origin header does not match this server; use --allowed-origin ORIGIN for a trusted origin"
            return httpx.Response(403, json={"error": {"message": msg, "type": "invalid_request_error"}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
    return httpx.MockTransport(handler)


class _Req:
    def __init__(self, method: str, headers: dict, body: bytes = b""):
        self.method, self.query_params = method, httpx.QueryParams()
        self.headers = httpx.Headers(headers)
        self._body = body

    async def stream(self):
        yield self._body


def test_the_mocked_rule_refuses_the_studio_page_origin():
    # the failure being fixed: the page's own origin, forwarded as it was, is a foreign origin to the engine
    r = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions", headers={"origin": "http://127.0.0.1:8090"})
    assert not main_origin_rule(r)
    assert main_origin_rule(httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions"))


def test_chat_through_the_relay_reaches_main_without_the_browser_origin():
    seen: list[httpx.Request] = []
    p = Proxy(lambda: "http://127.0.0.1:8080", lambda: "KEY")
    p._client = httpx.AsyncClient(transport=guarded_engine(seen))
    req = _Req("POST", {"origin": "http://127.0.0.1:8090", "content-type": "application/json",
                        "host": "127.0.0.1:8090"}, b'{"messages": []}')
    resp = asyncio.run(p.handle(req, "v1/chat/completions"))
    assert resp.status_code == 200
    assert len(seen) == 1
    assert "origin" not in {k.lower() for k in seen[0].headers}
    assert seen[0].headers["host"] == "127.0.0.1:8080"          # the engine's own address, not the page's
    assert seen[0].headers["authorization"] == "Bearer KEY"


def test_upstream_headers_drop_origin_in_any_case():
    p = Proxy(lambda: "http://127.0.0.1:8080")
    out = p.upstream_headers([("Origin", "http://a.test"), ("oRiGiN", "http://b.test"), ("accept", "*/*")])
    assert {k.lower() for k in out} == {"accept", "x-tfs-relayed"}


# -- teaching: main refuses `--slide` without a key (zig/src/server/slide_key.zig) ---------------------------------


def _slide_keyless(cmd: list[str], env: dict) -> bool:
    """main.zig's gate before any load: `--slide` with no --api-key, --api-key-file or TENSORFOLD_API_KEY exits 2."""

    if "--slide" not in cmd:
        return False
    return not ("--api-key" in cmd or "--api-key-file" in cmd or (env.get("TENSORFOLD_API_KEY") or "").strip(" ,"))


def test_starting_a_copy_that_learns_gives_it_a_key_main_accepts(tmp_path, monkeypatch):
    import json

    from fastapi.testclient import TestClient

    from tensorfold_web import living, native
    from tensorfold_web import settings as settings_mod
    from tensorfold_web.catalog import Catalog
    from tensorfold_web.engine import Engine
    from tensorfold_web.server import create_app
    from tensorfold_web.settings import Settings

    monkeypatch.setenv("TENSORFOLD_WEB_LIVING", str(tmp_path / "living"))
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "conf")
    monkeypatch.setattr(living, "teachable_here", lambda _b: None)
    (living.root() / "eden").mkdir(parents=True)
    (living.root() / "eden" / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))

    s = Settings(engine_url="http://127.0.0.1:9", remote=False)
    assert s.api_key is None                                      # sharing is off: until now, no key at all
    eng = Engine(s)
    launched = []

    def fake_start(model, *, native_model=False, drafter=None, slide=None, serve=None):
        caps = {"version": "main", "serve": {"flags": {"--host": {}, "--port": {}, "--slide": {}, "--slide-graph": {}}}}
        monkeypatch.setattr(native, "capabilities", lambda _b: caps)
        monkeypatch.setattr(native, "serve_flags", lambda _b: caps["serve"]["flags"])
        cmd, env, _notes = native.build_command("tensorfold-native", serve or model, s, slide=slide)
        launched.append((cmd, env))
        return {}

    eng.start = fake_start
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    c = TestClient(create_app(settings=s, engine=eng, catalog=cat))
    r = c.post("/api/engine/start", json={"model": "living:eden"})
    assert r.status_code == 200, r.text
    (cmd, env), = launched
    assert "--slide" in cmd and not _slide_keyless(cmd, env)
    assert env["TENSORFOLD_API_KEY"] == s.api_key and s.api_key not in " ".join(cmd)   # never on the command line
    saved = json.loads((tmp_path / "conf" / "settings.json").read_text())
    assert saved["api_key"] == s.api_key                         # kept, so the next start and the relay agree


def test_a_key_already_set_is_kept_for_learning(monkeypatch):
    from tensorfold_web import native
    from tensorfold_web.settings import Settings

    monkeypatch.setattr(native, "capabilities", lambda _b: {"serve": {"flags": {"--slide": {}, "--slide-graph": {}}}})
    monkeypatch.setattr(native, "serve_flags", lambda _b: {"--slide": {}, "--slide-graph": {}})
    cmd, env, _ = native.build_command("tensorfold-native", "/x/eden", Settings(api_key="tf-mine"),
                                       slide="/x/eden.facts.json")
    assert env["TENSORFOLD_API_KEY"] == "tf-mine" and not _slide_keyless(cmd, env)


# -- server ready and load: GET /health and GET /v1/load, as main answers them behind a key -------------------------


def keyed_main(asked: list, *, parked: bool = False, key: str = "tf-k"):
    """docs/api.md: with keys, /health is only {"status": "ok"}; /v1/load needs the key and returns the ledger."""

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        ok = request.headers.get("authorization") == f"Bearer {key}"
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/load":
            if not ok:
                return httpx.Response(401, json={"error": {"message": "missing or invalid API key"}})
            return httpx.Response(200, json={
                "instance": "0f3c", "timestamp": 1761940800.5, "model": "nemotron-3.5-lightning", "parked": parked,
                "lanes": 4, "http_active": {"chat_completions": 1, "completions": 0, "responses": 0, "messages": 0,
                                            "embeddings": 0, "speech": 0, "other": 0},
                "running": 1, "waiting": 2, "requests_started": 9, "requests_requeued": 0, "background_yields": 0,
                "waiting_prompt_tokens": 300, "fill_unknown": False})
        if request.url.path == "/metrics":
            return httpx.Response(200 if ok else 401, text="tensorfold:num_requests_running 0\n")
        return httpx.Response(404, json={"error": {"message": "not found"}})
    return httpx.MockTransport(handler)


def _engine(transport, key="tf-k"):
    from tensorfold_web.engine import Engine
    from tensorfold_web.settings import Settings

    eng = Engine(Settings(engine_url="http://127.0.0.1:8080", api_key=key, remote=True))
    eng._client = httpx.Client(transport=transport)
    return eng


def test_model_and_load_come_from_v1_load_behind_a_key():
    asked: list[str] = []
    h = _engine(keyed_main(asked)).health()
    assert h.ok and h.status == "running"
    assert h.model == "nemotron-3.5-lightning" and h.model_ids == ["nemotron-3.5-lightning"]
    assert "/v1/models" not in asked                       # the server's own ledger named it
    assert h.load["lanes"] == 4 and not h.parked
    assert (h.connections, h.waiting) == (3, 2)            # running and waiting, as /health's live block counts them


def test_a_parked_engine_is_reported_parked():
    h = _engine(keyed_main([], parked=True)).health()
    assert h.ok and h.parked and h.load["parked"] is True


def test_a_wrong_key_reads_as_unauthorized_not_as_running_without_a_model():
    h = _engine(keyed_main([]), key="tf-wrong").health()
    assert h.status == "unauthorized" and "refuses the API key" in (h.error or "")


def test_lanes_from_v1_load_reach_the_page_startup_figures():
    from tensorfold_web.engine import Engine
    from tensorfold_web.settings import Settings

    eng = Engine(Settings(engine_url="http://127.0.0.1:8080", api_key="tf-k", remote=False))
    eng._client = httpx.Client(transport=keyed_main([]))
    assert eng.info()["startup"]["streams"] == 4.0


def test_parse_load_refuses_what_is_not_the_ledger():
    from tensorfold_web.engine import parse_load

    assert parse_load({"status": "ok"}) is None and parse_load([]) is None
    assert parse_load({"model": "m", "parked": 0}) is None               # a bool, as main writes it
    assert parse_load({"model": "m", "parked": False, "lanes": True, "running": -1}) == {"model": "m", "parked": False}


def test_the_log_fallback_reads_mains_serving_line():
    from tensorfold_web.native import parse_startup

    # zig/src/server/serve.zig:247, with draft_line.zig's phrases
    line = ("[tensorfold] serving nemotron at http://127.0.0.1:8080/v1 (sampling: temperature 1.0, top_p 0.95; "
            "drafts: DFlash2 (z-lab/Qwen3.8-27B-DFlash2) + copies; context: 262144; loaded in 41.3s)")
    assert parse_startup(line) == {"loaded_s": 41.3, "drafts": 1.0, "context_tokens": 262144.0}
    unlimited = "serving m at http://127.0.0.1:8080/v1 (sampling: greedy; drafts: off; context: unlimited; loaded in 2.0s)"
    assert parse_startup(unlimited) == {"loaded_s": 2.0, "drafts": 0.0}
    none = "serving m at http://127.0.0.1:8080/v1 (sampling: greedy; drafts: none; context: 4096; loaded in 1.5s)"
    assert parse_startup(none)["drafts"] == 0.0


# -- teaching is Nemotron-only ---------------------------------------------------------------------------------------


def _home_with_copy(tmp_path, monkeypatch, name: str, family: str):
    import json

    from tensorfold_web import living
    from tensorfold_web import settings as settings_mod

    monkeypatch.setenv("TENSORFOLD_WEB_LIVING", str(tmp_path / "living"))
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "conf")
    monkeypatch.setattr(living, "teachable_here", lambda _b: None)        # the machine could learn
    (living.root() / name).mkdir(parents=True)
    (living.root() / name / "config.json").write_text(json.dumps({"model_type": family}))


def _app(engine_url="http://127.0.0.1:9"):
    from fastapi.testclient import TestClient

    from tensorfold_web.catalog import Catalog
    from tensorfold_web.engine import Engine
    from tensorfold_web.server import create_app
    from tensorfold_web.settings import Settings

    s = Settings(engine_url=engine_url, remote=False)
    eng = Engine(s)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    return s, eng, TestClient(create_app(settings=s, engine=eng, catalog=cat))


def test_a_lesson_on_another_family_is_refused_by_name(tmp_path, monkeypatch):
    from tensorfold_web.engine import Health, _Child

    _home_with_copy(tmp_path, monkeypatch, "qcopy", "qwen3_5")
    s, eng, c = _app()

    class Running:                                               # an engine process that is alive
        pid = 1

        def poll(self):
            return None
    eng.proc, eng.model = _Child(popen=Running()), "living:qcopy"
    eng.status = lambda reader="page": Health(ok=True, status="running", model="qcopy")
    r = c.post("/api/living/teach", json={"facts": [{"fact": "My cat is Pixel.", "question": "What is my cat called?"}]})
    assert r.status_code == 409
    assert "Nemotron 3.5 Lightning only" in r.json()["detail"] and "qwen3_5" in r.json()["detail"]


def test_another_family_serves_without_learning_and_needs_no_key(tmp_path, monkeypatch):
    _home_with_copy(tmp_path, monkeypatch, "qcopy", "qwen3_5")
    s, eng, c = _app()
    started = []
    eng.start = lambda model, **kw: started.append(kw) or {}
    notes = []
    eng.note = notes.append
    assert c.post("/api/engine/start", json={"model": "living:qcopy"}).status_code == 200
    assert started[0]["slide"] is None and s.api_key is None
    assert any("Nemotron 3.5 Lightning only" in n for n in notes)


def test_only_a_nemotron_copy_passes_the_family_check(tmp_path, monkeypatch):
    from tensorfold_web import living

    _home_with_copy(tmp_path, monkeypatch, "eden", "nemotron_h")
    (living.root() / "bare").mkdir()                              # no config.json: the family is unknown
    assert living.family_refusal("eden") is None
    assert "Nemotron 3.5 Lightning only" in living.family_refusal("bare")


# -- framing ---------------------------------------------------------------------------------------------------------


def _framing_ok(r) -> bool:
    return r.headers.get("x-frame-options") == "DENY" and \
        "frame-ancestors 'none'" in r.headers.get("content-security-policy", "")


def test_every_page_refuses_framing(tmp_path, monkeypatch):
    from tensorfold_web import settings as settings_mod

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "conf")
    _s, _eng, c = _app()
    for path in ("/", "/index.html", "/api/settings", "/manifest.webmanifest", "/no-such-page"):
        assert _framing_ok(c.get(path)), path
    refused = c.get("/api/settings", headers={"host": "rebind.example"})    # the gate's own refusal
    assert refused.status_code == 421 and _framing_ok(refused)


def test_an_engine_page_keeps_its_policy_but_not_its_frame_ancestors():
    from tensorfold_web.server import frame_policy

    assert frame_policy(None) == "frame-ancestors 'none'"
    assert frame_policy("default-src 'self'; frame-ancestors *; img-src data:") == \
        "default-src 'self'; img-src data:; frame-ancestors 'none'"


# -- pinned installs -------------------------------------------------------------------------------------------------


def test_every_install_command_names_an_exact_version():
    import pathlib
    import re

    from tensorfold_web import tooleval

    assert re.fullmatch(r"\d+\.\d+\.\d+", tooleval.VERSION)
    root = pathlib.Path(__file__).resolve().parents[1]
    sources = [*root.glob("tensorfold_web/*.py"), *root.glob("ui/src/**/*.tsx"), *root.glob("ui/src/**/*.ts"),
               *root.glob("tensorfold_web/ui_dist/assets/*.js")]
    # a package after `uv tool install`, `pip install` or `pipx install`; a bare mention (`uv tool install` puts…) is
    # no command
    command = re.compile(r"(?:uv tool install|pip3? install|pipx install)\s+(?!-)([A-Za-z][\w.-]*)(\S*)")
    seen = 0
    for path in sources:
        for m in command.finditer(path.read_text(encoding="utf-8", errors="replace")):
            name, rest = m.group(1), m.group(2)
            if name in ("puts", "for", "and", "the", "first", "it"):
                continue
            seen += 1
            assert rest.startswith("==") or "{" in rest, f"{path.name}: {m.group(0)!r} is not pinned"
    assert seen >= 3
