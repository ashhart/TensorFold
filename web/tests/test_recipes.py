import time

import pytest
from fastapi.testclient import TestClient

from tensorfold_web import recipes
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.mock_engine import serve as mock_serve
from tensorfold_web.server import create_app


@pytest.fixture
def mock():
    server = mock_serve(port=0, rate_tps=2000.0)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_every_recipe_writes_its_settings_with_the_address_key_and_model():
    c = recipes.Ctx(base="http://spark.local:8080", key="tf-abc", model="Nemotron", docker_base="http://host.docker.internal:8080")
    for r in recipes.RECIPES:
        text = r.snippet(c)
        if r.id == "cursor":                     # nothing to set: Cursor's servers cannot reach a private AI
            assert not r.checks and "private network" in text
            continue
        assert "Nemotron" in text or r.id == "open-webui", r.id
        # the key in every recipe a person pastes into a tool, never in a config file kept on disk (codex)
        assert ("tf-abc" in text) != (r.id == "codex"), r.id
        assert set(r.checks) <= set(recipes.CHECK_LABELS) and r.kind in recipes.KINDS
    assert "host.docker.internal:8080/v1" in recipes.SNIPPETS["librechat"](c)
    local = recipes.Ctx(base="http://127.0.0.1:8080", key="local", model="m")
    assert "env_key" not in recipes.SNIPPETS["codex"](local) and "apiKey" not in recipes.SNIPPETS["opencode"](local)


def test_the_open_webui_compose_reaches_the_engine_without_taking_its_port():
    text = recipes.compose(recipes.Ctx(base="http://127.0.0.1:8080", key="local", model="m"))
    assert "network_mode: host" in text and 'PORT: "3000"' in text and "http://127.0.0.1:8080/v1" in text
    # never a plain "open-webui": that name would start on another install's data, and migrate it
    assert "name: tensorfold-open-webui" in text and "container_name: tensorfold-open-webui" in text
    assert "open-webui:/app" not in text and "tensorfold-open-webui:/app" in recipes.SNIPPETS["open-webui"](
        recipes.Ctx(base="http://127.0.0.1:8080", key="local", model="m"))


def test_the_checks_against_an_engine(mock):
    c = recipes.Ctx(base=mock, key="local", model="Qwen3.8-27B")
    py = recipes.test(next(r for r in recipes.RECIPES if r.id == "python"), c)
    assert py["ok"] and [x["id"] for x in py["checks"]] == ["models", "chat_stream"]
    wrong = recipes.test(next(r for r in recipes.RECIPES if r.id == "python"), recipes.Ctx(base=mock, key="local", model="other"))
    assert not wrong["ok"] and "lists Qwen3.8-27B" in wrong["checks"][0]["detail"]
    codex = recipes.test(next(r for r in recipes.RECIPES if r.id == "codex"), c)
    assert not codex["ok"] and codex["checks"][1]["detail"].startswith("404")       # the mock has no Responses API
    gone = recipes.test(next(r for r in recipes.RECIPES if r.id == "python"), recipes.Ctx(base="http://127.0.0.1:9", key="local", model="m"))
    assert gone["checks"][0]["detail"].startswith("nothing answers")


def test_connect_routes(tmp_path, monkeypatch, mock):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url=mock, remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
    end = time.time() + 10
    while client.get("/api/engine").json()["status"] != "running" and time.time() < end:
        time.sleep(0.1)
    info = client.get("/api/connect").json()
    assert [r["id"] for r in info["snippets"]][:2] == ["open-webui", "librechat"] and info["kinds"]["agent"]
    # the served window, for the page's warning on the coding agents and for Claude Code's own setting
    cc = next(r for r in info["snippets"] if r["id"] == "claude-code")
    assert info["window"] == 32768 and cc["min_context"] == 32768 and "CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768" in cc["snippet"]
    # the same recipes for a tool on another computer: every line through TensorFold Web's address, with its key
    away = client.get("/api/connect", params={"where": "away", "relay": "https://box.example.ts.net:8443"}).json()
    snip = {r["id"]: r for r in away["snippets"]}
    assert away["where"] == "away" and "ANTHROPIC_BASE_URL=https://box.example.ts.net:8443/engine\n" in snip["claude-code"]["snippet"]
    assert "GENERIC_OPEN_AI_BASE_PATH=https://box.example.ts.net:8443/engine/v1" in snip["anythingllm"]["snippet"]
    assert "YOUR-KEY" in snip["jan"]["snippet"] and "compose" not in snip["open-webui"] and "-p 3000:8080" in snip["open-webui"]["snippet"]
    assert snip["jan"]["form"] == "desktop app" and "host.docker.internal" not in snip["librechat"]["snippet"]
    assert "host.docker.internal" in {r["id"]: r for r in info["snippets"]}["librechat"]["where_note"]
    assert client.get("/api/connect", params={"where": "away", "relay": "javascript:x"}).status_code == 400
    out = client.post("/api/connect/test", json={"recipe": "python"}).json()
    assert out["ok"] and out["checks"][1]["id"] == "chat_stream"
    assert client.post("/api/connect/test", json={"recipe": "nope"}).status_code == 404
    compose = client.get("/api/connect/compose/open-webui")
    assert "attachment" in compose.headers["content-disposition"] and "open-webui" in compose.text


def test_the_window_check_says_what_an_agent_needs(mock):
    c = recipes.Ctx(base=mock, key="local", model="Qwen3.8-27B")
    out = recipes.run_check("context", c, __import__("httpx").Client(timeout=10), {"min_context": 65536})
    assert not out["ok"] and "32,768 tokens served" in out["detail"] and "Longest conversation" in out["detail"]
    assert recipes.run_check("context", c, __import__("httpx").Client(timeout=10), {"min_context": 16384})["ok"]


def test_the_new_agents_are_written_for_the_place_and_window():
    away = recipes.Ctx(base="https://box.example.ts.net:8443/engine", key="tf-k", model="Nemo", window=65536, away=True)
    import json
    pi = json.loads(recipes.SNIPPETS["pi"](away))["providers"]["tensorfold"]
    assert pi["baseUrl"] == "https://box.example.ts.net:8443/engine/v1" and pi["models"][0]["contextWindow"] == 65536
    fox = json.loads(recipes.SNIPPETS["openfox"](away))
    assert fox["defaultModelSelection"] == "tensorfold/Nemo" and fox["providers"][0]["isLocal"] is False
    assert "base_url: https://box.example.ts.net:8443/engine/v1" in recipes.SNIPPETS["hermes"](away)
    assert "api_key: \n" in recipes.SNIPPETS["hermes"](recipes.Ctx(base="http://127.0.0.1:8080", key="local", model="m"))
    assert 'primary: "tensorfold/Nemo"' in recipes.SNIPPETS["openclaw"](away) and "modelRoles" in recipes.SNIPPETS["omp"](away)
    kinds = {r.id: r.kind for r in recipes.RECIPES}
    assert kinds["openclaw"] == kinds["hermes"] == "assistant" and "continue" not in kinds
