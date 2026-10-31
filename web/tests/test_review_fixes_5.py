"""The internal review's list "before publication" (13 October 2026), done before TensorFold Web goes to TensorFold's
repository: one test per item that can be tested outside a browser."""

import base64

import httpx
import pytest
from fastapi.testclient import TestClient

from tensorfold_web import fit, living, living_checks, living_versions, power, quality, recipes, service
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine, Health
from tensorfold_web.server import create_app

GIB = 1024**3


def _client(tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    return TestClient(create_app(settings=s, engine=Engine(s), catalog=cat, **kw)), s


def test_b1_the_engine_key_stays_out_of_files_a_person_may_share():
    c = recipes.Ctx(base="http://127.0.0.1:8080", key="tf-secret", model="m")
    compose = recipes.compose(c)
    assert "tf-secret" not in compose and "${TENSORFOLD_API_KEY:?" in compose
    assert "tf-secret" not in recipes.SNIPPETS["codex"](c) and "TENSORFOLD_API_KEY" in recipes.SNIPPETS["codex"](c)
    assert 'OPENAI_API_KEY: "local"' in recipes.compose(recipes.Ctx(base="http://127.0.0.1:8080", key="local", model="m"))


def test_b2_the_service_file_is_written_private(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "conf")
    monkeypatch.setattr(service.settings_mod, "write_private", lambda p, text: written.append(p))
    monkeypatch.setattr(service, "_ensure_token", lambda p: None)
    plan = service.Plan("linux", tmp_path / "unit.service", "[Unit]\n", ["studio"], tmp_path / "tok", [])
    monkeypatch.setattr(service, "_run", lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    service.install(plan, start=False)
    assert written == [tmp_path / "unit.service"]


def test_b3_the_one_open_route_reads_at_most_a_kilobyte(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch, token="t0k")
    assert c.post("/api/access/redeem", content=b'{"code": "' + b"x" * 5000 + b'"}').status_code == 413
    assert c.post("/api/access/redeem", json={"code": "nope"}).status_code == 403
    assert c.post("/api/access/redeem", content=b"not json").status_code == 400


def test_b4_the_studio_never_relays_to_its_own_relay(tmp_path, monkeypatch):
    for url in ("http://studio.ts.net/engine", "https://studio.ts.net/engine/", "http://h:8090/engine/v1"):
        with pytest.raises(ValueError, match="relay"):
            settings_mod.validate({"engine_url": url})
    c, _ = _client(tmp_path, monkeypatch)
    assert c.get("/engine/v1/models", headers={"x-tfs-relayed": "1"}).status_code == 508


def test_b10_quality_reads_the_answer_line_a_typographic_minus_and_a_stated_letter():
    assert quality.score("gsm8k", {"answer": "18"}, "Work.\nAnswer: 18.\n\nI hope this helps 2 you")[0]
    assert quality.score("gsm8k", {"answer": "-3"}, "Answer: −3")[0]
    ok, got = quality.score("mmlu", {"answer": "C"}, "C is correct because A and B are wrong")
    assert ok and got == "C"
    assert quality.score("mmlu", {"answer": "C"}, "A or C")[1] == "hedged between letters"


def test_b13_teach_removes_only_copies_and_reads_only_its_own_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_WEB_LIVING", str(tmp_path / "living"))
    (tmp_path / "living" / "notacopy").mkdir(parents=True)
    with pytest.raises(KeyError):
        living.remove("notacopy")
    assert (tmp_path / "living" / "notacopy").is_dir()
    for bad in ("..", "x", "../../etc", "20261013-120000-zzzz"):
        with pytest.raises(KeyError):
            living_versions.remove("eden", bad)
        with pytest.raises(KeyError):
            living_checks.get("eden", bad)


def test_b14_an_nvidia_smi_that_tells_nothing_leaves_amd_smi_its_turn(monkeypatch):
    class Mute(power.Source):
        name, scope, whole_machine = "nvidia", "GPU", False

        def read(self):
            return None

    class Amd(power.Source):
        name, scope, whole_machine = "amd", "APU", False

        def read(self):
            return power.Reading(watts=40.0, cpu_c=None, gpu_c=55.0)
    monkeypatch.setattr(power.platform, "system", lambda: "Linux")
    monkeypatch.setattr(power.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(power, "NvidiaSmi", Mute)
    monkeypatch.setattr(power, "AmdSmi", Amd)
    src, how = power.detect()
    assert isinstance(src, Amd) and how == "amd-smi"


def test_b14_a_slow_nvidia_smi_keeps_its_reason_fields(monkeypatch):
    class Done:
        stdout, stderr = "", ""                           # no answer, and no word of an unknown field
    monkeypatch.setattr(power.subprocess, "run", lambda *a, **k: Done())
    src = power.NvidiaSmi()
    fields = list(src._fields)
    assert src.read() is None and src._fields == fields


def test_b15_logos_are_sent_again_only_when_they_changed(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    png = "data:image/png;base64," + base64.b64encode(b"\x89PNG" + b"0" * 32).decode()
    c.put("/api/brands/falcon", json={"label": "Falcon", "kind": "model", "words": ["falcon"], "logo": png})
    r = c.get("/api/brands")
    tag = r.headers["etag"]
    assert r.status_code == 200 and r.json()["brands"][0]["logo"] == png
    assert c.get("/api/brands", headers={"if-none-match": tag}).status_code == 304
    c.put("/api/brands/falcon", json={"label": "Falcon 2", "kind": "model", "words": ["falcon"]})
    assert c.get("/api/brands", headers={"if-none-match": tag}).status_code == 200


def test_b17_systemd_quoting_escapes_what_it_quotes():
    assert service._quote("plain") == "plain"
    assert service._quote('a "b" c\\d') == '"a \\"b\\" c\\\\d"'


def test_b18_the_verdict_is_said_at_the_budget_a_start_uses():
    cap = fit.mlx_capacity(32 * GIB, None)
    found = None
    for tenths in range(50, 220):
        w = int(tenths / 10 * GIB)
        at_default = fit.verdict(cap, w, bytes_per_token=64_000, max_context=262_144)
        lower = fit.verdict(cap, w, bytes_per_token=64_000, max_context=262_144, requested_fraction=0.5)
        if at_default.level == "fits" and lower.context_tokens and lower.context_tokens < fit.COMFORT_TOKENS:
            found = lower
            break
    assert found is not None and found.level == "tight" and found.budget_fraction == 0.5


def test_b18_a_drafter_weighs_its_measured_size_before_the_hub_answers(monkeypatch):
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    e = next(x for x in cat.entries if x.drafter == "z-lab/Qwen3.8-27B-DFlash2")
    monkeypatch.setattr(cat, "size_of", lambda repo: None)
    e.size_bytes = 10 * GIB
    assert cat.weights_of(e) == 10 * GIB + int(e.drafter_size_bytes * e.drafter_fraction)


def test_b19_the_page_and_the_history_keep_their_own_rate_baselines(monkeypatch):
    s = settings_mod.Settings(engine_url="http://e.test:8080", remote=True)
    eng = Engine(s)
    totals = iter([100.0, 400.0, 200.0])
    monkeypatch.setattr("tensorfold_web.engine.parse_health", lambda raw: Health(
        ok=True, status="running", model="m", counters={"generation_tokens_total": next(totals)}))
    eng._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"status": "ok"})))
    eng.health(reader="page")
    eng.health(reader="history")
    eng.health(reader="page")
    assert eng._counters["page"]["generation_tokens_total"] == 200.0
    assert eng._counters["history"]["generation_tokens_total"] == 400.0

