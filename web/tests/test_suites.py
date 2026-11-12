import time

import pytest
from fastapi.testclient import TestClient

from tensorfold_web import bench, report
from tensorfold_web import settings as settings_mod
from tensorfold_web import suites as custom
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.mock_engine import serve as mock_serve
from tensorfold_web.server import create_app

SUITE = {"name": "Support replies", "about": "How it answers customers",
         "prompts": [{"user": "My order has not arrived. What can I do?", "max_tokens": 120},
                     {"user": "Rédige une réponse polie à un client mécontent.", "system": "You answer in French."}]}


@pytest.fixture
def mock():
    server = mock_serve(port=0, rate_tps=2000.0)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_a_suite_is_checked_before_it_is_kept():
    for bad, said in (({"prompts": [{"user": "x"}]}, "name"), ({"name": "a", "prompts": []}, "1 to 20"),
                      ({"name": "a", "prompts": [{"user": " "}]}, "prompt 1"),
                      ({"name": "a", "prompts": [{"user": "x", "max_tokens": 5}]}, "reply length"),
                      ({"name": "a", "prompts": [{"user": "x"}] * 21}, "1 to 20"), ([], "JSON object")):
        with pytest.raises(ValueError, match=said):
            custom.validate(bad)
    kept = custom.validate(SUITE)
    assert [p["id"] for p in kept["prompts"]] == ["p1", "p2"] and kept["prompts"][1]["max_tokens"] == 300


def test_the_digest_follows_what_is_sent_not_the_name():
    a = custom.validate(SUITE)
    renamed = {**a, "name": "Other name", "about": "changed"}
    edited = {**a, "prompts": [{**a["prompts"][0], "user": "Where is my parcel?"}, a["prompts"][1]]}
    assert custom.digest(a) == custom.digest(renamed) != custom.digest(edited)


def test_an_exported_suite_imports_as_the_same_suite(tmp_path):
    store = custom.CustomSuites(tmp_path / "bench.sqlite")
    first = store.save(SUITE)
    again = store.save(custom.export(first))
    assert again["id"] != first["id"] and custom.digest(again) == custom.digest(first)
    assert [s["name"] for s in store.list()] == ["Support replies", "Support replies"]
    with pytest.raises(KeyError):
        store.save(SUITE, "custom-missing")
    assert store.delete(first["id"]) and not store.delete(first["id"])


def test_a_profile_runs_a_custom_suite_and_records_its_digest(tmp_path, mock):
    store = bench.Store(tmp_path / "bench.sqlite")
    runner = bench.Bench(store, lambda: (mock, None, "m"))
    runner.power = False
    suite = runner.custom.save(SUITE)
    run = runner.start("profile", suites=["code", suite["id"]])
    end = time.time() + 30
    while runner.busy() and time.time() < end:
        time.sleep(0.05)
    got = store.run(run["id"])
    assert got["status"] == "done" and got["suites"] == ["code", suite["id"]]
    assert got["suites_version"]["custom"] == {suite["id"]: custom.digest(suite)}
    assert got["suite_names"][suite["id"]] == "Support replies"
    assert {r["suite"] for r in got["results"]} == {"code", suite["id"]}
    assert "Support replies (custom)" in report.html_report(got)
    # a profile with no choice runs the built-in suites only, so its figures compare with other machines'
    run2 = runner.start("profile", exactness=False)
    while runner.busy() and time.time() < end + 30:
        time.sleep(0.05)
    assert suite["id"] not in store.run(run2["id"])["suites"]


def test_custom_suite_routes(tmp_path, monkeypatch, mock):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url=mock, remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
    assert client.post("/api/bench/suites", json={"name": ""}).status_code == 400
    made = client.post("/api/bench/suites", json=SUITE).json()
    listed = client.get("/api/bench/suites").json()["custom"]
    assert [c["id"] for c in listed] == [made["id"]] and listed[0]["digest"] == custom.digest(made)
    changed = client.put(f"/api/bench/suites/{made['id']}", json={**SUITE, "name": "Support"}).json()
    assert changed["name"] == "Support" and changed["created"] == made["created"]
    exp = client.get(f"/api/bench/suites/{made['id']}/export")
    assert 'filename="suite-support.json"' in exp.headers["content-disposition"] and exp.json()["kind"] == custom.KIND
    assert client.put("/api/bench/suites/custom-nope", json=SUITE).status_code == 404
    assert client.delete(f"/api/bench/suites/{made['id']}").status_code == 200
    assert client.get("/api/bench/suites").json()["custom"] == []


def test_earlier_turns_are_sent_before_the_message_and_counted_in_the_digest():
    plain = custom.validate(SUITE)
    turns = [{"role": "user", "content": "Hi, my order 4411 is late."}, {"role": "assistant", "content": "Sorry; when was it due?"}]
    talk = custom.validate({**SUITE, "prompts": [{**SUITE["prompts"][0], "history": turns}, SUITE["prompts"][1]]})
    assert custom.digest(plain) == custom.digest(custom.validate(SUITE))   # a suite without turns keeps its digest
    assert custom.digest(talk) != custom.digest(plain)
    assert [m["role"] for m in custom.as_bench({**talk, "id": "x"})["prompts"][0]["messages"]] == ["user", "assistant", "user"]
    assert custom.export({**talk, "id": "x"})["prompts"][0]["history"] == turns
    for bad in ([{"role": "system", "content": "x"}], [{"role": "user", "content": ""}], "not a list"):
        with pytest.raises(ValueError):
            custom.validate({**SUITE, "prompts": [{**SUITE["prompts"][0], "history": bad}]})


def test_a_built_in_suite_copied_in_sends_the_same_messages():
    data = bench.load_suites()
    for suite in data["suites"]:
        copy = custom.from_builtin(suite)
        sent = custom.as_bench({**copy, "id": "x"})["prompts"]
        assert [p["messages"] for p in sent] == [p["messages"] for p in suite["prompts"]], suite["id"]
        assert [p["max_tokens"] for p in sent] == [p["max_tokens"] for p in suite["prompts"]]
        assert copy["name"].endswith("(my copy)")


def test_copy_route(tmp_path, monkeypatch, mock):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url=mock, remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
    made = client.post("/api/bench/suites/copy/chat").json()
    assert made["id"].startswith("custom-") and made["prompts"][0]["history"]
    assert client.post("/api/bench/suites/copy/nope").status_code == 404
