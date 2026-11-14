import json

import pytest
from fastapi.testclient import TestClient

from tensorfold_web import fleet
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.mock_engine import serve as mock_serve
from tensorfold_web.server import create_app


def test_normalize_url():
    assert fleet.normalize_url("spark.example.ts.net") == "http://spark.example.ts.net"
    assert fleet.normalize_url(" https://mac:8090/ ") == "https://mac:8090"
    for bad in ("", None, "ftp://x", "http://x/api", "http://x/?a=1", "http://"):
        with pytest.raises(ValueError):
            fleet.normalize_url(bad)


TOKEN = "s3cret-peer-token"


def _app(tmp_path, port, token):
    s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    return create_app(settings=s, engine=Engine(s), catalog=cat, token=token)


@pytest.fixture
def two(tmp_path, monkeypatch):
    """This TensorFold Web and another one (token TOKEN) behind http://peer.test, both on a mock engine."""

    server = mock_serve(port=0)
    port = server.server_address[1]
    (tmp_path / "here").mkdir()
    (tmp_path / "there").mkdir()
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "there")
    peer = _app(tmp_path, port, TOKEN)
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "here")
    here = TestClient(_app(tmp_path, port, "mine"), headers={"x-studio-token": "mine"})
    monkeypatch.setattr(fleet, "client", lambda: TestClient(peer, base_url="http://peer.test"))
    # both apps run in this process, so they share a hostname: the checks see the peer as another machine
    real = fleet.test
    monkeypatch.setattr(fleet, "test", lambda url, token, *, own_hostname, own_version, own_engine_version=None:
                        real(url, token, own_hostname="here-host", own_version=own_version,
                             own_engine_version=own_engine_version))
    yield here, tmp_path
    server.shutdown()


def test_check_steps_and_what_they_say(two):
    here, _ = two
    r = here.post("/api/fleet/test", json={"url": "peer.test", "token": TOKEN}).json()
    assert r["ok"], r
    assert [c["id"] for c in r["checks"]] == [s for s, _ in fleet.STEPS]
    assert {c["id"]: c["status"] for c in r["checks"]}["engine"] in ("pass", "warn")   # warn while the mock warms up

    r = here.post("/api/fleet/test", json={"url": "http://peer.test", "token": "wrong"}).json()
    assert not r["ok"]
    status = {c["id"]: c["status"] for c in r["checks"]}
    assert status["token"] == "fail" and status["other"] == status["engine"] == "skip"

    assert here.post("/api/fleet/test", json={"url": "http://peer.test/api"}).status_code == 400


def test_add_without_restart_token_kept_server_side(two):
    here, tmp = two
    assert here.post("/api/fleet", json={"url": "http://peer.test", "token": "wrong"}).status_code == 409
    added = here.post("/api/fleet", json={"url": "http://peer.test", "token": TOKEN, "name": "Node 2"}).json()
    assert added["name"] == "Node 2" and added["token_set"] and "token" not in added
    assert here.post("/api/fleet", json={"url": "http://peer.test", "token": TOKEN}).status_code == 409   # twice

    view = here.get("/api/fleet").json()["machines"]
    assert [m["self"] for m in view] == [True, False]
    other = view[1]
    assert other["reachable"] and other["engine"]["status"] in ("starting", "running") and "token" not in other
    assert TOKEN not in json.dumps(view)                     # the token never reaches a page

    stored = tmp / "here" / "fleet.json"
    assert oct(stored.stat().st_mode & 0o777) == "0o600"

    here.put(f"/api/fleet/{added['id']}", json={"token": "wrong"})
    assert here.get("/api/fleet").json()["machines"][1]["error"] == "its TensorFold Web refuses the token"
    assert here.delete(f"/api/fleet/{added['id']}").json() == {"removed": added["id"]}
    assert len(here.get("/api/fleet").json()["machines"]) == 1


def test_unreachable_machine_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    out = fleet.probe({"id": "x", "name": "gone", "url": "http://127.0.0.1:9", "token": "t"})
    assert not out["reachable"] and out["error"].startswith("not reachable")
    r = fleet.test("http://127.0.0.1:9", None, own_hostname="h", own_version="0")
    assert [c["status"] for c in r["checks"]] == ["fail"] + ["skip"] * (len(fleet.STEPS) - 1)


def test_another_studio_opens_signed_in_through_a_one_time_link(two):
    here, _ = two
    pid = here.post("/api/fleet", json={"url": "peer.test", "token": TOKEN, "name": "peer"}).json()["id"]
    r = here.post(f"/api/fleet/{pid}/open").json()
    assert r["signed_in"] and r["url"].startswith("http://peer.test/?code=") and TOKEN not in r["url"]
    code = r["url"].split("code=")[1]
    peer = fleet.client()
    assert peer.post("/api/access/redeem", json={"code": code}).json() == {"token": TOKEN}   # the browser's side
    assert peer.post("/api/access/redeem", json={"code": code}).status_code == 403        # once only


def test_a_peer_that_stalls_after_its_first_answer_fails_the_check_instead_of_crashing(tmp_path, monkeypatch):
    import httpx

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, json={"detail": "a TensorFold Web token is required"})
        raise httpx.ReadTimeout("the link stalled", request=request)
    monkeypatch.setattr(fleet, "client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    r = fleet.test("http://peer.test", "tok", own_hostname="h", own_version="0")
    by = {c["id"]: c["status"] for c in r["checks"]}
    assert not r["ok"] and by["reach"] == "pass" and by["studio"] == "pass" and by["token"] == "fail"
    assert "ReadTimeout" in next(c["detail"] for c in r["checks"] if c["id"] == "token")
    assert by["engine"] == "skip"


def test_two_tensorfold_releases_are_flagged_before_they_are_compared():
    """Since 1.0.10 Nemotron on CUDA can part from 1.0.9 at a near tie: KL across two releases is not like for like."""

    assert fleet.engine_skew("1.0.11", "1.0.11") is None
    assert fleet.engine_skew("1.0.11", None) is None and fleet.engine_skew(None, "1.0.9") is None
    assert fleet.engine_skew("1.0.11", "0.6.6 (Python)") is None             # the Python line: nothing to compare
    said = fleet.engine_skew("1.0.11", "1.0.9")
    assert said and said.startswith("it runs TensorFold 1.0.9, this one 1.0.11") and "KL" in said
