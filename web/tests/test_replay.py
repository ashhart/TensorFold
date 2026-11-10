import time

import pytest
from fastapi.testclient import TestClient

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


def _wait(client, timeout=30):
    end = time.time() + timeout
    while client.get("/api/replay").json()["busy"] and time.time() < end:
        time.sleep(0.1)


def test_a_conversation_replayed_on_two_configurations_and_compared(tmp_path, monkeypatch, mock):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url=mock, remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
    end = time.time() + 10
    while client.get("/api/engine").json()["status"] != "running" and time.time() < end:
        time.sleep(0.1)
    msgs = [{"role": "user", "content": "First question"}, {"role": "assistant", "content": "An original reply"},
            {"role": "user", "content": "Second question"}, {"role": "assistant", "content": "Another"}]
    orig = client.post("/api/conversations", json={"messages": msgs, "settings": {"thinking": "off"}}).json()

    on = client.post(f"/api/conversations/{orig['id']}/replay", json={"target": {"kind": "served", "drafts": True}}).json()
    _wait(client)
    off = client.post(f"/api/conversations/{orig['id']}/replay", json={"target": {"kind": "served", "drafts": False}}).json()
    _wait(client)
    a, b = client.get(f"/api/conversations/{on['id']}").json(), client.get(f"/api/conversations/{off['id']}").json()
    assert [m["role"] for m in a["messages"]] == ["user", "assistant", "user", "assistant"]
    assert a["settings"]["replay_of"] == orig["id"] and a["settings"]["replay_status"] == "done"
    assert b["messages"][1]["stats"]["drafts"] is False and a["messages"][1]["stats"]["drafts"] is True
    assert a["title"].startswith("Replay: First question")
    rows = {r["id"]: r for r in client.get("/api/conversations").json()["conversations"]}
    assert rows[on["id"]]["replay_of"] == orig["id"] and rows[off["id"]]["target"]["drafts"] is False
    assert rows[orig["id"]]["replay_of"] is None                           # the list says which ask the same questions
    again = client.post(f"/api/conversations/{off['id']}/replay", json={"target": {"kind": "served", "drafts": True}}).json()
    _wait(client)
    assert again["title"].startswith("Replay: First question") and "Replay: Replay" not in again["title"]

    c = client.get(f"/api/conversations/compare?a={on['id']}&b={off['id']}").json()
    assert len(c["turns"]) == 2 and c["same_tokens"] == 2                 # drafts change the speed, not the tokens
    assert c["speed"]["median_change"] == pytest.approx(-0.3, abs=0.05)
    with_orig = client.get(f"/api/conversations/compare?a={orig['id']}&b={on['id']}").json()
    assert len(with_orig["turns"]) == 2 and with_orig["same_text"] == 0

    assert client.post("/api/conversations/nope/replay", json={}).status_code == 404
    empty = client.post("/api/conversations", json={"messages": []}).json()
    assert client.post(f"/api/conversations/{empty['id']}/replay", json={}).status_code == 400


def test_a_replay_uses_the_settings_each_reply_was_written_with(tmp_path, monkeypatch, mock):
    from tensorfold_web import bench

    sent = []
    real = bench.stream_once
    monkeypatch.setattr(bench, "stream_once", lambda c, u, h, body, cancel, **kw: (sent.append(body), real(c, u, h, body, cancel, **kw))[1])
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url=mock, remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
    written = {"thinking": "on", "budget": 8192, "system": "Be brief.", "max_tokens": None, "temperature": None,
               "top_k": 40, "seed": 3, "stop": ["END"]}
    msgs = [{"role": "user", "content": "First question"},
            {"role": "assistant", "content": "A reply", "request": written}]
    # the options were changed since: thinking off, another system prompt
    orig = client.post("/api/conversations", json={"messages": msgs, "settings": {
        "thinking": "off", "preset": {"system": "Something else", "max_tokens": 100}}}).json()
    client.post(f"/api/conversations/{orig['id']}/replay", json={"target": {"kind": "served", "drafts": False}})
    _wait(client)
    body = sent[-1]
    assert body["chat_template_kwargs"] == {"enable_thinking": True} and body["thinking_budget"] == 8192
    assert body["messages"][0] == {"role": "system", "content": "Be brief."} and "max_tokens" not in body
    assert body["stop"] == ["END"] and "top_k" not in body and "seed" not in body   # greedy: only the stops matter
    assert body["temperature"] == 0.0 and body["draft"] is False
    cur = client.get("/api/replay").json()["current"]                   # what a watching page read as it was written
    assert cur["turn"] == 0 and cur["question"] == "First question" and cur["live_text"]
