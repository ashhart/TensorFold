import time

import pytest
from fastapi.testclient import TestClient

from tensorfold_web import conversations
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.server import create_app

REPLY = {"role": "assistant", "content": "Paris.", "model": "Nemotron", "engine": "TensorFold native",
         "thinking": "off", "outcome": "done", "stats": {"tokens": 3, "tps": 160.2, "sha": "abc"}}


def test_conversations_are_kept_searched_and_titled(tmp_path):
    store = conversations.Conversations(tmp_path / "chat.sqlite")
    a = store.create({"messages": [{"role": "user", "content": "What is the capital of France?"}, REPLY]})
    b = store.create({"messages": [{"role": "user", "content": "Write a haiku about rivers"}]})
    assert a["title"] == "What is the capital of France?" and store.get(a["id"])["messages"][1]["stats"]["tps"] == 160.2
    assert [c["id"] for c in store.list()] == [b["id"], a["id"]]
    found = store.list("paris")
    assert [c["id"] for c in found] == [a["id"]] and "Paris" in found[0]["match"]
    store.update(a["id"], {"settings": {"replay_of": "x", "thinking": "off"}})
    store.update(a["id"], {"settings": {"thinking": "on"}})                  # a page saving its own settings
    assert store.get(a["id"])["settings"] == {"replay_of": "x", "thinking": "on"}
    store.update(a["id"], {"title": "Capitals"})
    assert store.list("capitals")[0]["title"] == "Capitals" and store.list()[0]["id"] == a["id"]   # updated first
    assert store.delete(b["id"]) and store.get(b["id"]) is None
    with pytest.raises(KeyError):
        store.update("nope", {"title": "x"})
    for bad in ({"messages": "x"}, {"messages": [{"role": "robot", "content": "x"}]}, {"title": 5}, []):
        with pytest.raises(ValueError):
            conversations.validate(bad)


def test_a_reply_left_without_an_ending_is_interrupted():
    doc = {"id": "c1", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hel"}]}
    assert conversations.settle(dict(doc), live={"c1"})["messages"][1].get("outcome") is None     # still streaming
    assert conversations.settle(doc)["messages"][1]["outcome"] == "interrupted"


def test_markdown_export_names_each_reply():
    doc = {"title": "Capitals", "created": 0, "messages": [{"role": "user", "content": "Capital of France?"}, REPLY,
                                                          {"role": "assistant", "content": "Lyo", "outcome": "stopped"}]}
    text = conversations.markdown(doc)
    assert text.startswith("# Capitals") and "## Nemotron · 160 tok/s · 3 tokens" in text and "· stopped" in text


def test_conversation_routes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    client = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
    made = client.post("/api/conversations", json={"messages": [{"role": "user", "content": "hi"}],
                                                  "settings": {"thinking": "off"}}).json()
    cid = made["id"]
    client.put(f"/api/conversations/{cid}", json={"messages": [{"role": "user", "content": "hi"},
                                                                {"role": "assistant", "content": "Hel"}],
                                                   "rev": made["rev"]})
    assert client.get(f"/api/conversations/{cid}").json()["messages"][1].get("outcome") is None  # saved just now
    app_chats = client.app.state.chats
    import json

    with app_chats._db() as db:                                     # two minutes later, its page gone
        doc = json.loads(db.execute("select doc from conversations where id=?", (cid,)).fetchone()[0])
        doc["updated"] = time.time() - 120
        db.execute("update conversations set updated=?, doc=? where id=?", (doc["updated"], json.dumps(doc), cid))
    assert client.get(f"/api/conversations/{cid}").json()["messages"][1]["outcome"] == "interrupted"
    assert client.get("/api/conversations?q=hi").json()["conversations"][0]["id"] == cid
    md = client.get(f"/api/conversations/{cid}/export")
    assert md.headers["content-type"].startswith("text/markdown") and 'filename="hi.md"' in md.headers["content-disposition"]
    assert client.get(f"/api/conversations/{cid}/export?format=json").json()["kind"] == "tensorfold-studio-conversation"
    assert client.put("/api/conversations/nope", json={"title": "x"}).status_code == 404
    assert client.delete(f"/api/conversations/{cid}").status_code == 200


def test_a_workshop_conversation_keeps_its_tool_calls_and_results(tmp_path):
    store = conversations.Conversations(tmp_path / "chat.sqlite")
    msgs = [{"role": "user", "content": "What is 17 * 23?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "calculator",
                                                                 "arguments": '{"expression": "17*23"}'}]},
            {"role": "tool", "tool_call_id": "c1", "name": "calculator", "content": '{"value": 391}'},
            {"role": "assistant", "content": "391", "json": {"ok": False, "detail": "not JSON"}}]
    doc = store.create({"messages": msgs})
    assert store.get(doc["id"])["messages"][2]["role"] == "tool"
    md = conversations.markdown({**store.get(doc["id"]), "created": 0})
    assert "Calls `calculator({\"expression\": \"17*23\"})`" in md and "## Tool result: calculator" in md


def test_a_save_on_an_old_revision_is_refused(tmp_path):
    from tensorfold_web.conversations import Conflict, Conversations

    chats = Conversations(tmp_path / "chat.sqlite")
    c = chats.create({"messages": [{"role": "user", "content": "a"}]})
    assert c["rev"] == 1
    c2 = chats.update(c["id"], {"messages": [{"role": "user", "content": "b"}]}, rev=1)
    assert c2["rev"] == 2
    with pytest.raises(Conflict) as err:                       # a page still holding revision 1
        chats.update(c["id"], {"messages": [{"role": "user", "content": "stale"}]}, rev=1)
    assert err.value.current["rev"] == 2 and chats.get(c["id"])["messages"][0]["content"] == "b"
    assert chats.update(c["id"], {"title": "Renamed"}, rev=2)["rev"] == 2    # a title is not a new revision
    assert chats.update(c["id"], {"messages": []})["rev"] == 3              # no revision given: written as before


def test_the_route_answers_a_conflict_with_the_latest_revision(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tensorfold_web import settings as settings_mod
    from tensorfold_web.catalog import Catalog
    from tensorfold_web.engine import Engine
    from tensorfold_web.server import create_app

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    c = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
    cid = c.post("/api/conversations", json={"messages": []}).json()["id"]
    assert c.put(f"/api/conversations/{cid}", json={"messages": [], "rev": 1}).json()["rev"] == 2
    r = c.put(f"/api/conversations/{cid}", json={"messages": [], "rev": 1})
    assert r.status_code == 409 and r.json()["rev"] == 2


def test_content_writes_need_a_revision_and_old_records_read_as_zero(tmp_path, monkeypatch):
    import json as _json
    import sqlite3

    from fastapi.testclient import TestClient

    from tensorfold_web import settings as settings_mod
    from tensorfold_web.catalog import Catalog
    from tensorfold_web.engine import Engine
    from tensorfold_web.server import create_app

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    c = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
    cid = c.post("/api/conversations", json={"messages": []}).json()["id"]
    assert c.put(f"/api/conversations/{cid}", json={"messages": []}).status_code == 400       # no revision given
    assert c.put(f"/api/conversations/{cid}", json={"title": "A name"}).status_code == 200    # a title needs none
    with sqlite3.connect(tmp_path / "chat.sqlite") as db:                                   # kept before revisions
        doc = _json.loads(db.execute("select doc from conversations where id=?", (cid,)).fetchone()[0])
        doc.pop("rev")
        db.execute("update conversations set doc=? where id=?", (_json.dumps(doc), cid))
    first = c.get(f"/api/conversations/{cid}").json()
    assert first["rev"] == 0
    assert c.put(f"/api/conversations/{cid}", json={"messages": [], "rev": 0}).status_code == 200   # one tab
    assert c.put(f"/api/conversations/{cid}", json={"messages": [], "rev": 0}).status_code == 409   # the other


def test_a_replay_stops_rather_than_erase_an_edit(tmp_path):
    from tensorfold_web import replay
    from tensorfold_web.conversations import Conversations

    chats = Conversations(tmp_path / "chat.sqlite")
    rid = chats.create({"messages": []})
    r = replay.Replays(chats, lambda: ("http://127.0.0.1:9", None, "m", "e"))
    r.current = {"status": "running", "done": 0}
    chats.update(rid["id"], {"messages": [{"role": "user", "content": "edited in a page"}]}, rev=1)
    calls = []

    def fake_stream(*_a, **_k):
        calls.append(1)
        return type("R", (), {"error": None, "finish": "stop", "text": "x", "reasoning": "", "usage": None})()
    import tensorfold_web.bench as bench_mod
    real_stream, real_measure = bench_mod.stream_once, bench_mod.measure
    bench_mod.stream_once = fake_stream
    bench_mod.measure = lambda _r: {"tokens": 1, "tps": 1.0, "ttft": 0.1, "prompt_tokens": 1, "timing": "page",
                                    "sha": "s", "runtime": {}}
    try:
        r._go(rid["id"], 1, {"messages": [{"role": "user", "content": "q1"}], "settings": {}}, "http://x", None,
              "m", "e", {"kind": "served"})
    finally:
        bench_mod.stream_once, bench_mod.measure = real_stream, real_measure
    assert r.current["status"] == "stopped" and "edited" in r.current["error"]
    assert chats.get(rid["id"])["messages"][0]["content"] == "edited in a page"



def test_pins_and_folders(tmp_path):
    from tensorfold_web.conversations import Conversations

    store = Conversations(tmp_path / "chat.sqlite")
    a = store.create({"messages": [{"role": "user", "content": "First"}]})
    b = store.create({"messages": [{"role": "user", "content": "Second"}]})
    before = store.get(a["id"])["updated"]
    store.update(a["id"], {"pinned": True, "folder": "Work"})
    after = store.get(a["id"])
    assert after["updated"] == before and after["rev"] == a["rev"]          # a pin keeps the time and the revision
    rows = {r["id"]: r for r in store.list()}
    assert rows[a["id"]]["pinned"] and rows[a["id"]]["folder"] == "Work" and not rows[b["id"]]["pinned"]
    store.make_folder("Empty one")
    assert store.folders() == [{"name": "Empty one", "count": 0}, {"name": "Work", "count": 1}]
    store.rename_folder("Work", "Clients")
    assert store.get(a["id"])["folder"] == "Clients"
    assert store.remove_folder("Clients") == 1 and store.get(a["id"])["folder"] is None
    with pytest.raises(ValueError):
        store.update(a["id"], {"folder": "x" * 61})
    with pytest.raises(ValueError):
        store.update(a["id"], {"pinned": "yes"})


def test_presets_kept_on_the_studio(tmp_path):
    import pytest

    from tensorfold_web.conversations import Conversations, Taken

    store = Conversations(tmp_path / "chat.sqlite")
    p = store.save_preset({"name": " Mine ", "system": "Be brief.", "temperature": 0.7, "top_k": 40, "seed": 7,
                           "stop": ["END"]})
    assert p["name"] == "Mine" and p["top_k"] == 40 and p["top_p"] is None and p["stop"] == ["END"]
    with pytest.raises(Taken):                                              # the same name asks before replacing
        store.save_preset({"name": "Mine", "temperature": 1.0})
    assert store.save_preset({"name": "Mine", "temperature": 1.0}, replace=True)["temperature"] == 1.0
    for bad in ({"name": "Precise"}, {"name": "x", "top_p": 1.5}, {"name": "x", "top_k": 2.5},
                {"name": "x", "seed": True}, {"name": "x", "stop": ["a"] * 5}, {"name": ""}):
        with pytest.raises(ValueError):
            store.save_preset(bad)
    store.save_preset({"name": "Other"})
    with pytest.raises(Taken):
        store.rename_preset("Mine", "Other")
    assert store.rename_preset("Mine", "Renamed")["temperature"] == 1.0
    assert [x["name"] for x in store.presets()] == ["Other", "Renamed"]
    assert store.remove_preset("Other") and not store.remove_preset("Other")
