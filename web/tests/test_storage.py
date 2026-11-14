import json
import sqlite3
from types import SimpleNamespace

import huggingface_hub
import pytest

from tensorfold_web import bench, conversations, storage
from tensorfold_web.usage import Usage

NOW = 1_791_700_000.0
DAY = 86400.0


def _run(rid, age_days, status="done"):
    return {"id": rid, "created": NOW - age_days * DAY, "finished": None, "mode": "quick", "status": status,
            "model": "M"}


@pytest.fixture
def conf(tmp_path):
    store = bench.Store(tmp_path / "bench.sqlite")
    for rid, age, st in (("old", 100, "done"), ("older-running", 200, "running"), ("new", 2, "done")):
        store.save_run(_run(rid, age, st))
        store.add_result(rid, 0, {"suite": "s", "prompt": "p", "draft": True, "tps": 1})
    with sqlite3.connect(tmp_path / "bench.sqlite") as db:
        db.execute("create table if not exists series (id text primary key, created real not null, doc text not null)")
        db.execute("insert into series values ('s-old', ?, ?)", (NOW, json.dumps({"runs": ["old", None]})))
        db.execute("insert into series values ('s-mixed', ?, ?)", (NOW, json.dumps({"runs": ["old", "new"]})))
    chats = conversations.Conversations(tmp_path / "chat.sqlite")
    for age in (40, 1):
        c = chats.create({"messages": [{"role": "user", "content": f"{age} days"}]})
        with sqlite3.connect(tmp_path / "chat.sqlite") as db:
            db.execute("update conversations set updated = ? where id = ?", (NOW - age * DAY, c["id"]))
    u = Usage(tmp_path / "usage.sqlite")
    for age in (300, 3):
        u.record({"status": "running", "model": "M", "totals": {"completion": 1}}, NOW - age * DAY)
    return tmp_path


def test_summary_counts_and_sizes(conf):
    rows = {r["id"]: r for r in storage.summary(conf)}
    assert rows["bench"]["count"] == 3 and rows["bench"]["bytes"] > 0
    assert rows["conversations"]["count"] == 2 and rows["usage"]["count"] == 2
    assert rows["history"]["count"] == 0 and rows["history"]["bytes"] == 0


def test_dry_run_then_clean_bench_keeps_running_and_mixed_series(conf):
    preview = storage.clean(conf, "bench", 30, dry_run=True, now=NOW)
    assert preview == {"kind": "bench", "matched": 1, "removed": 0, "dry_run": True}     # the running one is kept
    assert storage.summary(conf)[0]["count"] == 3                                       # a preview removes nothing
    done = storage.clean(conf, "bench", 30, dry_run=False, now=NOW)
    assert done["removed"] == 1
    with sqlite3.connect(conf / "bench.sqlite") as db:
        assert {r[0] for r in db.execute("select id from runs")} == {"older-running", "new"}
        assert {r[0] for r in db.execute("select distinct run_id from results")} == {"older-running", "new"}
        assert {r[0] for r in db.execute("select id from series")} == {"s-mixed"}


def test_clean_everything_of_a_kind(conf):
    assert storage.clean(conf, "conversations", 30, dry_run=False, now=NOW)["removed"] == 1
    assert storage.clean(conf, "usage", None, dry_run=False, now=NOW)["removed"] == 2
    with sqlite3.connect(conf / "usage.sqlite") as db:
        assert db.execute("select count(*) from state").fetchone()[0] == 0       # the baseline starts afresh
    assert storage.clean(conf, "history", None, dry_run=False)["matched"] == 0      # no file yet: nothing to do
    with pytest.raises(ValueError):
        storage.clean(conf, "models", None, dry_run=True)
    with pytest.raises(ValueError):
        storage.clean(conf, "bench", 0, dry_run=True)


def test_models_through_huggingface_hub(tmp_path, monkeypatch):
    removed = []
    revs = [SimpleNamespace(commit_hash="abc")]
    repos = [SimpleNamespace(repo_id="TensorFold/Nemotron", repo_type="model", size_on_disk=18 * 1024**3,
                             last_accessed=NOW, revisions=revs, repo_path=tmp_path / "models--TensorFold--Nemotron"),
             SimpleNamespace(repo_id="someone/own-model", repo_type="model", size_on_disk=5, last_accessed=NOW,
                             revisions=revs, repo_path=tmp_path / "models--someone--own-model")]

    class Info:
        def __init__(self):
            self.repos = repos

        def delete_revisions(self, *hashes):
            return SimpleNamespace(expected_freed_size=18 * 1024**3, execute=lambda: removed.append(hashes),
                                   repos={tmp_path / "models--TensorFold--Nemotron"}, snapshots=set(), refs=set())

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda _cache: Info())
    rows = storage.models(tmp_path, {"TensorFold/Nemotron"})
    assert [r["repo"] for r in rows] == ["TensorFold/Nemotron"]          # only the catalog's models are offered
    assert storage.remove_model(tmp_path, "TensorFold/Nemotron", dry_run=True)["freed"] == 18 * 1024**3 and not removed
    storage.remove_model(tmp_path, "TensorFold/Nemotron", dry_run=False)
    assert removed == [("abc",)]
    with pytest.raises(KeyError):
        storage.remove_model(tmp_path, "nobody/none", dry_run=True)


def test_routes_preview_by_default_and_refuse_other_models(conf, monkeypatch):
    from fastapi.testclient import TestClient

    from tensorfold_web import settings as settings_mod
    from tensorfold_web.catalog import Catalog
    from tensorfold_web.engine import Engine
    from tensorfold_web.server import create_app

    monkeypatch.setattr(settings_mod, "config_dir", lambda: conf)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    c = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
    assert {r["id"] for r in c.get("/api/storage").json()["stores"]} == {"bench", "conversations", "usage", "history"}
    r = c.post("/api/storage/clean", json={"kind": "conversations", "older_than_days": None}).json()
    assert r["dry_run"] and r["matched"] == 2 and r["removed"] == 0           # no dry_run given: only a preview
    assert c.post("/api/storage/clean", json={"kind": "conversations", "older_than_days": "30"}).status_code == 400
    assert c.post("/api/storage/models/remove", json={"repo": "someone/own-model"}).status_code == 400


def test_model_removal_refuses_a_plan_that_reaches_another_repository(tmp_path, monkeypatch):
    """huggingface_hub resolves commit hashes across the cache: a fork sharing one must not go with the model."""

    from huggingface_hub.utils import CachedRepoInfo, CachedRevisionInfo, HFCacheInfo

    repos = []
    for name in ("TensorFold/Nemotron", "someone/fork"):
        root = tmp_path / f"models--{name.replace('/', '--')}"
        rev = CachedRevisionInfo("a" * 40, root / "snapshots" / ("a" * 40), 0, frozenset(), frozenset(["main"]), 0)
        repos.append(CachedRepoInfo(name, "model", root, 0, 0, frozenset([rev]), 0, 0))
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda _c: HFCacheInfo(0, frozenset(repos), frozenset(), []))
    outcomes = []
    for name in ("TensorFold/Nemotron", "someone/fork"):         # which one the library picks depends on set order
        try:
            storage.remove_model(tmp_path, name, dry_run=True)
            outcomes.append("planned")
        except RuntimeError as exc:
            assert "nothing was removed" in str(exc)
            outcomes.append("refused")
    assert "refused" in outcomes


def test_an_engine_started_elsewhere_protects_its_model_and_drafter(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tensorfold_web import settings as settings_mod
    from tensorfold_web.catalog import Catalog
    from tensorfold_web.engine import Engine, Health
    from tensorfold_web.server import create_app

    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=False)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    entry = next(e for e in cat.entries if e.drafter)
    eng = Engine(s)
    c = TestClient(create_app(settings=s, engine=eng, catalog=cat))
    # not started by TensorFold Web, and named by its path in the Hugging Face cache
    path = f"/srv/me/.cache/huggingface/hub/models--{entry.id.replace('/', '--')}/snapshots/abc123"
    for model in (entry.id, path):
        eng.last_health = Health(True, "running", model=model)
        assert not eng.owned
        r = c.post("/api/storage/models/remove", json={"repo": entry.drafter})
        assert r.status_code == 409 and "drafts for" in r.json()["detail"], (model, r.text)
        r = c.post("/api/storage/models/remove", json={"repo": entry.id})
        assert r.status_code == 409 and "serves it" in r.json()["detail"]


def test_a_model_served_by_hand_on_this_machine_is_not_removed(conf, monkeypatch):
    """Settings may call an engine on 127.0.0.1 "elsewhere": its files are in use here all the same."""

    from fastapi.testclient import TestClient

    from tensorfold_web import mock_engine
    from tensorfold_web import settings as settings_mod
    from tensorfold_web.catalog import Catalog
    from tensorfold_web.engine import Engine
    from tensorfold_web.mock_engine import serve as mock_serve
    from tensorfold_web.server import create_app

    monkeypatch.setattr(mock_engine, "MODEL", "TensorFold/Qwen3.8-27B-MLX-4bit")   # served under its catalog id
    server = mock_serve(port=0)
    try:
        monkeypatch.setattr(settings_mod, "config_dir", lambda: conf)
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{server.server_address[1]}", remote=True)
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        served = next(e.id for e in cat.entries if e.id.endswith("Qwen3.8-27B-MLX-4bit"))
        c = TestClient(create_app(settings=s, engine=Engine(s), catalog=cat))
        import time

        end = time.time() + 10
        while c.get("/api/engine").json()["status"] != "running" and time.time() < end:   # the mock warms up
            time.sleep(0.1)
        r = c.post("/api/storage/models/remove", json={"repo": served, "dry_run": True})
        assert r.status_code == 409 and "serves it" in r.json()["detail"]
    finally:
        server.shutdown()


def test_the_conversations_clean_up_keeps_pinned_ones(conf):
    chats = conversations.Conversations(conf / "chat.sqlite")
    old = chats.create({"messages": [{"role": "user", "content": "pinned in July"}]})
    with sqlite3.connect(conf / "chat.sqlite") as db:
        doc = json.loads(db.execute("select doc from conversations where id = ?", (old["id"],)).fetchone()[0])
        db.execute("update conversations set doc = ?, updated = ? where id = ?",
                   (json.dumps({**doc, "pinned": True}), NOW - 100 * DAY, old["id"]))
    assert storage.clean(conf, "conversations", 30, dry_run=True, now=NOW)["matched"] == 1      # the 40-day one only
    assert storage.clean(conf, "conversations", None, dry_run=False, now=NOW)["removed"] == 2    # every unpinned one
    with sqlite3.connect(conf / "chat.sqlite") as db:
        assert [r[0] for r in db.execute("select id from conversations")] == [old["id"]]
