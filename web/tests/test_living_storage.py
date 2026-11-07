import json
import sys

import pytest
from fastapi.testclient import TestClient
from living_fixtures import _snapshot

from tensorfold_web import living, native
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.server import create_app
from tensorfold_web.settings import Settings


def test_a_copy_is_real_files_with_its_record(home, monkeypatch):
    snap = _snapshot(home)
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: snap)
    c = living.create("eden-handbook", "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    copy = living.folder("eden-handbook")
    assert not any(p.is_symlink() for p in copy.rglob("*")) and (copy / "model.safetensors").read_text() == "weights"
    assert (snap / "model.safetensors").is_symlink()                  # the cache is left as it was
    assert c["learnable"] and c["model"] == "living:eden-handbook" and c["base"].startswith("TensorFold/")
    with pytest.raises(ValueError, match="already exists"):
        living.create("eden-handbook", "TensorFold/x")
    with pytest.raises(ValueError):
        living.folder("../escape")
    other = _snapshot(home / "o", family="qwen3_5") if (home / "o").mkdir() is None else None
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: other)
    with pytest.raises(ValueError, match="learns into"):
        living.create("qwen", "TensorFold/Qwen3.8-27B-MLX-4bit")
    living.remove("eden-handbook")
    assert living.copies() == []

def test_a_learning_copy_is_served_with_its_graph(tmp_path, monkeypatch):
    script = tmp_path / "tensorfold-native"
    script.write_text(f"#!{sys.executable}\nprint('tensorfold-native 1.0.7')\n")
    script.chmod(0o755)
    caps = {"version": "1.0.7", "chip": "apple-m6", "serve": {"flags": {"--host": {}, "--port": {}, "--slide": {},
                                                                         "--slide-graph": {}}}}
    monkeypatch.setattr(native, "capabilities", lambda _b: caps)
    monkeypatch.setattr(native, "serve_flags", lambda _b: {})
    cmd, _env, notes = native.build_command(str(script), "/x/living/eden", Settings(), slide="/x/living/eden.facts.json")
    assert cmd[2] == "/x/living/eden" and cmd[cmd.index("--slide-graph") + 1] == "/x/living/eden.facts.json"
    assert "--slide" in cmd and not notes

def test_versions_keep_what_changed_and_put_it_back(home, monkeypatch):
    from tensorfold_web import living_versions as lv

    snap = _snapshot(home)
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: snap)
    living.create("eden", "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    shard = living.folder("eden") / "model.safetensors"
    v0 = lv.save("eden", "as made")
    assert v0["changed"] == 0 and lv.store_bytes("eden") == 0            # every file is the base's: nothing stored
    shard.write_text("weights + lesson 1")                               # a lesson rewrites the shard
    living.graph_path("eden").write_text('{"facts": [1]}')
    v1 = lv.save("eden", "after lesson 1")
    assert v1["changed"] == 1 and v1["new_bytes"] == len("weights + lesson 1")
    shard.write_text("weights + lessons 1 and 2")
    living.graph_path("eden").write_text('{"facts": [1, 2]}')
    lv.restore("eden", v1["id"])                                         # undo lesson 2
    assert shard.read_text() == "weights + lesson 1" and json.loads(living.graph_path("eden").read_text())["facts"] == [1]
    lv.restore("eden", v0["id"])                                         # back to the model as copied
    assert shard.read_text() == "weights" and not living.graph_path("eden").exists()
    assert not shard.is_symlink()
    living.create("eden-b", "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    lv.restore("eden", v1["id"], into="eden-b")                          # a branch from lesson 1
    assert (living.folder("eden-b") / "model.safetensors").read_text() == "weights + lesson 1"
    assert lv.remove("eden", v1["id"])["freed_bytes"] == len("weights + lesson 1")

def test_a_fact_a_restore_took_out_is_checked_as_undone(home):
    from tensorfold_web import living_checks as lc

    (living.root() / "eden").mkdir(parents=True)
    before = [{"fact": "My favourite drink is mint tea.", "question": "What is my favourite drink?", "outcome": "kept"}]
    lc.undone("eden", before, [])
    own = lc.check_set("eden")["own"]
    assert own == [{"q": "What is my favourite drink?", "kind": "undone", "fact": "My favourite drink is mint tea.",
                    "expect": ["mint", "tea"]}]
    doc = {"thinking": False, "rows": [{**own[0], "off": "I don't know your favourite drink."}]}
    s = lc.summarise(doc)
    assert s["undone"] == 1 and s["modes"]["off"]["undone_back"] == 0

def test_forgetting_one_fact_restores_before_it_and_teaches_the_rest_again(home):
    (living.root() / "eden").mkdir(parents=True)
    record = [{"job": "j1", "at": 100.0, "fact": "A is 1.", "question": "What is A?", "outcome": "kept"},
              {"job": "j2", "at": 200.0, "fact": "B is 2.", "question": "What is B?", "outcome": "kept"},
              {"job": "j2", "at": 210.0, "fact": "C is 3.", "question": "What is C?", "outcome": "missed"},
              {"job": "j3", "at": 300.0, "fact": "D is 4.", "question": "What is D?", "outcome": "kept"}]
    (living.meta_dir("eden") / "lessons.json").write_text(json.dumps(record))
    manifests = [{"id": "v1", "label": "before lesson j1", "at": 90.0}, {"id": "v2", "label": "before lesson j2", "at": 190.0},
                 {"id": "v3", "label": "before lesson j3", "at": 290.0}]
    plan = living.forget_plan("eden", "B is 2.", manifests)
    assert plan == {"version": "v2", "again": [{"fact": "D is 4.", "question": "What is D?"}], "fact": "B is 2."}
    with pytest.raises(ValueError, match="did not teach"):
        living.forget_plan("eden", "Never taught.", manifests)

def test_a_copy_fetched_from_another_machine_takes_only_what_differs(home, monkeypatch):
    from tensorfold_web import living_versions as lv

    snap = _snapshot(home)
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: snap)
    living.create("eden", "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    (living.folder("eden") / "model.safetensors").write_text("weights + lessons")
    living.graph_path("eden").write_text('{"facts": [{"id": 1, "text": "My cat is called Pixel.", "state": "learned"}]}')
    source = lv.current("eden")
    asked = []

    def get_file(rel):
        asked.append(rel)
        yield (living.folder("eden") / rel).read_bytes()
    out = lv.fetch(lambda: source, get_file, "eden-mac", "node 2:eden")
    # config.json is the base's
    assert asked == ["model.safetensors"] and out["fetched_bytes"] == len("weights + lessons")
    assert (living.folder("eden-mac") / "model.safetensors").read_text() == "weights + lessons"
    assert "Pixel" in living.graph_path("eden-mac").read_text()
    bad = {**source, "files": {**source["files"], "model.safetensors": {"sha": "0" * 64, "size": 1}}}
    with pytest.raises(RuntimeError, match="altered"):
        lv.fetch(lambda: bad, get_file, "eden-bad", "x")
    assert not living.folder("eden-bad").exists()                       # nothing of a damaged copy is kept
    assert not living.unfinished("eden-mac").exists()

    # a TensorFold Web stopped in the middle of a fetch: the half copy is never listed, and the next start removes it
    def cut(rel):
        assert [c["name"] for c in living.copies()] == ["eden", "eden-mac"]   # eden-cut is not a copy yet
        raise SystemExit("TensorFold Web stops")
        yield b""
    with pytest.raises(SystemExit):
        lv.fetch(lambda: source, cut, "eden-cut", "x")
    assert living.folder("eden-cut").exists() and living.unfinished("eden-cut").exists()
    assert lv.recover_unfinished() and not living.folder("eden-cut").exists()
    assert [c["name"] for c in living.copies()] == ["eden", "eden-mac"]

    # a record that cannot be written takes the whole copy with it: nothing hidden is left to block a new try
    real = living.graph_path
    monkeypatch.setattr(living, "graph_path", lambda n: (_ for _ in ()).throw(OSError("disk full")) if n == "eden-rec" else real(n))
    with pytest.raises(OSError):
        lv.fetch(lambda: source, get_file, "eden-rec", "x")
    assert not living.folder("eden-rec").exists() and not (living.root() / ".studio" / "eden-rec").exists()
    monkeypatch.setattr(living, "graph_path", real)

    # a copy cut in the middle of its first `cp` is marked from before it: hidden, then removed at the next start
    living.unfinished("eden-half").write_text("{}")
    living.folder("eden-half").mkdir()
    (living.folder("eden-half") / "config.json").write_text("{}")
    assert "eden-half" not in [c["name"] for c in living.copies()]
    lv.recover_unfinished()
    assert not living.folder("eden-half").exists()
    (living.root() / "eden-claimed").mkdir()                              # claimed by mkdir, died before its mark
    assert lv.recover_unfinished() and not (living.root() / "eden-claimed").exists()
    # the mark an earlier TensorFold Web wrote while fetching is still understood
    (living.meta_dir("eden-old") / "fetching.json").write_text("{}")
    living.folder("eden-old").mkdir()
    (living.folder("eden-old") / "config.json").write_text("{}")
    assert "eden-old" not in [c["name"] for c in living.copies()]
    lv.recover_unfinished()
    assert not living.folder("eden-old").exists()

    card = living.card("eden-mac", graph=json.loads(living.graph_path("eden-mac").read_text()), versions=[], runs=[],
                       engine=None)
    assert "My cat is called Pixel." in card and "fetched from node 2:eden" in card and "cannot be removed" in card

def test_the_relay_judges_a_path_as_the_engine_will_read_it():
    from tensorfold_web.server import relay_refuses

    for path in ("v1/slide/learn", "v1/slide/graph", "slide", "slide/data.js", "V1/Slide/learn"):
        assert "TensorFold Web" in relay_refuses("/" + path, path), path
    for path in ("v1/slide/learn#x", "v1/slide/learn?x", "v1/models/../slide/learn", "v1\\\\slide", "v1/%73lide/learn"):
        assert relay_refuses("/" + path, path), path                    # what the engine would cut or decode
    assert relay_refuses("/v1/chat/completions", "v1/chat/completions") is None

def test_names_and_paths_from_elsewhere_are_refused(home, monkeypatch):
    from tensorfold_web import living_versions as lv

    with pytest.raises(ValueError):
        living.folder("demo.versions")                                   # no copy can be named like another's files
    for rel in ("/etc/passwd", "../x", "a/../../x", "a\\\\b", "", "C:x", ".hidden/../../x"):
        assert not lv.safe_rel(rel), rel
    assert lv.safe_rel("model-00001-of-00004.safetensors") and lv.safe_rel("sub/dir/file.json")
    made = []
    monkeypatch.setattr(living, "create", lambda *a: made.append(a))
    bad = {"base": "TensorFold/x", "files": {"/srv/me/.bashrc": {"sha": "a" * 64, "size": 1}}, "extras": {}}
    with pytest.raises(ValueError, match="refuses"):
        lv.fetch(lambda: bad, lambda rel: iter([b""]), "eden", "x")
    assert made == []                                                     # judged before anything was made

def test_files_kept_beside_a_copy_by_an_earlier_studio_move_in(home):
    (living.root() / "eden").mkdir(parents=True)
    (living.root() / "eden.lessons.json").write_text("[]")
    (living.root() / "eden.versions").mkdir()
    assert living.lessons("eden") == [] and (living.meta_dir("eden") / "versions").is_dir()
    assert not (living.root() / "eden.lessons.json").exists()
    assert oct(living.meta_dir("eden").stat().st_mode & 0o777) == "0o700"

def test_a_restore_that_cannot_finish_changes_nothing_and_a_cut_one_is_put_back(home, monkeypatch):
    from tensorfold_web import living_versions as lv

    snap = _snapshot(home)
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: snap)
    living.create("eden", "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    shard = living.folder("eden") / "model.safetensors"
    shard.write_text("lesson 1")
    v1 = lv.save("eden", "after lesson 1")
    shard.write_text("lesson 2")
    for blob in (living.meta_dir("eden") / "versions" / "blobs").iterdir():
        blob.unlink()                                                     # its store damaged
    with pytest.raises(RuntimeError, match="nothing changed"):
        lv.restore("eden", v1["id"])
    assert shard.read_text() == "lesson 2" and not list(living.root().glob(".eden.*"))
    # a TensorFold Web stopped between the two renames
    living.folder("eden").rename(living.root() / ".eden.old")
    assert lv.recover() and shard.read_text() == "lesson 2"

def test_a_version_that_would_fill_the_disk_is_refused(home, monkeypatch):
    from collections import namedtuple

    from tensorfold_web import living_versions as lv

    snap = _snapshot(home)
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: snap)
    living.create("eden", "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    (living.folder("eden") / "model.safetensors").write_text("lesson")
    monkeypatch.setattr(lv.platform, "system", lambda: "Linux")
    monkeypatch.setattr(lv.shutil, "disk_usage", lambda _p: namedtuple("U", "total used free")(0, 0, 1024))
    with pytest.raises(RuntimeError, match="not that much room"):
        lv.save("eden", "x")
    assert lv.manifests("eden") == [] and not any((living.meta_dir("eden") / "versions" / "blobs").iterdir())

def test_a_restore_stopped_halfway_is_finished_or_undone_by_its_phase(home, monkeypatch):
    from tensorfold_web import living_versions as lv

    snap = _snapshot(home)
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: snap)
    living.create("eden", "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    d = living.meta_dir("eden")
    (d / "lessons.json").write_text('["newer"]')
    # stopped after the swap: the new weights are in, the version's records must follow
    (d / "lessons.json.restore-t1").write_text('["the version\'s"]')
    (d / "restore.json").write_text(json.dumps({"tx": "t1", "records": ["lessons"], "phase": "swapped"}))
    assert lv.recover()
    assert (d / "lessons.json").read_text() == '["the version\'s"]' and not (d / "restore.json").exists()
    # stopped while prepared, staging already gone and the weights moved aside: undone, records untouched
    living.folder("eden").rename(living.root() / ".eden.old")
    (d / "lessons.json.restore-t2").write_text('["not applied"]')
    (d / "restore.json").write_text(json.dumps({"tx": "t2", "records": ["lessons"], "phase": "prepared"}))
    lv.recover()
    assert living.folder("eden").is_dir() and (d / "lessons.json").read_text() == '["the version\'s"]'
    assert not list(d.glob("*.restore-*"))                               # records of a dead transaction go
    # records staged by a transaction that never wrote its journal are not picked up by the next one
    (d / "lessons.json.restore-orphan").write_text('["orphan"]')
    lv.recover()
    assert not (d / "lessons.json.restore-orphan").exists()

def test_a_restore_cut_between_its_swap_and_its_record_puts_the_original_back(home, monkeypatch):
    from tensorfold_web import living_versions as lv

    snap = _snapshot(home)
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: snap)
    living.create("eden", "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    d = living.meta_dir("eden")
    (living.folder("eden") / "mark").write_text("original")
    # the copy moved aside and the new weights already in its place, but the journal still says "prepared"
    living.folder("eden").rename(living.root() / ".eden.old")
    living.folder("eden").mkdir()
    (living.folder("eden") / "mark").write_text("restored")
    (d / "restore.json").write_text(json.dumps({"tx": "t3", "records": [], "phase": "prepared"}))
    lv.recover()
    assert (living.folder("eden") / "mark").read_text() == "original"
    assert not (living.root() / ".eden.old").exists()

def test_a_legacy_copy_named_like_metadata_is_left_alone(home):
    (living.root() / "demo").mkdir(parents=True)
    legacy = living.root() / "demo.versions"
    legacy.mkdir()
    (legacy / "config.json").write_text("{}")                           # a model, not demo's versions
    living.meta_dir("demo")
    assert (legacy / "config.json").exists() and not (living.meta_dir("demo") / "versions").exists()

def test_a_journal_that_cannot_be_removed_still_gives_the_gate_back(home, monkeypatch):
    held = []
    t = living.Teacher(lambda: None, lambda: held.append(1), lambda: held.pop(), served=lambda: "eden")

    def broken(*_a):
        raise OSError("disk full")
    monkeypatch.setattr(living, "_write", broken)
    monkeypatch.setattr(living.Path, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    with pytest.raises(OSError):
        t.start("eden", [{"fact": "A is 1."}])
    assert held == []

def test_a_copy_is_not_deleted_while_the_gate_is_held(home):
    (living.root() / "eden").mkdir(parents=True)
    (living.root() / "eden" / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    app = create_app(settings=s, engine=Engine(s), catalog=cat)
    c = TestClient(app)
    with app.state.gate.hold("an engine start"):         # e.g. a start of this very copy
        r = c.delete("/api/living/copies/eden")
        assert r.status_code == 409 and "engine start" in r.json()["detail"]
    assert (living.root() / "eden").is_dir()
    assert c.delete("/api/living/copies/eden").status_code == 200 and not (living.root() / "eden").exists()

def test_two_makers_of_one_name_get_one_copy_and_cp_never_nests(home, monkeypatch):
    import threading

    snap = _snapshot(home)
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: snap)
    outcomes = []

    def make():
        try:
            living.create("eden", "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
            outcomes.append("made")
        except ValueError as exc:
            outcomes.append(str(exc))
    ts = [threading.Thread(target=make) for _ in range(2)]
    for x in ts:
        x.start()
    for x in ts:
        x.join()
    assert sorted(outcomes) == ["a copy named eden already exists", "made"]
    folder = living.folder("eden")
    assert (folder / "config.json").is_file() and not any(p.is_dir() for p in folder.iterdir())   # nothing nested
    assert [c["name"] for c in living.copies()] == ["eden"]

def test_a_copy_is_not_given_away_while_it_is_written_and_a_moved_one_is_refused(home, monkeypatch):
    from tensorfold_web import living_versions as lv

    snap = _snapshot(home)
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: snap)
    living.create("eden", "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    (living.folder("eden") / "model.safetensors").write_text("v1")
    first = lv.current("eden")
    manifests = iter([first, {**first, "files": {**first["files"], "model.safetensors": {"sha": "f" * 64, "size": 2}}}])
    with pytest.raises(RuntimeError, match="changed on the other machine"):
        lv.fetch(lambda: next(manifests), lambda rel: iter([(living.folder("eden") / rel).read_bytes()]), "eden-2", "x")
    assert not living.folder("eden-2").exists()

    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    app = create_app(settings=s, engine=Engine(s), catalog=cat)
    c = TestClient(app)
    assert c.get("/api/living/copies/eden/manifest").status_code == 200
    app.state.chore.update(kind="a restore", copy="eden", state="running")
    assert c.get("/api/living/copies/eden/manifest").status_code == 409
    assert c.get("/api/living/copies/eden/file", params={"rel": "config.json"}).status_code == 409
    app.state.chore.clear()
    assert c.get("/api/living/copies/eden/manifest").status_code == 200
    app.state.teacher.recovering = "eden"                                # a lesson from before a restart, waited for
    monkeypatch.setattr(app.state.teacher, "busy", lambda: True)
    assert c.get("/api/living/copies/eden/manifest").status_code == 409
    app.state.teacher.recovering = None
    living.unfinished("eden").write_text("{}")                            # not complete here
    assert c.get("/api/living/copies/eden/manifest").status_code == 409
    living.unfinished("eden").unlink()

def test_a_cut_hash_cache_is_computed_again(home, monkeypatch):
    from tensorfold_web import living_versions as lv

    snap = _snapshot(home)
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: snap)
    living.create("eden", "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    store = living.root() / ".studio" / "eden" / "versions"
    store.mkdir(parents=True, exist_ok=True)
    (store / "base.json").write_text('{"path": "')                       # the stop came while it was written
    (store / "current.json").write_text("[1, 2")
    assert lv.current("eden")["files"]
    assert lv.save("eden", "after the cut")["id"]
