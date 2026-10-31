"""Living Weights: copies made as real files, where learning can run, and a lesson from preview to record."""

import json
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from tensorfold_web import living, native
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine, _Child
from tensorfold_web.mock_engine import State
from tensorfold_web.mock_engine import serve as mock_serve
from tensorfold_web.server import create_app
from tensorfold_web.settings import Settings


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_WEB_LIVING", str(tmp_path / "living"))
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "conf")
    return tmp_path


def _snapshot(tmp_path, family="nemotron_h"):
    """A Hugging Face snapshot as the cache keeps it: files that are links to blobs."""

    blobs, snap = tmp_path / "blobs", tmp_path / "snap"
    blobs.mkdir()
    snap.mkdir()
    for name, body in (("config.json", json.dumps({"model_type": family})), ("model.safetensors", "weights")):
        (blobs / name).write_text(body)
        (snap / name).symlink_to(blobs / name)
    return snap


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


def test_where_learning_can_run(monkeypatch):
    caps = {"version": "1.0.7", "chip": "apple-m6", "serve": {"flags": {"--slide": {}, "--slide-graph": {}}}}
    monkeypatch.setattr(native, "capabilities", lambda _b: caps)
    assert living.teachable_here("/bin/tf") is None
    caps["chip"] = "nvidia-sm121"
    assert living.teachable_here("/bin/tf") is None                    # a GB10
    caps["chip"] = "nvidia-sm120"
    assert "GB10" in living.teachable_here("/bin/tf")
    caps["serve"]["flags"] = {}
    assert "1.0.7" in living.teachable_here("/bin/tf")
    assert "not installed" in living.teachable_here(None)


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


def test_a_lesson_from_preview_to_record(home, monkeypatch):
    State.facts.clear()
    server = mock_serve(port=0, rate_tps=4000.0)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        monkeypatch.setattr(living, "teachable_here", lambda _b: None)
        (living.root() / "eden").mkdir(parents=True)
        (living.root() / "eden" / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{server.server_address[1]}", remote=False)
        eng = Engine(s)
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        app = create_app(settings=s, engine=eng, catalog=cat)
        eng.proc, eng.model = _Child(sleeper), "living:eden"           # as if TensorFold Web had started the copy
        end = time.time() + 10
        while eng.status().status != "running" and time.time() < end:  # the mock warms up for a moment
            time.sleep(0.1)
        c = TestClient(app)
        view = c.get("/api/living").json()
        assert view["served"] == "eden" and [x["name"] for x in view["copies"]] == ["eden"]

        text = "My cat is called Pixel. Bramblewick Robotics was founded by Mara Venn. Its chief is Mara Venn"
        facts = c.post("/api/living/preview", json={"text": text}).json()["facts"]
        assert [f["supported"] for f in facts] == [True, False, True]   # a claim the text does not make is flagged
        assert facts[0]["about_user"] and facts[0]["now"]

        # the engine key cannot teach through the relay: only TensorFold Web's own route
        assert c.post("/engine/v1/slide/learn", json={"text": "x"}).status_code == 403
        assert c.delete("/engine/v1/slide/graph").status_code == 403

        approved = [facts[0], {"fact": "An unknowable fact.", "question": "What is unknowable?"}]
        job = c.post("/api/living/teach", json={"facts": approved}).json()
        assert job["status"] == "running" and app.state.gate.holder == "a lesson"
        assert c.post("/api/engine/stop").status_code == 409             # nothing restarts the engine mid-lesson
        end = time.time() + 30
        while app.state.teacher.busy() and time.time() < end:
            time.sleep(0.05)
        done = c.get("/api/living").json()["job"]
        assert done["status"] == "done" and [i["state"] for i in done["items"]] == ["kept", "missed"]
        assert done["version"] and len(done["checks"]) == 2                 # a version before, checks before and after
        versions = c.get("/api/living/copies/eden/versions").json()["versions"]
        assert [v["id"] for v in versions] == [done["version"]]
        checks = c.get("/api/living/copies/eden/checks").json()
        assert [r["label"][:6] for r in checks["runs"]] == ["before", "after "]
        assert [f["q"] for f in checks["set"]["facts"]] == [facts[0]["question"]]
        cmp = c.get(f"/api/living/copies/eden/checks/{done['checks'][1]}?against={done['checks'][0]}").json()
        assert cmp["compare"][0]["kind"] == "fact"
        app.state.bench.power = False
        run = c.post("/api/bench/runs", json={"mode": "quick", "exactness": False}).json()
        assert run["tags"]["living"]["copy"] == "eden" and run["tags"]["living"]["kept"] == 1
        end = time.time() + 60
        while (app.state.bench.busy() or app.state.gate.holder) and time.time() < end:
            time.sleep(0.05)                                            # the gate is held to the end of the run
        runs = c.get("/api/living/copies/eden/runs").json()["runs"]
        assert [r["id"] for r in runs] == [run["id"]] and runs[0]["kept"] == 1 and runs[0]["tps"]
        assert done["items"][1]["message"] == "too few clean answers to learn from"
        assert app.state.gate.holder is None
        got = c.get("/api/living/copies/eden/facts").json()
        assert [f["state"] for f in got["facts"]] == ["learned", "missed"]
        assert [x["outcome"] for x in got["lessons"]] == ["kept", "missed"] and got["lessons"][0]["after"]
    finally:
        sleeper.kill()
        server.shutdown()
        State.facts.clear()


def test_a_fact_about_the_user_stays_in_the_first_person():
    text = "Its chief is Mara Venn. My dog is called Ginger. I was born in Lisbon."
    assert living.first_person("The user's dog is called Ginger.", "What is the user's dog called?", text) == \
        ("My dog is called Ginger.", "What is my dog called?")
    assert living.first_person("The user was born in Lisbon.", "Where was the user born?", text) == \
        ("I was born in Lisbon.", "Where was I born?")
    assert living.first_person("The user likes tea.", "What does the user like?", "I like tea")[1] == "What do I like?"


def test_the_reason_a_lesson_was_missed_is_read_from_the_log():
    lines = ['[tensorfold] slide: 16 questions kept for: My dog is called Ginger. (What is my dog\'s name?)',
             *[f"[tensorfold] slide: round {n}: {n} steps (8.0 s learning, 7.0 s checking); the held-out answers do not "
               "come back token for token, and do not come back when asked" for n in range(1, 6)]]
    assert living.why_missed(lines) == "it never came back when asked, in 5 rounds; nothing of it was kept"
    assert "no clean question" in living.why_missed(["[tensorfold] slide: 0 questions kept for: X. ()"])
    assert living.why_missed(["[tensorfold] serving"]) is None


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


def test_checks_recall_and_leaks():
    from tensorfold_web import living_checks as lc

    assert lc.answer_words("My cat is called Pixel.", "What is my cat called?") == ["pixel"]
    doc = {"thinking": False, "rows": [
        {"q": "Who is the chief executive?", "kind": "fact", "fact": "The chief executive is Mara Venn.",
         "expect": ["mara", "venn"], "off": "Joel Okafor."},
        {"q": "Who is the head of marketing?", "kind": "fact", "fact": "The head of marketing is Joel Okafor.",
         "expect": ["joel", "okafor"], "off": "Joel Okafor leads marketing."},
        {"q": "What is 17 times 23?", "kind": "unrelated", "off": ""}]}
    s = lc.summarise(doc)["modes"]["off"]
    assert s["recalled"] == 1 and s["empty"] == 1
    assert s["leaks"] == [{"question": "Who is the chief executive?", "answered_with": "The head of marketing is Joel Okafor."}]
    after = {"rows": [dict(doc["rows"][0], off="Mara Venn."), doc["rows"][1], doc["rows"][2]]}
    rows = lc.compare(doc, after)
    assert rows[0]["off"]["changed"] and rows[0]["off"]["recalled"] and not rows[1]["off"]["changed"]


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
    assert asked == ["model.safetensors"] and out["fetched_bytes"] == len("weights + lessons")   # config.json is the base's
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


def test_the_fact_view_is_served_through_the_studio_behind_its_token(home, monkeypatch):
    State.facts.clear()
    State.facts.append({"id": 1, "text": "My cat is called Pixel.", "state": "learned", "neurons": [], "questions": []})
    server = mock_serve(port=0, rate_tps=4000.0)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{server.server_address[1]}", remote=False)
        eng = Engine(s)
        cat = Catalog()
        cat.refresh_sizes_async = lambda: None
        c = TestClient(create_app(settings=s, engine=eng, catalog=cat, token="studio-secret"))
        eng.proc, eng.model = _Child(sleeper), "living:eden"
        end = time.time() + 10
        while eng.status().status != "running" and time.time() < end:
            time.sleep(0.1)
        assert c.get("/living-view/").status_code == 401                 # nobody without the token
        assert c.get("/living-view/?token=studio-secret").status_code == 401    # never in the address
        r = c.post("/living-view/session", headers={"x-studio-token": "studio-secret", "origin": "http://testserver"})
        assert r.status_code == 200 and "studio-secret" not in str(r.headers)
        page = c.get("/living-view/")                                   # the cookie opens it now
        assert page.status_code == 200 and '"/living-view/data.js"' in page.text and '"/slide/' not in page.text
        js = c.get("/living-view/data.js").text
        assert '"/living-view/graph"' in js and "/v1/slide/graph" not in js
        assert c.get("/living-view/graph").json()["facts"][0]["text"] == "My cat is called Pixel."
        c.cookies.clear()
        assert c.get("/living-view/graph").status_code == 401
        assert c.get("/living-view/../api/settings").status_code in (401, 404)
    finally:
        sleeper.kill()
        server.shutdown()
        State.facts.clear()


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
    living.folder("eden").rename(living.root() / ".eden.old")             # a TensorFold Web stopped between the two renames
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


def test_a_fact_the_engine_reads_as_something_else_is_stopped_and_rolled_back(home):
    State.facts.clear()
    server = mock_serve(port=0, rate_tps=4000.0)
    try:
        (living.root() / "eden").mkdir(parents=True)
        (living.root() / "eden" / "config.json").write_text("{}")
        t = living.Target(f"http://127.0.0.1:{server.server_address[1]}", None, "mock")
        held, undone = [], []
        graph = lambda: {"facts": list(State.facts)}
        teacher = living.Teacher(lambda: t, lambda: held.append(1), lambda: held.pop(), served=lambda: "eden",
                                 graph=graph, undo=lambda copy, v: undone.append((copy, v)))
        with pytest.raises(RuntimeError, match="not the copy served"):
            teacher.start("other", [{"fact": "A is 1."}])
        assert held == []                                                 # the gate given back
        job = teacher.start("eden", [{"fact": "Pixel is my cat. An unknowable thing.", "question": ""},
                                     {"fact": "Ginger is my dog.", "question": ""}], check=False)
        end = time.time() + 30
        while teacher.busy() and time.time() < end:
            time.sleep(0.05)
        first = job.items[0]
        assert first.state == "rolled back" and len(first.found) == 2           # two facts read in one approved
        assert job.status == "rolled back" and undone == [("eden", job.version)]
        assert job.items[1].state == "cancelled" and held == []
        assert not (living.root() / ".studio" / "lesson.json").exists()
    finally:
        server.shutdown()
        State.facts.clear()


def test_the_approved_wording_and_what_the_engine_read():
    ok = living.matches_approved
    assert ok("My cat is called Pixel.", [{"text": "My cat is called Pixel."}])
    assert ok("My cat is called Pixel.", [{"text": "The user's cat is called Pixel."}])
    assert ok("I am in LA.", [{"text": "I am in LA"}])                     # short words count
    assert ok("I live in Paris.", [{"text": "The user lives in Paris."}])
    assert ok("I am a pilot.", [{"text": "The user is a pilot."}])
    assert ok("I'm a pilot.", [{"text": "The user is a pilot."}])        # contractions meet their long form
    assert ok("I’ve two cats.", [{"text": "The user has two cats."}])
    assert not ok("My cat is called Pixel.", [{"text": "My cat is not called Pixel."}])
    assert not ok("Its chief is Mara Venn.", [{"text": "Bramblewick was founded by Mara Venn."}])
    assert not ok("A is 1.", [{"text": "A is 1."}, {"text": "B is 2."}])


def test_a_lesson_that_cannot_write_its_journal_gives_the_gate_back(home, monkeypatch):
    held = []
    teacher = living.Teacher(lambda: None, lambda: held.append(1), lambda: held.pop(), served=lambda: "eden")

    def broken(*_a):
        raise OSError("disk full")
    monkeypatch.setattr(living, "_write", broken)
    with pytest.raises(OSError):
        teacher.start("eden", [{"fact": "A is 1."}])
    assert held == [] and not teacher.busy()


def test_recovery_trusts_only_a_real_graph_or_a_stopped_engine(home):
    import httpx

    (living.root() / ".studio").mkdir(parents=True)
    (living.root() / "eden").mkdir()
    journal = living.root() / ".studio" / "lesson.json"
    journal.write_text(json.dumps({"job": "j1", "copy": "eden", "started": 1.0, "items": []}))
    held = []
    t = living.Teacher(lambda: None, lambda: held.append(1), lambda: held.pop())
    t.recover(lambda: {"error": {"message": "busy"}})                    # an error page is not a graph
    time.sleep(0.5)
    assert held == [1] and journal.exists()

    def gone():
        raise httpx.ConnectError("refused")
    journal2 = living.root() / ".studio" / "lesson.json"
    held2 = []
    t2 = living.Teacher(lambda: None, lambda: held2.append(1), lambda: held2.pop())
    t2.recover(gone, alive=lambda: True)                                # nothing listens, but the process lives
    time.sleep(0.5)
    assert held2 == [1] and journal2.exists()


def test_recovery_waits_for_the_engine_or_its_absence(home):
    import httpx

    (living.root() / ".studio").mkdir(parents=True)
    (living.root() / "eden").mkdir()
    (living.root() / ".studio" / "lesson.json").write_text(json.dumps(
        {"job": "j1", "copy": "eden", "started": 1.0, "items": [{"fact": "A is 1.", "question": ""}]}))
    held = []

    def gone():
        raise httpx.ConnectError("refused")
    teacher = living.Teacher(lambda: None, lambda: held.append(1), lambda: held.pop())
    assert teacher.recover(gone, alive=lambda: False)
    end = time.time() + 5
    while teacher.busy() and time.time() < end:
        time.sleep(0.05)
    assert held == []                                                    # no engine: nothing is being learned


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


def test_the_run_after_asks_what_the_run_before_did_and_denial_is_by_sentence(home):
    from tensorfold_web import living_checks as lc

    (living.root() / "eden").mkdir(parents=True)
    t = type("T", (), {"chat": lambda self, m, **k: "Your cat is called Pixel."})()
    before = lc.run(t, "eden", "before", thinking=False, extra=[{"fact": "My cat is called Pixel.", "question": "What is my cat called?"}])
    lc.set_own("eden", [{"q": "Added later?", "kind": "own"}])
    after = lc.run(t, "eden", "after", same_as=before)
    assert [r["q"] for r in after["rows"]] == [r["q"] for r in before["rows"]]
    assert not lc.recalls("Pixel is not my cat.", ["pixel"])
    assert lc.recalls("Not Ginger. It is Pixel.", ["pixel"])


def test_a_lesson_cut_by_a_studio_restart_is_recorded_and_waited_for(home):
    (living.root() / "eden").mkdir(parents=True)
    (living.root() / ".studio").mkdir(exist_ok=True)
    (living.root() / ".studio" / "lesson.json").write_text(json.dumps(
        {"job": "j9", "copy": "eden", "started": 1.0, "items": [{"fact": "A is 1.", "question": "What is A?"}]}))
    held = []
    teacher = living.Teacher(lambda: None, lambda: held.append(1), lambda: held.pop())
    assert "interrupted" in teacher.recover(lambda: {"facts": []})
    assert living.lessons("eden")[0]["outcome"] == "interrupted"
    end = time.time() + 5
    while teacher.busy() and time.time() < end:
        time.sleep(0.05)
    assert held == [] and not (living.root() / ".studio" / "lesson.json").exists()


def test_a_denied_word_is_not_a_recall():
    from tensorfold_web import living_checks as lc

    assert not lc.recalls("Your cat is not Pixel.", ["pixel"])
    assert not lc.recalls("I don't know if it is Pixel.", ["pixel"])
    assert lc.recalls("Your cat is called Pixel.", ["pixel"])
    assert lc.recalls("Joel is 7.", ["7"])                                # one digit counts
    assert not lc.recalls("[error: timeout]", ["timeout"])


def test_a_stalled_error_body_is_given_up_in_time():
    import socket
    import threading

    import httpx

    from tensorfold_web import httpread

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]

    def stall():
        conn, _ = srv.accept()
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 500 Oops\r\nContent-Length: 100000\r\n\r\npartial")   # then nothing, forever
        time.sleep(30)
        conn.close()
    threading.Thread(target=stall, daemon=True).start()
    with httpx.Client(timeout=60) as c, c.stream("GET", f"http://127.0.0.1:{port}/") as r:
        began = time.monotonic()
        text = httpread.error_text(r, seconds=0.5)
        assert time.monotonic() - began < 3 and text == "partial"
    srv.close()


def test_a_lesson_inside_a_forgetting_leaves_the_gate_to_it(home):
    (living.root() / "eden").mkdir(parents=True)
    calls = []
    t = living.Teacher(lambda: None, lambda: calls.append("hold"), lambda: calls.append("release"),
                       served=lambda: "other")
    with pytest.raises(RuntimeError):
        t.start("eden", [{"fact": "A is 1."}], gate_held=True)
    assert calls == []                                                    # neither taken nor given


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


def test_recovery_ends_on_a_cancel_or_an_address_with_no_learner(home):
    import httpx

    (living.root() / ".studio").mkdir(parents=True)
    (living.root() / "eden").mkdir()
    journal = living.root() / ".studio" / "lesson.json"
    journal.write_text(json.dumps({"job": "j1", "copy": "eden", "started": 1.0, "items": []}))
    held = []

    def not_a_learner():                                                 # an engine started by hand without --slide
        raise httpx.HTTPStatusError("404", request=httpx.Request("GET", "http://x"), response=httpx.Response(404))
    t = living.Teacher(lambda: None, lambda: held.append(1), lambda: held.pop())
    assert t.recover(not_a_learner, alive=lambda: True)
    end = time.time() + 5
    while t.busy() and time.time() < end:
        time.sleep(0.05)
    assert held == [] and not journal.exists()

    journal.write_text(json.dumps({"job": "j2", "copy": "eden", "started": 1.0, "items": []}))
    t2 = living.Teacher(lambda: None, lambda: held.append(1), lambda: held.pop())
    t2.recover(lambda: {"error": "busy"}, alive=lambda: True)             # an error page keeps the gate...
    time.sleep(0.3)
    assert held == [1] and t2.busy()
    assert t2.cancel()                                                   # ...until Teach's cancel
    end = time.time() + 5
    while t2.busy() and time.time() < end:
        time.sleep(0.05)
    assert held == [] and not journal.exists()


def test_a_rollback_that_fails_is_written_down_and_a_good_one_keeps_every_entry(home):
    (living.root() / "eden").mkdir(parents=True)
    (living.root() / ".studio").mkdir(exist_ok=True)

    def broken(copy, version):
        raise RuntimeError("missing from the copy's store")
    t = living.Teacher(lambda: None, lambda: None, lambda: None, undo=broken)
    job = living.Job(id="j", copy="eden", items=[living.Item(id="a", fact="A is 1.", question="", state="kept"),
                                               living.Item(id="b", fact="B is 2.", question="", state="refused")],
                     source="studio", version="v1")
    with pytest.raises(RuntimeError):
        t._roll_back(job, 1)
    rows = living.lessons("eden")
    assert [r["outcome"] for r in rows] == ["kept", "refused"] and all("rollback failed" in r["message"] for r in rows)

    undone = []
    t2 = living.Teacher(lambda: None, lambda: None, lambda: None, undo=lambda c, v: undone.append(v))
    job2 = living.Job(id="k", copy="eden", items=[living.Item(id="a", fact="C is 3.", question="", state="missed"),
                                                living.Item(id="b", fact="D is 4.", question="", state="refused")],
                      source="studio", version="v2")
    t2._roll_back(job2, 1)
    rows = {r["fact"]: r["outcome"] for r in living.lessons("eden") if r.get("job") == "k"}
    assert undone == ["v2"] and rows == {"C is 3.": "missed", "D is 4.": "rolled back"}   # the missed one stays


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


def test_a_failed_create_leaves_a_store_left_by_hand_alone(home, monkeypatch):
    store = living.root() / ".studio" / "eden" / "versions"
    store.mkdir(parents=True)
    (store / "keep").write_text("blobs of a copy removed by hand")
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: (_ for _ in ()).throw(FileNotFoundError("not here")))
    with pytest.raises(ValueError, match="not downloaded"):
        living.create("eden", "TensorFold/x")
    assert (store / "keep").exists() and not living.folder("eden").exists()
    assert not (living.root() / ".studio" / "eden" / "unfinished.json").exists()


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


def test_what_the_checks_read_as_recalled_denied_or_leaked():
    from tensorfold_web import living_checks as lc

    assert not lc.recalls("I don’t know, maybe Pixel.", ["pixel"])         # a curly apostrophe denies too
    assert not lc.recalls("Nobody told me. Neither Pixel nor Ginger.", ["pixel"])
    rows = [{"kind": "fact", "q": "Where was Mara Venn born?", "fact": "Mara Venn was born in Lisbon.",
             "expect": ["lisbon"], "off": "Mara Venn was born in Lisbon."},
            {"kind": "fact", "q": "Who leads Bramblewick?", "fact": "Mara Venn is the chief executive of Bramblewick.",
             "expect": ["mara", "venn", "chief", "executive"]}]                # never asked: the run was cut
    s = lc.summarise({"rows": rows, "thinking": False})
    assert s["modes"]["off"] == {"recalled": 1, "asked": 1, "empty": 0, "errors": 0, "leaks": [], "undone_back": 0}


def test_a_partly_kept_lesson_counts_as_kept_everywhere(home):
    from tensorfold_web import living_checks as lc

    (living.root() / "eden").mkdir(parents=True)
    (living.root() / "eden" / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    job = living.Job(id="j", copy="eden", items=[], source="studio", version="v1")
    living.record("eden", job, living.Item(id="a", fact="A is 1.", question="What is A?", state="partly kept"))
    assert [c["taught"] for c in living.copies()] == [1]
    assert [r["fact"] for r in lc.check_set("eden")["facts"]] == ["A is 1."]
    plan = living.forget_plan("eden", "A is 1.", [{"id": "v1", "label": "before lesson j", "at": 0}])
    assert plan["version"] == "v1"


def test_the_users_sentence_stands_for_a_fact_only_when_it_is_about_that_fact():
    text = "I was born in Lisbon and my dog is called Ginger."
    fact, _ = living.first_person("The user was born in Lisbon.", "Where was the user born?", text)
    assert fact == "The user was born in Lisbon."                           # the compound sentence is not the fact
    fact, _ = living.first_person("The user's cat is called Pixel.", "What is the user's cat called?", "My cat is called Pixel.")
    assert fact == "My cat is called Pixel."


def test_why_missed_reads_this_facts_reason_only():
    lines = ["[tensorfold] slide: 0 questions kept for: Old fact. ()",
             "[tensorfold] slide: failed: pocket full for New fact."]
    assert living.why_missed(lines, "New fact.") is None
    assert "no clean question" in living.why_missed(lines, "Old fact.")


def test_a_learning_event_for_a_fact_never_announced_is_polled_and_nothing_found_means_the_worst():
    t = living.Teacher(lambda: None, lambda: None, lambda: None, graph=lambda: {"facts": [{"id": 7, "state": "learned"}]})
    assert t._learned_anyway(living.Item(id="x", fact="A is 1.", question="", found=[]))
    assert t._learned_anyway(living.Item(id="x", fact="A is 1.", question="", found=[{"id": 7, "text": None, "outcome": "found"}]))
    assert not t._learned_anyway(living.Item(id="x", fact="A is 1.", question="", found=[{"id": 8, "text": None, "outcome": "found"}]))


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


def test_a_run_of_checks_stops_on_teachs_cancel(home):
    (living.root() / "eden").mkdir(parents=True)
    s = settings_mod.Settings(engine_url="http://127.0.0.1:9", remote=True)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None
    app = create_app(settings=s, engine=Engine(s), catalog=cat)
    c = TestClient(app)
    assert c.post("/api/living/cancel").json() == {"cancelling": False}
    assert c.post("/api/living/cancel", json={"now": True}).json() == {"cancelling": False}
    app.state.chore.update(kind="checks", copy="eden", state="running")
    assert c.post("/api/living/cancel").json() == {"cancelling": True} and app.state.chore_cancel.is_set()


def test_stop_now_drops_the_fact_being_learned_and_rolls_back_what_the_engine_kept(home):
    State.facts.clear()
    State.learn_seconds = 5.0                                            # a fact the mock "learns" for a while
    server = mock_serve(port=0, rate_tps=4000.0)
    try:
        (living.root() / "eden").mkdir(parents=True)
        (living.root() / "eden" / "config.json").write_text("{}")
        t = living.Target(f"http://127.0.0.1:{server.server_address[1]}", None, "mock")
        held, undone = [], []
        teacher = living.Teacher(lambda: t, lambda: held.append(1), lambda: held.pop(), served=lambda: "eden",
                                 graph=lambda: {"facts": list(State.facts)}, undo=lambda copy, v: undone.append((copy, v)))
        job = teacher.start("eden", [{"fact": "Pixel is my cat.", "question": ""}, {"fact": "Ginger is my dog.", "question": ""}], check=False)
        end = time.time() + 10
        while job.items[0].state != "learning" and time.time() < end:
            time.sleep(0.05)
        time.sleep(0.3)                                                  # the engine has announced the fact
        asked = time.time()
        assert teacher.cancel(now=True)
        end = time.time() + 10
        while teacher.busy() and time.time() < end:
            time.sleep(0.05)
        assert time.time() - asked < 4                                   # not the mock's five seconds: the request was closed
        first = job.items[0]
        assert first.state == "rolled back" and "stopped at your request" in (first.message or "")
        assert job.status == "rolled back" and undone == [("eden", job.version)]   # the engine had kept it: undone
        assert job.items[1].state == "cancelled" and held == []
    finally:
        State.learn_seconds = 0.05
        server.shutdown()
        State.facts.clear()


def test_why_missed_reads_a_reason_many_lines_back():
    lines = ["[tensorfold] slide: 11 questions kept for: I have a dog named Boogie. (What is my dog's name?)"]
    lines += ["info: slide pocket: a step"] * 1500
    lines += ["[tensorfold] slide: round 5: 5 steps (14.3 s learning, 6.9 s checking); the held-out answers do not come " +
              "back token for token, and do not come back when asked"]
    assert "never came back when asked" in living.why_missed(lines, "I have a dog named Boogie.")


def test_a_lesson_whose_engine_stops_is_failed_not_done(home):
    """A fact that fails because the engine is gone ends the lesson as failed, the rest interrupted: a copy with a
    half-written lesson is never shown as a lesson done."""

    import httpx

    class Gone:
        key, base, model = None, "http://127.0.0.1:9", "living:eden"

        def chat(self, *_a, **_k):
            raise httpx.ConnectError("connection refused")

    (living.root() / "eden").mkdir(parents=True)
    (living.root() / "eden" / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    held: dict = {}                                 # the engine serves eden until the first fact is learned, then dies

    def served():
        return None if held and held["t"].job and held["t"].job.phase == "learning" else "eden"
    teacher = held["t"] = living.Teacher(lambda: Gone(), lambda: None, lambda: None, served=served)
    job = teacher.start("eden", [{"fact": "A is 1."}, {"fact": "B is 2."}, {"fact": "C is 3."}], check=False)
    end = time.time() + 20
    while teacher.busy() and time.time() < end:
        time.sleep(0.05)
    assert job.status == "failed" and "stopped while learning" in (job.error or "")
    assert [i.state for i in job.items] == ["failed", "interrupted", "interrupted"]
