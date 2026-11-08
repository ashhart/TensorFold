import json
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from tensorfold_web import living
from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine, _Child
from tensorfold_web.mock_engine import State
from tensorfold_web.mock_engine import serve as mock_serve
from tensorfold_web.server import create_app


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

def test_the_reason_a_lesson_was_missed_is_read_from_the_log():
    lines = ['[tensorfold] slide: 16 questions kept for: My dog is called Ginger. (What is my dog\'s name?)',
             *[f"[tensorfold] slide: round {n}: {n} steps (8.0 s learning, 7.0 s checking); the held-out answers do not "
               "come back token for token, and do not come back when asked" for n in range(1, 6)]]
    assert living.why_missed(lines) == "it never came back when asked, in 5 rounds; nothing of it was kept"
    assert "no clean question" in living.why_missed(["[tensorfold] slide: 0 questions kept for: X. ()"])
    assert living.why_missed(["[tensorfold] serving"]) is None

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

def test_a_lesson_inside_a_forgetting_leaves_the_gate_to_it(home):
    (living.root() / "eden").mkdir(parents=True)
    calls = []
    t = living.Teacher(lambda: None, lambda: calls.append("hold"), lambda: calls.append("release"),
                       served=lambda: "other")
    with pytest.raises(RuntimeError):
        t.start("eden", [{"fact": "A is 1."}], gate_held=True)
    assert calls == []                                                    # neither taken nor given

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

def test_a_lesson_whose_engine_stops_is_failed_not_done(home):

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
