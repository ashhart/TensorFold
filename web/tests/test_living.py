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

def test_a_fact_about_the_user_stays_in_the_first_person():
    text = "Its chief is Mara Venn. My dog is called Ginger. I was born in Lisbon."
    assert living.first_person("The user's dog is called Ginger.", "What is the user's dog called?", text) == \
        ("My dog is called Ginger.", "What is my dog called?")
    assert living.first_person("The user was born in Lisbon.", "Where was the user born?", text) == \
        ("I was born in Lisbon.", "Where was I born?")
    assert living.first_person("The user likes tea.", "What does the user like?", "I like tea")[1] == "What do I like?"

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

def test_a_failed_create_leaves_a_store_left_by_hand_alone(home, monkeypatch):
    store = living.root() / ".studio" / "eden" / "versions"
    store.mkdir(parents=True)
    (store / "keep").write_text("blobs of a copy removed by hand")
    monkeypatch.setattr(living, "snapshot_of", lambda _repo: (_ for _ in ()).throw(FileNotFoundError("not here")))
    with pytest.raises(ValueError, match="not downloaded"):
        living.create("eden", "TensorFold/x")
    assert (store / "keep").exists() and not living.folder("eden").exists()
    assert not (living.root() / ".studio" / "eden" / "unfinished.json").exists()

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
        # not the mock's five seconds: the request was closed
        assert time.time() - asked < 4
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
