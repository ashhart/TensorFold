"""The QA gate without its tools, and on a clip fetched from another app (a second local copy of this one); no GPU."""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
sys.path.insert(0, str(ROOT / "app"))
import qa  # noqa: E402
import remote  # noqa: E402
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
PRESET = "sharp" if "sharp" in server.PRESETS else next(iter(server.PRESETS))


def other_app():
    """This app again on a local port, as another machine's would be reached; (address, stop)."""

    import socket

    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    served = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=served.run, daemon=True).start()
    for _ in range(100):
        if served.started:
            break
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", lambda: setattr(served, "should_exit", True)


def wait(job_id: str) -> dict:
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("the job did not end")


def test_what_the_gate_needs_is_said():
    had = qa.tool
    try:
        qa.tool = lambda name: None  # a worker Spark as installed: no ffmpeg, no whisper
        can = qa.ready()
        assert can == {"sheet": False, "transcript": False, "runs": False, "missing": [can["missing"][0]]} and "ffmpeg" in can["missing"][0]
        assert client.get("/api/state").json()["qa"]["runs"] is False
        qa.tool = lambda name: "/usr/bin/" + name if name in ("ffmpeg", "ffprobe") else None
        can = qa.ready()
        assert can["runs"] and can["sheet"]
    finally:
        qa.tool = had


def test_a_gate_that_cannot_run_is_in_the_jobs_result():
    clip = server.HOME / "renders" / "made.mp4"
    clip.write_bytes(b"clip")
    job = {"id": "j1", "kind": "video", "status": "running", "params": {"prompt": "a walk", "preset": PRESET, "seed": 42}}
    had = qa.tool
    try:
        qa.tool = lambda name: None
        assert server.gated(job, clip) is True  # the clip is kept and the job goes on
    finally:
        qa.tool = had
    said = job["qa"]
    assert said["passed"] is None and said["skipped"] is True and said["takes"] == []
    assert "did not run" in said["note"] and "ffmpeg" in said["note"] and "/api/qa" in said["note"] and "renders/made.mp4" in said["note"]
    assert server.public({**job, "log": "", "started": None})["qa"] == said  # it reaches the page and the API


def test_the_head_runs_the_gate_on_a_workers_clip():
    """The worker made the clip and could not check it; the head fetches it by the worker's address and checks it."""

    theirs = server.HOME / "renders" / "worker-take.mp4"
    theirs.write_bytes(b"\0\0\0\x18ftypmp42" + b"x" * 5000)
    done = {"id": "w1", "kind": "video", "status": "done", "title": "their take", "outputs": ["renders/worker-take.mp4"], "error": None, "log": "",
            "created": 0, "started": 1.0, "finished": 2.0, "progress": None, "estimate": None,
            "params": {"prompt": "A man says <d>[English] Good morning.</d>", "seed": 43, "preset": PRESET}}
    server.jobs["w1"] = done
    seen = {}

    def check(clip, prompt, picture, brain=None):  # the gate itself is qa.py's, tested where its tools are
        seen.update(clip=Path(clip), prompt=prompt, size=Path(clip).stat().st_size)
        Path(picture).parent.mkdir(parents=True, exist_ok=True)
        Path(picture).write_bytes(b"jpg")
        return {"audio": {"result": "pass", "heard": "good morning", "note": "2 of the line's 2 words heard"}, "identity": {"result": "skipped", "note": "look yourself"},
                "sharpness": [], "times": [1.0, 2.5, 4.0], "passed": True}

    base, stop = other_app()
    had_check, had_ready = qa.check, qa.ready
    qa.check, qa.ready = check, lambda: {"sheet": True, "transcript": True, "runs": True, "missing": []}
    try:
        # by the worker's job: the clip, the prompt and the seed are read from it
        queued = client.post("/api/qa", json={"from": base, "job": "w1"})
        assert queued.status_code == 200 and queued.json()["kind"] == "qa", queued.text
        job = wait(queued.json()["id"])
        assert job["status"] == "done", job
        assert job["qa"]["passed"] is True and job["qa"]["from"] == base and job["qa"]["takes"][0]["seed"] == 43
        assert seen["prompt"] == done["params"]["prompt"] and seen["size"] == theirs.stat().st_size
        fetched = server.HOME / job["outputs"][0]
        assert fetched.is_file() and fetched.parent.name == "fetched" and fetched.read_bytes() == theirs.read_bytes()
        assert (server.HOME / job["qa"]["takes"][0]["sheet"]).is_file()
        # by path, with the prompt given; and a clip that is already here
        again = wait(client.post("/api/qa", json={"from": base.replace("http://", ""), "path": "renders/worker-take.mp4", "prompt": "p2", "seed": 7}).json()["id"])
        assert again["status"] == "done" and seen["prompt"] == "p2" and again["qa"]["takes"][0]["seed"] == 7
        here = wait(client.post("/api/qa", json={"path": "renders/worker-take.mp4", "prompt": "p3"}).json()["id"])
        assert here["status"] == "done" and seen["clip"] == theirs and here["qa"]["from"] is None
        # what cannot be done is said
        missing = wait(client.post("/api/qa", json={"from": base, "path": "renders/none.mp4"}).json()["id"])
        assert missing["status"] == "failed" and "has no file renders/none.mp4" in missing["error"]
        gone = wait(client.post("/api/qa", json={"from": base, "job": "w404"}).json()["id"])
        assert gone["status"] == "failed" and "has no job w404" in gone["error"]
        assert client.post("/api/qa", json={"from": "ftp://x", "path": "renders/a.mp4"}).status_code == 400
        assert client.post("/api/qa", json={"prompt": "p"}).status_code == 400
        qa.ready = lambda: {"sheet": False, "transcript": False, "runs": False, "missing": ["ffmpeg (sudo apt install ffmpeg)"]}
        refused = client.post("/api/qa", json={"path": "renders/worker-take.mp4"})
        assert refused.status_code == 400 and "ffmpeg" in refused.json()["error"]
    finally:
        qa.check, qa.ready = had_check, had_ready
        stop()
        server.jobs.pop("w1", None)


def test_a_take_rendered_elsewhere_is_logged_into_its_shot():
    """A clip rendered by another app is logged into a shot here, by that app's job, a path there, or a local file."""

    import slate

    film = client.post("/api/slate/projects", json={"name": "Hesitation"}).json()
    film["scenes"] = [{"id": "scene-1", "name": "Scene 1", "synopsis": "", "shots": [
        {"id": "shot-14", "name": "1.4 First Smiles", "intent": "", "spec": {}, "prompt": "# Subject\nLeo and Mia.", "takes": [], "tf": {"h3": "the compiled prompt"}},
        {"id": "shot-15", "name": "1.5 Twin", "takes": []}, {"id": "shot-16", "name": "1.5 Twin", "takes": []}]}]
    assert client.put(f"/api/slate/projects/{film['id']}", json=film).status_code == 200
    theirs = server.HOME / "renders" / "s1_4.mp4"
    theirs.write_bytes(b"\0\0\0\x18ftypmp42" + b"y" * 3000)
    server.jobs["w2"] = {"id": "w2", "kind": "video", "status": "done", "title": "t", "outputs": ["renders/s1_4.mp4"], "error": None, "log": "", "created": 0,
                         "started": 10.0, "finished": 215.0, "progress": None, "estimate": None, "engine": "cuda",
                         "qa": {"passed": None, "skipped": True, "takes": [], "note": "did not run"},
                         "params": {"prompt": "the prompt as rendered", "seed": 43, "preset": "sharp", "quality": "fh8", "frames": 124, "qa": False,
                                    "loras": [{"id": "camera-motion", "strength": 0.9}], "output": "x.mp4"}}
    server.jobs["w3"] = {**server.jobs["w2"], "id": "w3", "status": "running", "outputs": []}
    base, stop = other_app()
    try:
        # by the other app's job, logged into the project file (what a script does)
        got = client.post("/api/slate/takes", json={"projectId": film["id"], "shot": "1.4 first smiles", "from": base, "job": "w2", "notes": "Spark 5", "log": True})
        assert got.status_code == 200, got.text
        take = got.json()["take"]
        assert take["seed"] == 43 and take["prompt"] == "the prompt as rendered" and take["notes"] == "Spark 5" and take["rating"] == "good"
        assert take["settings"] == {"preset": "sharp", "quality": "fh8", "frames": 124, "loras": [{"id": "camera-motion", "strength": 0.9}]}
        assert take["source"] == {"from": base, "job": "w2", "engine": "cuda", "elapsed": 205.0, "path": "renders/s1_4.mp4"} and take["imported"] is True
        assert take["qa"]["skipped"] is True  # what the worker's gate said travels with the take
        media = server.HOME / take["mediaPath"]
        assert media.read_bytes() == theirs.read_bytes() and media.parent == server.HOME / "slate" / film["id"] / "takes" and media.name.startswith("shot-14-")
        saved = client.get(f"/api/slate/projects/{film['id']}").json()["scenes"][0]["shots"][0]["takes"]
        assert [t["id"] for t in saved] == [take["id"]] and got.json()["logged"] is True
        assert slate.best_take({"takes": saved}) == take["mediaPath"]  # it stands for the shot like any take
        # by a path there, with the seed said; not logged here: the page adds it to the film it holds
        second = client.post("/api/slate/takes", json={"projectId": film["id"], "shot": "shot-14", "from": base, "path": "renders/s1_4.mp4", "seed": "44", "rating": "circled",
                                                         "settings": {"preset": "sharp"}})
        assert second.status_code == 200 and second.json()["take"]["seed"] == 44 and second.json()["take"]["rating"] == "circled" and second.json()["logged"] is False
        assert second.json()["take"]["prompt"] == "the compiled prompt"  # none given: the shot's own
        assert len(client.get(f"/api/slate/projects/{film['id']}").json()["scenes"][0]["shots"][0]["takes"]) == 1
        # a file already in this studio folder (an upload)
        (server.HOME / "uploads" / "mine.mp4").write_bytes(b"clip of mine")
        local = client.post("/api/slate/takes", json={"projectId": film["id"], "shot": "shot-14", "path": "uploads/mine.mp4", "seed": 7}).json()["take"]
        assert (server.HOME / local["mediaPath"]).read_bytes() == b"clip of mine" and local["source"] == {"path": "uploads/mine.mp4"} and local["seed"] == 7
        # what cannot be imported says why, and nothing is written
        for body, code, word in (({"shot": "no such shot", "from": base, "job": "w2"}, 404, "no such shot"),
                                 ({"shot": "1.5 Twin", "from": base, "job": "w2"}, 400, "2 shots are called"),
                                 ({"shot": "shot-14", "from": base, "job": "w3"}, 400, "is running"),
                                 ({"shot": "shot-14", "from": base, "job": "w404"}, 404, "has no job"),
                                 ({"shot": "shot-14", "from": base, "path": "renders/none.mp4"}, 404, "has no file"),
                                 ({"shot": "shot-14", "from": base}, 400, "give the clip"),
                                 ({"shot": "shot-14", "from": "http://127.0.0.1:9", "path": "renders/a.mp4"}, 502, "did not answer"),
                                 ({"shot": "shot-14", "path": "../outside.mp4"}, 400, "give the clip"),
                                 ({"shot": "shot-14", "path": "projects"}, 400, "give the clip"),
                                 ({"shot": "shot-14", "from": base, "job": "w2", "seed": "x"}, 400, "whole number")):
            refused = client.post("/api/slate/takes", json={"projectId": film["id"], **body})
            assert refused.status_code == code and word in refused.json()["error"], (body, refused.status_code, refused.text)
        assert client.post("/api/slate/takes", json={"projectId": "nope", "shot": "x"}).status_code == 404
    finally:
        stop()
        server.jobs.pop("w2", None), server.jobs.pop("w3", None)


def test_another_apps_address_and_paths():
    assert remote.address("spark-5:7870") == "http://spark-5:7870" and remote.address(" http://192.0.2.20:7870/ ") == "http://192.0.2.20:7870"
    for bad in ("", "file:///etc/passwd", "http://host/files/x", "http://user:pw@host:1", "http://host:1?x=1"):
        try:
            remote.address(bad)
            raise AssertionError(f"taken as an address: {bad!r}")
        except ValueError:
            pass
    for bad in ("", "../x.mp4", "renders/../../x.mp4", "renders/x.sh", "projects/p.json"):
        try:
            remote.fetch("http://127.0.0.1:1", bad, Path(tempfile.mkdtemp()) / "x.mp4")
            raise AssertionError(f"fetched: {bad!r}")
        except ValueError:
            pass


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
