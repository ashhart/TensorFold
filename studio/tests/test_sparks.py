"""Sparks: a job's number of Sparks, the workers behind the queue; workers, ssh, rsync and tunnels are stood in for."""
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
os.environ["STUDIO_SPARK_POLL"] = "0.05"
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402
import job_build  # noqa: E402
import job_run  # noqa: E402
import sparks  # noqa: E402
import sparks_clip  # noqa: E402
import sparks_wire  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
hub = server.hub
hub.spark = True  # the scheduler is the same on any machine once ssh and the renders are stood in for
PRESET = "small" if "small" in server.PRESETS else next(iter(server.PRESETS))
QUALITY = "fh8"
TAKES = 0.35  # how long a stand-in render takes, here and on a worker


class Worker:
    """Another Spark's Studio app, as far as this one talks to it."""

    def __init__(self, name: str, **has):
        self.name, self.jobs, self.uploads, self.files, self.most, self.cancelled = name, {}, {}, {}, 0, []
        self.has = {"fasth3": True, "image": False, "ref2va": False, "ref2va_turbo": False, "ffmpeg": False, "loras": [], "text_encoders": [], **has}
        self.code, self.dead, self.fits, self.takes, self.fails = None, False, True, TAKES, False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def answer(self, body, status=200):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def handle_one(self, method):
                if outer.dead:  # the machine is gone: the connection drops without an answer
                    self.connection.close()
                    return
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else b""
                path = self.path
                if path == "/api/spark/self":
                    return self.answer({"name": outer.name, "version": "2.6.6", "guard": True, "code": outer.code or hub.code(), "disk_gb": 50,
                                        "active": sum(1 for j in outer.jobs.values() if j["status"] in ("queued", "running")),
                                        "memory": {"available_mb": 116000, "total_mb": 124000}, **outer.has})
                if path == "/api/jobs" and method == "GET":
                    return self.answer(list(outer.jobs.values()))
                if path == "/api/estimate":
                    return self.answer({"seconds": 40, "memory": {"fits": outer.fits, "reason": "" if outer.fits else "a clip with a LoRA needs about 76 GB; this machine has 40 GB for it, so it is refused",
                                                                  "need_mb": 57000, "available_mb": 116000}})
                if path == "/api/upload":
                    name = f"uploads/{outer.name}-{len(outer.uploads)}.png"
                    outer.uploads[name] = body
                    return self.answer({"path": name})
                if path == "/api/jobs" and method == "POST":
                    asked = json.loads(body)
                    job = {"id": f"{outer.name}-{len(outer.jobs)}", "status": "running", "params": asked["params"], "title": asked["title"], "kind": asked["kind"],
                           "outputs": [], "progress": {"stage": "picture", "at": 1, "of": 8}, "error": None, "elapsed": outer.takes, "log": "[tensorfold] engine: cuda\n"}
                    outer.jobs[job["id"]] = job
                    outer.most = max(outer.most, sum(1 for j in outer.jobs.values() if j["status"] == "running"))

                    def finish():
                        time.sleep(outer.takes)
                        if job["status"] != "running":
                            return
                        if outer.fails:
                            job.update(status="failed", error="ComfyUI stopped: the memory floor")
                            return
                        outer.files[f"renders/{job['id']}.mp4"] = f"made on {outer.name}".encode()
                        job.update(status="done", outputs=[f"renders/{job['id']}.mp4"])

                    threading.Thread(target=finish, daemon=True).start()
                    return self.answer(job)
                if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    job = outer.jobs[path.split("/")[3]]
                    job["status"] = "cancelled"
                    outer.cancelled.append(job["id"])
                    return self.answer(job)
                if path.startswith("/api/jobs/"):
                    job = outer.jobs.get(path.split("/")[3])
                    return self.answer(job) if job else self.answer({"error": "no such job"}, 404)
                if path.startswith("/files/"):
                    data = outer.files.get(path[len("/files/"):])
                    return self.answer(data) if data else self.answer({"error": "no such file"}, 404)
                return self.answer({"error": "not found"}, 404)

            def do_GET(self):
                self.handle_one("GET")

            def do_POST(self):
                self.handle_one("POST")

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"
        self.address = f"10.9.9.{self.http.server_address[1] % 250 + 1}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()


class Line:
    """A tunnel, stood in for: the worker is already on this machine."""

    def __init__(self, worker):
        self.worker, self.up, self.process = worker, False, None

    def open(self):
        self.up = True
        return self.worker.base

    def alive(self):
        return self.up

    def close(self):
        self.up = False


WORKERS: dict[str, Worker] = {}
SENT = []  # rsync, stood in for: (address, folder)
SHELLS = []  # every command ssh was asked to run


def stand_in_shell(address, command, timeout=30):
    SHELLS.append((address, command))
    worker = WORKERS.get(address)
    if worker is None or worker.dead:
        return 255, "ssh: connect to host: No route to host"
    return 0, "\n".join([f"host {worker.name}", "version 2.6.6", "pack yes", "install yes", "MemTotal: 127599612 kB", "MemAvailable: 118784000 kB", "app yes", "ours yes"])


def local_render(job, command, env, label=""):
    """This machine's render, stood in for: the clip appears at its path after a moment."""

    local_render.now += 1
    local_render.most = max(local_render.most, local_render.now)
    try:
        time.sleep(TAKES)
        if job["status"] == "cancelled":
            return False
        Path(command[-1]).write_bytes(b"made here")
        return True
    finally:
        local_render.now -= 1


local_render.now = local_render.most = 0
job_run.run = local_render
job_build.memory_of = lambda *args, **kwargs: {"fits": True, "reason": ""}  # this machine's guard has its own tests
hub.shell, hub.copy = stand_in_shell, lambda pack, address, folder, timeout=900: (SENT.append((address, folder)), (0, ""))[1]
hub.dial = lambda address: Line(WORKERS[address])
threading.Thread(target=hub.loop, daemon=True).start()


def fresh(*workers: Worker) -> None:
    """The list is these workers, each checked; the queue is empty."""

    for address in [one["address"] for one in hub.workers()]:
        hub.hang_up(address)
        hub.live.pop(address, None)
    hub.keep({"workers": [], "sparks": 1})
    WORKERS.clear()
    with server.lock:
        server.jobs.clear()
        server.order.clear()
    local_render.most = 0
    for worker in workers:
        WORKERS[worker.address] = worker
        added = client.post("/api/sparks/workers", json={"address": worker.address})
        assert added.status_code == 200, added.text
    settle()


def settle(limit: float = 10) -> None:
    until = time.time() + limit
    while time.time() < until and any(row["working"] for row in client.get("/api/sparks").json()["workers"]):
        time.sleep(0.05)


def queue(sparks_asked=None, **params) -> dict:
    body = {"kind": "video", "title": "clip", "params": {"prompt": "a walk", "preset": PRESET, "quality": QUALITY, "frames": 22, "shot_check": False, **params}}
    if sparks_asked is not None:
        body["sparks"] = sparks_asked
    done = client.post("/api/jobs", json=body)
    assert done.status_code == 200, done.text
    return done.json()


def finished(ids, limit: float = 30) -> list[dict]:
    until = time.time() + limit
    while time.time() < until:
        found = [client.get(f"/api/jobs/{one}").json() for one in ids]
        if all(job["status"] in ("done", "failed", "cancelled") for job in found):
            return found
        time.sleep(0.05)
    raise AssertionError([(job["status"], job.get("spark"), job["log"][-300:]) for job in found])


def test_an_address_is_a_host_and_nothing_ssh_reads_as_an_option():
    assert sparks.address_of(" 192.0.2.10 ") == "192.0.2.10" and sparks.address_of("user@spark-5") == "user@spark-5"
    for bad in ("", "-oProxyCommand=x", "a b", "host;rm", "user@", "http://192.0.2.20:7870", "a/b"):
        try:
            sparks.address_of(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert sparks.folder_of("slate-studio/") == "slate-studio"
    for bad in ("", "../x", "a/../b", "-x", "a b", "$HOME/x"):
        try:
            sparks.folder_of(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert client.post("/api/sparks/workers", json={"address": "-oProxyCommand=x"}).status_code == 400


def test_what_a_worker_can_render_decides_what_it_is_given():
    plain = {"fasth3": True, "loras": [], "ffmpeg": False}
    assert sparks.capable("video", {"prompt": "x"}, plain) == "" and sparks.capable("video", {"image": "uploads/a.png", "last_image": "uploads/b.png"}, plain) == ""
    assert "LoRA camera-motion" in sparks.capable("video", {"loras": [{"id": "camera-motion", "strength": 1}]}, plain)
    assert sparks.capable("video", {"loras": [{"id": "camera-motion"}]}, {**plain, "loras": ["camera-motion"]}) == ""
    assert "Ref2VA" in sparks.capable("video", {"references": ["uploads/face.png"]}, plain)
    assert "Turbo" in sparks.capable("video", {"references": ["uploads/face.png"], "ref_turbo": True}, {**plain, "ref2va": True})
    assert "ffmpeg" in sparks.capable("video", {"references": ["uploads/a.png"], "reference_videos": ["uploads/m.mp4"]}, {**plain, "ref2va": True})
    assert "text encoder" in sparks.capable("video", {"text_encoder": "other.safetensors"}, plain)
    assert sparks.capable("video", {"text_encoder": "other.safetensors"}, {**plain, "text_encoders": ["other.safetensors"]}) == ""
    assert "FastH3" in sparks.capable("video", {}, {}) and "only single clips" in sparks.capable("scout", {}, plain)
    # what stays on this Spark whatever the number, and why
    assert "chain" in sparks.here_only("long", {}) and "picture model" in sparks.here_only("studio", {}) and sparks.here_only("video", {}) == ""
    assert "this Spark" in sparks.here_only("edit", {})


def test_the_time_at_a_number_follows_the_rounds_the_clips_go_in():
    assert sparks.speedup(1, 8) == (1.0, True)
    assert sparks.speedup(5, 1)[0] == 1.0  # one clip is not faster on five
    had = dict(sparks.MEASURED)
    try:
        sparks.MEASURED.update({2: 1.9, 3: 2.5, 4: 3.6, 5: 3.7})
        two, three, eight = sparks.speedup(2, 8), sparks.speedup(3, 8), sparks.speedup(8, 16)
        assert two == (1.9, True) and three == (2.5, True)
        assert abs(sparks.speedup(4, 4)[0] - 4 * 3.6 / 4) < 0.01  # four clips on four: one round, less what the measured run lost
        assert eight[1] is False and eight[0] <= 8  # eight Sparks were not measured, and it says so
        said = hub.plan({"sparks": 1, "clips": 8, "seconds": 40})
        assert said["estimate"]["seconds"] == 320 and said["estimate"]["faster"] == 1.0
    finally:
        sparks.MEASURED.clear()
        sparks.MEASURED.update(had)


def test_the_number_is_held_to_the_sparks_there_are():
    one, two = Worker("spark-a"), Worker("spark-b")
    fresh(one, two)
    seen = client.get("/api/sparks").json()
    assert seen["supported"] and seen["limit"] == 8 and seen["available"] == 3 and seen["listed"] == 3
    assert [row["state"] for row in seen["workers"]] == ["ready", "ready"] and [row["name"] for row in seen["workers"]] == ["spark-a", "spark-b"]
    assert seen["workers"][0]["can"]["fasth3"] and seen["workers"][0]["guard"] and seen["workers"][0]["memory_gb"] == 116.0
    said = client.post("/api/sparks/plan", json={"sparks": 8, "kind": "video", "clips": 8, "seconds": 40}).json()
    assert said["chosen"] == 8 and said["available"] == 3 and said["used"] == 3 and said["count"] == "3 of 8 available" and said["mode"] == "parallel"
    assert "3 at a time" in said["says"] and said["estimate"]["one_spark"] == 320
    assert client.post("/api/sparks/plan", json={"sparks": 2, "clips": 8}).json()["count"] == "2 Sparks"
    # one clip: one Spark, and it says that a clip is not split
    alone = client.post("/api/sparks/plan", json={"sparks": 3, "kind": "video", "clips": 1}).json()
    assert alone["mode"] == "one" and "one Spark" in alone["says"] and "not in the app" in alone["says"]
    # a long video and a picture stay here
    assert client.post("/api/sparks/plan", json={"sparks": 3, "kind": "long"}).json()["used"] == 1
    # a LoRA no worker has: this Spark alone, and the reason for each worker
    lora = client.post("/api/sparks/plan", json={"sparks": 3, "kind": "video", "clips": 4, "params": {"loras": [{"id": "camera-motion"}]}}).json()
    assert lora["available"] == 1 and lora["used"] == 1 and len(lora["without"]) == 2 and "camera-motion" in lora["without"][0]["why"]
    # the API: sparks on a job, 1 to 8
    assert client.post("/api/jobs", json={"kind": "video", "sparks": 9, "params": {"prompt": "x", "preset": PRESET, "quality": QUALITY}}).status_code == 400
    assert client.post("/api/jobs", json={"kind": "video", "sparks": "many", "params": {"prompt": "x", "preset": PRESET, "quality": QUALITY}}).status_code == 400
    job = queue(8)
    assert job["sparks"]["chosen"] == 8 and job["sparks"]["available"] == 3 and job["sparks"]["used"] == 3
    finished([job["id"]])
    assert client.get("/api/state").json()["sparks"] == {"supported": server.SPARK, "limit": 8}


def test_queued_clips_render_one_per_spark_and_come_back_here():
    one, two = Worker("spark-a"), Worker("spark-b")
    fresh(one, two)
    picture = server.HOME / "uploads" / "first.png"
    picture.write_bytes(b"a first frame")
    started = time.time()
    ids = [queue(3, seed=10 + n, image="uploads/first.png", output=f"projects/p1/seg{n}.mp4" if n == 0 else None)["id"] for n in range(6)]
    done = finished(ids)
    wall = time.time() - started
    assert all(job["status"] == "done" for job in done), [(job["status"], job["error"]) for job in done]
    assert {job["spark"] for job in done} == {hub.name, "spark-a", "spark-b"}
    assert one.most == 1 and two.most == 1 and local_render.most == 1  # one job on a Spark at a time
    assert wall < 6 * TAKES * 0.75, wall  # three machines, not one after another
    made = hub.takes()
    for job in done:
        clip = server.HOME / job["outputs"][0]
        assert clip.is_file() and clip.read_bytes() == (b"made here" if job["spark"] == hub.name else f"made on {job['spark']}".encode())
        assert made[job["outputs"][0]]["spark"] == job["spark"] and made[job["outputs"][0]]["seed"] == job["params"]["seed"]
        assert made[job["outputs"][0]]["settings"]["frames"] == 22 and made[job["outputs"][0]]["sparks"] == 3
    assert done[0]["outputs"] == ["projects/p1/seg0.mp4"]  # a project's clip lands at the project's path wherever it was rendered
    # what a worker was sent: the picture uploaded and named by its path there; the checks left here; no path of ours
    theirs = next(iter(one.jobs.values()))["params"]
    assert theirs["image"].startswith("uploads/spark-a-") and b"a first frame" in one.uploads[theirs["image"]]
    assert theirs["qa"] is False and theirs["shot_check"] is False and "output" not in theirs and theirs["seed"] in range(10, 16)
    # the Library says where each clip was rendered
    shown = {item["path"]: item for item in client.get("/api/library").json()}
    assert all(shown[job["outputs"][0]]["spark"] == job["spark"] for job in done)
    assert client.get("/api/sparks/takes", params={"path": done[1]["outputs"][0]}).json()["spark"] == done[1]["spark"]


def test_one_spark_means_this_one_and_two_means_the_first_worker():
    one, two = Worker("spark-a"), Worker("spark-b")
    fresh(one, two)
    done = finished([queue(1)["id"] for _ in range(3)])
    assert {job["spark"] for job in done} == {hub.name} and not one.jobs and not two.jobs
    done = finished([queue(2)["id"] for _ in range(4)])
    assert {job["spark"] for job in done} == {hub.name, "spark-a"} and not two.jobs
    # no number given: the one set on the Sparks page
    assert client.post("/api/sparks", json={"sparks": 3}).json()["default"] == 3
    assert queue()["sparks"]["chosen"] == 3
    client.post("/api/sparks", json={"sparks": 1})
    # a worker switched off is not counted and not used
    client.post(f"/api/sparks/workers/{one.address}", json={"enabled": False})
    assert client.get("/api/sparks").json()["available"] == 2
    before = len(one.jobs)
    done = finished([queue(3)["id"] for _ in range(4)])
    assert len(one.jobs) == before and {job["spark"] for job in done} == {hub.name, "spark-b"}


def test_a_worker_that_stops_answering_is_marked_down_and_its_clip_rendered_elsewhere():
    one = Worker("spark-a")
    one.takes = 30  # it will not finish
    fresh(one)
    had = sparks_clip.LOST
    sparks_clip.LOST = 0.6
    try:
        ids = [queue(2)["id"] for _ in range(2)]
        until = time.time() + 10
        while time.time() < until and not one.jobs:
            time.sleep(0.05)
        assert one.jobs
        one.dead = True
        done = finished(ids, 30)
    finally:
        sparks_clip.LOST = had
    assert all(job["status"] == "done" and job["spark"] == hub.name for job in done), done
    moved = next(job for job in done if job["moved"])
    assert moved["moved"][0]["from"] == "spark-a" and "stopped answering" in moved["moved"][0]["why"]
    seen = client.get("/api/sparks").json()
    assert seen["workers"][0]["state"] == "down" and seen["available"] == 1 and "stopped answering" in seen["workers"][0]["why"]
    assert hub.state(one.address)["tunnel"] is None  # its tunnel is closed
    # it is not given another clip until it is checked again
    again = finished([queue(2)["id"] for _ in range(2)])
    assert all(job["spark"] == hub.name for job in again) and len(one.jobs) == 1
    # it answers again, busy on the clip taken back (no longer ours), so it is still given nothing
    one.dead, one.takes = False, TAKES
    client.post(f"/api/sparks/workers/{one.address}/check", json={})
    settle()
    row = client.get("/api/sparks").json()["workers"][0]
    assert row["state"] == "busy" and not row["usable"] and "not this Spark's" in row["why"]
    for job in one.jobs.values():
        job["status"] = "cancelled"
    client.post(f"/api/sparks/workers/{one.address}/check", json={})
    settle()
    assert client.get("/api/sparks").json()["workers"][0]["state"] == "ready"


def test_a_clip_a_workers_memory_guard_refuses_is_not_sent_and_a_failed_one_moves():
    one = Worker("spark-a")
    fresh(one)
    one.fits = False  # its guard says no before anything is sent
    done = finished([queue(2)["id"] for _ in range(2)])
    assert all(job["status"] == "done" and job["spark"] == hub.name for job in done)
    assert not one.jobs and not one.uploads
    moved = next(job for job in done if job["moved"])
    assert "refused" in moved["moved"][0]["why"] and "76 GB" in moved["moved"][0]["why"]
    assert client.get("/api/sparks").json()["workers"][0]["state"] == "ready"  # the worker is well: it said no
    # a render that fails there (its guard stopped its server): the clip is rendered here, and never there again
    one.fits, one.fails = True, True
    done = finished([queue(2)["id"] for _ in range(2)])
    assert all(job["status"] == "done" and job["spark"] == hub.name for job in done)
    assert any("memory floor" in m["why"] for job in done for m in job["moved"] or [])


def test_cancel_reaches_the_worker():
    one = Worker("spark-a")
    one.takes = 30
    fresh(one)
    ids = [queue(2)["id"] for _ in range(2)]
    until = time.time() + 10
    while time.time() < until and not one.jobs:
        time.sleep(0.05)
    there = next(job for job in [client.get(f"/api/jobs/{one_id}").json() for one_id in ids] if job["spark"] == "spark-a")
    assert there["status"] == "running" and "spark-a" in (there["progress"] or {"stage": "spark-a"})["stage"]
    assert client.post(f"/api/jobs/{there['id']}/cancel", json={}).json()["status"] == "cancelled"
    until = time.time() + 5
    while time.time() < until and not one.cancelled:
        time.sleep(0.05)
    assert one.cancelled == [next(iter(one.jobs))]
    finished(ids)
    assert client.get(f"/api/jobs/{there['id']}").json()["status"] == "cancelled"
    until = time.time() + 5
    while time.time() < until and hub.state(one.address)["job"]:
        time.sleep(0.05)
    assert hub.state(one.address)["job"] is None  # the worker is free again
    one.takes = TAKES


def test_a_clip_that_opens_on_another_waits_for_it():
    one = Worker("spark-a")
    fresh(one)
    had = job_run.last_frame
    job_run.last_frame = lambda video, image: Path(image).write_bytes(b"jpg of " + Path(video).read_bytes())
    try:
        first = queue(2, output="projects/p2/seg1.mp4")
        second = queue(2, first_frame_from="projects/p2/seg1.mp4", output="projects/p2/seg2.mp4")
        third = queue(2, output="projects/p2/seg3.mp4")
        done = finished([first["id"], second["id"], third["id"]])
    finally:
        job_run.last_frame = had
    assert all(job["status"] == "done" for job in done), [job["error"] for job in done]
    assert done[1]["started"] >= done[0]["finished"] - 0.01  # it waited
    assert done[2]["started"] < done[0]["finished"]  # and the one behind it did not
    if done[1]["spark"] == "spark-a":  # the frame was cut here and sent
        assert any(data.startswith(b"jpg of ") for data in one.uploads.values())


def test_a_worker_with_other_code_is_updated_before_it_is_used():
    one = Worker("spark-a")
    one.code = {**hub.code(), "app/server.py": "an older one"}
    fresh(one)
    seen = client.get("/api/sparks").json()
    assert seen["workers"][0]["state"] == "update" and seen["available"] == 1 and "app/server.py" in seen["workers"][0]["why"]
    done = finished([queue(2)["id"] for _ in range(2)])
    assert all(job["spark"] == hub.name for job in done) and not one.jobs  # nothing is sent to it as it is
    SENT.clear()
    SHELLS.clear()
    one.code = None  # what the copy brings
    client.post(f"/api/sparks/workers/{one.address}/update", json={})
    settle(40)
    assert SENT == [(one.address, hub.default_folder())]
    started = [command for _, command in SHELLS if "scripts/app.sh" in command]
    # its app is started on its own 127.0.0.1, never on the network, and nothing is run as root or written to /etc
    assert started and "HOST=127.0.0.1" in started[0] and "0.0.0.0" not in started[0]
    assert not any("sudo" in command or "/etc/" in command for _, command in SHELLS)
    assert client.get("/api/sparks").json()["workers"][0]["state"] == "ready"
    # a Spark ssh cannot reach says so
    one.dead = True
    client.post(f"/api/sparks/workers/{one.address}/check", json={})
    settle()
    row = client.get("/api/sparks").json()["workers"][0]
    assert row["state"] == "unreachable" and "key login" in row["why"]
    assert client.delete(f"/api/sparks/workers/{one.address}").json()["listed"] == 1


def test_the_tunnel_ends_on_the_workers_own_loopback_and_is_closed():
    """The command a tunnel is: local 127.0.0.1 to the worker's 127.0.0.1:7870, key login only, nothing else."""

    seen = {}

    class Process:
        pid, stderr = 0, None

        def __init__(self, command, **kwargs):
            seen["command"], seen["live"] = command, True

        def poll(self):
            return None if seen["live"] else 0

        def terminate(self):
            seen["live"] = False

        def wait(self, timeout=None):
            return 0

    had_popen, had_connect = sparks_wire.subprocess.Popen, sparks_wire.socket.create_connection
    sparks_wire.subprocess.Popen = Process
    sparks_wire.socket.create_connection = lambda *args, **kwargs: type("S", (), {"close": lambda self: None})()
    try:
        tunnel = sparks.Tunnel("192.0.2.10")
        base = tunnel.open()
        command = seen["command"]
        assert command[0] == "ssh" and "BatchMode=yes" in command and "-N" in command and command[-1] == "192.0.2.10"
        forward = command[command.index("-L") + 1]
        assert forward == f"127.0.0.1:{tunnel.port}:127.0.0.1:7870" and base == f"http://127.0.0.1:{tunnel.port}"
        assert tunnel.alive()
        tunnel.close()
        assert not seen["live"] and not tunnel.alive()
    finally:
        sparks_wire.subprocess.Popen, sparks_wire.socket.create_connection = had_popen, had_connect
    # stopping the app closes every tunnel it opened
    one = Worker("spark-a")
    fresh(one)
    line = hub.state(one.address)["tunnel"]
    assert line.up
    hub.close()
    assert not line.up and hub.state(one.address)["tunnel"] is None


def test_a_mac_shows_one_and_says_why():
    mac = sparks.Hub(server.HOME, ROOT, False, {}, [], threading.Lock(), threading.Event(), lambda: {}, server.inside, lambda job, worker: None)
    seen = mac.overview()
    assert seen["supported"] is False and seen["available"] == 1 and seen["workers"] == [] and "DGX Sparks" in seen["note"]
    said = mac.plan({"sparks": 5, "clips": 8})
    assert said["used"] == 1 and said["chosen"] == 1 and "Mac" in said["says"]
    job = {"kind": "video", "params": {}, "sparks": {"chosen": 5}}
    assert mac.team(job) == []
    mac.start()
    assert not mac.started  # no loop, no ssh, on a Mac


def test_nothing_here_runs_a_tool_without_a_time_limit():
    for path in [ROOT / "app/sparks.py", *sorted((ROOT / "app").glob("sparks_*.py"))]:
        tree = ast.parse(path.read_text())
        for call in [node for node in ast.walk(tree) if isinstance(node, ast.Call)]:
            if ast.unparse(call.func) == "subprocess.run":
                assert any(keyword.arg == "timeout" for keyword in call.keywords), ast.unparse(call)
        text = path.read_text()
        assert "sudo" not in text and "/etc/" not in text and "0.0.0.0" not in text


PAGE = r"""
const vm = require("vm"), fs = require("fs");
globalThis.calls = []; globalThis.answers = {};
globalThis.api = async (path, body) => { calls.push([path, body]); return answers[path] || {}; };
globalThis.render = () => {}; globalThis.toast = () => {}; globalThis.mmss = s => s + " s"; globalThis.tab = "Create";
globalThis.localStorage = { getItem: () => null, setItem() {} }; globalThis.document = { querySelector: () => null, activeElement: { matches: () => false } };
globalThis.el = (tag, attrs = {}, ...kids) => ({ tag, attrs, kids: kids.flat(), replaceChildren(...k) { this.kids = k; } });
globalThis.put = (node, ...kids) => node.kids.push(...kids.flat()); globalThis.field = (label, node) => el("div", {}, label, node); globalThis.bind = (o, k, n) => n;
globalThis.text = node => node == null || node === false ? "" : typeof node !== "object" ? String(node) : Array.isArray(node) ? node.map(text).join(" ") : text(node.kids);
globalThis.all = (node, tag, out = []) => { if (node && typeof node === "object") { if (node.tag === tag) out.push(node); (Array.isArray(node) ? node : node.kids || []).forEach(k => all(k, tag, out)); } return out; };
vm.runInThisContext(fs.readFileSync(process.argv[2], "utf8"));
vm.runInThisContext(`(async () => { const out = {};
  const five = { supported: true, limit: 8, available: 5, listed: 5, default: 1, this: { name: "head", state: "ready", version: "2.6.6", memory_gb: 110, disk_gb: 70, can: { fasth3: true, image: true, ref2va: true, ref2va_turbo: true, ffmpeg: true, loras: 22 } },
    workers: [{ address: "192.0.2.10", name: "spark-c", state: "ready", says: "idle", enabled: true, usable: true, guard: true, version: "2.6.6", memory_gb: 116.1, disk_gb: 40, can: { fasth3: true, loras: 0 }, why: "", working: "" },
              { address: "192.0.2.11", name: "spark-d", state: "down", says: "stopped answering", enabled: true, usable: false, unusable: "it stopped answering", why: "It stopped answering for 60 s while rendering", working: "", version: "2.6.6" }] };
  out.counts = [spkCount(five, 1), spkCount(five, 5), spkCount(five, 8), spkCount({ supported: false }, 1)];
  answers["/api/sparks"] = five; answers["/api/sparks/plan"] = { supported: true, count: "5 of 8 available", says: "One clip per Spark, 5 at a time: 8 clips on 5 Sparks.", mode: "parallel", without: [{ name: "spark-d", why: "it stopped answering" }], estimate: { seconds: 80, one_spark: 320, faster: 4, note: "an estimate" } };
  await spkLoad(); spkSet(8);
  const pick = sparksPick({ kind: "video", clips: 8 }); await new Promise(r => setTimeout(r, 20));
  const buttons = all(pick, "button").filter(b => /^[1-8]$/.test(text(b)));
  out.buttons = buttons.map(b => text(b)); out.on = buttons.filter(b => b.attrs.class.includes(" on")).map(b => text(b)); out.far = buttons.filter(b => b.attrs.class.includes(" far")).map(b => text(b));
  out.says = text(pick); out.sent = calls.filter(c => c[0] === "/api/sparks/plan").map(c => c[1])[0]; out.kept = calls.some(c => c[0] === "/api/sparks" && c[1] && c[1].sparks === 8);
  tab = "Sparks"; out.page = text(viewSparks());
  SPK.data = { supported: false, limit: 8, available: 1, note: "Multi-Spark rendering is for DGX Sparks: on a Mac every job renders on this machine.", workers: [], this: {} }; spkSet(5);
  const mac = sparksPick(); out.mac = [SPK.n, all(mac, "button").map(b => text(b)), text(mac)];
  console.log(JSON.stringify(out)); })().catch(e => { console.error(e); process.exit(1); });`);
"""


def test_the_page_draws_the_chooser_and_the_list():
    if not shutil.which("node"):
        return
    with tempfile.TemporaryDirectory() as scratch:
        (Path(scratch) / "harness.js").write_text(PAGE)
        done = subprocess.run(["node", str(Path(scratch) / "harness.js"), str(ROOT / "app/static/sparks.js")], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["counts"] == ["this Spark alone", "5 of 5 available Sparks", "5 of 8 available", "this machine"]
    assert out["buttons"] == list("12345678") and out["on"] == ["8"] and out["far"] == ["6", "7", "8"]
    assert "5 of 8 available" in out["says"] and "One clip per Spark" in out["says"] and "spark-d (it stopped answering)" in out["says"] and "Estimate" in out["says"]
    assert out["sent"] == {"sparks": 8, "kind": "video", "params": {}, "clips": 8, "seconds": 0} and out["kept"]
    for words in ("spark-c", "192.0.2.10", "116.1 GB of memory available", "FastH3 clips", "no LoRAs", "It stopped answering for 60 s", "Update and start", "Add a Spark", "5 available"):
        assert words in out["page"], words
    assert out["mac"][0] == 1 and out["mac"][1] == ["1"] and "DGX Sparks" in out["mac"][2]
    # every job the page queues carries the number
    page = "".join(path.read_text() for path in sorted((ROOT / "app/static").glob("*.js")))  # the page and its tab files
    assert 'body = { sparks: SPK.n, ...body }' in page and '"Sparks"' in page and "Sparks: viewSparks" in page


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
