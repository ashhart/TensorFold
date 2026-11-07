"""Voice, Fish Audio Drama 3, CUDA engine (app/voice_engines.py Cuda): a stand-in server script, no GPU or model."""
import json
import os
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
os.environ["STUDIO_GPU_LOCK"] = str(Path(tempfile.mkdtemp()) / "studio-gpu.lock.d")
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402
import job_run  # noqa: E402
import voice  # noqa: E402
import voice_machine  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
voice_machine.LOCK = Path(os.environ["STUDIO_GPU_LOCK"])


def wait(job_id: str, timeout: float = 30) -> dict:
    started = time.time()
    while time.time() - started < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.1)
    raise AssertionError("the job did not finish")


SERVE = r"""
import json, os, struct, sys, time, wave
args = sys.argv[1:]
assert args[args.index("--serve") + 1] == "1" and "--model" in args and "--codec" in args
log = open(os.environ["FAKE_DRAMA3_LOG"], "a")
print("info: drama3: weights on the device: 38.58 GB", file=sys.stderr, flush=True)
print("not json {", file=sys.stderr, flush=True)
print(json.dumps({"ready": True}), file=sys.stderr, flush=True)
for line in sys.stdin:
    request = json.loads(line)
    log.write(line); log.flush()
    if "FAIL" in request["text"]:
        print(json.dumps({"ok": False, "error": "TextTooLong"}), file=sys.stderr, flush=True)
        continue
    if "CRASH" in request["text"]:
        sys.exit(3)
    samples = struct.pack("<h", 100) * 44100
    if request.get("pcm"):
        with open(request["pcm"], "wb") as pcm:
            for i in range(4):
                pcm.write(samples[i * 22050:(i + 1) * 22050]); pcm.flush(); time.sleep(0.05)
    with wave.open(request["wav"], "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(samples)
    print("info: drama3: a log line between answers", file=sys.stderr, flush=True)
    print(json.dumps({"ok": True, "frames": 21, "eos": True, "audio_s": 1.0, "prompt_rows": 30, "ref_frames": 40 if request.get("ref_audio") else 0,
                      "encode_ms": 100.0 if request.get("ref_audio") else 0.0, "prefill_ms": 110.0, "first_audio_ms": 250.0,
                      "total_ms": 700.0, "fps": 30.0, "wav": request["wav"]}), file=sys.stderr, flush=True)
"""


def fake_cuda(folder: Path) -> "voice.Cuda":
    """A Cuda backend whose engine is a stand-in speaking the same protocol (--serve 1, JSON lines)."""

    engine = folder / "bin" / "drama3-generate"
    engine.parent.mkdir(parents=True)
    engine.write_text(f"#!{sys.executable}\n" + SERVE)
    engine.chmod(0o755)
    (folder / "lib").mkdir()
    (folder / "lib" / "libtensorfold_drama3.so").write_bytes(b"")
    model = folder / "drama-3"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text("{}")
    (folder / "codec-v1.safetensors").write_bytes(b"")
    os.environ.update(DRAMA3_CUDA_BIN=str(engine), DRAMA3_CUDA_MODEL=str(model), DRAMA3_CUDA_CODEC=str(folder / "codec-v1.safetensors"),
                      FAKE_DRAMA3_LOG=str(folder / "requests.jsonl"))
    try:
        one = voice.Cuda()
    finally:
        for key in ("DRAMA3_CUDA_BIN", "DRAMA3_CUDA_MODEL", "DRAMA3_CUDA_CODEC"):
            os.environ.pop(key)
    one.NEED_GB = 0  # the machine's memory is not this test's business
    return one


def requests_of(folder: Path) -> list[dict]:
    file = folder / "requests.jsonl"
    return [json.loads(line) for line in file.read_text().splitlines()] if file.is_file() else []


def test_the_cuda_backend_says_what_is_missing():
    folder = Path(tempfile.mkdtemp())
    one = fake_cuda(folder)
    assert one.check() == ""
    assert one.argv() == [str(one.bin), "--model", str(one.model), "--codec", str(one.codec), "--serve", "1"]
    (folder / "codec-v1.safetensors").unlink()
    one.known = None
    assert "codec_convert.py" in one.check() and "DRAMA3_CUDA_CODEC" in one.check()
    one.known, one.model = None, folder / "nowhere"
    assert "No Drama 3 checkpoint" in one.check()
    one.known, one.bin = None, folder / "nothing"
    why = one.check()
    assert "drama3-cuda" in why and "DRAMA3_CUDA_BIN" in why
    had = voice._backend
    try:
        voice._backend = one
        state = client.get("/api/state").json()["voice"]
        assert state["available"] is False and state["backend"] == "cuda" and state["credit"] == "Built with Fish Audio"
        assert state["server"]["loaded"] is False and state["live"] is True
        answer = client.post("/api/voice/takes", json={"text": "Hello."})
        assert answer.status_code == 400 and "DRAMA3_CUDA_BIN" in answer.json()["error"]
    finally:
        voice._backend = had


def test_a_cuda_take_runs_on_a_persistent_server_that_holds_the_lock_until_unloaded():
    folder = Path(tempfile.mkdtemp())
    one = fake_cuda(folder)
    had = voice._backend
    try:
        voice._backend = one
        job = client.post("/api/voice/takes", json={"text": "Hello there.", "seed": 5, "top_k": 25, "asr": False}).json()
        done = wait(job["id"])
        assert done["status"] == "done", done
        take = done["voice"]
        assert take["backend"] == "cuda" and take["duration"] == 1.0 and take["frames"] == 21 and take["frames_per_s"] == 30.0
        assert take["first_audio_ms"] == 250.0 and take["load_s"] is not None and take["device_gb"] == 38.58 and take["chunks"] == 1
        sent = requests_of(folder)[-1]
        assert sent["text"] == "Hello there." and sent["seed"] == 5 and sent["top_k"] == 25 and sent["temperature"] == 1.0
        assert sent["top_p"] == 0.9 and sent["max_frames"] == 1500 and sent["wav"].endswith(f"{job['id']}.wav") and "ref_audio" not in sent
        assert (server.HOME / take["path"]).is_file()
        # loaded after the take, holding the GPU lock: the next take does not load again
        assert one.loaded() and voice.lock_owner().startswith("studio-voice drama3")
        assert client.get("/api/voice/state").json()["server"]["loaded"] is True
        pid = one.process.pid
        job = client.post("/api/voice/takes", json={"text": "<|speaker:0|> Again. <|speaker:1|> Yes.", "asr": False}).json()
        take = wait(job["id"])["voice"]
        assert one.process.pid == pid and take["load_s"] is None and take["streamed"] is True
        assert requests_of(folder)[-1]["text"] == "<|speaker:0|> Again.\n<|speaker:1|> Yes."
        # a reference clip is sent as WAV with its text
        clip = server.HOME / "uploads" / "cuda-ref.wav"
        with wave.open(str(clip), "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(22050); w.writeframes(struct.pack("<hh", 0, 0) * 22050 * 3)
        job = client.post("/api/voice/takes", json={"text": "Cloned.", "ref_audio": "uploads/cuda-ref.wav", "ref_text": "the words", "consent": True, "asr": False}).json()
        assert wait(job["id"])["status"] == "done"
        sent = requests_of(folder)[-1]
        assert sent["ref_text"] == "the words" and sent["ref_audio"].endswith(".ref.wav") and Path(sent["ref_audio"]).is_file()
        if voice.tool("ffmpeg"):
            with wave.open(sent["ref_audio"]) as w:
                assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (1, 44100, 2)
        # an engine error fails the take and keeps the server
        job = client.post("/api/voice/takes", json={"text": "Please FAIL.", "asr": False}).json()
        failed = wait(job["id"])
        assert failed["status"] == "failed" and "TextTooLong" in failed["error"] and one.loaded()
        # any other job of this machine's queue unloads it first, and the lock goes back
        assert voice.release("a video job needs the GPU") is True
        assert not one.loaded() and not voice_machine.LOCK.exists() and one.events[-1]["why"] == "a video job needs the GPU"
        assert voice.release("again") is False
        # an engine that dies mid-take fails the take and gives the lock back
        job = client.post("/api/voice/takes", json={"text": "CRASH now.", "asr": False}).json()
        failed = wait(job["id"])
        assert failed["status"] == "failed" and "stopped" in failed["error"], failed
        assert not one.loaded() and not voice_machine.LOCK.exists()
    finally:
        one.release("test over")
        voice._backend = had


def test_a_long_cuda_take_keeps_one_voice_plays_while_made_and_unloads_when_idle():
    folder = Path(tempfile.mkdtemp())
    one = fake_cuda(folder)
    one.CHUNK_WORDS, one.idle_s = 4, 1
    had = voice._backend
    try:
        voice._backend = one
        job = client.post("/api/voice/takes", json={"text": "One two three. Four five six. Seven eight nine.", "seed": 9, "asr": False}).json()
        live = client.get(f"/api/voice/takes/{job['id']}/live")
        assert live.status_code == 200 and live.headers["content-type"] == "audio/wav"
        body = live.content
        assert body[:4] == b"RIFF" and body[8:16] == b"WAVEfmt " and body[36:40] == b"data"
        gap = int(44100 * one.GAP_S) * 2
        assert len(body) == 44 + 3 * 88200 + 2 * gap, len(body)
        take = wait(job["id"])["voice"]
        assert take["chunks"] == 3 and abs(take["duration"] - (3 + 2 * one.GAP_S)) < 1e-6
        with wave.open(str(server.HOME / take["path"])) as w:
            assert w.getnframes() == 3 * 44100 + 2 * int(44100 * one.GAP_S)
        sent = requests_of(folder)[-3:]
        assert [r["seed"] for r in sent] == [9, 10, 11] and "ref_audio" not in sent[0]
        assert sent[1]["ref_audio"] == sent[2]["ref_audio"] == sent[0]["wav"] and sent[1]["ref_text"] == "One two three."
        assert not list((server.HOME / "voice").glob(f"{job['id']}.part*.wav"))
        # five idle polls at most: the idle thread unloads it and gives the lock back
        started = time.time()
        while (one.loaded() or voice_machine.LOCK.exists()) and time.time() - started < 20:
            time.sleep(0.2)
        assert not one.loaded() and not voice_machine.LOCK.exists() and "idle" in one.events[-1]["why"]
    finally:
        one.release("test over")
        voice._backend = had


def test_an_h3_job_unloads_drama3_before_it_runs():
    folder = Path(tempfile.mkdtemp())
    one = fake_cuda(folder)
    had, built = voice._backend, job_run.build
    try:
        voice._backend = one
        assert wait(client.post("/api/voice/takes", json={"text": "Loaded.", "asr": False}).json()["id"])["status"] == "done"
        assert one.loaded() and voice_machine.LOCK.exists()
        seen = {}

        def build(kind, params, job_id, room=True):  # a stand-in render: what it sees of the lock and the server
            seen.update(loaded=one.loaded(), locked=voice_machine.LOCK.exists())
            out = server.HOME / "renders" / f"{job_id}.png"
            return [sys.executable, "-c", f"open({str(out)!r}, 'wb').write(b'png')"], {}, [str(out)]

        # while it is loaded a clip is measured against the memory the machine had before it loaded
        assert voice.room_mb() == int(one.before_gb * 1024)
        if server.SPARK:
            had_room = server.memory_guard.room_mb
            try:
                server.memory_guard.room_mb = lambda: 1000
                assert server.memory_of({}, {"WIDTH": 864, "HEIGHT": 480}, 121, False)["available_mb"] == voice.room_mb()
            finally:
                server.memory_guard.room_mb = had_room
        job_run.build = build
        job = {"id": "test-h3-yield", "kind": "image", "status": "running", "title": "t", "params": {}, "log": "", "outputs": [],
               "started": time.time(), "finished": None, "remote": None}
        server.perform(job)
        assert job["status"] == "done", job
        assert seen == {"loaded": False, "locked": False} and "Drama 3 was unloaded" in job["log"]
        assert voice.room_mb() is None
    finally:
        job_run.build = built
        one.release("test over")
        voice._backend = had


def test_a_cuda_take_with_missing_words_is_drawn_again_not_played_and_the_best_kept():
    folder = Path(tempfile.mkdtemp())
    one = fake_cuda(folder)
    had, had_ready, had_transcribe = voice._backend, voice.qa.ready, voice.qa.transcribe
    text = "<|speaker:0|> Did you hear that noise downstairs? <|speaker:1|> It's probably just the cat. <|speaker:0|> Cats don't knock on doors."
    full, short = "Did you hear that noise downstairs? It's probably just the cat. Cats don't knock on doors.", "Did you hear that noise downstairs? It's probably just the cat."
    try:
        voice._backend = one
        voice.qa.ready = lambda: {"transcript": True, "runs": True, "missing": []}
        voice.qa.transcribe = lambda clip: full if ".draw2" in str(clip) else short  # the first two draws drop the last turn
        job = client.post("/api/voice/takes", json={"text": text, "seed": 7}).json()
        done = wait(job["id"])
        take = done["voice"]
        sent = requests_of(folder)[-3:]
        assert [r["seed"] for r in sent] == [7, 1007, 2007] and "pcm" in sent[0] and "pcm" not in sent[1] and "pcm" not in sent[2]
        assert take["verify"][0]["action"] == "ok" and take["verify"][0]["attempts"] == 3 and take["asr"] == {**take["asr"], "score": 100, "by": "whisper"}
        assert take["retaken"]["played"][0]["action"] == "redraw" and take["retaken"]["played"][0]["missing"] == ["cats", "don't", "knock", "on", "doors"]
        assert take["streamed"] is False and (server.HOME / take["path"]).is_file() and not list((server.HOME / "voice").glob(f"{job['id']}.draw*"))
        assert "words missing" in done["log"]
        # every draw short: the best one is kept after two draws again, and said
        voice.qa.transcribe = lambda clip: short
        take = wait(client.post("/api/voice/takes", json={"text": text, "seed": 1}).json()["id"])["voice"]
        assert take["verify"][0]["action"] == "redraw" and take["verify"][0]["attempts"] == 3 and take["asr"]["score"] < 100
        # the whole text heard at once: no draw again
        voice.qa.transcribe = lambda clip: full
        count = len(requests_of(folder))
        take = wait(client.post("/api/voice/takes", json={"text": text, "seed": 2}).json()["id"])["voice"]
        assert len(requests_of(folder)) == count + 1 and take["retaken"] is None and take["streamed"] is True and take["verify"][0]["attempts"] == 1
    finally:
        one.release("test over")
        voice._backend, voice.qa.ready, voice.qa.transcribe = had, had_ready, had_transcribe



if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
