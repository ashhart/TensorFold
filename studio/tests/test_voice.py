"""Voice, Fish Audio Drama 3 (app/voice.py), text and MLX: no GPU, model or network (engines are stand-in scripts)."""
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
import slate  # noqa: E402
import voice  # noqa: E402
import voice_machine  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
voice_machine.LOCK = Path(os.environ["STUDIO_GPU_LOCK"])


FAKE = r'''
import json, sys, wave, struct
args = sys.argv[1:]
out, stats = args[args.index("--out") + 1], args[args.index("--stats") + 1]
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(struct.pack("<h", 0) * 44100 * 3)
json.dump({"frames": 65, "audio_s": 3.0, "decode_s": 1.0, "frames_per_s": 63.0, "rtf": 0.4, "total_s": 1.2,
           "peak_memory_gb": 48.0, "chunks": 1, "finish": ["stop"]}, open(stats, "w"))
print("[drama3] wrote", out, "seed", args[args.index("--seed") + 1], "lock", __import__("os").path.isdir(sys.argv[0] + ".lock"))
'''


class Fake:
    name, label = "mlx", "stand-in"

    def __init__(self, folder: Path):
        self.script = folder / "fake_drama3.py"
        self.script.write_text(FAKE)
        self.seen = []

    def check(self) -> str:
        return ""

    def command(self, text_file, out, stats, reference, seed, sampling):
        self.seen.append({"text": text_file.read_text(), "reference": reference, "seed": seed, "sampling": sampling,
                          "locked": voice_machine.LOCK.is_dir()})
        return [sys.executable, str(self.script), "--out", str(out), "--stats", str(stats), "--seed", str(seed)]


def wait(job_id: str, timeout: float = 30) -> dict:
    started = time.time()
    while time.time() - started < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.1)
    raise AssertionError("the job did not finish")


def test_a_sheet_becomes_a_direction_and_its_sample_line():
    sheet = {"name": "Old Tom", "ageGender": "a man in his late fifties", "accent": "a trace of Lisbon", "timbre": "gravelly",
             "pitch": "low", "pacing": "slow, patient", "energy": "", "texture": "smoke [and gravel]", "sampleLine": "Sit down.  Tell me."}
    text = voice.sheet_text(sheet)
    assert text == "[the voice of a man in his late fifties, a trace of Lisbon, gravelly, low, slow, patient, smoke (and gravel)] Sit down. Tell me."
    assert voice.spoken(text) == "Sit down. Tell me."
    assert voice.sheet_text({}) == voice.SAMPLE_LINE  # no description: the line alone, never an empty direction


def test_the_word_check_leaves_out_tags_and_directions():
    got = voice.match("<|speaker:0|> [whispering] Did you hear that? <|speaker:1|> No.", "Did you hear that? No.")
    assert got["score"] == 100 and got["found"] == 5 and got["words"] == 5
    assert voice.match("one two three four", "one two")["score"] < 100



def test_long_text_goes_in_parts_at_sentence_ends_with_its_speaker():
    short = "<|speaker:0|> Hello there. <|speaker:1|> Hi."
    assert voice.pieces(short, 60) == [short]
    text = "<|speaker:0|> One two three. Four five six. <|speaker:1|> [sighs] Seven eight nine ten. Eleven."
    parts = voice.pieces(text, 6)
    assert parts == ["<|speaker:0|> One two three. Four five six.", "<|speaker:1|> [sighs] Seven eight nine ten. Eleven."], parts
    parts = voice.pieces("<|speaker:1|> " + " ".join(f"Line {n} is here." for n in range(10)), 8)
    assert len(parts) == 5 and all(part.startswith("<|speaker:1|> Line") for part in parts)  # the speaker is said again
    assert voice.pieces("No tags. " * 30, 20)[1].startswith("No tags.")
    assert voice.one_speaker("<|speaker:0|> a <|speaker:0|> b") and voice.one_speaker("plain") and not voice.one_speaker("<|speaker:2|> a")



def test_a_take_is_checked_before_it_is_queued():
    had = voice._backend
    try:
        voice._backend = Fake(Path(tempfile.mkdtemp()))
        for body, says in (({"text": "  [sighs] "}, "no words"), ({"text": "<|speaker:16|> Hi"}, "speaker"),
                           ({"text": "Hi", "temperature": 3}, "temperature"), ({"text": "x" * (voice.TEXT_LIMIT + 1)}, "characters")):
            answer = client.post("/api/voice/takes", json=body)
            assert answer.status_code == 400 and says in answer.json()["error"], (body, answer.json())
        clip = server.HOME / "uploads" / "voice-ref.wav"
        with wave.open(str(clip), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(struct.pack("<h", 0) * 16000 * 4)
        answer = client.post("/api/voice/takes", json={"text": "Hi there.", "ref_audio": "uploads/voice-ref.wav", "ref_text": "hello"})
        assert answer.status_code == 400 and "consent" in answer.json()["error"]
    finally:
        voice._backend = had


def test_a_take_runs_under_the_gpu_lock_and_is_listed():
    had = voice._backend
    fake = Fake(Path(tempfile.mkdtemp()))
    try:
        voice._backend = fake
        job = client.post("/api/voice/takes", json={"text": "<|speaker:0|> Hello there. <|speaker:1|> Hi.", "seed": 7, "top_k": 20, "asr": False}).json()
        assert job["kind"] == "voice" and job["status"] in ("queued", "running")
        done = wait(job["id"])
        assert done["status"] == "done", done
        assert fake.seen[-1]["locked"] is False  # the command is built before the lock is taken ...
        assert not voice_machine.LOCK.exists()  # ... and the lock is given back after the run
        assert fake.seen[-1]["seed"] == 7 and fake.seen[-1]["sampling"] == {"temperature": 1.0, "top_p": 0.9, "top_k": 20}
        take = done["voice"]
        assert take["duration"] == 3.0 and take["frames_per_s"] == 63.0 and take["path"] == f"voice/{job['id']}.wav"
        listed = client.get("/api/voice/takes").json()
        assert listed[0]["id"] == job["id"]
        audio = client.get(f"/api/voice/takes/{job['id']}/audio")
        assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
        # a lock held by someone else: the take waits, and a cancelled take stops waiting
        voice_machine.LOCK.mkdir()
        (voice_machine.LOCK / "owner").write_text(f"trackc test {os.getpid()}\n")
        job = client.post("/api/voice/takes", json={"text": "Waiting.", "asr": False}).json()
        time.sleep(0.5)
        assert "GPU lock" in (client.get(f"/api/jobs/{job['id']}").json()["progress"] or {}).get("stage", "")
        client.post(f"/api/jobs/{job['id']}/cancel", json={})
        assert wait(job["id"])["status"] == "cancelled" and (voice_machine.LOCK / "owner").read_text().startswith("trackc")
        # a lock whose owner is gone is stale and taken
        (voice_machine.LOCK / "owner").write_text("trackc test 999999999\n")
        assert wait(client.post("/api/voice/takes", json={"text": "Stale.", "asr": False}).json()["id"])["status"] == "done"
        assert not voice_machine.LOCK.exists()
    finally:
        voice._backend = had
        if voice_machine.LOCK.exists():
            import shutil
            shutil.rmtree(voice_machine.LOCK)


def test_a_cast_voice_goes_into_a_shot_on_cast_stills_as_reference_audio():
    home = slate.HOME
    for name in ("leo.png", "mia.png"):
        (home / "uploads" / name).write_bytes(b"png")
    (home / "voice").mkdir(exist_ok=True)
    (home / "voice" / "tom.wav").write_bytes(b"wav")
    (home / "voice" / "long.wav").write_bytes(b"wav")
    project = {"id": "p-v", "name": "V", "defaults": {"durationSec": 8}, "tf": {"res": "480p", "quality": "fh8"},
               "characters": [{"id": "c1", "name": "Leo", "images": ["uploads/leo.png"]}, {"id": "c2", "name": "Mia", "images": ["uploads/mia.png"]}],
               "voices": [{"id": "v1", "name": "Leo voice", "characterId": "c1", "ageGender": "a young man", "timbre": "warm",
                           "drama": {"path": "voice/tom.wav", "seconds": 6.2, "use": True}},
                          {"id": "v2", "name": "Mia voice", "characterId": "c2", "ageGender": "a young woman", "drama": {"path": "voice/long.wav", "seconds": 18.0}}],
               "scenes": [{"id": "s1", "name": "S", "shots": [{"id": "sh", "name": "1", "spec": {"durationSec": 8}, "prompt": "# Subject\nThey talk.", "takes": [],
                                                               "tf": {"start": "cast", "cast": ["c1", "c2"], "lines": [{"characterId": "c2", "text": "Hi Leo."}, {"characterId": "c1", "text": "Hello."}]}}]}]}
    had = slate.ready
    try:
        slate.ready = lambda: {"ref2va": True, "ref2va_turbo": False}
        made = slate.compile_shot(project, "sh")
        assert made["plan"]["reference_audios"] == ["voice/tom.wav"]  # Mia's 18 s take is left out, and said
        assert any("18.0 s" in w for w in made["warnings"])
        assert "<Audio 1> is the voice timbre reference for <Subject 1>'s voice, containing the voice of a young man (warm)." in made["prompt"]
        assert "references the delivery from <Audio 1>" in made["prompt"]
        project["voices"][0]["drama"]["use"] = False  # unticked: no reference audio
        assert "reference_audios" not in slate.compile_shot(project, "sh")["plan"]
        project["scenes"][0]["shots"][0]["tf"]["start"] = "text"  # not on cast stills: never audio alone
        project["voices"][0]["drama"]["use"] = True
        assert "reference_audios" not in slate.compile_shot(project, "sh")["plan"]
    finally:
        slate.ready = had


STREAMING = r"""
import json, struct, sys, time
args = sys.argv[1:]
out, stats = args[args.index("--out") + 1], args[args.index("--stats") + 1]
print("[drama3] loaded /fake/drama-3-mlx in 3.7s", flush=True)
samples = struct.pack("<h", 7) * 44100 * 2
with open(out, "wb") as w:
    w.write(b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 44100, 88200, 2, 16) + b"data" + struct.pack("<I", 0xFFFFFFFF - 36))
    w.flush()
    for i in range(8):
        w.write(samples[i * 22050:(i + 1) * 22050]); w.flush(); time.sleep(0.05 if "--stream" in args else 0)
    w.seek(4); w.write(struct.pack("<I", 36 + len(samples))); w.seek(40); w.write(struct.pack("<I", len(samples)))
short = "--stream" in args and "--short" in args
json.dump({"frames": 43, "audio_s": 2.0, "frames_per_s": 73.0, "rtf": 0.35, "total_s": 0.7, "first_audio_s": 0.234, "stream": "--stream" in args,
           "peak_memory_gb": 49.4, "chunks": 2, "finish": ["eos", "stop_rule"],
           "verify": [{"words": 5, "matched": 5, "missing": [], "action": "ok", "attempts": 1, "heard": "One two three four five.", "asr_s": 0.1, "coverage": 1.0},
                      {"words": 4, "matched": 3, "missing": ["cat"], "action": "redraw" if short else "trim", "attempts": 2, "cut_s": 1.9, "heard": "It is the", "asr_s": 0.1}]},
          open(stats, "w"))
print("[drama3] wrote", out, "first audio 0.234s, total 0.7s", flush=True)
"""


class FakeStreaming(Fake):
    """The MLX engine from drama3-mlx 4a8a982f on: --stream writes a growing WAV, its stats carry its own check."""

    streams = True

    def __init__(self, folder: Path):
        super().__init__(folder)
        self.script.write_text(STREAMING)

    def command(self, text_file, out, stats, reference, seed, sampling, live=False):
        command = super().command(text_file, out, stats, reference, seed, sampling)
        self.seen[-1]["live"] = live
        return command + (["--stream"] if live else []) + (["--short"] if "SHORT" in self.seen[-1]["text"] else [])


def test_an_mlx_take_plays_while_made_and_its_score_is_the_engines_check():
    had, had_ready, had_transcribe = voice._backend, voice.qa.ready, voice.qa.transcribe
    fake = FakeStreaming(Path(tempfile.mkdtemp()))
    try:
        voice._backend = fake
        voice.qa.ready = lambda: {"transcript": True, "runs": True, "missing": []}
        voice.qa.transcribe = lambda clip: (_ for _ in ()).throw(AssertionError("transcribed twice"))
        state = client.get("/api/voice/state").json()
        assert state["live"] is True and state["checks"] is True
        job = client.post("/api/voice/takes", json={"text": "One two three four five. It is the cat.", "seed": 3}).json()
        body = client.get(f"/api/voice/takes/{job['id']}/live").content
        assert body[:4] == b"RIFF" and len(body) == 44 + 2 * 88200, len(body)  # the engine's own header is not sent twice
        done = wait(job["id"])
        assert done["status"] == "done", done
        take = done["voice"]
        assert fake.seen[-1]["live"] is True and take["streamed"] is True
        assert take["first_audio_ms"] == 234.0 and take["load_s"] == 3.7 and take["frames_per_s"] == 73.0
        assert [v["action"] for v in take["verify"]] == ["ok", "trim"] and take["verify"][1]["missing"] == ["cat"]
        assert take["asr"]["by"] == "engine" and take["asr"]["found"] == 8 and take["asr"]["words"] == 9 and take["asr"]["score"] == 89
        assert take["asr"]["missing"] == ["cat"]
        with wave.open(str(server.HOME / take["path"])) as w:
            assert w.getnframes() == 2 * 44100
        # a part short while it played: made once more without playing, where the engine can redraw it
        job = client.post("/api/voice/takes", json={"text": "SHORT take. It is the cat."}).json()
        take = wait(job["id"])["voice"]
        assert [s["live"] for s in fake.seen[-2:]] == [True, False] and take["retaken"]["played"][1]["action"] == "redraw"
        assert take["verify"][1]["action"] == "trim" and take["streamed"] is False and "made again without playing" in wait(job["id"])["log"]
        # not played as it is made: no --stream, so the engine may draw a part again
        job = client.post("/api/voice/takes", json={"text": "Quiet.", "live": False}).json()
        assert wait(job["id"])["status"] == "done" and fake.seen[-1]["live"] is False
        assert voice.estimate("One two three.", False, True) < voice.estimate("One two three.", False, False) + 1  # no Whisper after
    finally:
        voice._backend, voice.qa.ready, voice.qa.transcribe = had, had_ready, had_transcribe


MLX_SERVE = r"""
import json, os, struct, sys, wave
args = sys.argv[1:]
assert args[:3] == ["-I", "-m", "tensorfold.drama3"] and args[-1] == "--serve" and "--model" in args
log = open(os.environ["FAKE_DRAMA3_LOG"], "a")
print("[drama3] loading", file=sys.stderr, flush=True)
print(json.dumps({"ready": True, "engine": "tensorfold-drama3-mlx", "load_s": 0.1, "verify": True, "stream": True}), file=sys.stderr, flush=True)
for line in sys.stdin:
    request = json.loads(line)
    log.write(line); log.flush()
    samples = struct.pack("<h", 9) * 44100
    if request.get("pcm"):
        with open(request["pcm"], "wb") as pcm:
            pcm.write(samples)
    with wave.open(request["wav"], "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(samples)
    short = "SHORT" in request["text"] and request.get("stream")
    verify = [{"words": 4, "matched": 2 if short else 4, "missing": ["the", "cat"] if short else [], "action": "redraw" if short else "ok",
               "attempts": 1 if short else 2, "heard": "It is", "asr_s": 0.1}]
    print(json.dumps({"ok": True, "id": request.get("id"), "frames": 21, "eos": True, "audio_s": 1.0, "first_audio_ms": 210.0, "total_ms": 400.0,
                      "fps": 80.0, "rtf": 0.4, "wav": request["wav"], "pcm": request.get("pcm"), "stream": bool(request.get("stream")),
                      "chunks": [{"finish": "stop"}], "finish": ["stop"], "stop_rule": [False], "verify": verify, "peak_memory_gb": 47.5,
                      **({"ref_texts": ["Hello there."], "ref_asr_ms": 90.0} if request.get("ref_audio") and not request.get("ref_text") else {})}),
          file=sys.stderr, flush=True)
"""


def fake_mlx_server(folder: Path) -> "voice.Mlx":
    """An Mlx backend whose venv python stands in for `python -m tensorfold.drama3 --serve` (drama3-mlx 92c09a8e)."""

    python = folder / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(f"#!{sys.executable}\n" + MLX_SERVE)
    python.chmod(0o755)
    os.environ.update(DRAMA3_VENV=str(folder / "venv"), FAKE_DRAMA3_LOG=str(folder / "requests.jsonl"))
    try:
        one = voice.Mlx()
    finally:
        os.environ.pop("DRAMA3_VENV")
    one.known, one.streams, one.serves, one.NEED_GB = (time.time(), ""), True, True, 0
    return one


def requests_of(folder: Path) -> list[dict]:
    file = folder / "requests.jsonl"
    return [json.loads(line) for line in file.read_text().splitlines()] if file.is_file() else []


def test_dialogue_turns_go_on_lines_of_their_own():
    assert voice.dialogue("Hello there.") == "Hello there."
    assert voice.dialogue("<|speaker:0|> Hi. <|speaker:0|> Again.") == "<|speaker:0|> Hi.\n<|speaker:0|> Again."
    text = "[a direction] <|speaker:0|> [whispering] Did you? <|speaker:1|> No.  <|speaker:0|>Yes.\n<|speaker:1|> Fine."
    assert voice.dialogue(text) == "[a direction] <|speaker:0|> [whispering] Did you?\n<|speaker:1|> No.\n<|speaker:0|>Yes.\n<|speaker:1|> Fine."
    assert voice.dialogue(voice.dialogue(text)) == voice.dialogue(text)


def test_an_mlx_take_runs_on_the_served_engine_plays_while_made_and_is_retaken_when_short():
    folder = Path(tempfile.mkdtemp())
    one = fake_mlx_server(folder)
    had, had_ready, had_transcribe = voice._backend, voice.qa.ready, voice.qa.transcribe
    try:
        voice._backend = one
        voice.qa.ready = lambda: {"transcript": True, "runs": True, "missing": []}
        voice.qa.transcribe = lambda clip: (_ for _ in ()).throw(AssertionError("transcribed twice"))
        state = client.get("/api/voice/state").json()
        assert state["live"] is True and state["server"]["loaded"] is False
        text = "<|speaker:0|> [whispering] Did you hear that? <|speaker:1|> It is the cat. " + "And then more words follow here. " * 20
        job = client.post("/api/voice/takes", json={"text": text, "seed": 4}).json()
        body = client.get(f"/api/voice/takes/{job['id']}/live").content
        assert body[:4] == b"RIFF" and len(body) == 44 + 88200, len(body)
        take = wait(job["id"])["voice"]
        sent = requests_of(folder)
        assert len(sent) == 1  # the engine chunks long text itself: one request
        assert sent[0]["text"].startswith("<|speaker:0|> [whispering] Did you hear that?\n<|speaker:1|> It is the cat.")
        assert sent[0]["stream"] is True and sent[0]["pcm"].endswith("0.pcm") and sent[0]["seed"] == 4 and sent[0]["id"] == job["id"]
        assert take["backend"] == "mlx" and take["streamed"] is True and take["first_audio_ms"] == 210.0 and take["load_s"] is not None
        assert take["frames_per_s"] == 80.0 and take["peak_memory_gb"] == 47.5 and take["asr"]["by"] == "engine" and take["asr"]["score"] == 100
        assert one.loaded() and voice.lock_owner().startswith("studio-voice drama3")
        # a part short while it played: asked again, not streamed, on the same server
        pid = one.process.pid
        job = client.post("/api/voice/takes", json={"text": "SHORT: it is the cat.", "ref_audio": None}).json()
        take = wait(job["id"])["voice"]
        again = requests_of(folder)[-2:]
        assert again[0]["stream"] is True and again[1]["stream"] is False and "pcm" not in again[1] and one.process.pid == pid
        assert take["retaken"]["played"][0]["action"] == "redraw" and take["verify"][0]["action"] == "ok" and take["streamed"] is False
        assert take["load_s"] is None
        # a clip to clone without its words: the engine hears it, the app's Whisper does not run first
        one.hears = True
        clip = server.HOME / "uploads" / "mlx-ref.wav"
        with wave.open(str(clip), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(struct.pack("<h", 0) * 44100 * 3)
        job = client.post("/api/voice/takes", json={"text": "Cloned.", "ref_audio": "uploads/mlx-ref.wav", "ref_text": "", "consent": True}).json()
        done = wait(job["id"])
        assert done["status"] == "done", done
        sent = requests_of(folder)[-1]
        assert sent["ref_text"] == "" and sent["ref_audio"].endswith(".ref.wav") and done["voice"]["reference"]["text"] == "Hello there."
        # "live": false: one request, not streamed
        wait(client.post("/api/voice/takes", json={"text": "Quiet.", "live": False}).json()["id"])
        assert requests_of(folder)[-1]["stream"] is False
        assert voice.release("a video job needs the GPU") and not one.loaded() and not voice_machine.LOCK.exists()
    finally:
        one.release("test over")
        voice._backend, voice.qa.ready, voice.qa.transcribe = had, had_ready, had_transcribe


def test_an_mlx_engine_without_serve_runs_once_a_take():
    fake = FakeStreaming(Path(tempfile.mkdtemp()))
    assert not hasattr(fake, "serving")  # the per-take path, as 2.7.6 (test_an_mlx_take_plays_while_made_...)
    one = voice.Mlx()
    one.known, one.streams, one.serves = (time.time(), ""), True, False
    assert one.serving() is False
    command = one.command(Path("t.txt"), Path("o.wav"), Path("s.json"), None, 1, {"temperature": 1.0, "top_p": 0.9, "top_k": 30}, live=True)
    assert "--stream" in command and "--serve" not in command



def test_the_word_check_is_the_engines():
    import wordcheck
    text = "<|speaker:0|> [sighs] It's probably just the cat. <|speaker:1|> Cats don't knock on doors."
    assert wordcheck.check_text(text, "It's probably just the cat. Cats don't knock on doors.")["action"] == "ok"
    assert wordcheck.check_text(text, "It's probably just the cat. Cats don't knock on the")["reason"] == "the last word is missing"
    assert wordcheck.check_text("It costs 41 dollars.", "It costs forty-one dollars.")["action"] == "ok"
    assert wordcheck.check_text("Built with TensorFold.", "Built with Tensor Fold.")["action"] == "ok"  # a near-exact merge
    assert wordcheck.check_text("Go to the cat.", "Go to the.")["action"] == "redraw"  # the last content word


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
