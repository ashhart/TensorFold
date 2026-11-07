"""Clip lengths (app/lengths.py) and what a machine without ffmpeg is told; no GPU, network or ffmpeg needed."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
sys.path.insert(0, str(ROOT / "app"))
import lengths as L  # noqa: E402
import server  # noqa: E402
import engines  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
PRESET = "small" if server.SPARK else "draft"
QUALITY = "fh8"


def refused(call) -> str:
    try:
        call()
    except (HTTPException, ValueError) as error:
        return str(getattr(error, "detail", error))
    raise AssertionError("it was not refused")


def test_every_count_the_model_takes():
    counts = L.counts()
    assert counts[0] == 22 and counts[-1] == 362 and len(counts) == 21 and all((n - 5) % 17 == 0 for n in counts)
    # the four lengths the app has always offered are the same four counts
    assert [L.nearest(seconds) for seconds in (5, 8, 10, 15)] == [124, 192, 243, 362] == list(L.CLASSIC.values())
    assert L.nearest(6) == 141 and L.nearest(0.2) == 22 and L.nearest(99) == 362
    assert L.says(243) == "243 frames, 10.125 s" and L.says(192) == "192 frames, 8 s" and L.says(22) == "22 frames, 0.917 s"


def test_a_job_asks_by_frames_or_by_seconds():
    assert L.of({"frames": 209}) == 209 and L.of({"seconds": 10}) == 243 and L.of({"seconds": 8.7}) == 209 and L.of({}) == 124
    assert L.of({"frames": 209, "seconds": 5}) == 209  # frames is the exact one
    assert "the nearest is 226 or 243" in refused(lambda: L.of({"frames": 240}))
    assert "0.92 to 15.08 seconds" in refused(lambda: L.of({"seconds": 30}))
    assert "at most 243 frames" in refused(lambda: L.of({"frames": 362}, longest=243))


def test_the_job_gets_the_frames_it_asked_for():
    _, env, _ = server.build("video", {"prompt": "a harbour", "preset": PRESET, "quality": QUALITY, "frames": 209}, "j1")
    assert env["FRAMES"] == 209
    _, env, _ = server.build("video", {"prompt": "a harbour", "preset": PRESET, "quality": QUALITY, "seconds": 10}, "j2")
    assert env["FRAMES"] == 243
    _, env, _ = server.build("video", {"prompt": "a harbour", "preset": PRESET, "quality": QUALITY, "seconds": 12.5}, "j3")
    assert env["FRAMES"] == 294
    assert "17n + 5" in refused(lambda: server.build("video", {"prompt": "x", "preset": PRESET, "quality": QUALITY, "frames": 200}, "j4"))
    reply = client.post("/api/jobs", json={"kind": "video", "params": {"prompt": "x", "preset": PRESET, "quality": QUALITY, "frames": 200}})
    assert reply.status_code == 400 and "nearest is 192 or 209" in reply.json()["error"]


def test_the_page_is_told_the_lengths_and_the_prompt_the_true_end():
    state = client.get("/api/state").json()
    assert state["lengths"]["shortest"] == 22 and state["lengths"]["longest"] == 362 and state["lengths"]["step"] == 17 and state["lengths"]["fps"] == 24
    assert set(state["lengths"]["presets"]) == set(state["presets"]) and state["durations"] == {"5": 124, "8": 192, "10": 243, "15": 362}
    # a last frame stands at the clip's true last moment: 209 frames are 8.71 seconds
    assert "at 8.71 seconds" in server.lead(True, True, frames=209) and "at 10.12 seconds" in server.lead(True, True, 10)
    reply = client.post("/api/compose", json={"fields": {"scene": "a hall"}, "first_frame": True, "last_frame": True, "frames": 141}).json()
    assert "at 5.88 seconds" in reply["prompt"]


def test_the_time_estimate_of_the_usual_lengths_is_what_it_was():
    for seconds in (5, 8, 10, 15):
        assert server.clip_seconds({"seconds": seconds}) == seconds
    between = server.clip_seconds({"frames": 209})
    assert 8 < between < 10 and server.clip_seconds({"total_seconds": 60}) == 60
    assert client.post("/api/estimate", json={"preset": PRESET, "quality": QUALITY, "frames": 209}).status_code == 200


def test_a_machine_without_ffmpeg_is_told_before_the_job_is_queued():
    had = engines.media_tools
    try:
        engines.media_tools = lambda: {"ffmpeg": False, "ffprobe": False}
        assert server.without_ffmpeg("video", {"prompt": "x"}) == ""  # a single clip renders without it
        assert "long video" in server.without_ffmpeg("long", {}) and "timeline" in server.without_ffmpeg("export", {})
        assert "continues from" in server.without_ffmpeg("video", {"first_frame_from": "renders/a.mp4"})
        assert "reference videos" in server.without_ffmpeg("video", {"reference_videos": ["uploads/a.mp4"]})
        reply = client.post("/api/jobs", json={"kind": "long", "params": {"prompt": "x", "total_seconds": 30, "quality": QUALITY, "preset": PRESET}})
        assert reply.status_code == 400 and "needs ffmpeg" in reply.json()["error"] and "install ffmpeg" in reply.json()["error"]
        state = client.get("/api/state").json()
        assert state["ffmpeg"] is False and "ffmpeg is not installed" in state["ffmpeg_note"]
        assert client.post("/api/frame", json={"path": "renders/a.mp4"}).status_code == 400
    finally:
        engines.media_tools = had
    assert server.without_ffmpeg("long", {}) == "" or not had()["ffmpeg"]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
