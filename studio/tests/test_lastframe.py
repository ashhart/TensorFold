"""A clip's last frame: the prompt's opening line, the job's settings and a frame taken from a clip (needs ffmpeg)."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HOME = Path(tempfile.mkdtemp()) / "TensorFoldStudio"
os.environ["STUDIO_HOME"] = str(HOME)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import server  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

for name in ("renders", "uploads"):
    (HOME / name).mkdir(parents=True, exist_ok=True)
client = TestClient(server.app)


def picture(name):
    (HOME / "uploads" / name).write_bytes(b"png")
    return f"uploads/{name}"


def test_lead():
    assert server.lead(True, False) == server.LEAD
    assert server.lead(False, False) == ""
    both = server.lead(True, True, 5)
    assert both.startswith(server.LEAD[:-1]) and "at 5.17 seconds" in both and "<Picture 2>" in both
    assert "<Picture 2>" not in server.lead(False, True, 8) and "at 8.00 seconds" in server.lead(False, True, 8)
    assert server.compose({"scene": "a table"}, True, True, 10).splitlines()[0].count("fully referenced") == 2
    assert server.compose({"scene": "a table"}, False).startswith("integrated_multimodal_description")


def test_build_carries_the_last_frame():
    first, last = picture("a.png"), picture("b.png")
    base = {"prompt": "x", "preset": next(iter(server.PRESETS)), "quality": next(iter(server.QUALITIES)), "seconds": 5}
    _, env, _ = server.build("video", {**base, "image": first, "last_image": last}, "t1")
    assert env["LAST_FRAME"] == str(HOME / last) and env["IMAGE_FILE"] == str(HOME / first)
    _, env, _ = server.build("video", {**base, "last_image": last}, "t2")  # a last frame alone
    assert env["LAST_FRAME"] == str(HOME / last) and "IMAGE_FILE" not in env
    _, env, _ = server.build("video", {**base, "image": first, "last_same": True}, "t3")  # the clip returns to its start
    assert env["LAST_FRAME"] == str(HOME / first)
    _, env, _ = server.build("video", {**base, "image": first}, "t4")
    assert "LAST_FRAME" not in env
    for bad in ({"last_image": "uploads/missing.png"}, {"last_image": "../outside.png"}):
        try:
            server.build("video", {**base, **bad}, "t5")
        except HTTPException as error:
            assert error.status_code in (400, 403)
        else:
            raise AssertionError(f"{bad} was accepted")


def test_long_videos_refuse_a_last_frame():
    reply = client.post("/api/jobs", json={"kind": "long", "params": {"prompt": "x", "total_seconds": 30, "last_image": picture("c.png")}})
    assert reply.status_code == 400 and "single clips" in reply.text


def test_frame_from_a_clip():
    if not shutil.which("ffmpeg"):
        return
    clip = HOME / "renders" / "bars.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24:duration=2", "-pix_fmt",
                    "yuv420p", str(clip)], check=True)
    for at in ("last", "first", 1.0):
        reply = client.post("/api/frame", json={"path": "renders/bars.mp4", "at": at})
        assert reply.status_code == 200, reply.text
        made = HOME / reply.json()["path"]
        assert made.is_file() and made.parent == HOME / "uploads" and made.stat().st_size > 500
    assert client.post("/api/frame", json={"path": "renders/bars.mp4", "at": 99}).status_code == 400
    assert client.post("/api/frame", json={"path": "uploads/a.png", "at": "last"}).status_code == 400
    assert client.post("/api/frame", json={"path": "../../etc/passwd", "at": "last"}).status_code in (400, 403)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
