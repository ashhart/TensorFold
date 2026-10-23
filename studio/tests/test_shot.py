"""app/shot.py on clips made here with PyAV and NumPy, thresholds checked against the production run's figures below."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import server  # noqa: E402
import job_run  # noqa: E402
import shot  # noqa: E402

FOLDER = Path(tempfile.mkdtemp())
SIZE = (320, 184)  # the run's 1280x736 at a quarter


def scene(seed: int, shift: int = 0) -> np.ndarray:
    """A picture with large shapes, moved ``shift`` pixels to the right: a slow pan when the shift grows."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE[1], 0:SIZE[0]]
    image = np.zeros((SIZE[1], SIZE[0], 3), np.float32)
    for _ in range(7):
        cx, cy, radius = rng.integers(0, SIZE[0]), rng.integers(0, SIZE[1]), rng.integers(20, 70)
        image += ((xx - cx - shift) ** 2 + (yy - cy) ** 2 < radius ** 2)[..., None] * rng.integers(40, 255, 3)
    return np.clip(image + 30, 0, 255).astype(np.uint8)


def clip(name: str, frames: list[np.ndarray]) -> Path:
    import av

    path = FOLDER / name
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = SIZE[0], SIZE[1], "yuv420p"
        stream.options = {"crf": "14"}
        for frame in frames:
            out.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        out.mux(stream.encode())
    return path


def still(name: str, frame: np.ndarray) -> Path:
    from PIL import Image

    Image.fromarray(frame).save(FOLDER / name)
    return FOLDER / name


def test_one_shot_that_holds_its_keyframes_passes():
    frames = [scene(1, shift) for shift in range(0, 48)]
    report = shot.check(clip("pan.mp4", frames), still("first.png", frames[0]), still("last.png", frames[-1]))
    assert report["passed"] and report["first"]["held"] and report["last"]["held"] and report["cuts"] == [] and report["frames"] == 48
    assert report["first"]["difference"] < 4 and report["first"]["likeness"] > 0.99  # the run's held frames: 1.4 to 3.9


def test_a_cut_inside_the_shot_is_found_where_it_is():
    frames = [scene(1, shift) for shift in range(20)] + [scene(2, shift) for shift in range(28)]
    report = shot.check(clip("cut.mp4", frames), still("first.png", frames[0]), still("last2.png", frames[-1]))
    assert not report["passed"] and [cut["frame"] for cut in report["cuts"]] == [20] and report["cuts"][0]["seconds"] == round(20 / 24, 2)
    assert "1 cut inside the shot" in report["note"] and report["first"]["held"] and report["last"]["held"]
    # a prompt that writes a second shot asks for the cut: it is not looked for
    assert shot.one_shot("[Shot 1] a hall.") and not shot.one_shot("[Shot 1] a hall. [Shot 2] a close-up.")
    assert shot.check(FOLDER / "cut.mp4", single=False)["cuts"] is None and shot.check(FOLDER / "cut.mp4", single=False)["passed"]


def test_the_longest_stretch_without_a_cut_is_given_to_trim_to():
    # the run's own case: an establishing shot nobody asked for, a cut at frame 20, then the shot
    frames = [scene(1, shift) for shift in range(20)] + [scene(2, shift) for shift in range(28)]
    report = shot.check(clip("opens.mp4", frames))
    assert report["window"] == {"start": 20, "end": 48, "frames": 28, "in": round(20 / 24, 3), "out": 2.0, "whole": False}
    assert "the longest stretch without one is frames 20 to 47 (1.17 s)" in report["note"]
    # a clip without a cut is its own window, and a clip asked for as two shots has none
    whole = shot.check(clip("whole.mp4", [scene(1, shift) for shift in range(30)]))
    assert whole["window"] == {"start": 0, "end": 30, "frames": 30, "in": 0.0, "out": 1.25, "whole": True}
    assert shot.check(FOLDER / "opens.mp4", single=False)["window"] is None
    # two cuts: the longest of the three stretches; of two equal ones the later
    assert shot.window(124, [{"frame": 14}, {"frame": 100}]) == {"start": 14, "end": 100, "frames": 86, "in": round(14 / 24, 3), "out": round(100 / 24, 3), "whole": False}
    assert shot.window(100, [{"frame": 50}])["start"] == 50
    assert shot.window(90, [])["whole"] and shot.window(90, None)["frames"] == 90
    assert shot.window(90, [{"frame": 0}, {"frame": 90}, {"frame": 400}])["whole"]  # a cut at an end is none
    # for a clip that has no job: the cuts are looked for
    from fastapi.testclient import TestClient

    shutil.copy(FOLDER / "opens.mp4", server.HOME / "renders" / "opens.mp4")
    told = TestClient(server.app).get("/api/shot/window", params={"path": "renders/opens.mp4"})
    assert told.status_code == 200 and told.json()["start"] == 20 and told.json()["in"] == round(20 / 24, 3)
    assert TestClient(server.app).get("/api/shot/window", params={"path": "renders/none.mp4"}).status_code == 404
    alone = shot.longest_take(FOLDER / "opens.mp4")
    assert alone["start"] == 20 and alone["end"] == 48 and alone["of"] == 48 and [cut["frame"] for cut in alone["cuts"]] == [20]


def test_fast_movement_and_a_fade_are_not_cuts():
    fast = [scene(3, 9 * shift) for shift in range(40)]  # the whole stretch moves: no frame stands out from its neighbours
    assert shot.check(clip("fast.mp4", fast))["cuts"] == []
    a, b = scene(4).astype(np.float32), scene(5).astype(np.float32)
    fade = [a.astype(np.uint8)] * 10 + [((1 - t) * a + t * b).astype(np.uint8) for t in np.linspace(0, 1, 24)] + [b.astype(np.uint8)] * 10
    assert shot.check(clip("fade.mp4", fade))["cuts"] == []


def test_a_keyframe_that_is_not_held_is_said():
    frames = [scene(1, shift) for shift in range(30)]
    report = shot.check(clip("pan30.mp4", frames), still("other.png", scene(9)), still("last30.png", frames[-1]))
    assert not report["passed"] and not report["first"]["held"] and report["last"]["held"]
    assert report["note"] == "the clip does not open on its first frame"
    report = shot.check(FOLDER / "pan30.mp4", still("first30.png", frames[0]), still("other.png", scene(9)))
    assert report["note"] == "the clip does not end on its last frame"


def test_a_keyframe_is_compared_as_the_renderer_fitted_it():
    from PIL import Image

    frames = [scene(1, shift) for shift in range(24)]
    # the 720p preset: generated at 736 rows, 8 cropped away top and bottom (here 184 -> 180)
    cropped = [frame[2:-2] for frame in frames]
    import av

    path = FOLDER / "crop.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt, stream.options = SIZE[0], SIZE[1] - 4, "yuv420p", {"crf": "14"}
        for frame in cropped:
            out.mux(stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")))
        out.mux(stream.encode())
    assert shot.check(path, still("first.png", frames[0]), None, canvas=SIZE)["first"]["held"]
    # a last frame of another shape: centre-cropped beside a first frame; alone, stretched on a Mac, cropped by ComfyUI
    tall = Image.new("RGB", (SIZE[0], SIZE[1] * 2))
    tall.paste(Image.fromarray(frames[-1]), (0, SIZE[1] // 2))  # the middle half is the frame
    tall.save(FOLDER / "tall.png")
    Image.fromarray(frames[-1]).resize((SIZE[0], SIZE[1] * 2)).save(FOLDER / "stretched.png")
    whole = clip("pan24.mp4", frames)
    assert shot.check(whole, FOLDER / "first.png", FOLDER / "tall.png")["last"]["held"]
    assert shot.check(whole, None, FOLDER / "stretched.png")["last"]["held"]
    assert shot.check(whole, None, FOLDER / "tall.png", alone_stretched=False)["last"]["held"]
    assert not shot.check(whole, None, FOLDER / "tall.png")["last"]["held"]


def test_the_verdict_goes_beside_the_clip_and_a_retake_is_made_only_when_asked():
    good = [scene(1, shift) for shift in range(40)]
    bad = good[:15] + [scene(2, shift) for shift in range(25)]
    first, last = still("f.png", good[0]), still("l.png", good[-1])
    target = server.HOME / "renders" / "t1.mp4"
    env = {"IMAGE_FILE": str(first), "LAST_FRAME": str(last), "WIDTH": str(SIZE[0]), "HEIGHT": str(SIZE[1])}
    job = lambda **more: {"id": "t1", "kind": "video", "status": "running", "params": {"prompt": "[Shot 1] a hall.", "seed": 5, "preset": "custom", **more}}
    # not asked: the failed clip is reported with the seed a retake would use, and nothing is rendered
    shutil.copy(clip("bad.mp4", bad), target)
    one = job()
    had_build, had_run, renders = job_run.build, job_run.run, []
    try:
        job_run.build = lambda kind, params, job_id: (["true"], env, [str(target)])
        job_run.run = lambda j, command, e, label="": renders.append(j["params"]["seed"]) or bool(shutil.copy(clip("good.mp4", good), target))
        assert server.shot_checked(one, target, env) and renders == []
        assert one["shot"]["passed"] is False and one["shot"]["offer"] == 6 and len(one["shot"]["takes"]) == 1
        assert one["shot"]["takes"][0]["cuts"][0]["frame"] == 15 and one["params"]["seed"] == 5
        # the job's result says where to trim to: the 25 frames after the cut
        assert one["shot"]["takes"][0]["window"]["start"] == 15 and one["shot"]["takes"][0]["window"]["frames"] == 25
        # asked: one retake at the next seed, the failed take kept
        two = job(retake=True)
        assert server.shot_checked(two, target, env) and renders == [6]
        assert two["shot"]["passed"] is True and [take["seed"] for take in two["shot"]["takes"]] == [5, 6] and two["params"]["seed"] == 6
        kept = server.HOME / two["shot"]["takes"][0]["kept"]
        assert kept.is_file() and not shot.check(kept)["passed"] and shot.check(target)["passed"]
        # a clip that passes is never rendered again, and one the QA gate judged is left to the gate
        three = job(retake=True)
        assert server.shot_checked(three, target, env) and renders == [6] and three["shot"]["passed"] and "offer" not in three["shot"]
        shutil.copy(FOLDER / "bad.mp4", target)
        four = {**job(retake=True), "qa": {"passed": True, "takes": []}}
        assert server.shot_checked(four, target, env) and renders == [6] and four["shot"]["offer"] == 6
        # a clip that cannot be read does not lose the job
        five = job()
        assert server.shot_checked(five, FOLDER / "missing.mp4", env) and five["shot"]["passed"] is None and five["shot"]["error"]
    finally:
        job_run.build, job_run.run = had_build, had_run


# the run's figures the thresholds were set from: likeness to the frame before, and change against the local median
RUN_CUTS = [(-0.03, 10.8), (0.00, 16.1), (0.01, 4.2), (0.02, 38.2), (0.06, 28.5), (0.09, 7.8), (0.11, 36.3), (0.11, 28.4), (0.17, 15.7),
            (0.41, 19.9), (0.48, 8.1), (0.66, 2.4)]
RUN_NOT_CUTS = [(0.63, 1.1), (0.64, 1.1), (0.70, 1.0), (0.74, 3.5), (0.76, 3.4), (0.77, 1.1), (0.79, 1.2), (0.89, 3.9), (0.96, 4.2)]
RUN_HELD = [(1.4, 0.999), (3.9, 1.000), (2.1, 0.996), (3.7, 1.000)]  # the extremes of the 63 held keyframes
RUN_MISSED = [(74.4, 0.247)]


def test_the_thresholds_separate_the_runs_own_clips():
    cut = lambda likeness, ratio: likeness < shot.CUT_LIKENESS and ratio >= shot.CUT_RATIO
    assert all(cut(*pair) for pair in RUN_CUTS) and not any(cut(*pair) for pair in RUN_NOT_CUTS)
    held = lambda difference, likeness: difference <= shot.HELD_DIFFERENCE and likeness >= shot.HELD_LIKENESS
    assert all(held(*pair) for pair in RUN_HELD) and not any(held(*pair) for pair in RUN_MISSED)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
