"""The crop of a clip on a DGX Spark (spark/spark_finish.py): the 720p preset's workflow, no GPU, ComfyUI or ffmpeg."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spark"))


def test_the_720p_crop_is_made_in_the_workflow_without_ffmpeg():
    import spark_finish as G  # crop_frames and the `call` it makes live here

    had_call, had_which = G.call, G.shutil.which
    workflow = lambda: {"13": {"class_type": "VAEDecode", "inputs": {}}, "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0]}}}
    try:
        G.shutil.which = lambda name, **_: None  # no ffmpeg on this machine
        G.call = lambda path, data=None: {"ImageCrop": {}}
        w = workflow()
        assert G.crop_frames(w, "1280x720", 1280, 736) is True
        assert w["19"]["inputs"] == {"image": ["13", 0], "width": 1280, "height": 720, "x": 0, "y": 8} and w["15"]["inputs"]["images"] == ["19", 0]
        # neither a crop node nor ffmpeg: said before anything is rendered, never skipped
        G.call = lambda path, data=None: {}
        try:
            G.crop_frames(workflow(), "1280x720", 1280, 736)
            raise AssertionError("it was not refused")
        except SystemExit as stop:
            assert "ffmpeg is not installed" in str(stop)
        # an older ComfyUI with ffmpeg: the crop is made after the render, as before
        G.shutil.which = lambda name, **_: "/usr/bin/ffmpeg"
        w = workflow()
        assert G.crop_frames(w, "1280x720", 1280, 736) is False and "19" not in w
        try:
            G.crop_frames(workflow(), "1280x800", 1280, 736)
            raise AssertionError("it was not refused")
        except SystemExit as stop:
            assert "cannot centre-crop" in str(stop)
    finally:
        G.call, G.shutil.which = had_call, had_which


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
