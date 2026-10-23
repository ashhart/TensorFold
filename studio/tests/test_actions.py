"""The builders and checks of app/actions.py, and the first-frame plate of spark/spark_workflows.py."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import actions  # noqa: E402
from fastapi import HTTPException  # noqa: E402

PACK = Path(__file__).resolve().parent.parent


def studio(spark=True, installed=True):
    root = Path(tempfile.mkdtemp())
    home, prefix, models = root / "TensorFoldStudio", root / "prefix", root / "models"
    for name in ("renders", "uploads", "scouts", "loras"):
        (home / name).mkdir(parents=True)
    if installed:
        for node in ("ComfyUI-H3-ContactSheet", "ComfyUI-H3-Video-Upsampler", "h3-relay"):
            (prefix / "ComfyUI/custom_nodes" / node).mkdir(parents=True)
        for file in ("Workflows-Comfy/checkpoints/sam3.1_multiplex_fp16.safetensors",
                     "Workflows-Comfy/latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors",
                     "MiniMax-H3-Comfy/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors"):
            (models / file).parent.mkdir(parents=True, exist_ok=True)
            (models / file).write_bytes(b"x")
        for lora in ("turnaround", "video-upscaler", "person-remover"):
            (home / "loras" / f"{lora}.safetensors").write_bytes(b"x")
    os.environ.update(PREFIX=str(prefix), SPARK_MODELS=str(models))
    os.environ.pop("LORA_BUNDLE", None)

    def inside(path):
        resolved = (home / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if home.resolve() not in (resolved, *resolved.parents):
            raise HTTPException(400, "path is outside the studio folder")
        return resolved

    router, build, title, source = actions.attach(home, PACK, spark, inside, lambda p: str(Path(p).resolve().relative_to(home.resolve())))
    (home / "uploads/a.png").write_bytes(b"x")
    (home / "renders/c.mp4").write_bytes(b"x")
    state = next(r.endpoint for r in router.routes if r.path == "/api/actions")
    return home, build, title, source, state


def refused(call, *args):
    try:
        call(*args)
    except HTTPException as error:
        return error.detail
    return None


def test_turnaround_command_and_outputs():
    home, build, title, source, _ = studio()
    command, env, outputs = build("turnaround", {"image": "uploads/a.png", "size": 512, "strength": 0.7, "seed": 5}, "j1")
    assert command[1].endswith("scripts/turnaround.sh") and command[2].endswith("uploads/a.png") and command[3].endswith("scouts/j1/turn")
    assert (env["SIZE"], env["STRENGTH"], env["SEED"]) == (512, "0.7", 5) and env["TURNAROUND_LORA"].endswith("loras/turnaround.safetensors")
    assert outputs[0].endswith("scouts/j1/turn_sheet.png") and len(outputs) == 6   # under scouts: the Library lists them
    assert title("turnaround", {"image": "uploads/a.png"}) == "Turnaround sheet: a.png"
    assert source("turnaround", {"image": "uploads/a.png"}) == "uploads/a.png"


def test_turnaround_checks():
    _, build, *_ = studio()
    assert "512, 1024 or 2048" in refused(build, "turnaround", {"image": "uploads/a.png", "size": 700}, "j")
    assert "strength" in refused(build, "turnaround", {"image": "uploads/a.png", "strength": 3}, "j")
    assert "<Picture 1>" in refused(build, "turnaround", {"image": "uploads/a.png", "prompt": "turn it"}, "j")
    assert "choose a picture" in refused(build, "turnaround", {}, "j")
    assert "not a picture" in refused(build, "turnaround", {"image": "renders/c.mp4"}, "j")
    assert "outside" in refused(build, "turnaround", {"image": "/etc/passwd"}, "j")


def test_upscale():
    _, build, *_ = studio()
    command, _, outputs = build("upscale", {"video": "renders/c.mp4"}, "j2")
    assert command[command.index("--megapixels") + 1] == "2" and command[command.index("--denoise") + 1] == "0.1"
    assert outputs == [str(Path(outputs[0]).parent / "j2.mp4")] and "/renders/" in outputs[0]
    assert "megapixels" in refused(build, "upscale", {"video": "renders/c.mp4", "megapixels": 9}, "j")
    assert "re-draw" in refused(build, "upscale", {"video": "renders/c.mp4", "denoise": 0.9}, "j")
    assert "not a clip" in refused(build, "upscale", {"video": "uploads/a.png"}, "j")


def test_remove_and_mask():
    home, build, *_ = studio()
    command, _, outputs = build("mask", {"video": "renders/c.mp4", "person": "  man in   a gray shirt "}, "j3")
    assert command[command.index("--person") + 1] == "man in a gray shirt" and "--lora" not in command
    assert [Path(o).name for o in outputs] == ["person_mask.mp4", "person_matte.mp4", "person_plate.png"]
    command, _, outputs = build("remove", {"video": "renders/c.mp4", "person": "woman", "clean_frame": "uploads/a.png", "silent": True}, "j4")
    assert "--silent" in command and command[command.index("--clean-frame") + 1].endswith("uploads/a.png")
    assert command[command.index("--window") + 1] == "22" and command[command.index("--steps") + 1] == "12"
    assert outputs[0].endswith("renders/j4.mp4")
    assert "who to remove" in refused(build, "remove", {"video": "renders/c.mp4"}, "j")
    assert "window" in refused(build, "remove", {"video": "renders/c.mp4", "person": "x", "window": 30}, "j")
    assert "not a picture" in refused(build, "remove", {"video": "renders/c.mp4", "person": "x", "clean_frame": "renders/c.mp4"}, "j")


def test_not_installed_and_mac():
    _, build, _, _, state = studio(installed=False)
    assert "setup-workflows.sh" in refused(build, "upscale", {"video": "renders/c.mp4"}, "j")
    assert all(not one["ready"] and one["why"] for one in state().values())
    home, build, _, _, state = studio(spark=False)   # a Mac: the turnaround sheet alone, on the MLX engine
    os.environ["H3_MODEL_DIR"] = str(home / "h3")
    assert "Ref2VA is not installed" in refused(build, "turnaround", {"image": "uploads/a.png"}, "j")
    (home / "h3/Ref2VA/transformer").mkdir(parents=True)
    assert "LoRA hub" in refused(build, "turnaround", {"image": "uploads/a.png"}, "j")   # the Spark's file is not the Mac's
    (home / "loras/turnaround.mlx.safetensors").write_bytes(b"x")
    command, env, _ = build("turnaround", {"image": "uploads/a.png", "prompt": "turn <Picture 1> round"}, "j")
    assert env["TURNAROUND_LORA"].endswith("turnaround.mlx.safetensors") and env["PROMPT"] == "turn <Picture 1> round" and "PREFIX" not in env
    assert all("DGX Spark only" in one["why"] for kind, one in state().items() if kind != "turnaround") and state()["turnaround"]["ready"]
    assert refused(build, "upscale", {"video": "renders/c.mp4"}, "j").startswith("DGX Spark only")
    _, _, _, _, state = studio()
    assert all(one["ready"] for one in state().values())


def test_lora_from_a_bundle():
    home, build, *_ = studio()
    (home / "loras/turnaround.safetensors").unlink()
    assert "LoRA hub" in refused(build, "turnaround", {"image": "uploads/a.png"}, "j")
    kept = home / "lora-bundle/02-transformation-morphing/turnaround/minimax_h3_five_view_512_s1500.safetensors"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"x")
    _, env, _ = build("turnaround", {"image": "uploads/a.png"}, "j")
    assert env["TURNAROUND_LORA"] == str(kept) and env["STRENGTH"] == "1"


def test_plate_fills_from_other_frames():
    import numpy as np

    sys.path.insert(0, str(PACK / "spark"))
    import spark_workflows as sw

    height, width, count = 64, 96, 12
    yy, xx = np.mgrid[0:height, 0:width]
    back = np.stack([xx * 2 % 256, yy * 3 % 256, (xx + yy) % 256], -1).astype(np.uint8)   # a still background
    frames, masks = np.repeat(back[None], count, 0).copy(), np.zeros((count, height, width), bool)
    for i in range(count):                                   # a block that walks across it
        masks[i, 20:44, 6 * i:6 * i + 20] = True
        frames[i][masks[i]] = (200, 30, 30)
    folder = Path(tempfile.mkdtemp())

    def fake(path, w, h):
        return frames if "source" in str(path) else np.repeat((masks * 255).astype(np.uint8)[..., None], 3, -1)

    sw.read_frames = fake
    share = sw.plate(folder / "source.mp4", folder / "matte.mp4", {"width": width, "height": height}, folder / "plate.png")
    from PIL import Image

    made = np.asarray(Image.open(folder / "plate.png")).astype(int)
    assert share == 1.0 and np.abs(made - back.astype(int)).max() <= 1   # the first frame's hole is the background again


def test_mend_paints_over_dull_green_residue():
    import numpy as np

    sys.path.insert(0, str(PACK / "spark"))
    import spark_workflows as sw

    import importlib

    sw = importlib.reload(sw)                        # the test above reads its frames from memory
    width, height, count = 160, 96, 12
    rng = np.random.default_rng(3)
    back = np.clip(rng.normal(120, 6, (height, width, 3)), 0, 255).astype(np.uint8)
    frames = np.repeat(back[None], count, 0).copy()
    frames[9, 30:60, 40:80] = (40, 110, 40)          # a dull green block: under the mask's own green, still a stain
    folder = Path(tempfile.mkdtemp())
    assert sw.write_frames(frames, folder / "clip.mp4", {"width": width, "height": height}, 6)
    mended, left = sw.mend(folder / "clip.mp4", {"width": width, "height": height})
    out = sw.read_frames(folder / "clip.mp4", width, height).astype(int)
    assert mended == 1 and left == 0, (mended, left)
    assert np.abs(out[9, 30:60, 40:80].mean((0, 1)) - 120).max() < 6      # the block is the background again
    assert np.abs(out[4].astype(int) - frames[4]).mean() < 6              # a clean frame is left as it was


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
