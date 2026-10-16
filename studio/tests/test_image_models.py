"""The picture models (Qwen 2.1 Turbo, FLUX3-K Fast and Full): what a job asks for, and FLUX3-K's Spark workflow."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "spark"))


def test_only_installed_models_are_offered_and_asked_for():
    import picture_models
    import server
    from fastapi import HTTPException

    had = picture_models.image_models
    try:
        picture_models.image_models = lambda: ["qwen"]
        assert server.image_model_env({}) == {"IMAGE_MODEL": "qwen"}
        for name, said in (("flux3k-fast", "FLUX3-K Fast is not installed"), ("flux9", "unknown picture model")):
            try:
                server.image_model_env({"image_model": name})
                raise AssertionError(name + " was accepted")
            except HTTPException as error:
                assert said in error.detail, error.detail
        picture_models.image_models = lambda: ["qwen", "flux3k-fast", "flux3k-full"]
        assert server.image_model_env({"image_model": "flux3k-full"}) == {"IMAGE_MODEL": "flux3k-full"}
        # a scout job carries the choice to scripts/image.sh
        _, env, _ = server.build("scout", {"prompt": "a lighthouse", "count": 2, "seed": 3, "image_model": "flux3k-fast"}, "j1")
        assert env["IMAGE_MODEL"] == "flux3k-fast"
        labels = [m["label"] for m in server.state()["image_models"]]
        assert labels == ["Qwen 2.1 Turbo", "FLUX3-K Fast", "FLUX3-K Full"], labels
    finally:
        picture_models.image_models = had


def test_the_flux3_workflow_follows_bfl():
    import spark_generate as G
    import flux3_workflows as F

    class Args:
        model, width, height, steps = "flux3k-fast", 1344, 768, None

    with tempfile.TemporaryDirectory() as folder:
        had = F.MODELS
        try:
            F.MODELS = Path(folder)
            (Path(folder) / G.FLUX3_DIR).mkdir()
            (Path(folder) / G.FLUX3_DIR / "system_prompts.json").write_text(json.dumps({"t2i": "You are the prompt upsampler."}))
            w = G.flux3_workflow(Args, "a lighthouse", 7)
            # the upsampler writes the caption inside the workflow, and the text encoder reads it
            assert w["21"]["class_type"] == "TextGenerate" and w["21"]["inputs"]["system_prompt"] == "You are the prompt upsampler."
            assert w["21"]["inputs"]["prompt"] == "a lighthouse" and w["5"]["inputs"]["text"] == ["21", 0]
            assert w["3"]["inputs"]["type"] == "flux3" and w["20"]["inputs"]["clip_name"] == G.FLUX3_UPSAMPLER
            sampler = w["12"]["inputs"]
            assert (sampler["steps"], sampler["cfg"], sampler["seed"]) == (16, 1.0, 7)
            assert w["1"]["inputs"]["unet_name"] == "flux3_image_klein_sd_bf16.safetensors"
            Args.model = "flux3k-full"
            w = G.flux3_workflow(Args, "a lighthouse", 7, caption="Full-color photograph of a lighthouse.")
            assert (w["12"]["inputs"]["steps"], w["12"]["inputs"]["cfg"]) == (50, 3.5)
            assert "21" not in w and w["5"]["inputs"]["text"] == "Full-color photograph of a lighthouse."
        finally:
            F.MODELS = had


def test_an_edit_goes_to_the_picked_model():
    import picture_edit as E
    from fastapi import HTTPException

    home = Path(tempfile.mkdtemp())
    (home / "uploads").mkdir()
    for name in ("room.png", "man.png"):
        (home / "uploads" / name).write_bytes(b"png")
    models = home / "models/Qwen-Image-2.1-Comfy/diffusion_models"
    models.mkdir(parents=True)
    (models / "qwen_image_2.1_turbo_bf16.safetensors").write_bytes(b"x")
    os.environ["SPARK_MODELS"] = str(home / "models")
    installed = [{"id": "qwen", "label": "Qwen 2.1 Turbo", "note": ""}, {"id": "flux3k-fast", "label": "FLUX3-K Fast", "note": ""}]
    router, build = E.attach(home, ROOT, True, lambda asked, target: [{"id": "x", "path": "/x", "strength": 1.0, "trigger": ""}] if asked else [],
                             lambda path: home / path, lambda path: str(path), lambda: installed)
    state = next(r.endpoint for r in router.routes if r.path == "/api/edit/state")()
    assert [m["id"] for m in state["models"]] == ["qwen", "flux3k-fast"] and state["ready"] and state["qwen"]
    _, env, outs = build({"image": "uploads/room.png", "images": ["uploads/man.png"], "prompt": "the man of image 2 in the room",
                          "image_model": "flux3k-fast"}, "e1")
    assert env["IMAGE_MODEL"] == "flux3k-fast" and env["IMAGES"].endswith("man.png") and len(outs) == 1
    assert build({"image": "uploads/room.png", "prompt": "night"}, "e2")[1]["IMAGE_MODEL"] == "qwen"
    for params, said in (({"image_model": "flux3k-full"}, "not installed"), ({"image_model": "flux9"}, "unknown picture model"),
                         ({"image_model": "flux3k-fast", "loras": [{"id": "x"}]}, "are Qwen 2.1 Turbo's"),
                         ({"image_model": "flux3k-fast", "base": True}, "are Qwen 2.1 Turbo's")):
        try:
            build({"image": "uploads/room.png", "prompt": "night", **params}, "e3")
            raise AssertionError(f"{params} was accepted")
        except HTTPException as error:
            assert said in error.detail, error.detail


def test_the_flux3_edit_workflow_follows_bfl():
    import spark_edit as SE
    import spark_generate as G
    import flux3_workflows as F

    class Args:
        model, steps, steps_set = "flux3k-fast", 40, False

    with tempfile.TemporaryDirectory() as folder:
        had = F.MODELS
        try:
            F.MODELS = Path(folder)
            (Path(folder) / G.FLUX3_DIR).mkdir()
            (Path(folder) / G.FLUX3_DIR / "system_prompts.json").write_text(json.dumps({"t2i": "t2i", "i2i": "You edit."}))
            w = SE.flux3_workflow(Args, ["room.png", "man.png"], ["shown_0.png", "shown_1.png"], "put him in the room", 9)
            # every picture is a reference latent on the prompt and the negative; the result takes the first one's size
            refs = w["17"]["inputs"]
            assert w["17"]["class_type"] == "ReferenceMultiLatent" and refs["resolution"] == "keep resolution"
            assert refs["images.image0"] == ["30", 0] and refs["images.image1"] == ["31", 0] and w["31"]["inputs"]["image"] == "man.png"
            assert w["12"]["inputs"]["latent_image"] == ["17", 2] and w["7"]["class_type"] == "ConditioningZeroOut"
            assert (w["12"]["inputs"]["steps"], w["12"]["inputs"]["cfg"], w["12"]["inputs"]["seed"]) == (16, 1.0, 9)
            # the upsampler sees both pictures (one batch), BFL's i2i system prompt and its user turn
            up = w["21"]["inputs"]
            assert up["system_prompt"] == "You edit." and up["image"] == ["71", 0] and up["sampling_mode.presence_penalty"] == 0.0
            assert up["prompt"].startswith("Reference images follow in order") and "put him in the room" in up["prompt"]
            assert w["5"]["inputs"]["text"] == ["21", 0]
            Args.model = "flux3k-full"
            w = SE.flux3_workflow(Args, ["room.png"], [], "A room at night.", 9, made=(1024, 576))
            assert (w["12"]["inputs"]["steps"], w["12"]["inputs"]["cfg"]) == (50, 3.5) and "21" not in w
            assert w["5"]["inputs"]["text"] == "A room at night." and w["7"]["inputs"]["text"] == ""
            assert w["12"]["inputs"]["latent_image"] == ["6", 0] and w["6"]["inputs"]["width"] == 1024
        finally:
            F.MODELS = had


def test_the_upsampler_is_shown_the_pictures_on_one_canvas():
    import spark_edit as SE
    from PIL import Image

    with tempfile.TemporaryDirectory() as folder:
        Image.new("RGB", (1600, 900), "red").save(Path(folder) / "wide.png")
        Image.new("RGBA", (300, 600), (0, 0, 0, 0)).save(Path(folder) / "tall.png")
        one = SE.upsampler_pictures([str(Path(folder) / "wide.png")], Path(folder))
        assert Image.open(one[0]).size[0] * Image.open(one[0]).size[1] <= 1024 * 1024
        two = SE.upsampler_pictures([str(Path(folder) / "wide.png"), str(Path(folder) / "tall.png")], Path(folder))
        sizes = {Image.open(p).size for p in two}
        assert len(sizes) == 1 and next(iter(sizes))[0] <= 724 * 2  # one batch: the same size, over white
        assert Image.open(two[1]).convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_an_unknown_model_is_refused_by_the_script():
    if sys.platform.startswith("linux"):
        return  # scripts/image.sh hands over to spark/image.sh, whose argparse refuses it
    done = subprocess.run(["bash", str(ROOT / "scripts/image.sh"), "x", "/tmp/never.png"], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "IMAGE_MODEL": "flux9"})
    assert done.returncode == 2 and "IMAGE_MODEL is qwen" in done.stderr


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
