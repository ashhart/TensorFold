"""app/picture_edit.py and the hub's picture LoRAs: the job a request becomes and what is refused; no GPU."""
import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import loras  # noqa: E402
import picture_edit as E  # noqa: E402
from fastapi import HTTPException  # noqa: E402

ANGLES = {"id": "qwen-multiple-angles", "name": "Multiple Angles", "path": "/loras/angles.safetensors", "strength": 0.9, "trigger": "<mva>", "lead": "<mva>"}


def studio(spark=True, picked=()):
    """A studio folder with one picture, and the step's build function over a stubbed hub."""
    home = Path(tempfile.mkdtemp())
    (home / "uploads").mkdir()
    (home / "uploads/cat.png").write_bytes(b"png")
    def resolve(asked, target="video"):
        assert target == "image"
        return [dict(p) for p in picked if any(a.get("id") == p["id"] for a in asked or [])]
    def inside(path):
        return home / path
    _, build = E.attach(home, ROOT, spark, resolve, inside, lambda p: str(p.relative_to(home)))
    return home, build


def refused(call) -> str:
    try:
        call()
    except HTTPException as error:
        return str(error.detail)
    raise AssertionError("it was not refused")


def test_the_caption_is_the_authors_grammar():
    assert E.angle_prompt("front-right-quarter", "eye", False) == "<mva> front-right quarter view, eye-level shot"
    assert E.angle_prompt("right", "high", True) == "<mva> right side view, high-angle shot close-up"
    assert E.angle_prompt("back", "top", False, "") == "back view, top-down shot"
    assert len(E.DIRECTIONS) == 12 and [d[2] for d in E.DIRECTIONS] == list(range(0, 360, 30)) and len(E.HEIGHTS) == 4
    assert "camera direction" in refused(lambda: E.angle_prompt("sideways", "eye", False))
    for _, angles in E.SETS.values():
        assert 1 <= len(angles) <= E.LIMIT and len({E.label(*a) for a in angles}) == len(angles)
        for a in angles:
            E.angle_prompt(*a)


def models(monkey_home: Path) -> dict:
    """Environment under which the step finds the image model on a Spark."""
    folder = monkey_home / "models/Qwen-Image-2.1-Comfy/diffusion_models"
    folder.mkdir(parents=True)
    (folder / "qwen_image_2.1_turbo_bf16.safetensors").write_bytes(b"x")
    return {"SPARK_MODELS": str(monkey_home / "models")}


def with_env(env, call):
    import os
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return call()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_a_set_of_angles_becomes_one_job_with_a_picture_each():
    home, build = studio(picked=[ANGLES])
    env = models(home)
    angles = [{"direction": "front", "height": "eye"}, {"direction": "back", "height": "high", "close": True}]
    command, job_env, outputs = with_env(env, lambda: build({"image": "uploads/cat.png", "angles": angles, "loras": [{"id": ANGLES["id"]}], "seed": 7}, "j1"))
    assert command[:2] == ["bash", str(ROOT / "scripts/edit.sh")] and command[2] == str(home / "uploads/cat.png")
    plan = json.loads(Path(job_env["SET_FILE"]).read_text())
    assert [p for p, _ in plan] == ["<mva> front view, eye-level shot", "<mva> back view, high-angle shot close-up"]
    assert [Path(o).name for o in outputs] == ["edit_front_eye.png", "edit_back_high_close.png"] and [o for _, o in plan] == outputs
    assert all(Path(o).parent == home / "scouts/j1" for o in outputs)  # beside the scout images: in the Library
    assert job_env["LORA"] == "/loras/angles.safetensors:0.9" and job_env["SEED"] == 7 and job_env["BASE"] == "" and "QUANTIZE" not in job_env


def test_words_get_a_loras_trigger_once():
    home, build = studio(picked=[ANGLES])
    env = models(home)
    _, job_env, outputs = with_env(env, lambda: build({"image": "uploads/cat.png", "prompt": "back view, eye-level shot", "loras": [{"id": ANGLES["id"]}]}, "j2"))
    assert json.loads(Path(job_env["SET_FILE"]).read_text())[0][0] == "<mva> back view, eye-level shot" and Path(outputs[0]).name == "edit.png"
    _, job_env, _ = with_env(env, lambda: build({"image": "uploads/cat.png", "prompt": "<mva> back view, eye-level shot", "loras": [{"id": ANGLES["id"]}]}, "j3"))
    assert json.loads(Path(job_env["SET_FILE"]).read_text())[0][0] == "<mva> back view, eye-level shot"
    _, job_env, _ = with_env(env, lambda: build({"image": "uploads/cat.png", "prompt": "the same statue at night"}, "j4"))
    assert json.loads(Path(job_env["SET_FILE"]).read_text())[0][0] == "the same statue at night" and job_env["LORA"] == ""


def test_what_is_refused_before_anything_is_queued():
    home, build = studio(picked=[ANGLES])
    env = models(home)
    ok = {"image": "uploads/cat.png", "prompt": "night"}
    assert "pick the picture" in with_env(env, lambda: refused(lambda: build({"prompt": "night"}, "a")))
    assert "pick the picture" in with_env(env, lambda: refused(lambda: build({"image": "uploads/none.png", "prompt": "night"}, "a")))
    assert "say what changes" in with_env(env, lambda: refused(lambda: build({"image": "uploads/cat.png"}, "a")))
    assert "Multiple Angles LoRA" in with_env(env, lambda: refused(lambda: build({"image": "uploads/cat.png", "angles": [{"direction": "back"}]}, "a")))
    assert "multiple of 32" in with_env(env, lambda: refused(lambda: build({**ok, "resolution": 1000}, "a")))
    assert "base model is not installed" in with_env(env, lambda: refused(lambda: build({**ok, "base": True}, "a")))
    many = [{"direction": d[0], "height": h[0]} for d in E.DIRECTIONS for h in E.HEIGHTS][:E.LIMIT + 1]
    assert "at most" in with_env(env, lambda: refused(lambda: build({"image": "uploads/cat.png", "angles": many, "loras": [{"id": ANGLES["id"]}]}, "a")))
    twice = [{"direction": "back", "height": "eye"}] * 2
    assert "different" in with_env(env, lambda: refused(lambda: build({"image": "uploads/cat.png", "angles": twice, "loras": [{"id": ANGLES["id"]}]}, "a")))
    assert "not installed here" in with_env({"SPARK_MODELS": str(home / "nowhere")}, lambda: refused(lambda: build(ok, "a")))


def lora_file(path: Path, keys: list[str]) -> None:
    head = json.dumps({k: {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]} for k in keys}).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"\0\0")


def test_the_hub_tells_a_picture_lora_from_a_clip_lora():
    folder = Path(tempfile.mkdtemp())
    picture, clip = folder / "p.safetensors", folder / "c.safetensors"
    lora_file(picture, ["diffusion_model.transformer_blocks.0.attn.to_k.lora_A.weight", "diffusion_model.transformer_blocks.0.attn.to_k.lora_B.weight"])
    lora_file(clip, ["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"])
    assert loras.check(picture, "image") == 1 and loras.check(clip) == 1
    for path, target in ((picture, "video"), (clip, "image")):
        try:
            loras.check(path, target)
        except ValueError as error:
            assert "not a LoRA for" in str(error)
        else:
            raise AssertionError("the wrong model's LoRA passed the check")
    presets = json.loads((ROOT / "app/lora-presets.json").read_text())
    pictures = [r for r in presets if r.get("target") == "image"]
    assert [r["id"] for r in pictures] == [E.ANGLE_LORA] and pictures[0]["status"] == "ready" and pictures[0]["needs"] == "image"
    assert "non-commercial" in pictures[0]["licence"]  # the model it runs on is under the Qwen Research License


def test_a_job_refuses_the_other_models_lora():
    home = Path(tempfile.mkdtemp())
    (home / "loras").mkdir()
    with_env({"LORA_OFFLINE": "1", "LORA_BUNDLE": str(home / "no-bundle")}, lambda: None)
    _, resolve, _ = loras.attach(home, ROOT, True)
    lora_file(home / "loras/qwen-multiple-angles.safetensors", ["diffusion_model.transformer_blocks.0.attn.to_k.lora_A.weight"])
    lora_file(home / "loras/camera-motion.safetensors", ["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"])
    assert resolve([{"id": "qwen-multiple-angles"}], "image")[0]["path"].endswith("qwen-multiple-angles.safetensors")
    assert "for pictures" in refused(lambda: resolve([{"id": "qwen-multiple-angles"}]))
    assert "for clips" in refused(lambda: resolve([{"id": "camera-motion"}], "image"))


def test_further_pictures_and_the_size_of_the_result():
    home, build = studio()
    (home / "uploads/man.png").write_bytes(b"png")
    (home / "uploads/room.jpg").write_bytes(b"jpg")
    # one picture, nothing said about the size: the result has the picture's own (2.6.5 returned the model's)
    _, env, _ = build({"image": "uploads/cat.png", "prompt": "at night", "resolution": 1024}, "j1")
    assert env["SIZE"] == "source" and env["IMAGES"] == "" and "WIDTH" not in env
    # further pictures go in the order given, after the one that is edited
    command, env, _ = build({"image": "uploads/cat.png", "images": ["uploads/man.png", "uploads/room.jpg"], "prompt": "the man of image 2 holds the cat"}, "j2")
    assert command[-1] == str(home / "uploads/cat.png")
    assert env["IMAGES"].split("\n") == [str(home / "uploads/man.png"), str(home / "uploads/room.jpg")]
    # the model's own size, and a size of one's own
    _, env, _ = build({"image": "uploads/cat.png", "prompt": "x", "size": "model", "resolution": 768}, "j3")
    assert env["SIZE"] == "model" and env["RESOLUTION"] == 768
    _, env, _ = build({"image": "uploads/cat.png", "prompt": "x", "width": 1280, "height": 720}, "j4")
    assert (env["WIDTH"], env["HEIGHT"]) == (1280, 720)
    assert "at most 10 pictures" in refused(lambda: build({"image": "uploads/cat.png", "images": ["uploads/man.png"] * 10, "prompt": "x"}, "j5"))
    assert "no picture at uploads/gone.png" in refused(lambda: build({"image": "uploads/cat.png", "images": ["uploads/gone.png"], "prompt": "x"}, "j6"))
    assert "size is" in refused(lambda: build({"image": "uploads/cat.png", "prompt": "x", "size": "huge"}, "j7"))
    assert "outside" in refused(lambda: build({"image": "uploads/cat.png", "prompt": "x", "width": 4096, "height": 4096}, "j8"))
    assert "a width and a height" in refused(lambda: build({"image": "uploads/cat.png", "prompt": "x", "width": 1024}, "j9"))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
