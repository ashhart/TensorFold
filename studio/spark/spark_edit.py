"""Edit a picture on a DGX Spark (CUDA): Qwen-Image-2.1 or FLUX3-K with the picture as reference, via ComfyUI."""
# Credit: the Qwen team (Qwen-Image-2.1), Black Forest Labs (FLUX 3 Image Klein, its upsampler), ComfyUI.
import time
import subprocess
import argparse
import json
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import edit_plan  # noqa: E402  (the noise seed and the sizes, shared with the Mac build)
import spark_generate as sg  # noqa: E402
# FLUX3-K, a separate contribution: drop this import and the `flux3` lines of main() to remove it
from flux3_workflows import FLUX3, flux3_edit_workflow as flux3_workflow, upsampler_pictures  # noqa: E402,F401


def prepared(path: str, size: tuple[int, int], folder: Path, index: int) -> Path:
    """A picture at exactly ``size`` (Lanczos, as ComfyUI's node would do it), so the node leaves it alone."""

    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        picture = ImageOps.exif_transpose(opened)
        picture = picture.convert("RGBA" if "A" in picture.getbands() or "transparency" in picture.info else "RGB")
        if picture.size != size:
            picture = picture.resize(size, Image.Resampling.LANCZOS)
        target = folder / f"edit_{index}.png"
        picture.save(target)
    return target


def workflow(args, pictures: list[str], prompt: str, seed: int, loras: list[tuple[str, float]], made: tuple[int, int] | None = None) -> dict:
    w = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": sg.QWEN_BASE if args.base else sg.QWEN, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": sg.QWEN_TEXT, "type": "qwen_image", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": sg.QWEN_VAE}},
        # the pictures go to the text encoder and in as latents, edit made at the first's size; resolution 0 keeps sizes
        "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["3", 0], "prompt": prompt, "negative_prompt": args.negative,
                                                                "vae": ["4", 0], "resolution": 0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": f"studio/edit_{uuid.uuid4().hex[:8]}"}},
    }
    for index, picture in enumerate(pictures):
        w[str(20 + index)] = {"class_type": "LoadImage", "inputs": {"image": picture}}
        w["5"]["inputs"][f"images.image_{index + 1}"] = [str(20 + index), 0]
    model = ["1", 0]
    for index, (name, strength) in enumerate(loras):
        w[str(80 + index)] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [str(80 + index), 0]
    latent = ["5", 2]
    if made:  # a size asked for by hand in another shape than the picture's
        w["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": made[0], "height": made[1], "batch_size": 1}}
        latent = ["6", 0]
    if args.base:
        w["12"] = {"class_type": "KSampler", "inputs": {"model": model, "seed": seed, "steps": args.steps, "cfg": args.cfg, "sampler_name": "euler",
                                                         "scheduler": "simple", "positive": ["5", 0], "negative": ["5", 1], "latent_image": latent, "denoise": 1.0}}
    else:
        w["8"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
        w["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        w["10"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": sg.TURBO_SIGMAS}}
        w["11"] = {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["5", 0], "negative": ["5", 1], "cfg": 1.0}}
        w["12"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["8", 0], "guider": ["11", 0], "sampler": ["9", 0], "sigmas": ["10", 0], "latent_image": latent}}
    return w


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image", required=True, action="append", help="the picture to edit; repeat for further pictures the prompt draws on (image 2, image 3, ...)")
    parser.add_argument("--prompt", default="", help="what to change (one picture, written to -o)")
    parser.add_argument("-o", "--output", default="")
    parser.add_argument("--set", default="", help='several edits of the one picture from one load: JSON [[prompt, output], ...], or @FILE holding it')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base", action="store_true", help="the 40-step base model (when it has been downloaded) instead of Turbo's 8 steps")
    parser.add_argument("--steps", type=int, default=40, help="base model only")
    parser.add_argument("--cfg", type=float, default=1.0, help="base model only; 1 runs the prompt alone")
    parser.add_argument("--negative", default="")
    parser.add_argument("--size", choices=("source", "model"), default="source",
                        help="source: the result has the picture's own size (the default); model: about --resolution pixels squared, keeping its shape")
    parser.add_argument("--resolution", type=int, default=1024, help="with --size model: the picture is brought to about this many pixels squared")
    parser.add_argument("--most", type=int, default=0, help="with --size source: the most pixels an edit is made at (0: 1536x1536)")
    parser.add_argument("--width", type=int, default=0, help="with --height: another size for the result (the edit shifts when its shape differs from the picture's)")
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--lora", action="append", default=[], help="an image LoRA, PATH or PATH:STRENGTH (repeat to stack)")
    parser.add_argument("--model", choices=("qwen", *FLUX3), default="qwen",
                        help="qwen: Qwen-Image-2.1 (the default); flux3k-fast or flux3k-full: FLUX3-K (oneshot-setup-spark.sh --flux3)")
    parser.add_argument("--no-upsample", dest="upsample", action="store_false",
                        help="FLUX3-K: send the prompt as written (a full caption of the result) instead of BFL's prompt upsampler's")
    args = parser.parse_args()
    args.steps_set = "--steps" in sys.argv[1:]
    flux3 = args.model in FLUX3
    if flux3 and (args.base or args.lora):
        sys.exit("image LoRAs and the base model are Qwen-Image-2.1's: FLUX3-K takes neither")
    for path in args.image:
        if not Path(path).is_file():
            sys.exit(f"no picture at {path}")
    if len(args.image) > edit_plan.PICTURES:
        sys.exit(f"at most {edit_plan.PICTURES} pictures in one edit, got {len(args.image)}")
    if bool(args.width) != bool(args.height):
        sys.exit("--width and --height go together")
    if args.width and edit_plan.refused(args.width, args.height):
        sys.exit(edit_plan.refused(args.width, args.height))
    if args.resolution % 32 or not 256 <= args.resolution <= 2048:
        sys.exit("--resolution is a multiple of 32 between 256 and 2048")
    edits = json.loads(Path(args.set[1:]).read_text() if args.set.startswith("@") else args.set) if args.set else [[args.prompt, args.output]]
    if not edits or any(not str(p).strip() or not o for p, o in edits):
        sys.exit("give --prompt and -o, or --set with a prompt and an output for each edit")
    # after a Library action ComfyUI's unloading leaves too little for Qwen-Image: start from an empty server
    last = sg.STATE.read_text().strip() if sg.STATE.exists() else ""
    if sg.up() and last.startswith("workflow"):
        print("[tensorfold] restarting ComfyUI: a picture edit after a Library action starts from a server with nothing loaded", flush=True)
        subprocess.run(["bash", str(sg.HERE / "comfy.sh"), "stop"], check=False)
        for _ in range(60):
            if not sg.up():
                break
            time.sleep(1)
        time.sleep(5)
    sg.ensure_server("flux3" if flux3 else "image")
    if flux3 and not sg.installed("unet", FLUX3[args.model][0]):
        sys.exit(f"{args.model} is not installed: oneshot-setup-spark.sh --flux3")
    if args.base and not sg.installed("unet", sg.QWEN_BASE):
        sys.exit("the base model is not installed: oneshot-setup-spark.sh --image-edit fetches Qwen-Image-2.1 (14 GB; Qwen Research License, non-commercial)")
    loras = sg.hub_loras(args.lora)
    for name, _ in loras:
        if not sg.installed("lora", name):
            sys.exit(f"ComfyUI does not list the LoRA {name}")
    # the size the edit is made at and the size that comes back, and the noise for this seed and these pictures
    sizes = [edit_plan.picture_size(path) for path in args.image]
    plan = edit_plan.canvas(*sizes[0], args.size, args.resolution, (args.width, args.height) if args.width else None, args.most)
    made, size = tuple(plan["made"]), tuple(plan["size"])
    if plan["note"]:
        print(f"[tensorfold] {plan['note']}", flush=True)
    noise = edit_plan.noise_seed(args.seed, args.image)
    try:
        with tempfile.TemporaryDirectory(prefix="studio-edit-") as scratch:
            fitted = [tuple(plan["source"])] + [edit_plan.beside(*one, made) for one in sizes[1:]]
            pictures = [sg.upload(str(prepared(path, fit, Path(scratch), index))) for index, (path, fit) in enumerate(zip(args.image, fitted))]
            shown = [sg.upload(str(p)) for p in upsampler_pictures(args.image, Path(scratch))] if flux3 and args.upsample else []
        for prompt, output in edits:
            other = made if tuple(plan["source"]) != made else None
            if flux3:
                item, seconds = sg.render(flux3_workflow(args, pictures, shown, prompt, noise, other), "12")
            else:
                item, seconds = sg.render(workflow(args, pictures, prompt, noise, loras, other), "12")
            sg.fetch(item, Path(output))
            if size != made:  # a side that is not on the model's grid: the result is brought back to the picture's own
                from PIL import Image

                with Image.open(output) as opened:
                    opened.resize(size, Image.Resampling.LANCZOS).save(output)
            print("[tensorfold] " + json.dumps({"output": str(output), "engine": "cuda", "model": args.model, "edit": prompt[:80], "seed": args.seed,
                                                "noise_seed": noise, "turbo": not args.base, **({"upsampled": args.upsample} if flux3 else {}), "loras": [name for name, _ in loras], "pictures": len(pictures),
                                                "size": list(size), "made_at": list(made), "total_s": round(seconds, 1)}), flush=True)
    finally:
        for name in sg.UPLOADED:
            (sg.PREFIX / "ComfyUI" / "input" / name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
