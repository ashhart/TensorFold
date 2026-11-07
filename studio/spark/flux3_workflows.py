"""FLUX3-K (Black Forest Labs' FLUX 3 Image Klein) on a DGX Spark: ComfyUI workflows for a picture and for an edit."""
import json
import sys
import uuid
from pathlib import Path

from spark_comfy import MODELS, fetch, render

# FLUX3-K Fast (distilled, 16 steps, no guidance) and Full (base, 50 steps, CFG 3.5); BFL's upsampler writes captions
FLUX3_DIR = "FLUX3-K-Comfy"
FLUX3 = {"flux3k-fast": ("flux3_image_klein_sd_bf16.safetensors", 16, 1.0),
         "flux3k-full": ("flux3_image_klein_base_bf16.safetensors", 50, 3.5)}
FLUX3_TEXT = "qwen3vl_4b_bf16.safetensors"
FLUX3_UPSAMPLER = "qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors"
FLUX3_VAE = "flux2-vae.safetensors"


def flux3_system_prompt(task: str = "t2i") -> str:
    """A system prompt of BFL's upsampler (system_prompts.json by the model): t2i for a picture, i2i for an edit."""

    path = MODELS / FLUX3_DIR / "system_prompts.json"
    try:
        return json.loads(path.read_text())[task]
    except (OSError, ValueError, KeyError):
        sys.exit(f"FLUX3-K's prompt upsampler has no system prompt at {path}: run oneshot-setup-spark.sh --flux3")


def flux3_workflow(args, text: str, seed: int, caption: str = "") -> dict:
    """FLUX3-K on ComfyUI's FLUX 3 nodes as in BFL's templates: upsampler caption, Qwen3-VL text encoder, sampler."""

    unet, steps, cfg = FLUX3[args.model]
    w = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX3_TEXT, "type": "flux3", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX3_VAE}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": args.width, "height": args.height, "batch_size": 1}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
        "12": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": seed, "steps": args.steps or steps,
                                                    "cfg": cfg, "sampler_name": "euler", "scheduler": "simple",
                                                    "positive": ["5", 0], "negative": ["7", 0], "latent_image": ["6", 0],
                                                    "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": f"studio/image_{uuid.uuid4().hex[:8]}"}},
    }
    if caption:
        w["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": caption}}
    else:
        w["20"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX3_UPSAMPLER, "type": "flux3", "device": "default"}}
        w["21"] = {"class_type": "TextGenerate", "inputs": {
            "clip": ["20", 0], "prompt": text, "max_length": 3584, "sampling_mode": "on",
            "sampling_mode.temperature": 1.0, "sampling_mode.top_k": 1, "sampling_mode.top_p": 0.95,
            "sampling_mode.min_p": 0.0, "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": 0,
            "sampling_mode.presence_penalty": 1.5, "thinking": False, "use_default_template": True, "mtp": "auto",
            "system_prompt": flux3_system_prompt()}}
        w["22"] = {"class_type": "PreviewAny", "inputs": {"source": ["21", 0]}}
        w["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ["21", 0]}}
    return w


def flux3_images(args, text: str, target: Path, seeds: list[int]) -> None:
    caption = "" if args.upsample else text
    for seed in seeds:
        item, seconds = render(flux3_workflow(args, text, seed, caption), "12")
        out = target.with_name(f"{target.stem}_s{seed}{target.suffix}") if args.seeds else target
        fetch(item, out)
        print("[tensorfold] " + json.dumps({"output": str(out), "engine": "cuda", "model": args.model,
                                            "width": args.width, "height": args.height, "seed": seed,
                                            "steps": args.steps or FLUX3[args.model][1], "upsampled": args.upsample,
                                            "total_s": round(seconds, 1)}), flush=True)


# BFL's upsampler: the user turn with the pictures (flux3-mlx's pup.py), pixel budget by count, grid, JPEG quality
I2I_SCAFFOLD = "Reference images follow in order (the first image, the second image, ...).\nUser request: {}\nOutput the caption."
UPSAMPLER_PIXELS = {1: 1024 * 1024, 2: 724 * 724}
UPSAMPLER_GRID, UPSAMPLER_JPEG = 32, 92


def upsampler_pictures(paths: list[str], folder: Path) -> list[Path]:
    """The pictures as BFL's upsampler sees them (over white, size budget, JPEG), centred on one canvas, one batch."""

    import io
    from PIL import Image, ImageOps

    budget = UPSAMPLER_PIXELS.get(len(paths), 512 * 512)
    shown = []
    for path in paths:
        with Image.open(path) as opened:
            picture = ImageOps.exif_transpose(opened).convert("RGBA")
        picture = Image.alpha_composite(Image.new("RGBA", picture.size, (255, 255, 255, 255)), picture).convert("RGB")
        w, h = (max(UPSAMPLER_GRID, round(v / UPSAMPLER_GRID) * UPSAMPLER_GRID) for v in picture.size)
        if w * h > budget:
            scale = (budget / (w * h)) ** 0.5
            w, h = (max(UPSAMPLER_GRID, int(v * scale) // UPSAMPLER_GRID * UPSAMPLER_GRID) for v in (w, h))
        buffer = io.BytesIO()
        picture.resize((w, h), Image.Resampling.LANCZOS).save(buffer, format="JPEG", quality=UPSAMPLER_JPEG)
        shown.append(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
    wide, tall = max(p.width for p in shown), max(p.height for p in shown)
    out = []
    for index, picture in enumerate(shown):
        if picture.size != (wide, tall):
            canvas = Image.new("RGB", (wide, tall), (255, 255, 255))
            canvas.paste(picture, ((wide - picture.width) // 2, (tall - picture.height) // 2))
            picture = canvas
        target = folder / f"shown_{index}.png"
        picture.save(target)
        out.append(target)
    return out


def flux3_edit_workflow(args, pictures: list[str], shown: list[str], prompt: str, seed: int, made: tuple[int, int] | None = None) -> dict:
    """FLUX3-K's edit on ComfyUI's FLUX 3 nodes: the pictures as reference latents, the caption from BFL's upsampler."""

    unet, steps, cfg = FLUX3[args.model]
    w = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX3_TEXT, "type": "flux3", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX3_VAE}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        # Fast is distilled without guidance (its template zeroes the negative); Full runs an empty prompt against it
        "7": ({"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}} if args.model == "flux3k-fast"
              else {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}}),
        # every picture goes to the prompt and to the negative, at the size it was brought to (edit_plan.py)
        "17": {"class_type": "ReferenceMultiLatent", "inputs": {"positive": ["5", 0], "negative": ["7", 0], "vae": ["4", 0],
                                                               "resolution": "keep resolution"}},
        "12": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": seed, "steps": args.steps if args.steps_set else steps,
                                                    "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "positive": ["17", 0],
                                                    "negative": ["17", 1], "latent_image": ["17", 2], "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": f"studio/edit_{uuid.uuid4().hex[:8]}"}},
    }
    for index, picture in enumerate(pictures):
        w[str(30 + index)] = {"class_type": "LoadImage", "inputs": {"image": picture}}
        w["17"]["inputs"][f"images.image{index}"] = [str(30 + index), 0]
    if made:  # a size asked for by hand in another shape than the picture's
        w["6"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": made[0], "height": made[1], "batch_size": 1}}
        w["12"]["inputs"]["latent_image"] = ["6", 0]
    if shown:  # BFL's upsampler writes the caption from the pictures and the prompt, inside the same workflow
        batch = ["50", 0]
        for index, picture in enumerate(shown):
            w[str(50 + index)] = {"class_type": "LoadImage", "inputs": {"image": picture}}
            if index:
                w[str(70 + index)] = {"class_type": "ImageBatch", "inputs": {"image1": batch, "image2": [str(50 + index), 0]}}
                batch = [str(70 + index), 0]
        w["20"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX3_UPSAMPLER, "type": "flux3", "device": "default"}}
        w["21"] = {"class_type": "TextGenerate", "inputs": {
            "clip": ["20", 0], "prompt": I2I_SCAFFOLD.format(prompt), "image": batch, "max_length": 3584, "sampling_mode": "on",
            "sampling_mode.temperature": 1.0, "sampling_mode.top_k": 1, "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.0,
            "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": 0, "sampling_mode.presence_penalty": 0.0,
            "thinking": False, "use_default_template": True, "mtp": "auto", "system_prompt": flux3_system_prompt("i2i")}}
        w["22"] = {"class_type": "PreviewAny", "inputs": {"source": ["21", 0]}}
        w["5"]["inputs"]["text"] = ["21", 0]
    return w
