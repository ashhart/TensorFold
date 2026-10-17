#!/usr/bin/env python3
"""Edit a picture: Qwen-Image-2.1 with the picture as its reference, on mflux's Qwen-Image-2.1 pipeline (MLX)."""
# Credit: the Qwen team for Qwen-Image-2.1 and Qwen-Image-2.1-Turbo, Filip Strand for mflux.

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import edit_plan


def lora_specs(specs: list[str]) -> tuple[list[str], list[float]]:
    """--lora values, PATH or PATH:STRENGTH, as (paths, strengths)."""

    paths, strengths = [], []
    for spec in specs:
        path, _, strength = spec.rpartition(":")
        try:
            value = float(strength)
        except ValueError:
            path, value = spec, 1.0
        if not path:
            path, value = spec, 1.0
        if not Path(path).is_file():
            raise SystemExit(f"no LoRA at {path}")
        paths.append(path)
        strengths.append(value)
    return paths, strengths


def own_schedule(nodes: tuple[float, ...]) -> None:
    """Make mflux's default schedule the checkpoint's raw noise levels, shifted for the size, no terminal stretch."""

    import mlx.core as mx
    from mflux.models.common.config import config as mflux_config
    from mflux.models.common.schedulers.linear_scheduler import LinearScheduler

    class Nodes(LinearScheduler):
        def _get_sigmas(self):
            if self.config.num_inference_steps != len(nodes):
                raise ValueError(f"this checkpoint runs on its own {len(nodes)} steps")
            return LinearScheduler.shift_sigmas(mx.array(list(nodes), dtype=mx.float32), self.config, use_terminal=False)

    mflux_config.LinearScheduler = Nodes


def own_sizes(sizes: list[tuple[int, int]]) -> None:
    """Make mflux leave the pictures at the sizes edit_plan.py brought them to; other shapes keep mflux's own rule."""

    from mflux.models.qwen21.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator

    rule, planned = QwenImage21LatentCreator.dimensions, {width / height: (width, height) for width, height in sizes}
    QwenImage21LatentCreator.dimensions = staticmethod(lambda resolution, ratio: planned.get(ratio) or rule(resolution, ratio))


def prepared(path: str, size: tuple[int, int], folder: Path, index: int) -> Path:
    """A picture at exactly ``size`` (Lanczos, as mflux would do it), its rotation tag applied."""

    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        picture = ImageOps.exif_transpose(opened).convert("RGBA")
        if picture.size != size:
            picture = picture.resize(size, Image.Resampling.LANCZOS)
        target = folder / f"edit_{index}.png"
        picture.save(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir")
    parser.add_argument("--image", required=True, action="append", help="the picture to edit; repeat for further pictures the prompt draws on (image 2, image 3, ...)")
    parser.add_argument("--prompt", default="", help="what to change (one picture, written to -o)")
    parser.add_argument("-o", "--output", default="")
    parser.add_argument("--set", default="", help='several edits of the one picture from one load: JSON [[prompt, output], ...], or @FILE holding it')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40, help="for a checkpoint without its own schedule")
    parser.add_argument("--size", choices=("source", "model"), default="source",
                        help="source: the result has the picture's own size (the default); model: about --resolution pixels squared, keeping its shape")
    parser.add_argument("--resolution", type=int, default=1024, help="with --size model: the picture is brought to about this many pixels squared")
    parser.add_argument("--most", type=int, default=0, help="with --size source: the most pixels an edit is made at (0: 1536x1536)")
    parser.add_argument("--width", type=int, default=0, help="with --height: another size for the result (the edit shifts when its shape differs from the picture's)")
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--lora", action="append", default=[], help="an image LoRA, PATH or PATH:STRENGTH (repeat to stack)")
    parser.add_argument("--quantize", type=int, default=0, choices=(0, 4, 6, 8), help="bits for the weights; 0 keeps bfloat16 (8 for a 64 GB Mac)")
    args = parser.parse_args()
    for path in args.image:
        if not Path(path).is_file():
            parser.error(f"no picture at {path}")
    if len(args.image) > edit_plan.PICTURES:
        parser.error(f"at most {edit_plan.PICTURES} pictures in one edit, got {len(args.image)}")
    if bool(args.width) != bool(args.height):
        parser.error("--width and --height go together")
    if args.width and edit_plan.refused(args.width, args.height):
        parser.error(edit_plan.refused(args.width, args.height))
    edits = json.loads(Path(args.set[1:]).read_text() if args.set.startswith("@") else args.set) if args.set else [[args.prompt, args.output]]
    if not edits or any(not str(p).strip() or not o for p, o in edits):
        parser.error("give --prompt and -o, or --set with a prompt and an output for each edit")
    paths, strengths = lora_specs(args.lora)
    index = json.loads((Path(args.model_dir) / "model_index.json").read_text())
    nodes = tuple(float(v) for v in index.get("sample_sigmas") or ())
    steps = len(nodes) or args.steps

    import mlx.core as mx
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

    if nodes:
        own_schedule(nodes)
    # the size the edit is made at and the size that comes back, and the noise for this seed and these pictures
    sizes = [edit_plan.picture_size(path) for path in args.image]
    plan = edit_plan.canvas(*sizes[0], args.size, args.resolution, (args.width, args.height) if args.width else None, args.most)
    made, size = tuple(plan["made"]), tuple(plan["size"])
    if plan["note"]:
        print(f"[tensorfold] {plan['note']}", flush=True)
    fitted = [tuple(plan["source"])] + [edit_plan.beside(*one, made) for one in sizes[1:]]
    own_sizes(fitted)
    noise = edit_plan.noise_seed(args.seed, args.image)
    scratch = tempfile.TemporaryDirectory(prefix="studio-edit-")
    pictures = [str(prepared(path, fit, Path(scratch.name), index)) for index, (path, fit) in enumerate(zip(args.image, fitted))]
    started = time.perf_counter()
    model = QwenImage21Edit(quantize=args.quantize or None, model_path=args.model_dir, model_config=ModelConfig.qwen_image_21(),
                            lora_paths=paths or None, lora_scales=strengths or None)
    load_s = time.perf_counter() - started
    for prompt, output in edits:
        mark = time.perf_counter()
        result = model.generate_image(seed=noise, prompt=prompt, num_inference_steps=steps, image_paths=pictures,
                                      width=made[0], height=made[1], output_resolution=args.resolution)
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        picture = result.image
        if picture.size != size:  # a side that is not on the model's grid: the result is brought back to the picture's own
            from PIL import Image

            picture = picture.resize(size, Image.Resampling.LANCZOS)
        picture = picture.convert("RGB") if target.suffix.lower() in (".jpg", ".jpeg") else picture
        picture.save(target)
        print("[tensorfold] " + json.dumps({"output": str(target), "engine": "mlx", "edit": prompt[:80], "seed": args.seed, "noise_seed": noise, "steps": steps,
                                            "turbo": bool(nodes), "loras": [Path(p).name for p in paths], "pictures": len(pictures),
                                            "size": list(picture.size), "made_at": list(made),
                                            "load_s": round(load_s, 1), "total_s": round(time.perf_counter() - mark, 1),
                                            "peak_gib": round(mx.get_peak_memory() / 2**30, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
