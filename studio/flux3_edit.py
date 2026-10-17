#!/usr/bin/env python3
"""Edit a picture with FLUX3-K (Black Forest Labs' FLUX 3 Image Klein) on a Mac, through flux3-mlx."""
# Runs in $PREFIX/flux3-venv, never the Studio's venv; sizes and noise seed from edit_plan.py. FLUX Kommunity License.

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edit_plan  # noqa: E402  (the sizes and the noise seed, shared with the other builds)

STEPS = {"flux3k-fast": 16, "flux3k-full": 50}


def fitted(path: str, size: tuple[int, int]):
    """The picture as the model is given it: upright, RGB, at exactly ``size`` (Lanczos)."""

    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        picture = ImageOps.exif_transpose(opened).convert("RGB")
    return picture if picture.size == size else picture.resize(size, Image.Resampling.LANCZOS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir", help="BFL's checkpoint folder: model.safetensors, ae.safetensors, text_encoder/")
    parser.add_argument("--model", choices=sorted(STEPS), required=True)
    parser.add_argument("--image", required=True, action="append", help="the picture to edit; repeat for further pictures")
    parser.add_argument("--prompt", default="")
    parser.add_argument("-o", "--output", default="")
    parser.add_argument("--set", default="", help="several edits from one load: JSON [[prompt, output], ...], or @FILE holding it")
    parser.add_argument("--upsampler", default="", help="the prompt upsampler's folder (MLX); empty sends the prompt as written")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--size", choices=("source", "model"), default="source")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--most", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
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

    sizes = [edit_plan.picture_size(path) for path in args.image]
    plan = edit_plan.canvas(*sizes[0], args.size, args.resolution, (args.width, args.height) if args.width else None, args.most)
    made, size = tuple(plan["made"]), tuple(plan["size"])
    if plan["note"]:
        print(f"[tensorfold] {plan['note']}", flush=True)
    noise = edit_plan.noise_seed(args.seed, args.image) % 2**32  # flux3-mlx draws its noise from a 32-bit seed
    references = [fitted(path, fit) for path, fit in
                  zip(args.image, [tuple(plan["source"])] + [edit_plan.beside(*one, made) for one in sizes[1:]])]

    import mlx.core as mx
    from flux3_mlx import Flux3Klein

    started = time.perf_counter()
    pipe = Flux3Klein.from_pretrained(args.model_dir, upsampler=args.upsampler or None, low_memory=False)
    captions, upsample_s = [str(p) for p, _ in edits], 0.0
    if args.upsampler:  # the upsampler sees the pictures as given, and is let go before the pictures are made
        mark = time.perf_counter()
        originals = [fitted(path, one) for path, one in zip(args.image, sizes)]
        captions = [pipe.upsample(text, originals) for text in captions]
        upsample_s = time.perf_counter() - mark
        pipe._upsampler = None
        mx.clear_cache()
    steps = args.steps or STEPS[args.model]
    for caption, (prompt, output) in zip(captions, edits):
        mark = time.perf_counter()
        image, _ = pipe(caption, images=references, width=made[0], height=made[1], steps=steps, seed=noise, upsample=False,
                        callback=lambda i, n: print(f"[tensorfold] step {i}/{n}", file=sys.stderr, flush=True))
        if image.size != size:  # a side that is not on the model's grid: back to the picture's own
            from PIL import Image

            image = image.resize(size, Image.Resampling.LANCZOS)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        print("[tensorfold] " + json.dumps({
            "output": str(output), "engine": "mlx", "model": args.model, "edit": str(prompt)[:80], "seed": args.seed,
            "noise_seed": noise, "pictures": len(references), "size": list(size), "made_at": list(made), "steps": steps,
            "upsampled": bool(args.upsampler), "caption": caption, "edit_s": round(time.perf_counter() - mark, 1),
            "upsample_s": round(upsample_s, 1), "total_s": round(time.perf_counter() - started, 1),
            "peak_gib": round(mx.get_peak_memory() / 2**30, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
