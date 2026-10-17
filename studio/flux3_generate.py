#!/usr/bin/env python3
"""Render images with FLUX3-K (Black Forest Labs' FLUX 3 Image Klein) on a Mac, through flux3-mlx."""
# Runs in $PREFIX/flux3-venv (oneshot-setup.sh --flux3), never the Studio's venv. FLUX Kommunity License.

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

STEPS = {"flux3k-fast": 16, "flux3k-full": 50}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir", help="BFL's checkpoint folder: model.safetensors, ae.safetensors, text_encoder/")
    parser.add_argument("--model", choices=sorted(STEPS), required=True)
    parser.add_argument("--upsampler", default="", help="the prompt upsampler's folder (MLX); empty sends the prompt as written")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", help="comma-separated seeds: one image each from one load, written as NAME_s<seed>.png")
    args = parser.parse_args()
    prompt = Path(args.prompt_file).read_text().strip() if args.prompt_file else args.prompt
    if not prompt:
        parser.error("give --prompt or --prompt-file")
    for name, value in (("width", args.width), ("height", args.height)):
        if value % 16:
            parser.error(f"--{name} must be a multiple of 16, got {value}")

    import mlx.core as mx
    from flux3_mlx import Flux3Klein

    started = time.perf_counter()
    pipe = Flux3Klein.from_pretrained(args.model_dir, upsampler=args.upsampler or None, low_memory=False)
    load_s = time.perf_counter() - started
    caption, upsample_s = prompt, 0.0
    if args.upsampler:
        mark = time.perf_counter()
        caption = pipe.upsample(prompt, [])
        upsample_s = time.perf_counter() - mark
        pipe._upsampler = None  # the 9B upsampler is not needed for the pictures
        mx.clear_cache()
    steps = args.steps or STEPS[args.model]
    seeds = [int(v) for v in args.seeds.split(",")] if args.seeds else [args.seed]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    outputs = [target.with_name(f"{target.stem}_s{seed}{target.suffix}") for seed in seeds] if args.seeds else [target]
    mark = time.perf_counter()
    for seed, path in zip(seeds, outputs):
        image, _ = pipe(caption, width=args.width, height=args.height, steps=steps, seed=seed, upsample=False,
                        callback=lambda i, n: print(f"[tensorfold] step {i}/{n}", file=sys.stderr, flush=True))
        image.save(path)
    denoise_s = time.perf_counter() - mark
    print("[tensorfold] " + json.dumps({
        "output": [str(path) for path in outputs] if args.seeds else args.output, "engine": "mlx", "model": args.model,
        "width": args.width, "height": args.height, "images": len(seeds), "steps": steps,
        "upsampled": bool(args.upsampler), "caption": caption, "load_s": round(load_s, 1),
        "upsample_s": round(upsample_s, 1), "per_image_s": round(denoise_s / len(seeds), 1),
        "total_s": round(time.perf_counter() - started, 1), "peak_gib": round(mx.get_peak_memory() / 2**30, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
