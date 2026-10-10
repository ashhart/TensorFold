"""GLM-5.3-Flash image input against the Python frontend, on a fixed image set (#565's gate).

    python -I tools/zig/glm_vision_fixtures.py --model DIR --out DIR [--tower zig-out/bin/tf-glm-vision]

Draws the image set (each case a preprocessing path: padding only, an antialiased downscale, an upscale, alpha over
white, a palette with a transparent entry, gray with alpha, EXIF orientation, baseline JPEG), then for each image
writes what TensorFold 0.6.6's --vision gives (vision/images.py's decode: EXIF orientation, alpha over white; then
mlx-vlm's Glm5NextImageProcessor and tower, the code 0.6.6 serves):
  patches.f32   (grid_h*grid_w, 1176) normalized patches, temporal frames duplicated
  embed.f32 / block0.f32 / blocks.f32 / features.f32   the tower's stages, features the rows that replace <|image|>
and meta.json. With --tower it then runs tf-glm-vision on each image: the native patches compared value by value,
the native tower's rows by cosine (bf16 against bf16).

Needs TensorFold 0.6.6 with its vision extra (mlx-vlm 0.7.x) and Pillow. Loads only model.visual.* (about 1 GB).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw


def draw_shapes(size: tuple[int, int]) -> Image.Image:
    w, h = size
    im = Image.new("RGB", size, (245, 240, 230))
    d = ImageDraw.Draw(im)
    d.ellipse((w * 0.08, h * 0.25, w * 0.42, h * 0.7), fill=(210, 40, 40))
    d.rectangle((w * 0.55, h * 0.25, w * 0.9, h * 0.7), fill=(40, 70, 200))
    for x in range(int(w * 0.1), int(w * 0.9), 6):
        d.line((x, h * 0.8, x, h * 0.92), fill=(20, 20, 20), width=2)
    d.text((w * 0.4, h * 0.06), "GLM 7", fill=(0, 0, 0))
    return im


def fixtures(folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []

    def save(name: str, im: Image.Image, **kw) -> None:
        p = folder / name
        im.save(p, **kw)
        out.append(p)

    save("shapes.png", draw_shapes((517, 389)))                                     # padding only
    big = draw_shapes((2000, 1500))
    px = big.load()
    for y in range(0, 1500, 3):                                                      # fine detail for the antialias
        for x in range(0, 2000, 7):
            px[x, y] = ((x * 3) % 256, (y * 5) % 256, (x + y) % 256)
    save("large.png", big)                                                           # antialiased downscale
    save("tiny.png", draw_shapes((20, 15)))                                          # upscale to the token floor
    logo = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    d = ImageDraw.Draw(logo)
    d.rounded_rectangle((4, 4, 92, 92), 16, fill=(30, 30, 40, 255))
    d.polygon([(26, 28), (70, 28), (30, 68), (70, 68), (70, 74), (24, 74), (64, 34), (26, 34)], fill=(255, 255, 255, 230))
    save("logo.png", logo)                                                           # RGBA upscale, alpha over white
    pal = draw_shapes((517, 389)).quantize(32)
    pal.info["transparency"] = 0
    save("palette.png", pal, transparency=0)                                         # palette, a transparent entry
    la = Image.new("LA", (240, 180), (128, 0))
    d = ImageDraw.Draw(la)
    for r in range(80, 0, -8):
        d.ellipse((120 - r, 90 - r, 120 + r, 90 + r), fill=(255 - 3 * r, 255 - 2 * r))
    save("gray-alpha.png", la)                                                       # gray with alpha
    rot = draw_shapes((300, 200))
    exif = Image.Exif()
    exif[0x0112] = 6                                                                 # rotate 90 degrees clockwise
    save("rotated.jpg", rot, quality=90, exif=exif.tobytes())                        # EXIF orientation
    save("shapes.jpg", draw_shapes((517, 389)), quality=85)                          # baseline JPEG
    return out


def reference(model: Path, image_path: Path, out: Path, tower, processor, max_tokens: int) -> dict:
    from tensorfold.vision.images import DEFAULT_LIMITS, _decode

    out.mkdir(parents=True, exist_ok=True)
    image = _decode(image_path.read_bytes(), "auto", DEFAULT_LIMITS, DEFAULT_LIMITS.max_total_pixels).to_pil()
    done = processor([image], return_tensors="np", min_image_tokens=min(16, max_tokens), max_image_tokens=max_tokens)
    patches = np.asarray(done["pixel_values"], dtype=np.float32)
    grid = np.asarray(done["image_grid_thw"], dtype=np.int64)
    dtype = tower.patch_embed.proj.weight.dtype
    x = mx.array(patches).astype(dtype)
    g = mx.array(grid, dtype=mx.int32)
    h = tower.patch_embed(x)
    embed = h
    pos = tower._rotary_embeddings(g)
    cu = mx.array([0, int(h.shape[0])])
    block0 = None
    for i, block in enumerate(tower.blocks):
        h = block(h, cu, pos)
        if i == 0:
            block0 = h
    blocks = h
    features = tower(x, g)
    mx.eval(embed, block0, blocks, features)
    for name, a in (("patches", patches), ("embed", embed), ("block0", block0), ("blocks", blocks), ("features", features)):
        np.asarray(a.astype(mx.float32) if isinstance(a, mx.array) else a, dtype=np.float32).tofile(out / f"{name}.f32")
    meta = {"image": str(image_path), "size": list(image.size), "mode": Image.open(image_path).mode, "grid_thw": grid[0].tolist(),
            "max_tokens": max_tokens, "tokens": int(features.shape[0]), "patch_width": int(patches.shape[1]), "dtype": str(dtype)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="the GLM-5.3-Flash checkpoint folder")
    ap.add_argument("--out", required=True, help="where the images and their references go")
    ap.add_argument("--max-tokens", type=int, default=4096, help="the visual-token budget an image is sized to")
    ap.add_argument("--tower", help="tf-glm-vision: also compare the native patches and tower with each reference")
    args = ap.parse_args()

    from mlx_vlm.models.glm5_next.config import VisionConfig
    from mlx_vlm.models.glm5_next.processing import Glm5NextImageProcessor
    from mlx_vlm.models.glm5_next.vision import VisionModel

    model, out = Path(args.model), Path(args.out)
    config = json.loads((model / "config.json").read_text())
    proc_cfg = json.loads((model / "processor_config.json").read_text())["image_processor"]
    proc_cfg.pop("image_processor_type", None)
    processor = Glm5NextImageProcessor(**proc_cfg)
    tower = VisionModel(VisionConfig.from_dict(config["vision_config"]))
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for shard in sorted({f for k, f in index.items() if k.startswith("model.visual.")}):
        for k, v in mx.load(str(model / shard)).items():
            if k.startswith("model.visual."):
                weights[k[len("model.visual."):]] = v
    tower.load_weights(list(tower.sanitize(weights).items()), strict=True)
    mx.eval(tower.parameters())

    failures = 0
    for image in fixtures(out / "images"):
        ref = out / image.name.replace(".", "-")
        meta = reference(model, image, ref, tower, processor, args.max_tokens)
        line = f"{image.name:15} {meta['size'][0]}x{meta['size'][1]} {meta['mode']:5} grid {meta['grid_thw']} {meta['tokens']} tokens"
        if args.tower:
            run = subprocess.run([args.tower, str(model), str(ref), "features", str(image)], capture_output=True, text=True)
            text = run.stdout + run.stderr
            p = re.search(r"patches: (\d+) of (\d+) values differ", text)
            c = re.search(r"cosine mean ([0-9.]+) worst ([0-9.]+)", text)
            if run.returncode != 0 or not p or not c:
                failures += 1
                line += f"  FAIL: {text.strip().splitlines()[-1] if text.strip() else run.returncode}"
            else:
                differ = int(p.group(1))
                exact = differ == 0
                # PNG patches must be bit-exact; JPEG patches differ by decoder (ImageIO against libjpeg-turbo)
                failures += (not exact) and image.suffix == ".png"
                line += f"  patches {'equal' if exact else f'{differ} of {p.group(2)} differ'}, rows' cosine {c.group(1)} (worst {c.group(2)})"
        print(line)
    print("PASS" if failures == 0 else f"FAIL: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
