#!/usr/bin/env python3
"""Step-5's image path in torch: vLLM's Step3-VL preprocessing, StepFun's encoder, downsamplers and projector."""
# Usage: step5_vision_ref.py --model DIR --stepfun SRC/inference --image PNG --out DIR (pieces' pixels, rows, layout)

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.transforms import InterpolationMode

MEAN, STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
MAX_SIDE = 3024


def window(long: int, short: int) -> int:  # ImagePatcher.determine_window_size
    if long < 728:
        return short if long / short > 1.5 else 0
    return min(short, 504) if long / short > 4 else 504


def crop_size(side: int, win: int) -> int:  # get_image_size_for_crop, one side
    ratio = side / win
    if ratio < 1:
        return side
    return win * (int(ratio) + 1 if ratio - side // win > 0.2 else int(ratio))


def starts(side: int, win: int) -> list[int]:  # slide_window, one side
    n = 1 if side <= win else math.ceil((side - win) / win + 1)
    out = [win * i for i in range(n)]
    if len(out) > 1 and out[-1] + win > side:
        out[-1] = side - win
    return out


def pieces(img: Image.Image) -> list[tuple[str, torch.Tensor, bool]]:
    """vLLM's ImagePatcher and Step3VisionProcessor: (kind, pixels [3, S, S], newline after) crops, then overview."""
    w, h = img.size
    if min(w, h) < 32 and (w / h > 4 or w / h < 1 / 4):
        side = max(w, h)
        padded = Image.new(img.mode, (side, side), 0)
        padded.paste(img, (0, 0))
        img, (w, h) = padded, (side, side)
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        w, h = int(w * s), int(h * s)
    img = img.resize((w, h), Image.Resampling.BILINEAR)

    def tf(size):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD),
                                   transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR, antialias=True)])

    out = []
    win = window(max(w, h), min(w, h))
    if win:
        cw, ch = crop_size(w, win), crop_size(h, win)
        grid = img.resize((cw, ch), Image.Resampling.BILINEAR) if (cw, ch) != (w, h) else img
        xs, ys = starts(cw, win), starts(ch, win)
        for (r, y0), (c, x0) in product(enumerate(ys), enumerate(xs)):
            crop = grid.crop((x0, y0, x0 + win, y0 + win))
            out.append(("crop", tf(504)(crop), c + 1 == len(xs) and r + 1 < len(ys)))
    out.append(("overview", tf(728)(img), False))
    return out


def layout(ps) -> list[int]:
    """The placeholder's expansion: <patch_start> 81 x <im_patch> <patch_end> [<patch_newline>], then the overview."""
    ids = []
    for kind, _, newline in ps:
        if kind == "crop":
            ids += [128003] + [128001] * 81 + [128005] + ([128004] if newline else [])
        else:
            ids += [128000] + [128001] * 169 + [128002]
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--stepfun", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fp32", action="store_true", help="run the tower in fp32 (a ground truth for bf16 runs)")
    ap.add_argument("--pieces", type=int, nargs="*", help="only these pieces (by index)")
    a = ap.parse_args()
    sys.path.insert(0, str(a.stepfun))
    from vision_encoder import Step5VisionConfig, Step5VisionEncoder

    cfg = json.loads((a.model / "config.json").read_text())
    idx = json.loads((a.model / "model.safetensors.index.json").read_text())["weight_map"]
    sd = {}
    for f in sorted({v for k, v in idx.items() if k.startswith(("vision_model.", "vit_large_projector"))}):
        sd.update({k: v for k, v in load_file(str(a.model / f)).items() if k.startswith(("vision_model.", "vit_large_projector"))})
    enc = Step5VisionEncoder(Step5VisionConfig(**{k: v for k, v in cfg["vision_config"].items() if k != "model_type"})).to(torch.float32 if a.fp32 else torch.bfloat16)
    enc.load_state_dict({k[len("vision_model."):]: v for k, v in sd.items() if k.startswith("vision_model.")})
    dt = torch.float32 if a.fp32 else torch.bfloat16
    proj = sd["vit_large_projector.weight"].to(dt)
    a.out.mkdir(parents=True, exist_ok=True)
    ps = pieces(Image.open(a.image).convert("RGB"))
    meta = []
    with torch.inference_mode():
        for i, (kind, px, newline) in enumerate(ps):
            if a.pieces is not None and i not in a.pieces:
                continue
            x = enc(px[None].to(dt))  # [1, g*g, 1536]
            g = px.shape[-1] // 14
            x = x.transpose(1, 2).reshape(1, -1, g, g)
            x = enc.vit_downsampler2(enc.vit_downsampler1(x))
            rows = torch.nn.functional.linear(x.flatten(2).transpose(1, 2), proj)[0]
            np.save(a.out / f"piece{i}_pixels.npy", px.float().numpy())
            np.save(a.out / f"piece{i}_rows.npy", rows.float().numpy())
            meta.append({"kind": kind, "side": px.shape[-1], "rows": rows.shape[0], "newline": newline})
    (a.out / "layout.json").write_text(json.dumps({"pieces": meta, "ids": layout(ps)}) + "\n")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
