"""LFM2-VL picture features on the CPU from transformers' Lfm2VlModel: vision.py MODEL_DIR OUT."""

import base64
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration

from images import cap
from tensorfold.vision.images import ImageSource, load_images

NAMES = ("turned", "banner", "photo")  # one tile, a wide tile, six tiles and a thumbnail


def features(processor, model, pic):
    """Lfm2VlModel.get_image_features over one picture's tiles, padded only to its longest tile."""
    out = processor(text=["<image>"], images=[[pic]], return_tensors="pt", add_special_tokens=False)
    pv, mask = out["pixel_values"], out["pixel_attention_mask"]
    n = int(mask.sum(1).max())
    tiles = model.get_image_features(pixel_values=pv[:, :n].to(torch.bfloat16), spatial_shapes=out["spatial_shapes"],
                                     pixel_attention_mask=mask[:, :n]).pooler_output
    return torch.cat(tiles).to(torch.bfloat16)


def picked(rows):
    """The rows kept whole: the first, the quarters and the last."""
    return sorted({0, rows // 4, rows // 2, 3 * rows // 4, rows - 1})


@torch.inference_mode()
def main(model_dir, out_path):
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Lfm2VlForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16).model
    pictures = {c["name"]: c["url"] for c in json.loads((Path(__file__).parent / "images.json").read_text())["cases"]}
    cases = []
    for name in NAMES:
        (img,) = load_images([ImageSource(pictures[name])])
        f = features(processor, model, cap(img.to_pil()))
        wide = f.float()
        rows = picked(f.shape[0])
        cases.append({"name": name, "rows": f.shape[0], "norms": wide.norm(dim=1).tolist(), "sums": wide.sum(dim=1).tolist(),
                      "picked": rows, "values": base64.b64encode(f[rows].view(torch.int16).numpy().tobytes()).decode()})
    with open(out_path, "w") as out:
        json.dump({"cases": cases}, out)
        out.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
