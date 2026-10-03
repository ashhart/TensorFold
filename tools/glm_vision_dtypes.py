"""Show GLM's CUDA vision tower both ways: tensors in their stored dtypes, and an all-bf16 copy of the same tower.

    python tools/glm_vision_dtypes.py MODEL_DIR IMAGE [--device cuda] [--server URL [--server-b URL]]
    python tools/glm_vision_dtypes.py --synthetic [IMAGE]   # tiny mixed-dtype tower, random pixels, CPU (IMAGE: --server only)

Prints the dtype of every tensor kind, then max/mean abs difference of the image features (both outputs widened to F32,
as `encode` hands them to the engine in bf16 after this). With --server it runs one image prompt twice: against URL
twice (run-to-run baseline) or, with --server-b, against two servers (say, the stored-dtype build and the bf16-cast
one), printing both replies' token ids and the first position where they differ."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import re
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import torch

from tensorfold.vision.glm_cuda import build_tower, keep_stored_dtypes

TINY = {"model_type": "glm5_next_vision", "hidden_size": 32, "out_hidden_size": 32, "depth": 2, "patch_size": 4,
        "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3, "intermediate_size": 64, "num_heads": 2,
        "projection_intermediate_size": 48, "attention_bias": True}


def synthetic(path: Path) -> None:
    """F32 norms, an F16 last linear, BF16 elsewhere: the mix the tests use."""

    from safetensors.torch import save_file
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    (path / "config.json").write_text(json.dumps({"model_type": "glm5_next", "image_token_id": 99,
                                                  "vision_config": TINY, "text_config": {"hidden_size": 32}}))
    torch.manual_seed(0)
    weights = {}
    for name, value in Glm5NextVisionModel(Glm5NextVisionConfig(**TINY)).state_dict().items():
        norm = "norm" in name and name.endswith("weight")
        weights["model.visual." + name] = (1 + 0.05 * torch.randn_like(value) if norm else value).to(
            torch.float32 if norm else torch.float16 if name == "merger.down_proj.weight" else torch.bfloat16)
    save_file(weights, str(path / "model.safetensors"))


def pixels_for(model_dir: Path, image: str | None, config: dict):
    if image is None:                                    # synthetic: two random images, 8x8 and 4x16 patches
        grid = torch.tensor([[1, 8, 8], [1, 4, 16]])
        width = config["in_channels"] * config["temporal_patch_size"] * config["patch_size"] ** 2
        return torch.randn(int(grid.prod(-1).sum()), width), grid
    from PIL import Image

    from tensorfold.vision.glm_processing import GLMImageProcessor

    done = GLMImageProcessor.from_directory(model_dir).processor.image_processor(
        [Image.open(image).convert("RGB")], return_tensors="np", min_image_tokens=16, max_image_tokens=1024)
    return torch.tensor(done["pixel_values"]), torch.tensor(done["image_grid_thw"])


def encode(tower, pixels, grid, device):
    with torch.inference_mode():
        pixels = pixels.to(device=device, dtype=tower.patch_embed.proj.weight.dtype)
        return tower(pixels, grid_thw=grid.to(device), return_dict=True).pooler_output.float().cpu()


def chat(url: str, image: str, prompt: str) -> list[int]:
    data = "data:image/png;base64," + base64.b64encode(Path(image).read_bytes()).decode()
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data}},
                                                      {"type": "text", "text": prompt}]}],
            "temperature": 0, "max_tokens": 48, "return_token_ids": True}
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)["tensorfold"]["token_ids"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir", nargs="?")
    ap.add_argument("image", nargs="?")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--server")
    ap.add_argument("--server-b")
    ap.add_argument("--prompt", default="Describe this image in detail.")
    args = ap.parse_args()
    scratch = tempfile.TemporaryDirectory()
    image = args.image
    if args.synthetic:
        model_dir, image = Path(scratch.name), args.model_dir
        synthetic(model_dir)
    elif args.model_dir and args.image:
        model_dir = Path(args.model_dir)
    else:
        ap.error("give MODEL_DIR and IMAGE, or --synthetic")
    stored = build_tower(model_dir, args.device)
    cast = copy.deepcopy(stored)                         # the old loader's result: every parameter cast to bf16
    for p in cast.parameters():
        p.data = p.data.to(torch.bfloat16)
    keep_stored_dtypes(stored)
    keep_stored_dtypes(cast)

    kinds = defaultdict(set)                             # tensor kind (blocks.N. removed) -> dtypes in the checkpoint
    for name, p in stored.named_parameters():
        kinds[re.sub(r"blocks\.\d+", "blocks.*", name)].add(str(p.dtype).removeprefix("torch."))
    for kind, dtypes in sorted(kinds.items()):
        print(f"  {kind:42s} {','.join(sorted(dtypes))}")

    pixels, grid = pixels_for(model_dir, None if args.synthetic else image, stored.config.to_dict())
    a, b = encode(stored, pixels, grid, args.device), encode(cast, pixels, grid, args.device)
    diff = (a - b).abs()
    print(f"features {tuple(a.shape)}: mean |stored| {a.abs().mean():.3e}; "
          f"max |stored - bf16| {diff.max():.3e}, mean {diff.mean():.3e}")
    if args.server and image:
        first, second = chat(args.server, image, args.prompt), chat(args.server_b or args.server, image, args.prompt)
        at = next((i for i, (x, y) in enumerate(zip(first, second)) if x != y), None)
        print("reply A ids:", first, "\nreply B ids:", second,
              "\nidentical" if first == second else f"\nfirst difference at token {at} (A {len(first)}, B {len(second)})")


if __name__ == "__main__":
    main()
