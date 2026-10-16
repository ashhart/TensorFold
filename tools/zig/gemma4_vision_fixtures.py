"""Gemma 4 images (the 31B's tower, the 12B's embedder) against transformers' on drawn images (tf-gemma4-vision)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open


def draw_shapes(size: tuple[int, int]) -> Image.Image:
    w, h = size
    im = Image.new("RGB", size, (245, 240, 230))
    d = ImageDraw.Draw(im)
    d.ellipse((w * 0.08, h * 0.25, w * 0.42, h * 0.7), fill=(210, 40, 40))
    d.rectangle((w * 0.55, h * 0.25, w * 0.9, h * 0.7), fill=(40, 70, 200))
    for x in range(int(w * 0.1), int(w * 0.9), 6):
        d.line((x, h * 0.8, x, h * 0.92), fill=(20, 20, 20), width=2)
    d.text((w * 0.4, h * 0.06), "GEMMA 4", fill=(0, 0, 0))
    return im


def fixtures(folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []

    def save(name: str, im: Image.Image, **kw) -> None:
        p = folder / name
        im.save(p, **kw)
        out.append(p)

    save("shapes.png", draw_shapes((517, 389)))                                     # a mild downscale
    big = draw_shapes((1200, 800))
    px = big.load()
    for y in range(0, 800, 3):                                                       # fine detail for the bicubic filter
        for x in range(0, 1200, 7):
            px[x, y] = ((x * 3) % 256, (y * 5) % 256, (x + y) % 256)
    save("wide.png", big)                                                            # 3:2, the full 280-token budget
    save("tall.png", draw_shapes((180, 1400)))                                       # one 48-px column wide
    save("tiny.png", draw_shapes((20, 15)))                                          # upscaled
    save("square.png", draw_shapes((672, 672)))                                      # already a multiple of 48
    return out


def tensors(model: Path, prefix: str) -> dict[str, torch.Tensor]:
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for name, shard in index.items():
        if name.startswith(prefix):
            with safe_open(model / shard, "pt") as f:
                out[name[len(prefix):]] = f.get_tensor(name)
    return out


def dequantize(w: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, group: int, bits: int) -> torch.Tensor:
    """MLX affine: `bits`-wide codes packed little-endian through each row's uint32 words; a scale and bias a group."""
    words = (w.to(torch.int64) & 0xFFFFFFFF).numpy().astype("<u4")
    stream = np.unpackbits(words.view(np.uint8).reshape(w.shape[0], -1), axis=1, bitorder="little")
    codes = stream.reshape(w.shape[0], -1, bits) @ (1 << np.arange(bits))
    codes = torch.from_numpy(codes.astype(np.float32))
    s = scales.float().repeat_interleave(group, dim=1)
    b = biases.float().repeat_interleave(group, dim=1)
    return (codes * s + b).to(torch.bfloat16)


def projection_width(model: Path) -> tuple[int, int]:
    """embed_vision.embedding_projection's (bits, group): its own config entry, else the checkpoint's."""
    q = json.loads((model / "config.json").read_text())["quantization"]
    entry = q.get("embed_vision.embedding_projection", q)
    return entry["bits"], entry["group_size"]


def build(model: Path, dtype: torch.dtype):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel

    cfg = Gemma4Config.from_pretrained(model)
    cfg.vision_config._attn_implementation = "eager"
    tower = Gemma4VisionModel(cfg.vision_config)
    missing, unexpected = tower.load_state_dict(tensors(model, "vision_tower."), strict=False)
    if [m for m in missing if "inv_freq" not in m] or unexpected:
        raise SystemExit(f"vision tower tensors: missing {missing}, unexpected {unexpected}")
    embed = Gemma4MultimodalEmbedder(cfg.vision_config, cfg.text_config)
    q = tensors(model, "embed_vision.embedding_projection.")
    bits, group = projection_width(model)
    embed.embedding_projection.weight.data = dequantize(q["weight"], q["scales"], q["biases"], group, bits)
    return tower.to(dtype).eval(), embed.to(dtype).eval(), cfg


def build_unified(model: Path, dtype: torch.dtype):
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedVisionEmbedder

    cfg = Gemma4UnifiedConfig.from_pretrained(model)
    embedder = Gemma4UnifiedVisionEmbedder(cfg.vision_config, cfg.text_config)
    state = tensors(model, "vision_embedder.")
    q = tensors(model, "embed_vision.embedding_projection.")
    projection = q["weight"]
    if "scales" in q:  # an oQ pack's quantized projection
        bits, group = projection_width(model)
        projection = dequantize(q["weight"], q["scales"], q["biases"], group, bits)
    state["multimodal_embedder.embedding_projection.weight"] = projection
    embedder.load_state_dict(state, strict=True)
    return embedder.to(dtype).eval()


@torch.no_grad()
def unified_stages(embedder, pixels: torch.Tensor, positions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Gemma4UnifiedVisionEmbedder.forward's steps: the patch norms and Linear, positions and pos_norm, projection."""
    dtype = next(embedder.parameters()).dtype
    x = embedder.patch_ln2(embedder.patch_dense(embedder.patch_ln1(pixels.to(dtype))))
    out = {"patch": x[0]}
    axes = torch.arange(2)
    x = embedder.pos_norm(x + embedder.pos_embedding[positions.long(), axes].sum(-2))
    out["posed"] = x[0]
    out["features"] = embedder(pixels.to(dtype), positions).pooler_output[0]
    return out


def merged(pil, image: Image.Image, max_tokens: int):
    """The PIL processor's 16-px patches (its 8-bit resize, the engine's) through the unified processor's merge."""
    from transformers.models.gemma4_unified.image_processing_gemma4_unified import patches_merge

    done = pil([image], return_tensors="pt", max_soft_tokens=max_tokens)
    real = ~(done["image_position_ids"] == -1).all(dim=-1)[0]
    teacher, positions = done["pixel_values"][0][real], done["image_position_ids"][0][real]
    count = int(done["num_soft_tokens_per_image"][0])
    pixels, merged_positions = patches_merge(teacher.unsqueeze(0), positions.unsqueeze(0), count)
    return pixels, merged_positions, (positions.max(dim=0).values + 1).tolist()


def unified_reference(model: Path, image_path: Path, out: Path, runs, processor, pil, max_tokens: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    pixels, positions, grid = merged(pil, image, max_tokens)
    np.asarray(pixels[0], dtype=np.float32).tofile(out / "patches.f32")
    np.asarray(positions[0], dtype=np.int32).tofile(out / "positions.i32")
    for suffix, embedder in runs.items():
        for name, a in unified_stages(embedder, pixels, positions).items():
            np.asarray(a.float(), dtype=np.float32).tofile(out / f"{name}{suffix}.f32")
    # the checkpoint's own processor (torchvision's resize) beside the PIL one, as levels of 1/255
    own = processor.image_processor([image], return_tensors="pt", max_soft_tokens=max_tokens)["pixel_values"]
    own = own[0][: pixels.shape[1]]
    levels = int(((own - pixels[0]).abs().max() * 255).round())
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = processor(text=[text], images=[image], return_tensors="np", images_kwargs={"max_soft_tokens": max_tokens})
    (out / "prompt.ids.json").write_text(json.dumps(ids["input_ids"][0].tolist()) + "\n")
    meta = {"image": str(image_path), "size": list(image.size), "patches": grid[0] * grid[1], "grid_wh": grid,
            "tokens": int(pixels.shape[1]), "max_tokens": max_tokens, "prompt": text, "torchvision_levels": levels}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


@torch.no_grad()
def stages(tower, embed, pixels: torch.Tensor, positions: torch.Tensor) -> dict[str, torch.Tensor]:
    dtype = next(tower.parameters()).dtype
    pad = (positions == -1).all(dim=-1)
    x = tower.patch_embedder(pixels.to(dtype), positions, pad)
    out = {"embed": x[~pad]}
    mask = torch.zeros(1, 1, x.shape[1], x.shape[1], dtype=dtype)
    mask[..., pad[0]] = torch.finfo(dtype).min
    cos_sin = tower.encoder.rotary_emb(x, positions)
    for i, layer in enumerate(tower.encoder.layers):
        x = layer(x, attention_mask=mask, position_embeddings=cos_sin, position_ids=positions)
        if i == 0:
            out["block0"] = x[~pad]
    out["blocks"] = x[~pad]
    pooled = tower(pixel_values=pixels.to(dtype), pixel_position_ids=positions).last_hidden_state
    out["pooled"] = pooled
    out["features"] = embed(pooled)
    return out


def reference(model: Path, image_path: Path, out: Path, runs, processor, pil, max_tokens: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    done = pil([image], return_tensors="pt", max_soft_tokens=max_tokens)
    pixels, positions = done["pixel_values"], done["image_position_ids"]
    real = ~(positions == -1).all(dim=-1)[0]
    np.asarray(pixels[0][real], dtype=np.float32).tofile(out / "patches.f32")
    np.asarray(positions[0][real], dtype=np.int32).tofile(out / "positions.i32")
    for suffix, (tower, embed) in runs.items():
        for name, a in stages(tower, embed, pixels, positions).items():
            np.asarray(a.float().reshape(-1, a.shape[-1]), dtype=np.float32).tofile(out / f"{name}{suffix}.f32")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = processor(text=[text], images=[image], return_tensors="np", images_kwargs={"max_soft_tokens": max_tokens})
    (out / "prompt.ids.json").write_text(json.dumps(ids["input_ids"][0].tolist()) + "\n")
    if "mm_token_type_ids" in ids:
        (out / "prompt.types.json").write_text(json.dumps(ids["mm_token_type_ids"][0].tolist()) + "\n")
    grid = positions[0][real].max(dim=0).values + 1
    meta = {"image": str(image_path), "size": list(image.size), "patches": int(real.sum()), "grid_wh": grid.tolist(),
            "tokens": int(done["num_soft_tokens_per_image"][0]), "max_tokens": max_tokens, "prompt": text}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="the 31B (mlx-community/gemma-4-31B-it-4bit) or the converted 12B")
    ap.add_argument("--out", required=True, help="where the images and their references go")
    ap.add_argument("--max-tokens", type=int, default=280, help="soft tokens an image is sized to: 70, 140 or 280")
    args = ap.parse_args()
    from transformers import AutoProcessor

    model, out = Path(args.model), Path(args.out)
    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil

    processor = AutoProcessor.from_pretrained(model)
    # the PIL processor, whose 8-bit resize the engine matches (AutoProcessor's torchvision one differs by a level)
    config = json.loads((model / "processor_config.json").read_text())["image_processor"]
    pil = Gemma4ImageProcessorPil(**{k: v for k, v in config.items() if k != "image_processor_type"})
    if json.loads((model / "config.json").read_text())["model_type"] == "gemma4_unified":
        runs = {"": build_unified(model, torch.bfloat16), "_fp32": build_unified(model, torch.float32)}
        for path in fixtures(out / "images"):
            meta = unified_reference(model, path, out / path.stem, runs, processor, pil, args.max_tokens)
            size, grid = meta["size"], meta["grid_wh"]
            print(f"{path.name}: {size[0]}x{size[1]} -> grid {grid[0]}x{grid[1]}, {meta['tokens']} tokens, "
                  f"torchvision's resize {meta['torchvision_levels']} levels from PIL's")
        return 0
    tower, embed, _ = build(model, torch.bfloat16)
    tower32, embed32, _ = build(model, torch.float32)
    runs = {"": (tower, embed), "_fp32": (tower32, embed32)}
    for path in fixtures(out / "images"):
        meta = reference(model, path, out / path.stem, runs, processor, pil, args.max_tokens)
        print(f"{path.name}: {meta['size'][0]}x{meta['size'][1]} -> grid {meta['grid_wh'][0]}x{meta['grid_wh'][1]}, {meta['patches']} patches, {meta['tokens']} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
