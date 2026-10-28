"""Conditioning rows for h3_generate.py: keyframes and references through the VAEs and text encoder, and layout."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from h3_media import QWEN_FRAMES_PER_BLOCK, encoded_frames, sample_video
from h3_text import _text_encoder, vision_stats

def encode_first_frame(root: Path, image, width: int, height: int, patch):
    """Conditioning rows for the keyframes (one or a list, first then last) from minimax-h3-mlx's video VAE."""

    from minimax_h3_mlx.load import load_video_vae
    from minimax_h3_mlx.pipeline import encode_keyframe_rows

    vae = load_video_vae(root / "video_vae")
    rows = encode_keyframe_rows(vae, list(image) if isinstance(image, (list, tuple)) else [image], height, width, patch)
    mx.eval(rows)
    del vae
    gc.collect()
    mx.clear_cache()
    return rows


_POSITION_SCALE = 32  # the spatial axes of the rotary positions, as in tensorfold.families.h3.packing


def encode_references(root: Path, images, patch):
    """Rows of the reference pictures (VAE posterior mean, normalised, patchified) and each latent (height, width)."""

    from minimax_h3_mlx.load import load_video_vae
    from minimax_h3_mlx.pipeline import PIXEL_MEAN, PIXEL_STD

    from tensorfold.families.h3.packing import patchify

    vae = load_video_vae(root / "video_vae")
    cfg = vae.config
    latents_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
    latents_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
    pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    rows, shapes = [], []
    for image in images:
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None, :, None]
        pixels = (pixels / 255.0 - pixel_mean) / pixel_std
        moments = vae._encode_clip(mx.array(pixels).transpose(0, 2, 3, 4, 1))  # (1, 1, h, w, 2 C), channels last
        mean = moments[..., :cfg.latent_channels].transpose(0, 4, 1, 2, 3).astype(mx.float32)
        if mean.shape[2] != 1:
            raise ValueError(f"a reference picture should encode to one latent frame, got {mean.shape[2]}")
        normalized = (mean - latents_mean) / latents_std
        rows.append(patchify(normalized, patch))
        shapes.append((int(mean.shape[3]), int(mean.shape[4])))
    rows = mx.concatenate(rows)
    mx.eval(rows)
    del vae
    gc.collect()
    mx.clear_cache()
    return rows, tuple(shapes)


def reference_layout(shapes):
    """A ``layout`` packing [text | references | audio | video]; reference rows are held like condition rows."""

    from tensorfold.families.h3 import packing

    def grid(size: int, patch: int, sqrt_area: float) -> np.ndarray:
        ratio = size / sqrt_area
        left = (1.0 - ratio) / 2.0
        return np.linspace(left, left + ratio, size // patch, endpoint=False) * _POSITION_SCALE

    def build(text_tags, latent_frames, latent_height, latent_width, audio_latents, patch, keyframes=()):
        if keyframes:
            raise ValueError("reference pictures cannot be combined with a first or last frame")
        _, ph, pw = patch
        text_tags = np.asarray(text_tags, dtype=np.int64)
        text, count = int(text_tags.shape[0]), len(shapes)
        base = packing.layout(np.concatenate([text_tags, np.full(count, packing.TAG_TEXT, dtype=np.int64)]),
                              latent_frames, latent_height, latent_width, audio_latents, patch)
        positions = np.array(base.position_ids)
        blocks = []
        for index, (height, width) in enumerate(shapes):
            sqrt_area = np.sqrt(height * width)
            hh, ww = np.meshgrid(grid(height, ph, sqrt_area), grid(width, pw, sqrt_area), indexing="ij")
            block = np.empty((hh.size, 3), dtype=np.float64)
            block[:, 0] = float(text + index)
            block[:, 1], block[:, 2] = hh.reshape(-1), ww.reshape(-1)
            blocks.append(block.astype(np.float32))
        held = sum(block.shape[0] for block in blocks)
        positions = np.concatenate([positions[:text], *blocks, positions[text + count:]])
        rows = int(positions.shape[0])
        audio_start = text + held
        video_start = audio_start + audio_latents * packing.AUDIO_CHANNELS
        video_rows = np.concatenate([np.arange(text, audio_start), np.arange(video_start, rows)])
        audio_rows = np.arange(audio_start, video_start)
        tags = np.empty(rows, dtype=np.int64)
        tags[:text] = text_tags
        tags[audio_rows] = packing.TAG_AUDIO
        tags[video_rows] = packing.TAG_VIDEO
        return packing.Layout(rows, mx.array(positions), mx.array(tags.astype(np.int32)),
                              mx.array(video_rows.astype(np.int32)), mx.array(audio_rows.astype(np.int32)),
                              mx.array(np.arange(text, dtype=np.int32)), held)

    return build


def video_patches(frames: np.ndarray, mean, std, patch: int, merge: int, budget: int | None = None,
                  sampled: int | None = None):
    """Sampled frames as the vision tower's patch rows and (blocks, grid h, grid w), Qwen3-VL processor layout."""

    from PIL import Image

    count, height, width = frames.shape[:3]
    factor = patch * merge
    sampled = sampled or count
    # the processor's own test: the sampled count rounded to whole blocks (half to even), times the frame's area
    if budget is not None and round(sampled / QWEN_FRAMES_PER_BLOCK) * QWEN_FRAMES_PER_BLOCK * height * width > budget:
        beta = math.sqrt(sampled * height * width / budget)
        height, width = (max(factor, math.floor(side / beta / factor) * factor) for side in (height, width))
        frames = np.stack([np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BICUBIC))
                           for frame in frames])
    pixels = (frames.astype(np.float32) / 255.0 - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    blocks, grid_h, grid_w = count // QWEN_FRAMES_PER_BLOCK, height // patch, width // patch
    pixels = pixels.transpose(0, 3, 1, 2).reshape(blocks, QWEN_FRAMES_PER_BLOCK, 3, grid_h // merge, merge, patch,
                                                  grid_w // merge, merge, patch)
    rows = pixels.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(blocks * grid_h * grid_w, -1)
    return np.ascontiguousarray(rows), (blocks, grid_h, grid_w)


def reference_request(encoder, prompt: str, items, video_stats, budget: int | None = None):
    """Ids, tags, patch rows and grids of a Ref2VA request with videos or audio (``build_ref2va_presentation``)."""

    from tensorfold.families.h3.packing import TAG_TEXT, TAG_VIDEO

    tokenizer, processor = encoder.tokenizer, encoder._processor.image_processor
    patch, merge = encoder.vision_config.patch_size, encoder.merge_size
    start, end = encoder.vision_start_token_id, encoder.vision_end_token_id
    image_pad, video_pad = encoder.image_token_id, encoder.model_config.video_token_id
    ids, tags, rows, grids, image_grids, video_grids = [], [], [], [], [], []
    counts = {"image": 0, "video": 0, "audio": 0}

    def label(value: str):
        tokens = tokenizer(value, add_special_tokens=False)["input_ids"]
        ids.extend(tokens)
        tags.extend([TAG_TEXT] * len(tokens))

    def block(pad: int, tokens: int):
        ids.extend([start] + [pad] * tokens + [end])
        tags.extend([TAG_VIDEO] * (tokens + 2))

    for item in items:
        if item.get("sound") is not None:
            counts["audio"] += 1
            label(f"<Audio {counts['audio']}>: ")
        if item["kind"] == "image":
            counts["image"] += 1
            label(f"<Picture {counts['image']}>: ")
            vision = processor(images=[item["image"]], return_tensors="np")
            grid = [int(v) for v in np.asarray(vision["image_grid_thw"])[0]]
            block(image_pad, grid[0] * grid[1] * grid[2] // merge**2)
            rows.append(np.asarray(vision["pixel_values"], np.float32))
            grids.append(grid)
            image_grids.append(grid)
        elif item["kind"] == "video":
            counts["video"] += 1
            label(f"<Video {counts['video']}>: ")
            picked, stamps, sampled = sample_video(item["frames"])
            patches, (blocks, grid_h, grid_w) = video_patches(picked, *video_stats, patch, merge, budget, sampled)
            for stamp in stamps:
                label(f"<{stamp:.1f} seconds>")
                block(video_pad, grid_h * grid_w // merge**2)
            rows.append(patches)
            grids.extend([[1, grid_h, grid_w]] * blocks)
            video_grids.append([blocks, grid_h, grid_w])
    label(prompt)
    return ids, np.array(tags, dtype=np.int64), np.concatenate(rows), grids, image_grids, video_grids


def encode_reference_text(root: Path, prompt: str, items, weights=None, checkpoint_stats: bool = True):
    """Text rows and tags of a Ref2VA request whose references include a video or an audio clip."""

    from minimax_h3_mlx.text_encoder import _masked_scatter
    from mlx_vlm.models.qwen3_vl.language import LanguageModel

    encoder = _text_encoder(root, True, weights, checkpoint_stats)
    with open(root / "processor" / "video_preprocessor_config.json") as handle:
        budget = json.load(handle).get("size", {}).get("longest_edge")
    ids, tags, rows, grids, image_grids, video_grids = reference_request(
        encoder, prompt, items, vision_stats(root, "video_preprocessor_config.json"), budget)
    input_ids = mx.array(np.array([ids], dtype=np.int32))
    hidden, deepstack = encoder.vision(mx.array(rows).astype(encoder.dtype), mx.array(np.array(grids, np.int32)),
                                       output_hidden_states=True)
    embeds = encoder.language.embed_tokens(input_ids)
    mask = (input_ids == encoder.image_token_id) | (input_ids == encoder.model_config.video_token_id)
    expanded = mx.broadcast_to(mask[..., None], embeds.shape)
    features = hidden.astype(embeds.dtype)
    if int(expanded.sum().item()) != features.size:
        raise ValueError(f"the request reserves {int(mask.sum().item())} vision rows and the vision tower made "
                         f"{features.shape[0]}")
    embeds = _masked_scatter(embeds, expanded, features)
    position_ids, _ = LanguageModel.get_rope_index(
        encoder, input_ids, image_grid_thw=mx.array(np.array(image_grids, np.int32)) if image_grids else None,
        video_grid_thw=mx.array(np.array(video_grids, np.int32)) if video_grids else None, attention_mask=None)
    states = encoder._hidden_states(input_ids, position_ids, inputs_embeds=embeds, visual_pos_masks=mask,
                                    deepstack_visual_embeds=deepstack)
    mx.eval(states)
    del encoder
    gc.collect()
    mx.clear_cache()
    return states, tags


def encode_reference_rows(root: Path, items, patch):
    """Video rows, audio rows (or None) and one ``packing.Reference`` per item; videos cut to 17 n + 5 frames."""

    from minimax_h3_mlx.load import load_audio_vae, load_video_vae
    from minimax_h3_mlx.pipeline import PIXEL_MEAN, PIXEL_STD

    from tensorfold.families.h3.packing import Reference, patchify

    video_rows, audio_rows, geometry = [], [], []
    visual = [item for item in items if item["kind"] != "audio"]
    shapes = {}
    if visual:
        vae = load_video_vae(root / "video_vae")
        cfg = vae.config
        latents_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
        latents_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
        pixel_mean, pixel_std = np.array(PIXEL_MEAN, np.float32), np.array(PIXEL_STD, np.float32)
        for item in visual:
            if item["kind"] == "image":
                frames = np.asarray(item["image"], dtype=np.uint8)[None]
                chunks = [frames]
            else:
                frames = item["frames"][: encoded_frames(item["frames"].shape[0])]
                short = -frames.shape[0] % cfg.clip_length
                frames = np.concatenate([frames, np.repeat(frames[-1:], short, axis=0)]) if short else frames
                chunks = [frames[at:at + cfg.clip_length] for at in range(0, frames.shape[0], cfg.clip_length)]
            means = []
            for chunk in chunks:  # channels last: (1, frames, height, width, 3)
                pixels = (chunk.astype(np.float32) / 255.0 - pixel_mean) / pixel_std
                moments = vae._encode_clip(mx.array(pixels[None]))
                means.append(moments[..., :cfg.latent_channels].astype(mx.float32))
                mx.eval(means[-1])
            mean = mx.concatenate(means, axis=1)
            if item["kind"] == "video" and cfg.token_drop > 0:
                mean = mean[:, :-cfg.token_drop]
            mean = mean.transpose(0, 4, 1, 2, 3)
            if item["kind"] == "image" and mean.shape[2] != 1:
                raise ValueError(f"a reference picture should encode to one latent frame, got {mean.shape[2]}")
            video_rows.append(patchify((mean - latents_mean) / latents_std, patch))
            mx.eval(video_rows[-1])
            shapes[id(item)] = tuple(int(v) for v in mean.shape[2:])
        del vae
        gc.collect()
        mx.clear_cache()
    lengths = {}
    if any(item.get("sound") is not None for item in items):
        vae = load_audio_vae(root / "audio_vae")
        cfg = vae.config
        latents_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, 1, -1)
        latents_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, 1, -1)
        for item in items:
            if item.get("sound") is None:
                continue
            mean, _ = vae.encode(mx.array(item["sound"])[:, None, :])  # each stereo side is a batch item
            mean = mean.astype(mx.float32).transpose(0, 2, 1)  # (2, latents, channels)
            audio_rows.append(((mean - latents_mean) / latents_std).reshape(-1, mean.shape[-1]))
            mx.eval(audio_rows[-1])
            lengths[id(item)] = int(mean.shape[1])
        del vae
        gc.collect()
        mx.clear_cache()
    for item in items:
        if item["kind"] == "audio":
            geometry.append(Reference("audio", audio_latents=lengths[id(item)]))
        else:
            count, tall, wide = shapes[id(item)]
            geometry.append(Reference(item["kind"], tall, wide, count, lengths.get(id(item), 0)))
    return (mx.concatenate(video_rows) if video_rows else None,
            mx.concatenate(audio_rows) if audio_rows else None, tuple(geometry))
