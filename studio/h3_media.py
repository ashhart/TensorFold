"""Reference media for h3_generate.py: limits, probing and reading pictures, videos and sound with ffmpeg."""

from __future__ import annotations

import json
import math
import shutil
import subprocess

import numpy as np

from tensorfold.families.h3 import config as h3

REFERENCE_MULTIPLE = 32  # a reference picture's sides, as the canvas's: whole 2x2 patches of 16x latents
REFERENCE_LIMIT = 9  # pictures the Ref2VA model takes in one request


def reference_size(width: int, height: int, canvas_width: int, canvas_height: int) -> tuple[int, int]:
    """A reference picture's size: down (never up) to the clip's pixel area, aspect kept, sides multiples of 32."""

    scale = min(1.0, math.sqrt((canvas_width * canvas_height) / (width * height)))
    return (max(REFERENCE_MULTIPLE, round(width * scale / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE),
            max(REFERENCE_MULTIPLE, round(height * scale / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE))


def prepare_references(paths, canvas_width: int, canvas_height: int):
    from PIL import Image

    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        size = reference_size(*image.size, canvas_width, canvas_height)
        images.append(image if image.size == size else image.resize(size, Image.Resampling.LANCZOS))
    return images


VIDEO_LIMIT = 3  # reference videos the Ref2VA model takes in one request
AUDIO_LIMIT = 3  # reference audio clips
TOTAL_LIMIT = 12  # references of all kinds
CLIP_SECONDS = (2.0, 15.0)  # the length of one reference video or audio clip
TOTAL_SECONDS = 15.0  # the summed length of the reference videos, and of the audio clips
CLIP_TOLERANCE = 0.05  # seconds a container's duration may miss a limit by
ASPECT_LIMIT = 4.0  # a reference video is between 1:4 and 4:1
MODEL_SHORT_EDGE, MODEL_MAX_PIXELS = 768, 768 * 1344  # the model's own canvas for a reference video
QWEN_SAMPLE_STRIDE = 12  # the text encoder sees a reference video at 2 fps: every 12th of its 24 fps frames
QWEN_FRAMES_PER_BLOCK = 2  # and reads two sampled frames as one vision block
FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 17, 5


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"{name} is not on the PATH: reference videos and audio are read with it (brew install ffmpeg)")
    return path


def probe_media(path) -> dict:
    """What a media file holds: ``video`` (displayed width, height) or None, ``audio`` (channels) or None, seconds."""

    done = subprocess.run([_tool("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                          capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"cannot read {path}: {done.stderr.strip() or 'not a media file'}")
    data = json.loads(done.stdout)
    found = {"video": None, "audio": None, "seconds": 0.0}
    for stream in data.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "video" and found["video"] is None and not stream.get("disposition", {}).get("attached_pic"):
            width, height = int(stream["width"]), int(stream["height"])
            num, _, den = str(stream.get("sample_aspect_ratio", "1:1")).partition(":")
            if num.isdigit() and den.isdigit() and int(num) > 0 and int(den) > 0:
                width = round(width * int(num) / int(den))
            turn = [side.get("rotation") for side in stream.get("side_data_list", []) if "rotation" in side]
            turn = float(turn[0]) if turn else float(stream.get("tags", {}).get("rotate", 0) or 0)
            if round(turn / 90.0) % 2:  # ffmpeg shows a rotated clip upright, so its sides swap
                width, height = height, width
            found["video"] = (width, height)
            found["seconds"] = max(found["seconds"], float(stream.get("duration") or 0.0))
        elif kind == "audio" and found["audio"] is None:
            found["audio"] = int(stream.get("channels") or 0)
            found["audio_seconds"] = float(stream.get("duration") or 0.0)
    found["seconds"] = found["seconds"] or float(data.get("format", {}).get("duration") or 0.0)
    found["audio_seconds"] = found.get("audio_seconds") or found["seconds"]
    return found


def check_clip_lengths(kind: str, lengths) -> None:
    """The model card's limits for reference videos or audio clips: each 2 to 15 s, 15 s in all."""

    low, high = CLIP_SECONDS
    for path, seconds in lengths:
        if not low - CLIP_TOLERANCE <= seconds <= high + CLIP_TOLERANCE:
            raise SystemExit(f"a reference {kind} is {low:g} to {high:g} seconds long, and {path} is {seconds:.1f} s")
    total = sum(seconds for _, seconds in lengths)
    if total > TOTAL_SECONDS + CLIP_TOLERANCE:
        raise SystemExit(f"the reference {kind}s are at most {TOTAL_SECONDS:g} seconds in all, got {total:.1f} s")


def reference_video_size(width: int, height: int, canvas_width: int, canvas_height: int,
                         mode: str = "match") -> tuple[int, int]:
    """A reference video's size, sides multiples of 32: ``full`` the model's canvas, ``match`` also the clip's area."""

    ratio = width / height
    if not 1.0 / ASPECT_LIMIT <= ratio <= ASPECT_LIMIT:
        raise SystemExit(f"a reference video is between 1:4 and 4:1, got {width}x{height}")
    wide, tall = (MODEL_SHORT_EDGE * ratio, float(MODEL_SHORT_EDGE)) if ratio >= 1.0 else \
        (float(MODEL_SHORT_EDGE), MODEL_SHORT_EDGE / ratio)
    if wide * tall > MODEL_MAX_PIXELS:
        scale = math.sqrt(MODEL_MAX_PIXELS / (wide * tall))
        wide, tall = wide * scale, tall * scale
    if width * height < wide * tall:
        wide, tall = float(width), float(height)
    if mode == "match" and wide * tall > canvas_width * canvas_height:
        scale = math.sqrt(canvas_width * canvas_height / (wide * tall))
        wide, tall = wide * scale, tall * scale
    return (max(REFERENCE_MULTIPLE, round(wide / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE),
            max(REFERENCE_MULTIPLE, round(tall / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE))


def read_video_frames(path, width: int, height: int, limit: int) -> np.ndarray:
    """Up to ``limit`` frames as (frames, height, width, 3) uint8 at 24 fps, LANCZOS-scaled as the model's pipeline."""

    done = subprocess.run([_tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-vf",
                           f"fps={h3.FPS},scale={width}:{height}:flags=lanczos,setsar=1", "-frames:v", str(limit),
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True)
    size = width * height * 3
    if done.returncode or len(done.stdout) < size:
        raise SystemExit(f"cannot read frames from {path}: {done.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(done.stdout[: len(done.stdout) // size * size], dtype=np.uint8).reshape(-1, height, width, 3)


def read_waveform(path, rate: int, seconds: float, channels: int) -> np.ndarray:
    """The first ``seconds`` of a file's sound as (2, samples) float32 at ``rate``; mono is repeated on both sides."""

    mix = ["-ac", "2"] if channels > 2 else []
    done = subprocess.run([_tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", *mix, "-ar",
                           str(rate), "-t", f"{seconds:.6f}", "-f", "f32le", "pipe:1"], capture_output=True)
    count = min(channels, 2)
    if done.returncode or len(done.stdout) < 4 * count:
        raise SystemExit(f"cannot read sound from {path}: {done.stderr.decode(errors='replace').strip()}")
    wave = np.frombuffer(done.stdout[: len(done.stdout) // (4 * count) * 4 * count], dtype=np.float32)
    wave = wave.reshape(-1, count).T[:, : round(seconds * rate)]
    return np.ascontiguousarray(np.repeat(wave, 2, axis=0) if count == 1 else wave)


def encoded_frames(frames: int) -> int:
    """Frames of a reference video the video VAE encodes: the count cut down to ``17 n + 5`` (22 at least)."""

    return max(1, (frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK) * FRAMES_PER_CHUNK + LATENTS_PER_CHUNK


def sample_video(frames: np.ndarray):
    """Frames the text encoder sees (every 12th, pairs, last repeated if odd), block times, and the count sampled."""

    picked = frames[::QWEN_SAMPLE_STRIDE]
    sampled = int(picked.shape[0])
    times = [index / (h3.FPS / QWEN_SAMPLE_STRIDE) for index in range(sampled)]
    if sampled % QWEN_FRAMES_PER_BLOCK:
        picked, times = np.concatenate([picked, picked[-1:]]), times + times[-1:]
    stamps = [(times[i] + times[i + 1]) / 2 for i in range(0, len(times), QWEN_FRAMES_PER_BLOCK)]
    return picked, stamps, sampled


def prepare_reference_media(images, videos, audios, width: int, height: int, frames: int, rate: int,
                            video_sound: bool = False, video_size: str = "match"):
    """Every reference as a dict (kind, path, name, image/frames/sound), in model order: pictures, videos, audio."""

    items = [{"kind": "image", "path": path, "image": image, "name": f"<Picture {index + 1}>"}
             for index, (path, image) in enumerate(zip(images, prepare_references(images, width, height), strict=True))]
    seconds = frames / h3.FPS
    probes = [(path, probe_media(path)) for path in videos]
    for path, probe in probes:
        if probe["video"] is None:
            raise SystemExit(f"{path} has no picture: a reference video is a video file")
    check_clip_lengths("video", [(path, probe["seconds"]) for path, probe in probes])
    sounds = [(path, probe_media(path)) for path in audios]
    for path, probe in sounds:
        if not probe["audio"]:
            raise SystemExit(f"{path} has no sound: a reference audio clip is an audio file (or a video with sound)")
    check_clip_lengths("audio clip", [(path, probe["audio_seconds"]) for path, probe in sounds])
    spoken = 0
    for index, (path, probe) in enumerate(probes):
        size = reference_video_size(*probe["video"], width, height, video_size)
        item = {"kind": "video", "path": path, "frames": read_video_frames(path, *size, frames),
                "name": f"<Video {index + 1}>", "sound": None}
        if video_sound and probe["audio"]:
            spoken += 1
            item["sound"] = read_waveform(path, rate, min(seconds, item["frames"].shape[0] / h3.FPS), probe["audio"])
            item["sound_name"] = f"<Audio {spoken}>"
        items.append(item)
    for index, (path, probe) in enumerate(sounds):
        items.append({"kind": "audio", "path": path, "name": f"<Audio {spoken + index + 1}>",
                      "sound": read_waveform(path, rate, seconds, probe["audio"])})
    return items
