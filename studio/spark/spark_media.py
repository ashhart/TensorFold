"""Reference videos and sound on a DGX Spark: limits, probing, and copies prepared with ffmpeg for ComfyUI's loaders."""
import json
import math
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

# MiniMax's limits for one Ref2VA request, the same as on the Mac (h3_generate.py)
PICTURE_LIMIT, VIDEO_LIMIT, AUDIO_LIMIT, TOTAL_LIMIT = 9, 3, 3, 12
CLIP_SECONDS, TOTAL_SECONDS, CLIP_TOLERANCE = (2.0, 15.0), 15.0, 0.05
ASPECT_LIMIT = 4.0  # a reference video is between 1:4 and 4:1
MODEL_SHORT_EDGE, MODEL_MAX_PIXELS, REFERENCE_MULTIPLE = 768, 768 * 1344, 32
FPS, AUDIO_RATE = 24, 32000


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        sys.exit(f"{name} is not on the PATH: reference videos and audio are read with it (sudo apt install ffmpeg)")
    return path


def probe_media(path) -> dict:
    """What a media file holds: ``video`` (displayed width, height) or None, ``audio`` (channels) or None, seconds."""

    done = subprocess.run([_tool("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                          capture_output=True, text=True)
    if done.returncode:
        sys.exit(f"cannot read {path}: {done.stderr.strip() or 'not a media file'}")
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
            sys.exit(f"a reference {kind} is {low:g} to {high:g} seconds long, and {path} is {seconds:.1f} s")
    total = sum(seconds for _, seconds in lengths)
    if total > TOTAL_SECONDS + CLIP_TOLERANCE:
        sys.exit(f"the reference {kind}s are at most {TOTAL_SECONDS:g} seconds in all, got {total:.1f} s")


def reference_video_size(width: int, height: int, canvas_width: int, canvas_height: int, mode: str = "match") -> tuple[int, int]:
    """A reference video's size, sides multiples of 32: the model's canvas, not enlarged; ``match``: the clip's area."""

    ratio = width / height
    if not 1.0 / ASPECT_LIMIT <= ratio <= ASPECT_LIMIT:
        sys.exit(f"a reference video is between 1:4 and 4:1, got {width}x{height}")
    wide, tall = (MODEL_SHORT_EDGE * ratio, float(MODEL_SHORT_EDGE)) if ratio >= 1.0 else (float(MODEL_SHORT_EDGE), MODEL_SHORT_EDGE / ratio)
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


def prepared_video(path, size: tuple[int, int], limit: int, folder: Path) -> tuple[Path, int]:
    """A reference video for ComfyUI's loader: 24 fps, scaled to ``size`` (bounds rows), lossless: (file, frames)."""

    target = folder / f"video_{uuid.uuid4().hex[:8]}.mp4"
    done = subprocess.run([_tool("ffmpeg"), "-v", "error", "-y", "-i", str(path), "-map", "0:v:0", "-an", "-vf",
                           f"fps={FPS},scale={size[0]}:{size[1]}:flags=lanczos,setsar=1", "-frames:v", str(limit),
                           "-c:v", "libx264rgb", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "rgb24", str(target)],
                          capture_output=True, text=True)
    if done.returncode or not target.is_file():
        sys.exit(f"cannot read frames from {path}: {done.stderr.strip()}")
    count = subprocess.run([_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                            "stream=nb_read_frames", "-of", "csv=p=0", str(target)], capture_output=True, text=True).stdout.strip()
    frames = int(count) if count.isdigit() else 0
    if frames < 22:
        sys.exit(f"cannot read frames from {path}: {frames} frames at {FPS} a second, and a reference video needs 22")
    return target, frames


def prepared_sound(path, seconds: float, folder: Path) -> Path:
    """The first ``seconds`` of a file's sound as a stereo wav at the audio decoder's rate (mono is repeated)."""

    target = folder / f"sound_{uuid.uuid4().hex[:8]}.wav"
    done = subprocess.run([_tool("ffmpeg"), "-v", "error", "-y", "-i", str(path), "-map", "0:a:0", "-vn", "-ac", "2", "-ar",
                           str(AUDIO_RATE), "-t", f"{seconds:.6f}", "-c:a", "pcm_f32le", str(target)], capture_output=True, text=True)
    if done.returncode or not target.is_file() or target.stat().st_size < 1024:
        sys.exit(f"cannot read sound from {path}: {done.stderr.strip()}")
    return target
