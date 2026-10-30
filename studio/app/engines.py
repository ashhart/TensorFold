"""What this machine can render and how long it takes: engines, models, ffmpeg, hand-set sizes and time estimates."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from fastapi import HTTPException

from settings import (CUSTOM_PIXELS, CUSTOM_SIDES, CUSTOM_STEPS, DURATIONS, FASTH3_5S, LONGEST, MEASURED_8S, REF2VA_5S,
                      REF2VA_ADAPTER, REF2VA_TURBO_5S, SPARK, SPARK_COMFY_5S, SPARK_ENGINE_LIB, SPARK_FASTH3_5S, SPARK_REF2VA,
                      SPARK_REF2VA_5S, SPARK_REF2VA_TURBO_5S, ZIG_ENGINE, ZIG_FASTH3_5S, ZIG_FROM_IMAGE, ZIG_LENGTH_POWER)


def plan(total: int, piece: int = LONGEST) -> list[int]:
    """Clip lengths in seconds adding up to at least ``total``: whole pieces, then the shortest clip for the rest."""

    if piece not in DURATIONS:
        raise HTTPException(400, f"a segment is one of {sorted(DURATIONS)} seconds")
    pieces = [piece] * (total // piece)
    rest = total - piece * len(pieces)
    if rest > 0:
        pieces.append(min(d for d in DURATIONS if d >= rest))
    return pieces


def zig_ready() -> bool:
    """Whether FastH3 will run on the native engine here (scripts/video.sh decides the same way)."""

    if not (ZIG_ENGINE / "tf-h3-dit").is_file() or os.environ.get("FASTH3_ENGINE") == "mlx":
        return False
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return re.search(r"Apple M([5-9]|[1-9][0-9])", chip) is not None


try:
    M5 = re.search(r"Apple M([5-9]|[1-9][0-9])", subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                                               capture_output=True, text=True).stdout) is not None
except OSError:
    M5 = False


def turbo_ready() -> bool:
    """Whether MiniMax H3's own transformer and its Turbo adapter are installed (oneshot-setup.sh --turbo)."""

    model = Path(os.environ.get("H3_MODEL_DIR", Path.home() / "h3-models/MiniMax-H3"))
    return any((model / "FL2VA" / "transformer").glob("*.safetensors")) and any((ZIG_ENGINE.parent / "adapters").glob("lightx2v*"))



def text_encoders() -> dict[str, str]:
    """Other MiniMax H3 text encoder files on this machine: file name -> what TEXT_ENCODER is set to for it."""

    if SPARK:
        folder = SPARK_REF2VA.parent / "FastH3-Comfy/text_encoders"
        own = {"qwen3vl_32b_minimax_h3_int8_convrot.safetensors"}
        return {f.name: f.name for f in sorted(folder.glob("*minimax_h3*.safetensors")) if f.name not in own}
    folder = Path(os.environ.get("H3_MODEL_DIR", Path.home() / "h3-models/MiniMax-H3")).parent / "text_encoders"
    return {f.name: str(f) for f in sorted(folder.glob("*.safetensors"))}



def own_text_encoder(asked: str) -> str | None:
    """TEXT_ENCODER for a file of the user's choosing (a path on a Mac, a ComfyUI file name on a Spark), or None."""
    if not asked.endswith(".safetensors"):
        return None
    if SPARK:
        name = Path(asked).name
        return name if (SPARK_REF2VA.parent / "FastH3-Comfy/text_encoders" / name).is_file() else None
    path = Path(asked).expanduser()
    return str(path) if path.is_file() else None

def ref2va_ready() -> bool:
    """Whether MiniMax H3's Ref2VA transformer is installed here (oneshot-setup.sh --ref2va)."""

    if SPARK:
        return any(SPARK_REF2VA.glob("diffusion_models/minimax_h3_ref2va_pruned_*.safetensors"))
    model = Path(os.environ.get("H3_MODEL_DIR", Path.home() / "h3-models/MiniMax-H3"))
    return any((model / "Ref2VA" / "transformer").glob("*.safetensors"))


def custom(params: dict) -> tuple[dict, bool]:
    """(environment, whether anything was set) for a size or a number of passes chosen by hand."""

    env = {}
    if params.get("preset") == "custom":
        try:
            width, height = int(params["width"]), int(params["height"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "a custom size needs a width and a height")
        low, high = CUSTOM_SIDES
        if width % 32 or height % 32 or not (low <= width <= high and low <= height <= high):
            raise HTTPException(400, f"width and height are multiples of 32 between {low} and {high}")
        if width * height > CUSTOM_PIXELS:
            raise HTTPException(400, f"at most {CUSTOM_PIXELS} pixels (1344x768), got {width}x{height}")
        env.update(WIDTH=str(width), HEIGHT=str(height))
    if params.get("steps") not in (None, ""):
        try:
            steps = int(params["steps"])
        except (TypeError, ValueError):
            raise HTTPException(400, "passes is a whole number")
        if not CUSTOM_STEPS[0] <= steps <= CUSTOM_STEPS[1]:
            raise HTTPException(400, f"passes is between {CUSTOM_STEPS[0]} and {CUSTOM_STEPS[1]}")
        env["STEPS"] = str(steps)
    if params.get("text_encoder") not in (None, "", "regular"):
        asked = str(params["text_encoder"]).strip()
        found = text_encoders().get(asked) or own_text_encoder(asked)
        if not found:
            raise HTTPException(400, f"no text encoder file at {asked} on this machine (a .safetensors file"
                                     + (" in ComfyUI's text_encoders folder)" if SPARK else ")"))
        env["TEXT_ENCODER"] = found
    return env, bool(env)


def ref2va_turbo_ready() -> bool:
    if SPARK:
        return ((SPARK_REF2VA / "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors").is_file()
                and (SPARK_REF2VA / "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors").is_file())
    return ref2va_ready() and (ZIG_ENGINE.parent / "adapters" / REF2VA_ADAPTER).is_file()


def media_seconds(path: str) -> tuple[float | None, bool | None]:
    """(seconds, whether it has sound) of a media file, or (None, None) without ffprobe: the renderer checks again."""

    tool = shutil.which("ffprobe") or shutil.which("ffprobe", path="/opt/homebrew/bin:/usr/local/bin")
    if tool is None:
        return None, None
    try:
        done = subprocess.run([tool, "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json",
                               path], capture_output=True, text=True, timeout=30)  # in a request: never without a limit
    except subprocess.TimeoutExpired:
        return None, None
    try:
        data = json.loads(done.stdout)
        return float(data["format"]["duration"]), any(one.get("codec_type") == "audio" for one in data["streams"])
    except (KeyError, TypeError, ValueError):
        return None, None


def estimate(preset: str, quality: str, seconds: float, text_only: bool = False, referenced: bool = False,
             ref_turbo: bool = False, ref_media: bool = False) -> int | None:
    if referenced and ref_media:
        return None  # a reference video or audio clip adds its own rows to every pass: the time depends on its length
    if referenced:
        # the same MLX engine as the 20-step qualities, whose time grows with the length to the power 1.6
        if SPARK:
            base = (SPARK_REF2VA_TURBO_5S if ref_turbo else SPARK_REF2VA_5S).get(preset)
        else:
            base = (REF2VA_TURBO_5S if ref_turbo else REF2VA_5S).get(preset)
        return None if base is None or seconds > LONGEST else int(base * (seconds / 5.0) ** 1.6)
    if seconds > LONGEST:
        # every part after the first opens on a frame
        parts = [estimate(preset, quality, piece) for piece in plan(seconds)]
        return None if None in parts else int(sum(parts) + 2 * len(parts))
    if SPARK:
        base = (SPARK_FASTH3_5S if SPARK_ENGINE_LIB.exists() else SPARK_COMFY_5S).get(quality, {}).get(preset)
        # the pass time grows a little faster than the length: attention over all rows is part of every pass
        return None if base is None else int(base * (seconds / 5.0) ** 1.2 * (1.0 if text_only else 1.5))
    if zig_ready() and preset in ZIG_FASTH3_5S.get(quality, {}):
        return int(ZIG_FASTH3_5S[quality][preset] * (seconds / 5.0) ** ZIG_LENGTH_POWER[preset] * (1.0 if text_only else ZIG_FROM_IMAGE))
    if quality.startswith("fh") and not M5:
        return None  # FastH3 on an earlier chip runs with 8-bit weights on the MLX engine; no time is measured there
    if quality in FASTH3_5S and preset in FASTH3_5S[quality]:
        return int(FASTH3_5S[quality][preset] * (seconds / 5.0) ** 1.3)
    if quality.startswith("fh"):
        return None
    base = MEASURED_8S.get(preset)
    if base is None:
        return None
    scale = {"standard": 1.0, "high": 1.45, "full": 3.0, "fast": 0.58}[quality] * (seconds / 8.0) ** 1.6
    return int(base * scale)


def environment(extra: dict) -> dict:
    env = dict(os.environ)
    env.update({key: str(value) for key, value in extra.items() if value not in (None, "")})
    env["PATH"] = "/opt/homebrew/bin:" + env.get("PATH", "")
    return env


def media_tools() -> dict:
    """Whether ffmpeg and ffprobe are on this machine, looked for where the jobs will look."""

    path = environment({})["PATH"]
    return {name: bool(shutil.which(name, path=path)) for name in ("ffmpeg", "ffprobe")}


FFMPEG_HOW = "sudo apt install ffmpeg" if SPARK else "brew install ffmpeg"
FFMPEG_NOTE = ("ffmpeg is not installed on this machine (" + FFMPEG_HOW + "). Without it: no long videos, no timeline export, "
               "no frame taken from a clip (continue from a clip, end on a clip), no reference videos or audio, no silenced "
               "sound track" + (", and a clip keeps ComfyUI's own sound track instead of the limited 256 kb/s one. Single clips, "
                                "the 720p crop included, and pictures render without it." if SPARK else "."))


def without_ffmpeg(kind: str, params: dict) -> str:
    """Why this job cannot run without ffmpeg ("" when it can, or ffmpeg is there): said before it is queued."""

    if media_tools()["ffmpeg"]:
        return ""
    what = ("joining the clips of a timeline" if kind == "export" else "joining the clips of a long video" if kind == "long"
            else "taking the last frame of the clip this one continues from" if params.get("first_frame_from")
            else "taking the first frame of the clip this one leads into" if params.get("last_frame_from")
            else "reading reference videos and audio" if params.get("reference_videos") or params.get("reference_audios")
            else "silencing the clip's sound track" if params.get("silent") else "")
    return f"{what} needs ffmpeg, which is not installed on this machine: {FFMPEG_HOW}" if what else ""
