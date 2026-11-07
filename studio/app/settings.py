"""Where the Studio keeps its files, what each preset and quality means, and the measured times the estimates use."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi import HTTPException

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
HOME = Path(os.environ.get("STUDIO_HOME", Path.home() / "TensorFoldStudio"))
for name in ("projects", "uploads", "renders", "scouts", "exports"):
    (HOME / name).mkdir(parents=True, exist_ok=True)

import lengths  # noqa: E402  (which frame counts the model takes, beside this file)
sys.path.insert(0, str(PACK / "spark"))
import memory_guard  # noqa: E402,F401  (a DGX Spark's memory: what a clip needs and whether this machine has it, in spark/)

FPS = 24
# Seconds -> frames (17n + 5) for the four classic lengths; a job may ask for any count the model takes (lengths.py).
DURATIONS = lengths.CLASSIC
# Longer videos are chains of clips, each opening on the last frame of the one before, joined at the end.
LONG = {30: "30 s", 60: "1 min", 120: "2 min", 300: "5 min", 600: "10 min", 900: "15 min", 1800: "30 min"}
LONGEST = max(DURATIONS)
# name -> (label, final size, environment). Generation happens at half size where X2 is set.
PRESETS = {
    "draft": ("Draft 1344x768 (generated at 672x384, 2x decoder)", (1344, 768), {"X2": "1"}),
    "2k": ("2K 2048x1152 (generated at 1024x576, 2x decoder)", (2048, 1152), {"TWOK": "1"}),
    "qhd": ("2K 2560x1440 (generated at 1280x736, 2x decoder)", (2560, 1440), {"QHD": "1"}),
    "small": ("Small 864x480 (native)", (864, 480), {"WIDTH": "864", "HEIGHT": "480"}),
    "native": ("Native 1344x768 (no 2x decoder)", (1344, 768), {"WIDTH": "1344", "HEIGHT": "768"}),
    "p720": ("720p 1280x720 (native)", (1280, 720), {"WIDTH": "1280", "HEIGHT": "736", "CROP": "1280x720"}),
    "p960": ("960p 1728x960 (generated at 480p, 2x decoder)", (1728, 960), {"X2": "1", "WIDTH": "1728", "HEIGHT": "960"}),
    "sharp": ("Wasserman Sharp Spec 1376x768 (native, one pass, no upscale)", (1376, 768), {"WIDTH": "1376", "HEIGHT": "768"}),
}
# Sam Wasserman's locked recipe: this size, FastH3 8 passes, seed 42 and a retake at the next seed, 8 or 10 seconds.
SHARP = {"preset": "sharp", "quality": "fh8", "seed": 42, "seconds": [8, 10]}
QA_TAKES = 2  # his QA gate (qa.py) follows each such clip: one that fails is rendered again at the next seed
QUALITIES = {
    "standard": ("Standard: Turbo 5 passes + base-model sound", {}),
    "high": ("High: 20 steps, no adapter, fast recipe (about 1.5x slower)", {"QUALITY": "high"}),
    "full": ("Full: plain 20 steps, no adapter (about 3x slower)", {"QUALITY": "full"}),
    "fast": ("Fast: Turbo 3 passes, adapter sound", {"POINTS": "4", "REVOICE": "0"}),
    "fh8": ("FastH3: 8 passes, sparse attention", {"ENGINE": "fasth3", "STEPS": "8"}),
    "fh4": ("FastH3: 4 passes, fastest and softer", {"ENGINE": "fasth3", "STEPS": "4"}),
    "fh20": ("FastH3: 20 passes, for prompts with several actions", {"ENGINE": "fasth3", "STEPS": "20"}),
}
SCOUT_SIZES = {"draft": (1344, 768), "2k": (1024, 576), "qhd": (1280, 736), "small": (864, 480),
               "native": (1344, 768), "p720": (1280, 736), "p960": (864, 480), "sharp": (1376, 768)}
# Measured on a Mac Studio M5 Ultra, 8 second clips, standard quality, image step included (README).
MEASURED_8S = {"draft": 108, "2k": 299, "qhd": 635, "small": 174, "native": 709}
# FastH3 on the native engine (TensorFold's Zig + Metal runtime), 5 second text-to-video clips, same machine.
ZIG_ENGINE = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio")) / "zig-engine"
# Measured through this app at 480p and 720p; the other cells follow from the measured pass and decoder times (README).
ZIG_FASTH3_5S = {"fh4": {"small": 43, "p960": 53, "p720": 125, "qhd": 137, "draft": 31, "2k": 60, "native": 104},
                 "fh8": {"small": 66, "p960": 76, "p720": 205, "qhd": 217, "draft": 44, "2k": 93, "native": 175},
                 "fh20": {"small": 129, "p960": 139, "p720": 445, "qhd": 457, "draft": 83, "2k": 192, "native": 388}}
# How the time grows with the clip's length, per preset (a 10 s 720p clip took 388 s against 205 s for 5 s).
ZIG_LENGTH_POWER = {"small": 1.15, "p960": 1.15, "draft": 1.15, "2k": 1.15, "p720": 0.95, "qhd": 0.95, "native": 0.95}
# A clip that starts from an image carries the image's rows and its vision tokens through every pass.
ZIG_FROM_IMAGE = 1.13  # measured through this app: 72 s from an image against 64 s from text, 480p, 8 passes
# FastH3 on the MLX engine (machines without the native engine), 5 second clips: quality -> preset -> seconds.
FASTH3_5S = {"fh4": {"small": 51, "p960": 61, "p720": 112, "qhd": 123},
             "fh8": {"small": 78, "p960": 88, "p720": 192, "qhd": 203},
             "fh20": {"small": 174, "p960": 184, "p720": 434, "qhd": 445}}
# Reference to video (MiniMax H3 Ref2VA), plain 20 passes or the 4-step adapter, one reference, 5 s: preset -> seconds.
REF2VA_5S = {"small": 223}
REF2VA_TURBO_5S = {"small": 88}
# The same on a DGX Spark (oneshot-setup-spark.sh --ref2va), through ComfyUI's own blocks.
SPARK_REF2VA = Path(os.environ.get("SPARK_MODELS", Path.home() / "spark-studio-models")) / "MiniMax-H3-Comfy"
SPARK_REF2VA_5S = {"small": 321}
SPARK_REF2VA_TURBO_5S = {"small": 107}
REF2VA_ADAPTER = "lightx2v_ref2v_4step_v0.1_ourlayout.safetensors"
REFERENCE_LIMIT = 9
# MiniMax's limits: 3 videos and 3 audio clips, each 2 to 15 s and 15 s in all per kind, 12 files of all kinds.
REFERENCE_VIDEO_LIMIT, REFERENCE_AUDIO_LIMIT, REFERENCE_TOTAL = 3, 3, 12
REFERENCE_CLIP_SECONDS, REFERENCE_KIND_SECONDS = (2.0, 15.0), 15.0
VIDEO_SUFFIXES, AUDIO_SUFFIXES = (".mp4", ".mov"), (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")
# A size typed in by hand: multiples of 32, and no more pixels than the largest size the model is run at here.
CUSTOM_SIDES, CUSTOM_PIXELS, CUSTOM_STEPS = (256, 1536), 1344 * 768, (1, 50)
LEAD = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."

# On a DGX Spark (Linux, CUDA) the scripts hand over to spark/: no 2x decoder and no Turbo, so native sizes and FastH3.
SPARK = sys.platform.startswith("linux")
if SPARK:
    PRESETS = {k: v for k, v in PRESETS.items() if "X2" not in v[2] and k not in ("2k", "qhd")}
    QUALITIES = {k: v for k, v in QUALITIES.items() if k.startswith("fh")}
# FastH3 on a DGX Spark, 5 s clips, models loaded, TensorFold's CUDA engine with the 8-bit decoder (README receipts).
SPARK_ENGINE_LIB = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio")) / "tf-h3" / "libtf_h3.so"
SPARK_FASTH3_5S = {"fh4": {"small": 45, "p720": 111, "native": 113},
                   "fh8": {"small": 73, "p720": 183, "native": 195, "sharp": 202},  # sharp: Sam Wasserman's 330 to 380 s at 8 s
                   "fh20": {"small": 158, "p720": 399, "native": 441}}
# On ComfyUI's own blocks in bf16 (no engine built); 720p is scaled from 1344x768 by its rows.
SPARK_COMFY_5S = {"fh4": {"small": 76, "p720": 155, "native": 172},
                  "fh8": {"small": 118, "p720": 260, "native": 289},
                  "fh20": {"small": 244, "p720": 575, "native": 642}}


GPU_NAME: list[str] = []  # asked of the driver once: see machine()


def machine() -> tuple[str, str]:
    """(platform, label) for the header; the GPU's name is asked of nvidia-smi once, with a time limit, and kept."""

    if SPARK:
        if not GPU_NAME:
            try:
                GPU_NAME.append(subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                               text=True, timeout=5).stdout.strip().splitlines()[0])
            except (OSError, IndexError, subprocess.TimeoutExpired):
                return "spark", "DGX SPARK · CUDA · H3-CUDA"  # not kept: asked again on a later call
        return "spark", f"DGX SPARK · {GPU_NAME[0].upper()} · H3-CUDA"
    # no chip name: the same recipe runs from an M1 with 64 GB up
    return "mac", "MAC · APPLE SILICON MX · H3-METAL"


def inside(path: str | Path) -> Path:
    """A path under the studio folder, or 400."""

    resolved = (HOME / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if HOME.resolve() not in (resolved, *resolved.parents):
        raise HTTPException(400, "path is outside the studio folder")
    return resolved


def relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(HOME.resolve()))
