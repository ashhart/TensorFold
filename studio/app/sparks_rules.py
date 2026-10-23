"""Sparks: which jobs may go to a worker, what a worker must have for a clip, and the measured speedup."""

from __future__ import annotations

import math
from pathlib import Path

LIMIT = 8        # the most Sparks a job may ask for. Five exist where this was written; six to eight are not measured.
MOVES = 2        # a clip is taken back from a worker at most this often; after that it renders on this Spark
DISK_GB = 2      # a worker with less free disk than this is not given a clip
KINDS = ("video",)  # what is handed to a worker: single clips
# speedup of 8 FastH3 clips on 1 to 5 Sparks, one measured run (docs/multi-spark.md); n Sparks take ceil(8 / n) rounds
MEASURED_CLIPS = 8
MEASURED = {1: 1.0, 2: 2.03, 3: 2.70, 4: 3.81, 5: 4.05}
MAC = "Multi-Spark rendering is for DGX Sparks: on a Mac every job renders on this machine."
ONE_CLIP = ("Splitting a single clip across Sparks is measured (1.40x on 2, 1.73x on 3, 1.77x on 4 for a 5 s 864x480 clip) "
            "and is not in the app yet.")


def capable(kind: str, params: dict, info: dict | None) -> str:
    """Why a worker cannot render this job ("" when it can), from what its app says it has (/api/spark/self)."""

    info = info or {}
    if kind not in KINDS:
        return "only single clips are handed to another Spark"
    if not info.get("fasth3"):
        return "FastH3 is not installed there"
    media = bool(params.get("reference_videos") or params.get("reference_audios"))
    if params.get("references") or media:
        if not info.get("ref2va"):
            return "MiniMax H3 Ref2VA (reference to video) is not installed there"
        if params.get("ref_turbo") and not info.get("ref2va_turbo"):
            return "the Ref2VA Turbo adapter is not installed there"
        if media and not info.get("ffmpeg"):
            return "reference videos and audio need ffmpeg, which is not installed there"
    missing = [str(one.get("id")) for one in params.get("loras") or [] if isinstance(one, dict) and one.get("id") not in (info.get("loras") or [])]
    if missing:
        return "the LoRA " + ", ".join(missing) + " is not installed there"
    encoder = params.get("text_encoder")
    if encoder not in (None, "", "regular") and Path(str(encoder)).name not in (info.get("text_encoders") or []):
        return "the text encoder file is not on that Spark"
    return ""


def here_only(kind: str, params: dict) -> str:
    """Why a job renders on this Spark whatever the number says ("" for a single clip, which may go to a worker)."""

    if kind == "long":
        return "A long video is a chain: each piece opens on the last frame of the one before, so its pieces render one after another, on this Spark."
    if kind == "studio":
        return "A clip that starts from a new picture renders on this Spark: the picture model is here."
    if kind not in KINDS:
        return "Pictures, edits, Library actions and exports render on this Spark: only clips are handed to other Sparks."
    return ""


def speedup(used: int, clips: int) -> tuple[float, bool]:
    """(times faster than one Spark, whether measured) for ``clips`` on ``used`` Sparks: rounds times what was kept."""

    used = max(1, min(used, clips))
    measured = used in MEASURED
    known = MEASURED.get(used) or MEASURED[max(MEASURED)]
    kept = min(1.0, known / (MEASURED_CLIPS / math.ceil(MEASURED_CLIPS / min(used, max(MEASURED)))))
    return max(1.0, clips / math.ceil(clips / used) * kept), measured
