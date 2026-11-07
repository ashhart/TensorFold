"""A clip's finish on a DGX Spark: the sound made again from a lossless track, and the centre crop."""
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from spark_comfy import PREFIX, call, fetch
from spark_media import AUDIO_RATE, FPS

# ComfyUI's AAC can overshoot full scale: the sound also leaves as a lossless track HEADROOM_DB down, remade later
HEADROOM_DB, PEAK_DBFS, SOUND_BITRATE = 12, -1.0, "256k"


def save_sound(workflow: dict, audio: list, prefix: str) -> None:
    """Add the lossless copy of the decoded sound to a workflow (nodes 17 and 18)."""

    if os.environ.get("AUDIO_EQ", "on") == "off":
        return
    workflow["17"] = {"class_type": "AudioAdjustVolume", "inputs": {"audio": audio, "volume": -HEADROOM_DB}}
    workflow["18"] = {"class_type": "SaveAudio", "inputs": {"audio": ["17", 0], "filename_prefix": prefix}}


def finish_sound(item, target: Path, frames: int) -> None:
    """Remake the clip's sound from the lossless track: full level, peaks limited to PEAK_DBFS, the picture's length."""

    sound = getattr(item, "sound", None)
    if sound is None:
        return
    if shutil.which("ffmpeg") is None:
        # not silently: the track ComfyUI wrote can overshoot full scale (see above)
        print("[tensorfold] sound: left as ComfyUI wrote it, at about 130 kb/s and without the limiter: ffmpeg is not "
              "installed on this machine (sudo apt install ffmpeg)", flush=True)
        (PREFIX / "ComfyUI" / "output" / sound.get("subfolder", "") / sound["filename"]).unlink(missing_ok=True)
        return
    import numpy as np

    track = target.with_suffix(".sound.flac")
    fetch(sound, track)
    try:
        # the peak between the samples too: measured at four times the rate
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(track), "-ar", str(4 * AUDIO_RATE), "-f", "f32le", "pipe:1"],
                             capture_output=True).stdout
        wave = np.frombuffer(raw, np.float32)
        if not len(wave) or not float(np.abs(wave).max()):
            print("[tensorfold] sound: the decoded track is silent", flush=True)
            return
        peak = 20 * math.log10(float(np.abs(wave).max())) + HEADROOM_DB
        chain = f"volume={HEADROOM_DB}dB"
        if peak > PEAK_DBFS:
            chain += f",alimiter=limit={10 ** (PEAK_DBFS / 20):.4f}:level=disabled:attack=2:release=40,aresample={AUDIO_RATE}"
        out = target.with_suffix(".sound.mp4")
        done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(target), "-i", str(track), "-map", "0:v:0", "-map", "1:a:0",
                               "-c:v", "copy", "-af", chain, "-c:a", "aac", "-b:a", SOUND_BITRATE, "-ar", str(AUDIO_RATE),
                               "-t", f"{frames / FPS:.6f}", "-movflags", "+faststart", str(out)], capture_output=True, text=True)
        if done.returncode or not out.is_file() or not out.stat().st_size:
            out.unlink(missing_ok=True)
            print(f"[tensorfold] sound: left as ComfyUI wrote it ({done.stderr.strip()[-200:]})", flush=True)
            return
        out.replace(target)
        print(f"[tensorfold] sound: peak {peak:+.1f} dBFS as decoded, " + (f"limited to {PEAK_DBFS:g} dBFS" if peak > PEAK_DBFS
              else "left at its level"), flush=True)
    finally:
        track.unlink(missing_ok=True)
        (PREFIX / "ComfyUI" / "output" / sound.get("subfolder", "") / sound["filename"]).unlink(missing_ok=True)


def crop_frames(workflow: dict, crop: str, width: int, height: int) -> bool:
    """Centre-crop to ``crop`` (WxH) in the workflow (no ffmpeg); False: no crop node, ``crop_file`` crops after."""

    try:
        wide, tall = (int(side) for side in crop.lower().split("x"))
    except ValueError:
        sys.exit(f"--crop is WIDTHxHEIGHT, got {crop}")
    if wide > width or tall > height or (width - wide) % 2 or (height - tall) % 2:
        sys.exit(f"cannot centre-crop {width}x{height} frames to {wide}x{tall}")
    if "ImageCrop" in call("/object_info/ImageCrop"):
        workflow["19"] = {"class_type": "ImageCrop", "inputs": {"image": ["13", 0], "width": wide, "height": tall,
                                                                "x": (width - wide) // 2, "y": (height - tall) // 2}}
        workflow["15"]["inputs"]["images"] = ["19", 0]
        return True
    if shutil.which("ffmpeg") is None:
        sys.exit(f"the clip is generated at {width}x{height} and cropped to {wide}x{tall}, and this machine can do neither crop: "
                 "its ComfyUI has no ImageCrop node and ffmpeg is not installed (sudo apt install ffmpeg)")
    return False


def crop_file(target: Path, crop: str) -> None:
    """The crop after the render, with ffmpeg: only for a ComfyUI without a crop node (see ``crop_frames``)."""

    wide, tall = crop.lower().split("x")
    cropped = target.with_suffix(".crop.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(target), "-vf", f"crop={wide}:{tall}", "-c:v",
                    "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy",
                    str(cropped)], check=True)
    cropped.replace(target)
