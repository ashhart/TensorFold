"""Motion Previs worker plumbing (ffmpeg, frames, progress events); part of a port of Motion Previs Studio 4.1.0."""
# Copyright 2026 Sam Wasserman, Apache-2.0, modified: https://github.com/wassermanproductions/motion-previs-studio

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def tool(name: str) -> str:
    """ffmpeg or ffprobe: MOTION_PREVIS_FFMPEG / MOTION_PREVIS_FFPROBE as in the original, then PATH."""

    override = os.environ.get(f"MOTION_PREVIS_{name.upper()}")
    if override:
        return override
    for folder in ("/opt/homebrew/bin", "/usr/local/bin"):
        if (Path(folder) / name).is_file():
            return str(Path(folder) / name)
    return shutil.which(name) or name


def models_dir() -> Path:
    return Path(os.environ.get("MOTION_MODELS") or Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio")) / "motion-models")


def emit(**event) -> None:
    sys.stdout.write("@@" + json.dumps(event) + "\n")
    sys.stdout.flush()


class Progress:
    """One stage at a time; `at` of `of` are frames or seconds of that stage, as counted, not estimated."""

    def __init__(self) -> None:
        self.stage, self.started, self.timings, self.notes = None, 0.0, {}, []

    def start(self, stage: str, message: str) -> None:
        self.finish()
        self.stage, self.started = stage, time.time()
        emit(stage=stage, at=0, of=1, message=message)

    def step(self, at: float, of: float, message: str) -> None:
        emit(stage=self.stage, at=round(at, 2), of=round(max(of, 1e-6), 2), message=message)

    def finish(self) -> None:
        if self.stage:
            self.timings[self.stage] = round(self.timings.get(self.stage, 0.0) + time.time() - self.started, 3)
            emit(stage=self.stage, done=True, seconds=self.timings[self.stage])
        self.stage = None

    def note(self, text: str) -> None:
        self.notes.append(text)
        emit(note=text)


def ffmpeg(args: list[str], seconds: float = 0.0, tick=None) -> None:
    """Run ffmpeg; `tick(done_seconds)` follows its own -progress output."""

    command = [tool("ffmpeg"), "-v", "error", "-y", "-nostdin", *args]
    if tick:
        command[1:1] = ["-progress", "pipe:1", "-nostats"]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE if tick else subprocess.DEVNULL, stderr=errors, text=True)
        if tick:
            for line in process.stdout:
                if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                    try:
                        tick(min(max(int(line.split("=")[1]) / 1e6, 0.0), seconds or 1e9))
                    except ValueError:
                        pass
        if process.wait():
            errors.seek(0)
            raise RuntimeError("ffmpeg failed: " + errors.read().decode(errors="replace")[-600:])


def ffprobe(path: Path) -> dict:
    out = subprocess.run([tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name:format=duration", "-of", "json",
                          str(path)], capture_output=True, text=True)
    data = json.loads(out.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        top, _, bottom = rate.partition("/")
        fps = float(top) / float(bottom or 1)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {"width": int(stream.get("width") or 0), "height": int(stream.get("height") or 0), "frameRate": fps,
            "duration": float((data.get("format") or {}).get("duration") or 0)}


def read_frames(video: Path, fps: float, width: int, height: int, pixels: str = "rgb24"):
    """Frames of `video` sampled at `fps` and scaled, as numpy arrays (height, width, channels)."""

    import numpy as np

    channels = 1 if pixels == "gray" else 3
    size = width * height * channels
    process = subprocess.Popen([tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(video), "-vf",
                                f"fps={fps},scale={width}:{height}:flags=bicubic", "-f", "rawvideo", "-pix_fmt", pixels, "-"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            data = process.stdout.read(size)
            if len(data) < size:
                break
            yield np.frombuffer(data, np.uint8).reshape(height, width, channels).copy()
    finally:
        process.stdout.close()
        process.kill() if process.poll() is None else None
        process.wait()


class Encoder:
    """Frames in, an H.264 mp4 out: one encoded frame per frame given (frameEncode.cjs: veryfast, crf 18, yuv420p)."""

    def __init__(self, target: Path, width: int, height: int, fps: float, pixels: str = "bgr24", size: tuple[int, int] | None = None):
        self.errors = tempfile.TemporaryFile()
        scale = ["-vf", f"scale={size[0]}:{size[1]}:flags=bicubic"] if size and size != (width, height) else []
        self.process = subprocess.Popen([tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pixels, "-s",
                                         f"{width}x{height}", "-r", str(fps), "-i", "-", *scale, "-an", "-c:v", "libx264",
                                         "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(target)],
                                        stdin=subprocess.PIPE, stderr=self.errors)

    def write(self, frame) -> None:
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        if self.process.wait():
            self.errors.seek(0)
            raise RuntimeError("ffmpeg failed while encoding: " + self.errors.read().decode(errors="replace")[-400:])
        self.errors.close()


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
