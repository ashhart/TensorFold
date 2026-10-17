"""A line-art copy of a clip (white lines on black) with ffmpeg alone: bilateral smoothing, morphological gradient."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException

PATH = os.pathsep.join([os.environ.get("PATH", ""), "/opt/homebrew/bin", "/usr/local/bin"])
# (smoothing before the edges, level under which a difference is black, gain)
DETAIL = {"fine": ("bilateral=sigmaS=4:sigmaR=0.08,format=gray,gblur=sigma=0.6", 4, 10),
          "medium": ("bilateral=sigmaS=6:sigmaR=0.12,format=gray,gblur=sigma=0.8", 5, 9),
          "bold": ("bilateral=sigmaS=8:sigmaR=0.2,bilateral=sigmaS=8:sigmaR=0.2,format=gray,gblur=sigma=1.0", 5, 9)}


def chain(detail: str, width: int | None = None, height: int | None = None) -> str:
    """The ffmpeg filter graph from [0:v] to [v]."""

    smooth, floor, gain = DETAIL[detail]
    size = f"scale={width}:{height}:flags=lanczos," if width and height else ""
    return (f"[0:v]{size}{smooth},split[a][b];[a]dilation[d];[d][b]blend=all_mode=difference,"
            f"lutyuv=y='clip((val-16-{floor})*{gain},0,255)':u=128:v=128,format=yuv420p[v]")


def make(source: Path, target: Path, detail: str = "medium", width: int | None = None, height: int | None = None) -> Path:
    """`source` as line art in `target` (24 fps, no sound: a line-art reference carries the picture only)."""

    if detail not in DETAIL:
        raise ValueError(f"detail is one of {', '.join(DETAIL)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.stem + ".part.mp4")
    done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-filter_complex", chain(detail, width, height),
                           "-map", "[v]", "-an", "-r", "24", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(part)],
                          capture_output=True, text=True, env={**os.environ, "PATH": PATH})
    if done.returncode or not part.is_file() or not part.stat().st_size:
        part.unlink(missing_ok=True)
        raise RuntimeError((done.stderr or "ffmpeg wrote no file").strip()[-300:])
    part.replace(target)
    return target


def attach(home: Path):
    router = APIRouter()

    @router.post("/api/lineart")
    def lineart(body: dict) -> dict:
        """A line-art copy of a video in the studio's folder, kept beside the uploads: {"path", "detail"}."""

        given = Path(str(body.get("path") or "")).expanduser()
        given = (given if given.is_absolute() else home / given).resolve()
        if home.resolve() not in given.parents or not given.is_file():
            raise HTTPException(400, "the video is not in the studio's folder")
        detail = body.get("detail") or "medium"
        if detail not in DETAIL:
            raise HTTPException(400, f"detail is one of {', '.join(DETAIL)}")
        mark = hashlib.sha256(f"{given}:{given.stat().st_size}:{given.stat().st_mtime_ns}:{detail}".encode()).hexdigest()[:10]
        target = home / "uploads" / f"{mark}_lineart.mp4"
        if not target.is_file():
            try:
                make(given, target, detail)
            except RuntimeError as error:
                raise HTTPException(400, f"no line art could be made of that file: {error}")
        return {"path": str(target), "detail": detail, "from": str(given)}

    return router
