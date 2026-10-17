"""The studio folder over HTTP: uploads, frames taken from clips, the Library, probes, Production projects and files."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import engines
import shot
from engines import FFMPEG_HOW, environment, media_seconds
from job_run import frame_at, last_frame
from job_state import jobs, lock
from settings import AUDIO_SUFFIXES, HOME, SPARK, VIDEO_SUFFIXES, inside, relative

router = APIRouter()
hub = None  # set by server.py: the multi-Spark hub, which knows the Spark and seed of each clip (sparks.py)


@router.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "image.png").suffix.lower()
    # .safetensors: a LoRA for the hub; sound files: reference audio
    if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".safetensors") + VIDEO_SUFFIXES + AUDIO_SUFFIXES:
        raise HTTPException(400, "images (png, jpg, webp), videos (mp4, mov) and sound (wav, mp3, m4a, aac, flac, ogg) only")
    target = HOME / "uploads" / f"{uuid.uuid4().hex[:10]}{suffix}"
    with open(target, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"path": relative(target)}


@router.post("/api/frame")
def take_frame(body: dict) -> dict:
    """A still from a Library clip saved to the Library: ``at`` is "last", "first" or seconds into the clip."""

    source = inside(str(body.get("path") or ""))
    if not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES:
        raise HTTPException(400, "a frame is taken from a video in the studio's folder")
    if not engines.media_tools()["ffmpeg"]:
        raise HTTPException(400, f"taking a frame of a clip needs ffmpeg, which is not installed on this machine: {FFMPEG_HOW}")
    at = body.get("at", "last")
    name = f"{source.stem[:40]}_{'last' if at == 'last' else 'first' if at == 'first' else str(at).replace('.', 's')}_{uuid.uuid4().hex[:6]}.png"
    target = HOME / "uploads" / name
    try:
        if at == "last":
            last_frame(source, target)
        else:
            seconds = 0.0 if at == "first" else float(at)
            length, _ = media_seconds(str(source))
            if seconds < 0 or (length and seconds > length):
                raise HTTPException(400, f"the clip is {length:.1f} seconds long")
            frame_at(source, target, seconds)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        raise HTTPException(400, "no frame could be read there")
    if not target.is_file():
        raise HTTPException(400, "no frame could be read there")
    return {"path": relative(target)}


@router.get("/api/library")
def library() -> list[dict]:
    """Every clip and still under the studio folder, newest first."""

    found = []
    for folder, kinds in (("renders", (".mp4",)), ("exports", (".mp4",)), ("scouts", (".png",)),
                          ("uploads", (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov")), ("projects", (".mp4",)), ("slate", (".mp4",))):
        for path in (HOME / folder).rglob("*"):
            if path.suffix.lower() in kinds and path.is_file():
                found.append({"path": relative(path), "kind": "video" if path.suffix.lower() in (".mp4", ".mov")
                              else "image", "folder": folder, "modified": path.stat().st_mtime,
                              "size": path.stat().st_size})
    found = sorted(found, key=lambda item: -item["modified"])[:400]
    made = hub.takes()  # which Spark rendered a clip, and at which seed (sparks.py)
    for item in found:
        if item["path"] in made:
            item.update(spark=made[item["path"]].get("spark"), seed=made[item["path"]].get("seed"))
    return found


@router.delete("/api/library")
def discard(path: str) -> dict:
    """Take a clip or still out of the library: it is moved to the studio's trash folder, not erased."""

    target = inside(path)
    if not target.is_file() or relative(target).split("/")[0] not in ("renders", "exports", "scouts", "uploads", "projects", "slate") \
            or target.suffix.lower() not in (".mp4", ".mov", ".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(400, "not a library file")
    with lock:
        if any(job["status"] in ("queued", "running") and relative(target) in json.dumps(job["params"]) for job in jobs.values()):
            raise HTTPException(400, "a queued or running job uses this file")
    kept = HOME / "trash" / (time.strftime("%Y%m%d-%H%M%S-") + relative(target).replace("/", "__"))
    kept.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(kept))
    return {"trash": relative(kept)}


@router.get("/api/probe")
def probe(path: str) -> dict:
    target = inside(path)
    if not engines.media_tools()["ffprobe"]:
        raise HTTPException(400, f"measuring a clip needs ffprobe, which comes with ffmpeg and is not installed on this machine: {FFMPEG_HOW}")
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                              "stream=codec_type,width,height,nb_frames:format=duration", "-of", "json", str(target)],
                             capture_output=True, text=True, env=environment({}), timeout=30)  # in a request: never without a limit
    except subprocess.TimeoutExpired:
        raise HTTPException(400, "the file could not be read in 30 seconds")
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams") or []
    stream = next((one for one in streams if one.get("codec_type") == "video"), {})
    # sound: whether the file has an audio track (a reference video's own sound, a reference audio clip)
    return {"width": stream.get("width"), "height": stream.get("height"), "frames": stream.get("nb_frames"),
            "duration": float((data.get("format") or {}).get("duration") or 0),
            "sound": any(one.get("codec_type") == "audio" for one in streams)}


@router.get("/api/shot/window")
def shot_window(path: str) -> dict:
    """The longest stretch of a clip without a cut (shot.py), for trimming a take to, whatever made the clip."""

    target = inside(path)
    if not target.is_file():
        raise HTTPException(404, "no such file")
    try:
        return shot.longest_take(target)
    except LookupError as error:
        raise HTTPException(400, f"the clip's frames cannot be read on this machine: {error}")
    except Exception as error:  # noqa: BLE001 - not a video, or a damaged one
        raise HTTPException(400, f"not a clip whose frames can be read: {str(error)[-200:]}")


def project_file(project_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", project_id):
        raise HTTPException(400, "bad project id")
    return HOME / "projects" / project_id / "project.json"


@router.get("/api/projects")
def projects() -> list[dict]:
    out = []
    for file in sorted((HOME / "projects").glob("*/project.json"), key=lambda f: -f.stat().st_mtime):
        try:
            data = json.loads(file.read_text())
            data["id"]
        except (ValueError, KeyError, TypeError):
            continue  # a file that was cut off while being written must not take the Production tab down
        out.append({"id": data["id"], "name": data.get("name", data["id"]), "segments": len(data.get("segments", [])),
                    "modified": file.stat().st_mtime})
    return out


@router.post("/api/projects")
def new_project(body: dict) -> dict:
    project_id = time.strftime("p%m%d-%H%M%S")
    data = {"id": project_id, "name": body.get("name") or "Untitled", "preset": "small" if SPARK else "draft", "quality": "fh8" if SPARK else "standard",
            "segments": [], "timeline": []}
    file = project_file(project_id)
    file.parent.mkdir(parents=True, exist_ok=True)
    fresh = file.with_suffix(".json.new")
    fresh.write_text(json.dumps(data, indent=1))
    fresh.replace(file)
    return data


@router.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    file = project_file(project_id)
    if not file.is_file():
        raise HTTPException(404, "no such project")
    return json.loads(file.read_text())


@router.put("/api/projects/{project_id}")
def save_project(project_id: str, body: dict) -> dict:
    file = project_file(project_id)
    if not file.is_file():
        raise HTTPException(404, "no such project")
    body["id"] = project_id
    fresh = file.with_suffix(".json.new")  # written beside it and swapped in: a reader never sees half a file
    fresh.write_text(json.dumps(body, indent=1))
    fresh.replace(file)
    return body


@router.get("/api/wait")
def wait(ms: int = 1000):
    """Holds a response open: a page asked for with ?shot=1 loads an image from here so a headless screenshot waits."""

    time.sleep(min(max(ms, 0), 10000) / 1000)
    return JSONResponse({"waited": ms})


@router.get("/files/{path:path}")
def files(path: str):
    target = inside(path)
    if not target.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(target)
