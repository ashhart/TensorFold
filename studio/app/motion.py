"""Motion Previs routes, queue, pack import and hand-off; the analysis is a modified port of Motion Previs Studio."""
# Copyright 2026 Sam Wasserman, Apache-2.0, modified: https://github.com/wassermanproductions/motion-previs-studio

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

import motion_worker as worker
import pace

HOME = Path.home() / "TensorFoldStudio"
PACK = Path(__file__).resolve().parent.parent
FPS = 24                      # the Studio's clips
SHORTEST, LONGEST = 2.0, 15.0  # one reference clip for MiniMax H3 Ref2VA: 2 to 15 seconds (3 clips, 15 s in all)
REFERENCE_CLIPS = 3
VIDEO = (".mp4", ".mov", ".m4v", ".webm")
SOURCES = ("uploads", "renders", "exports", "projects", "slate", "motion")
DEFAULT_LAYERS = ["depth", "ai-depth", "pose", "camera", "edges", "masks"]  # App.tsx selectedLayers
# what "Use as motion reference" can hand on: the layer's name here -> its file in the pack
HANDOFF_LAYERS = {"reference": "reference.mp4", "openpose": "openpose_pose.mp4", "pose": "pose_high_contrast.mp4",
                  "ai-depth": "ai_depth.mp4", "depth": "depth.mp4", "edges": "edges.mp4", "lineart": "lineart.mp4",
                  "masks": "motion_mask.mp4", "normals": "normals_proxy.mp4"}
# measured on one shot: the source clip follows the body closest, then depth, then the skeleton (never the default)
SUGGESTED = {"camera-only": ["ai-depth", "depth", "reference"], "actor-motion": ["ai-depth", "depth", "reference"],
             "object-motion": ["ai-depth", "depth", "reference"], "full-scene": ["reference"]}
# a pack ZIP may hold these and nothing else is taken out of it
PACK_FILES = ({name for name in worker.BUNDLE_FILES.values() if name}
              | {"bundle_manifest.json", "analysis_manifest.json", "preview.jpg", "pose_high_contrast.webm", "openpose_pose.webm", "ai_depth.webm"})
ZIP_LIMIT = 2 * 1024 ** 3


def inside(path: str | Path) -> Path:
    resolved = (HOME / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if HOME.resolve() not in (resolved, *resolved.parents):
        raise HTTPException(400, "path is outside the studio folder")
    return resolved


def relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(HOME.resolve()))


def pack_folder(pack: str | Path) -> Path:
    """A pack by its id, or by the path of its folder under the studio's motion folder."""

    name = Path(str(pack)).name
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
        raise HTTPException(400, "bad pack id")
    folder = HOME / "motion" / name
    if not (folder / "bundle_manifest.json").is_file():
        raise HTTPException(404, f"no motion pack {name}")
    return folder


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def camera_words(folder: Path, seconds: float | None = None, until: float | None = None) -> dict:
    """The pack's camera move in words; `until` reads only the first seconds of it."""

    camera = read_json(folder / "camera_motion.json")
    if not camera or not isinstance(camera.get("frames"), list):
        return {"sentence": "", "moves": [], "confidence": 0.0, "reliable": False}
    if until and until < float(camera.get("duration") or 0):
        camera = {**camera, "duration": until, "frames": [f for f in camera["frames"] if f.get("time", 0) <= until]}
    try:
        return worker.describe_camera(camera, seconds)
    except (KeyError, TypeError, ValueError):  # a pack from elsewhere whose camera file is not in the expected shape
        return {"sentence": "", "moves": [], "confidence": 0.0, "reliable": False}


def has_sound(path: Path) -> bool:
    out = subprocess.run([worker.tool("ffprobe"), "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return "audio" in out.stdout


def soft_mouth(clip: Path, pose: dict, target: Path, speed: float, source_size: tuple[int, int]) -> int:
    """`clip` with the tracked mouth blurred (the model copies lip movement), to `target`; returns frames smoothed."""

    try:
        import numpy as np
    except ImportError:
        return 0
    frames = [f for f in pose.get("frames") or [] if isinstance(f, dict)]
    meta = worker.ffprobe(clip)
    width, height, count = meta["width"], meta["height"], round(meta["duration"] * FPS)
    if not frames or not width or not count:
        return 0
    wide, tall = source_size
    scale = max(width / wide, height / tall)  # the cover crop of motion_reference
    left, top = (wide * scale - width) / 2, (tall * scale - height) / 2
    times = np.array([float(f.get("time") or 0) for f in frames])
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    masks, radii, done = [], [], 0
    for n in range(count):
        frame = frames[int(np.argmin(np.abs(times - n / FPS * speed)))]
        marks = frame.get("landmarks") or []
        mask = np.zeros((height, width), np.float32)
        if len(marks) > 10 and min(float(marks[i].get("visibility") or 0) for i in (9, 10)) > 0.5:
            x = (marks[9]["x"] + marks[10]["x"]) / 2 * wide * scale - left
            y = (marks[9]["y"] + marks[10]["y"]) / 2 * tall * scale - top
            radius = max(abs(marks[7]["x"] - marks[8]["x"]) * wide * scale * 0.62, 0.016 * width)
            reach = np.sqrt(((xs - x) / radius) ** 2 + ((ys - (y + 0.15 * radius)) / (0.8 * radius)) ** 2)
            mask = np.clip((1.25 - reach) / 0.5, 0, 1)  # full inside, feathered over the outer quarter
            radii.append(radius)
            done += 1
        masks.append((mask * 255).astype(np.uint8).tobytes())
    if not done:
        return 0
    sigma = max(float(np.median(radii)) * 0.45, 3.0)
    command = [worker.tool("ffmpeg"), "-v", "error", "-y", "-i", str(clip), "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}", "-r", str(FPS),
               "-i", "-", "-filter_complex", f"[0:v]format=gbrp,split[a][b];[b]gblur=sigma={sigma:.1f}[soft];[1:v]format=gbrp[m];[a][soft][m]maskedmerge,format=yuv420p[v]",
               "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-c:a", "copy", "-frames:v", str(count),
               "-movflags", "+faststart", str(target)]
    made = subprocess.run(command, input=b"".join(masks), capture_output=True)
    if made.returncode or not target.is_file():
        target.unlink(missing_ok=True)
        return 0
    return done


def motion_reference(pack, layer, mode, seconds, width, height, *, timing: str = "stretch", fit: str = "cover", mouth: str = "") -> dict:
    """One pack layer as a `<Video 1>` clip of the asked size and length plus the camera in words; calls no model."""

    folder = pack_folder(pack)
    manifest = read_json(folder / "bundle_manifest.json", {})
    mode = mode or (manifest.get("planning") or {}).get("subjectMode") or "full-scene"
    if mode not in worker.MODES:
        raise HTTPException(400, f"the reference mode is one of {', '.join(worker.MODES)}")
    try:
        seconds, width, height = float(seconds), int(width), int(height)
    except (TypeError, ValueError):
        raise HTTPException(400, "seconds, width and height are numbers")
    if not SHORTEST <= seconds <= LONGEST:
        raise HTTPException(400, f"a reference clip is {SHORTEST:g} to {LONGEST:g} seconds")
    if width % 2 or height % 2 or not (64 <= width <= 4096 and 64 <= height <= 4096):
        raise HTTPException(400, "width and height are even numbers between 64 and 4096")
    if timing not in ("stretch", "trim") or fit not in ("cover", "contain"):
        raise HTTPException(400, "timing is stretch or trim; fit is cover or contain")
    layer = None if layer in (None, "", "none") else layer
    sound = False
    probe = worker.ffprobe(folder / "reference.mp4") if (folder / "reference.mp4").is_file() else {"duration": 0, "width": 0, "height": 0}
    clip = None
    if layer is not None:
        if layer not in HANDOFF_LAYERS:
            raise HTTPException(400, f"the layer is one of {', '.join(HANDOFF_LAYERS)}, or none")
        source = folder / HANDOFF_LAYERS[layer]
        if not source.is_file():
            raise HTTPException(400, f"this pack has no {HANDOFF_LAYERS[layer]}")
        probe = worker.ffprobe(source)
        if not probe["duration"]:
            raise HTTPException(400, f"{HANDOFF_LAYERS[layer]} in this pack cannot be read")
        frames = round(seconds * FPS)
        clip = folder / "handoff" / f"{layer}-{width}x{height}-{seconds:g}s-{timing}-{fit}.mp4"
        clip.parent.mkdir(exist_ok=True)
        shape = (f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height}" if fit == "cover"
                 else f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
        retime = f"setpts=PTS*{seconds / probe['duration']:.6f}," if timing == "stretch" else ""
        try:
            worker.ffmpeg(["-i", str(source), "-vf", f"{retime}fps={FPS},{shape},setsar=1,tpad=stop_mode=clone:stop_duration={seconds + 1:g}",
                           "-frames:v", str(frames), "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(clip)])
        except RuntimeError as error:
            raise HTTPException(500, str(error))
        # the shot's own sound rides along at its own speed, whatever the layer; a retimed clip gets none
        origin = folder / "reference.mp4"
        if timing == "trim" and origin.is_file() and has_sound(origin):
            voiced = clip.with_suffix(".sound.mp4")
            try:
                worker.ffmpeg(["-i", str(clip), "-i", str(origin), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                               "-b:a", "192k", "-t", f"{frames / FPS:.6f}", "-movflags", "+faststart", str(voiced)])
                voiced.replace(clip)
                sound = True
            except RuntimeError:
                voiced.unlink(missing_ok=True)
        if mouth == "soft" and layer == "reference" and fit == "cover":
            quiet = clip.with_name(clip.stem + "-softmouth.mp4")
            softened = soft_mouth(clip, read_json(folder / "pose_landmarks.json", {}) or {}, quiet,
                                  probe["duration"] / seconds if timing == "stretch" else 1.0, (probe["width"], probe["height"]))
            if softened:
                clip = quiet
    length = float(probe["duration"] or seconds)
    said = camera_words(folder, seconds if timing == "stretch" else min(seconds, length), None if timing == "stretch" else seconds)
    return {"kind": "motion-reference", "schema": 1, "pack": folder.name, "layer": layer, "file": HANDOFF_LAYERS.get(layer), "mode": mode,
            "clip": relative(clip) if clip else None, "clip_path": str(clip) if clip else None, "seconds": seconds, "width": width,
            "height": height, "fps": FPS, "frames": round(seconds * FPS) if clip else 0, "timing": timing, "fit": fit, "sound": sound,
            "mouth": "soft" if clip and clip.name.endswith("-softmouth.mp4") else "",
            "source": {"seconds": length, "width": probe["width"], "height": probe["height"],
                       "speed": round(length / seconds, 4) if timing == "stretch" else 1.0},
            "camera": said["sentence"], "camera_reliable": said["reliable"], "camera_confidence": said["confidence"],
            "camera_moves": said["moves"], "instruction": worker.mode_instruction(mode), "tag": "<Video 1>",
            "limits": {"clips": REFERENCE_CLIPS, "min_seconds": SHORTEST, "max_seconds": LONGEST, "total_seconds": LONGEST},
            "made": time.time()}


def attach(home: Path, pack: Path, spark: bool) -> APIRouter:
    global HOME, PACK
    HOME, PACK = home, pack
    root = HOME / "motion"
    root.mkdir(parents=True, exist_ok=True)
    router = APIRouter()
    jobs: dict[str, dict] = {}
    order: list[str] = []
    lock, wake = threading.Lock(), threading.Event()
    running: dict = {"process": None, "id": None}
    seen: dict = {"at": 0.0, "value": None}
    prefix = Path(os.environ.get("PREFIX", Path.home() / ".local/opt/tensorfold-studio"))

    def python() -> str:
        """The interpreter the analysis runs in: its own environment when the installer made one."""

        own = prefix / "motion-venv" / "bin" / "python"
        return os.environ.get("MOTION_PYTHON") or (str(own) if own.is_file() else sys.executable)

    def environment() -> dict:
        env = dict(os.environ)
        env["PATH"] = "/opt/homebrew/bin:" + env.get("PATH", "")
        env.setdefault("MOTION_MODELS", str(prefix / "motion-models"))
        return env

    def able(fresh: bool = False) -> dict:
        """What the analysis can do on this machine, asked of the worker's own interpreter (kept for a minute)."""

        if fresh or seen["value"] is None or time.time() - seen["at"] > 60:
            try:
                out = subprocess.run([python(), str(Path(__file__).with_name("motion_worker.py")), "--probe"], capture_output=True,
                                     text=True, env=environment(), timeout=120)
                value = json.loads(out.stdout.strip().splitlines()[-1])
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
                value = {"python": python(), "ffmpeg": bool(shutil.which("ffmpeg", path=environment()["PATH"])), "pose": False,
                         "camera": False, "ai_depth": False, "pose_models": [], "why": {"probe": str(error)[:200]}}
            seen.update(at=time.time(), value=value)
        return seen["value"]

    def public(job: dict) -> dict:
        out = {k: job.get(k) for k in ("id", "status", "title", "pack", "stages", "stage", "notes", "error", "created", "started", "finished",
                                       "params", "timings", "memory")}
        out["elapsed"] = (job.get("finished") or time.time()) - job["started"] if job.get("started") else None
        out["log"] = job["log"][-2000:]
        return out

    def follow(job: dict, line: str) -> None:
        if not line.startswith("@@"):
            job["log"] = (job["log"] + line)[-12000:]
            return
        try:
            event = json.loads(line[2:])
        except ValueError:
            return
        if "note" in event:
            job["notes"].append(event["note"])
        if "failed" in event:
            job["error"] = event["failed"]
        if event.get("finished"):
            job["timings"], job["memory"] = event.get("timings"), event.get("memory")
        stage = event.get("stage")
        if stage in job["stages"]:
            state = job["stages"][stage]
            if event.get("done"):
                state.update(state="done", seconds=event.get("seconds"), at=state.get("of") or 1, of=state.get("of") or 1)
            else:
                job["stage"] = stage
                state.update(state="running", at=event.get("at", 0), of=event.get("of", 1), message=event.get("message", ""))

    def work() -> None:
        while True:
            wake.wait()
            with lock:
                job = next((jobs[i] for i in order if jobs[i]["status"] == "queued"), None)
                if job is None:
                    wake.clear()
                    continue
                job["status"], job["started"] = "running", time.time()
            folder = root / job["id"]
            try:
                folder.mkdir(parents=True, exist_ok=True)
                spec = folder / ".job.json"
                spec.write_text(json.dumps({**job["params"], "id": job["id"], "folder": str(folder),
                                            "source": str(inside(job["params"]["source"])),
                                            "studio_version": (PACK / "VERSION").read_text().strip()}))
                process = subprocess.Popen([python(), "-u", str(Path(__file__).with_name("motion_worker.py")), str(spec)], cwd=PACK,
                                           env=environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           start_new_session=True)
                running.update(process=process, id=job["id"])
                for line in process.stdout:
                    follow(job, line)
                code = process.wait()
                running.update(process=None, id=None)
                spec.unlink(missing_ok=True)
                if job["status"] == "cancelled":
                    continue
                if code or not (folder / "bundle_manifest.json").is_file():
                    raise RuntimeError(job.get("error") or f"the analysis exited with status {code}")
                job["status"], job["pack"] = "done", job["id"]
            except Exception as error:  # noqa: BLE001 - a failed analysis must not stop the queue
                if job["status"] != "cancelled":
                    job["status"], job["error"] = "failed", str(error)
            finally:
                job["finished"] = time.time()
                if job["status"] != "done":  # nothing half-made is left looking like a pack
                    shutil.rmtree(folder, ignore_errors=True)
                    folder.with_suffix(".zip").unlink(missing_ok=True)

    threading.Thread(target=work, daemon=True).start()

    def paced(folder: Path) -> dict | None:
        """How fast the shot's quickest movement is and whether its sound holds hits (pace.py), measured once."""

        kept, clip = folder / "pace.json", folder / "reference.mp4"
        found = read_json(kept)
        if found is None and clip.is_file():
            try:
                found = pace.measure(clip)
                if found["verdict"] != "not measured":
                    kept.write_text(json.dumps(found))
            except Exception:  # noqa: BLE001 - a pack is still a pack without it
                found = None
        return found

    def describe(folder: Path) -> dict:
        manifest = read_json(folder / "bundle_manifest.json", {})
        port = manifest.get("port") or {}
        files = {name: relative(folder / name) for name in sorted(PACK_FILES) if (folder / name).is_file()}
        layers = [layer for layer, name in HANDOFF_LAYERS.items() if name in files]
        mode = (manifest.get("planning") or {}).get("subjectMode") or "full-scene"
        size = port.get("frameSize") or (read_json(folder / "analysis_manifest.json", {}) or {}).get("frameSize") or {}
        archive = folder.with_suffix(".zip")
        return {"id": folder.name, "created": manifest.get("createdAt"), "modified": (folder / "bundle_manifest.json").stat().st_mtime,
                "source": manifest.get("sourceName"), "range": manifest.get("range"), "mode": mode, "fps": manifest.get("sampleFps"),
                "app": manifest.get("app"), "version": manifest.get("version"), "imported": (folder / ".imported.json").is_file(),
                "size": size, "files": files, "layers": layers, "suggested": next((l for l in SUGGESTED.get(mode, []) if l in layers), "reference" if "reference" in layers else None),
                "camera": port.get("cameraDescription") or camera_words(folder), "camera_summary": (read_json(folder / "camera_motion.json", {}) or {}).get("summary"),
                "pose": (read_json(folder / "pose_landmarks.json", {}) or {}).get("summary"), "quality": read_json(folder / "quality_report.json"),
                "pace": paced(folder),
                "notes": port.get("notes") or [], "timings": port.get("timings"), "memory": port.get("peakMemoryMB"),
                "zip": relative(archive) if archive.is_file() else None, "instruction": worker.mode_instruction(mode)}

    @router.get("/api/motion/state")
    def state(fresh: bool = False) -> dict:
        can = able(fresh)
        return {"modes": {"camera-only": ["Camera only", "Keep just the camera move and timing. Replace the subject and world."],
                          "actor-motion": ["Actor motion", "Keep the body motion and the camera move. Replace who it is and where."],
                          "object-motion": ["Object motion", "Keep an object's or vehicle's path and the camera move."],
                          "full-scene": ["Full scene", "Keep camera, blocking, subject motion and depth rhythm."]},
                "layers": {"depth": "Depth proxy (ffmpeg)", "ai-depth": "AI depth (Depth-Anything-small)", "pose": "Pose and OpenPose skeleton",
                           "camera": "Camera move", "edges": "Edges", "lineart": "Line art", "masks": "Motion mask", "normals": "Normals proxy"},
                "default_layers": DEFAULT_LAYERS, "handoff_layers": HANDOFF_LAYERS, "suggested": SUGGESTED, "limits": [SHORTEST, LONGEST],
                "reference_clips": REFERENCE_CLIPS, "fps": [4, 24, 12], "pose_models": can.get("pose_models") or [],
                "available": {"ffmpeg": bool(can.get("ffmpeg")), "pose": bool(can.get("pose")), "camera": bool(can.get("camera")),
                              "ai_depth": bool(can.get("ai_depth"))},
                "packages": {key: can.get(key) for key in ("numpy", "cv2", "mediapipe", "onnxruntime")}, "python": can.get("python"),
                "own_environment": (prefix / "motion-venv" / "bin" / "python").is_file() or bool(os.environ.get("MOTION_PYTHON")),
                "setup": "bash oneshot-setup-spark.sh --motion" if spark else "bash oneshot-setup.sh --motion",
                "missing": [text for ok, text in (
                    (can.get("pose"), "Pose is not available on this machine" + (": MediaPipe is not installed." if not can.get("mediapipe") else
                                                                                 ": no pose model was downloaded." if not can.get("pose_models") else ".")),
                    (can.get("camera"), "The camera solve is not available on this machine: OpenCV is not installed."),
                    (can.get("ai_depth"), "AI depth is not available on this machine: the ffmpeg depth proxy stands in.")) if not ok],
                "credit": {"name": worker.SOURCE["name"], "author": worker.SOURCE["author"], "url": worker.SOURCE["url"], "license": worker.SOURCE["license"]}}

    @router.get("/api/motion/sources")
    def sources() -> list[dict]:
        """Clips under the studio folder that an analysis can start from, newest first."""

        found = []
        for name in SOURCES[:-1]:
            for path in (HOME / name).rglob("*"):
                if path.suffix.lower() in VIDEO and path.is_file():
                    found.append({"path": relative(path), "folder": name, "modified": path.stat().st_mtime, "size": path.stat().st_size})
        return sorted(found, key=lambda item: -item["modified"])[:200]

    @router.post("/api/motion/jobs")
    def submit(body: dict) -> dict:
        source = inside(str(body.get("source") or ""))
        if not source.is_file() or source.suffix.lower() not in VIDEO or relative(source).split("/")[0] not in SOURCES:
            raise HTTPException(400, "pick a video clip from the studio folder, or upload one")
        if not able().get("ffmpeg"):
            raise HTTPException(400, "ffmpeg is not installed on this machine")
        meta = worker.ffprobe(source)
        if not meta["width"] or not meta["duration"]:
            raise HTTPException(400, "this file has no video stream that can be read")
        try:
            start, end = float(body.get("start", 0)), float(body.get("end", 0))
        except (TypeError, ValueError):
            raise HTTPException(400, "the range is a start and an end in seconds")
        if start < 0 or end > meta["duration"] + 0.05 or end <= start:
            raise HTTPException(400, f"the range lies inside the clip, 0 to {meta['duration']:.2f} seconds")
        if not SHORTEST - 1e-6 <= end - start <= LONGEST + 1e-6:
            raise HTTPException(400, f"a shot is {SHORTEST:g} to {LONGEST:g} seconds long (this range is {end - start:.2f} s)")
        mode = body.get("mode") or "full-scene"
        if mode not in worker.MODES:
            raise HTTPException(400, f"the reference mode is one of {', '.join(worker.MODES)}")
        layers = body.get("layers") if body.get("layers") is not None else DEFAULT_LAYERS
        if not isinstance(layers, list) or any(layer not in worker.LAYERS for layer in layers):
            raise HTTPException(400, f"layers are among {', '.join(worker.LAYERS)}")
        try:
            fps = float(body.get("fps") or 12)
        except (TypeError, ValueError):
            raise HTTPException(400, "the sample rate is a number")
        if not 4 <= fps <= 24:
            raise HTTPException(400, "the sample rate is 4 to 24 frames a second")
        if body.get("resolution") not in (None, "auto", "720p"):
            raise HTTPException(400, "resolution is auto or 720p")
        job_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        params = {"source": relative(source), "source_name": source.name, "start": start, "end": min(end, meta["duration"]), "mode": mode,
                  "layers": [layer for layer in worker.LAYERS if layer in layers], "fps": fps, "resolution": body.get("resolution") or "auto",
                  "pose": worker.normalize_pose_settings(body.get("pose") if isinstance(body.get("pose"), dict) else None),
                  "planning": {k: str(v)[:400] for k, v in (body.get("planning") or {}).items()
                               if k in ("projectTitle", "sceneTitle", "shotTitle", "creativeIntent", "visualStyle")}}
        job = {"id": job_id, "status": "queued", "title": body.get("title") or source.name, "params": params, "pack": None, "stage": None,
               "stages": {name: {"state": "waiting", "at": 0, "of": 1, "seconds": None, "message": ""} for name in worker.STAGES},
               "notes": [], "error": None, "log": "", "created": time.time(), "started": None, "finished": None, "timings": None, "memory": None}
        with lock:
            jobs[job_id] = job
            order.append(job_id)
        wake.set()
        return public(job)

    @router.get("/api/motion/jobs")
    def list_jobs() -> list[dict]:
        return [public(jobs[i]) for i in reversed(order)][:30]

    @router.get("/api/motion/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        if job_id not in jobs:
            raise HTTPException(404, "no such analysis")
        return public(jobs[job_id])

    @router.post("/api/motion/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such analysis")
        if job["status"] in ("queued", "running"):
            job["status"] = "cancelled"
            process = running["process"] if running["id"] == job_id else None
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)  # the worker and whichever ffmpeg it is waiting on
                except ProcessLookupError:
                    pass
        return public(job)

    @router.get("/api/motion/packs")
    def packs() -> list[dict]:
        found = [describe(file.parent) for file in root.glob("*/bundle_manifest.json")]
        return sorted(found, key=lambda item: -item["modified"])[:100]

    @router.get("/api/motion/packs/{pack_id}")
    def get_pack(pack_id: str) -> dict:
        return describe(pack_folder(pack_id))

    @router.delete("/api/motion/packs/{pack_id}")
    def discard(pack_id: str) -> dict:
        """Move a pack and its ZIP to the studio's trash folder; nothing is erased."""

        folder = pack_folder(pack_id)
        kept = HOME / "trash" / (time.strftime("%Y%m%d-%H%M%S-") + "motion__" + folder.name)
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(kept))
        if folder.with_suffix(".zip").is_file():
            shutil.move(str(folder.with_suffix(".zip")), str(kept) + ".zip")
        current = read_json(root / "handoff.json")
        if current and current.get("pack") == folder.name:
            (root / "handoff.json").unlink(missing_ok=True)
        return {"trash": relative(kept)}

    @router.post("/api/motion/packs/import")
    async def import_pack(file: UploadFile = File(...)) -> dict:
        """Take a production pack ZIP (this app's, or one exported by Motion Previs Studio) in place of an analysis."""

        if not (file.filename or "").lower().endswith(".zip"):
            raise HTTPException(400, "a production pack is a .zip file")
        pack_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        archive, folder = root / f"{pack_id}.zip", root / pack_id
        with open(archive, "wb") as handle:
            shutil.copyfileobj(file.file, handle)
        try:
            with zipfile.ZipFile(archive) as bundle:
                entries = [item for item in bundle.infolist() if not item.is_dir()]
                if len(entries) > 400 or sum(item.file_size for item in entries) > ZIP_LIMIT:
                    raise HTTPException(400, "this ZIP is too large to be a production pack")
                # files by their own names only, wherever they sit in the ZIP: nothing is written outside the pack
                wanted = {}
                for item in entries:
                    name = item.filename.replace("\\", "/").split("/")[-1]
                    if name in PACK_FILES and name not in wanted:
                        wanted[name] = item
                if "bundle_manifest.json" not in wanted or "reference.mp4" not in wanted:
                    raise HTTPException(400, "this ZIP is not a Motion Previs production pack: it has no bundle_manifest.json and reference.mp4")
                manifest = json.loads(bundle.read(wanted["bundle_manifest.json"]))
                if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
                    raise HTTPException(400, "this pack's bundle_manifest.json is not in the production pack layout")
                folder.mkdir(parents=True)
                for name, item in wanted.items():
                    with bundle.open(item) as source, open(folder / name, "wb") as target:
                        shutil.copyfileobj(source, target)
        except (zipfile.BadZipFile, ValueError, KeyError) as error:
            shutil.rmtree(folder, ignore_errors=True)
            archive.unlink(missing_ok=True)
            raise HTTPException(400, f"this ZIP cannot be read as a production pack: {error}")
        except HTTPException:
            shutil.rmtree(folder, ignore_errors=True)
            archive.unlink(missing_ok=True)
            raise
        (folder / ".imported.json").write_text(json.dumps({"from": Path(file.filename or "pack.zip").name, "at": time.time()}))
        return describe(folder)

    @router.post("/api/motion/reference")
    def reference(body: dict) -> dict:
        """Make the hand-off for a pack and keep it as the current motion reference (read by the create wizard)."""

        if not body.get("pack"):
            raise HTTPException(400, "which pack?")
        out = motion_reference(body["pack"], body.get("layer"), body.get("mode"), body.get("seconds", 5), body.get("width", 864),
                               body.get("height", 480), timing=body.get("timing") or "stretch", fit=body.get("fit") or "cover",
                               mouth=body.get("mouth") or "")
        if body.get("keep", True):
            (root / "handoff.json").write_text(json.dumps(out, indent=1))
        return out

    @router.get("/api/motion/handoff")
    def handoff() -> dict:
        current = read_json(root / "handoff.json")
        if current and current.get("clip") and not (HOME / current["clip"]).is_file():
            current = None
        return {"reference": current}

    @router.delete("/api/motion/handoff")
    def forget() -> dict:
        (root / "handoff.json").unlink(missing_ok=True)
        return {"reference": None}

    return router
