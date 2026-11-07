#!/usr/bin/env python3
"""Motion Previs analysis, a modified port of Motion Previs Studio 4.1.0: `motion_worker.py job.json` or `--probe`."""
# Copyright 2026 Sam Wasserman, Apache-2.0, modified: https://github.com/wassermanproductions/motion-previs-studio

from __future__ import annotations

import json
import resource
import shutil
import sys
import time
from pathlib import Path

from motion_camera import analyze_camera, describe_camera
from motion_depth import DEPTH_MODEL, ai_depth
from motion_media import Encoder, Progress, clamp, emit, ffmpeg, ffprobe, models_dir, tool
from motion_pack import BUNDLE_FILES, SOURCE, mode_instruction, planning_data, quality_report, write_texts, zip_folder
from motion_pose import POSE_MODELS, analyze_pose, draw_openpose, draw_pose, empty_pose, normalize_pose_settings, openpose_json

PORT_NAME = "Wasserman Slate AI Studio · Motion Previs (Python port of Motion Previs Studio by Sam Wasserman)"
# its own process (packages in their own venv; a cancel signals the group); a layer that cannot run is skipped
STAGES = ("prepare", "pose", "camera", "depth", "encode", "pack")
MODES = ("camera-only", "actor-motion", "object-motion", "full-scene")
LAYERS = ("depth", "ai-depth", "pose", "camera", "edges", "lineart", "masks", "normals")


def probe() -> dict:
    """What this interpreter can do. A missing package turns its layer off; it never fails the others."""

    out = {"python": sys.executable, "version": sys.version.split()[0], "models": str(models_dir()),
           "ffmpeg": bool(shutil.which(tool("ffmpeg")) or Path(tool("ffmpeg")).is_file()), "numpy": None, "cv2": None,
           "mediapipe": None, "onnxruntime": None, "pose_models": [], "depth_model": False, "why": {}}
    for name in ("numpy", "cv2", "mediapipe", "onnxruntime"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "installed")
        except Exception as error:  # noqa: BLE001 - a broken wheel is the same as a missing one here
            out["why"][name] = f"{type(error).__name__}: {error}"[:200]
    out["pose_models"] = [key for key, name in POSE_MODELS.items() if (models_dir() / name).is_file()]
    out["depth_model"] = (models_dir() / DEPTH_MODEL).is_file()
    out["pose"] = bool(out["mediapipe"] and out["numpy"] and out["cv2"] and out["pose_models"])
    out["camera"] = bool(out["cv2"] and out["numpy"])
    out["ai_depth"] = bool(out["onnxruntime"] and out["numpy"] and out["depth_model"])
    return out


def peak_memory_mb() -> dict:
    scale = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, kilobytes on Linux
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1048576
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale / 1048576
    return {"analysis": round(own), "largest_ffmpeg": round(children)}


# ---------- one job ----------

def run(job: dict) -> None:
    folder = Path(job["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    can, progress = probe(), Progress()
    layers = [layer for layer in job["layers"] if layer in LAYERS]
    job["layers"] = layers
    settings = normalize_pose_settings(job.get("pose"))
    fps = clamp(float(job.get("fps") or 12), 4, 24)
    source = Path(job["source"])
    meta = ffprobe(source)
    if not meta["width"] or not meta["height"]:
        raise RuntimeError("this file has no video stream")
    start = clamp(float(job["start"]), 0, max(meta["duration"] - 0.1, 0))
    end = clamp(float(job["end"]), start + 0.1, meta["duration"] or start + 8)
    span = {"start": start, "end": end, "duration": end - start}
    job["range"] = span
    names = {name: folder / name for name in ("reference.mp4", "depth.mp4", "ai_depth.mp4", "edges.mp4", "lineart.mp4", "motion_mask.mp4",
                                              "normals_proxy.mp4", "animatic.mp4", "contact_sheet.jpg", "preview.jpg",
                                              "pose_high_contrast.mp4", "openpose_pose.mp4", "combined_reference_depth_pose.mp4")}
    landscape = meta["width"] >= meta["height"]
    hd = job.get("resolution") == "720p"
    # the original's sizes (long edge 1280 reference, 960 control; or short edge 720), a smaller source not enlarged
    even = lambda edge, side: f"trunc(min({edge}\\,{side})/2)*2"
    reference_scale = ((f"scale=-2:{even(720, 'ih')}" if landscape else f"scale={even(720, 'iw')}:-2") if hd
                       else (f"scale={even(1280, 'iw')}:-2" if landscape else f"scale=-2:{even(1280, 'ih')}"))
    control_scale = (("scale=-2:720" if landscape else "scale=720:-2") if hd else
                     (f"scale={even(960, 'iw')}:-2" if landscape else f"scale=-2:{even(960, 'ih')}"))

    def control(target: Path, filters: str, label: str, done: float, of: float, origin: Path | None = None) -> None:
        ffmpeg(["-i", str(origin or names["reference.mp4"]), "-vf", filters, "-r", f"{fps:g}", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p", str(target)], span["duration"],
               lambda at: progress.step(done + at, of, label))

    # 1. prepare: the trimmed, normalised reference clip
    progress.start("prepare", "Trimming and normalising the reference clip")
    ffmpeg(["-ss", f"{start}", "-t", f"{span['duration']}", "-i", str(source), "-map", "0:v:0", "-map", "0:a?", "-vf", reference_scale,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(names["reference.mp4"])], span["duration"], lambda at: progress.step(at, span["duration"], "Trimming and normalising the reference clip"))
    ffmpeg(["-ss", "0", "-i", str(names["reference.mp4"]), "-frames:v", "1", "-q:v", "3", str(names["preview.jpg"])])
    reference = ffprobe(names["reference.mp4"])
    if not reference["duration"]:
        raise RuntimeError("the trimmed clip is empty: check the range")

    # 2. pose
    progress.start("pose", "Tracking pose")
    pose = None
    if "pose" not in layers:
        why = "The pose layer was not selected for this pack."
    elif not can["pose"]:
        why = "Pose is not available on this machine: " + ("no pose model was downloaded." if can["mediapipe"] and can["cv2"]
                                                             else "MediaPipe is not installed (oneshot-setup.sh --motion adds it).")
        progress.note(why)
    else:
        try:
            pose = analyze_pose(names["reference.mp4"], reference, fps, settings, progress)
        except Exception as error:  # noqa: BLE001 - the other layers do not depend on this one
            why = f"Pose did not run: {str(error)[:200]}"
            progress.note(why)
    pose_data = pose or empty_pose(reference, fps, settings, why)

    # 3. camera
    progress.start("camera", "Solving the subject-masked camera move")
    camera = None
    if "camera" in layers:
        if can["camera"]:
            camera = analyze_camera(names["reference.mp4"], reference, min(fps, 12), pose, progress)
            if not pose:
                progress.note("Camera solved without a pose track: corners were taken from the edges of the frame first.")
        else:
            progress.note("The camera solve is not available on this machine: OpenCV is not installed (oneshot-setup.sh --motion adds it).")

    # 4. depth: the ffmpeg proxy (always, as in the original), its normals, and Depth-Anything
    progress.start("depth", "Rendering depth")
    wants_ai = "ai-depth" in layers and settings["depthModel"] == "depth-anything"
    control(names["depth.mp4"], f"fps={fps:g},{control_scale},format=gray,eq=contrast=1.45:brightness=0.03,boxblur=2:1", "Rendering the depth proxy",
            0, span["duration"] * (2 if "normals" in layers else 1))
    if "normals" in layers:
        control(names["normals_proxy.mp4"], f"fps={fps:g},format=gray,sobel,eq=contrast=1.7", "Rendering the normals proxy", span["duration"],
                span["duration"] * 2, names["depth.mp4"])
    if wants_ai:
        if can["ai_depth"]:
            try:
                ai_depth(names["reference.mp4"], names["ai_depth.mp4"], reference, fps, progress)
            except Exception as error:  # noqa: BLE001 - the proxy stands in
                names["ai_depth.mp4"].unlink(missing_ok=True)
                progress.note(f"AI depth did not run, the ffmpeg proxy stands in: {str(error)[:200]}")
        else:
            progress.note("AI depth is not available on this machine (oneshot-setup.sh --motion adds Depth-Anything-small): the ffmpeg proxy stands in.")

    # 5. encode: skeleton videos, edges, line art, motion mask
    progress.start("encode", "Rendering the control layers")
    passes = [(key, name, filters) for key, name, filters in (
        ("edges", "edges.mp4", f"fps={fps:g},{control_scale},edgedetect=low=0.08:high=0.28,format=gray,eq=contrast=1.35"),
        ("lineart", "lineart.mp4", f"fps={fps:g},{control_scale},format=gray,sobel,negate,eq=contrast=1.2"),
        ("masks", "motion_mask.mp4", f"fps={fps:g},{control_scale},tblend=all_mode=difference,format=gray,eq=contrast=3.2:brightness=0.02,boxblur=1:1"))
        if key in layers]
    count = len(pose["frames"]) if pose else 0
    of = len(passes) * span["duration"] + 2 * count + span["duration"]
    done = 0.0
    for key, name, filters in passes:
        control(names[name], filters, f"Rendering {name}", done, of)
        done += span["duration"]
    keypoints = None
    if pose:
        width, height = reference["width"], reference["height"]
        for name, draw, label in (("openpose_pose.mp4", draw_openpose, "Rendering OpenPose skeleton"),
                                  ("pose_high_contrast.mp4", draw_pose, "Rendering pose video")):
            encoder = Encoder(names[name], width, height, fps)
            try:
                for index, frame in enumerate(pose["frames"]):
                    encoder.write(draw(frame, width, height))
                    progress.step(done + index + 1, of, f"{label} {index + 1}/{count}")
            finally:
                encoder.close()
            done += count
        keypoints = openpose_json(pose, width, height)
        best_depth = names["ai_depth.mp4"] if names["ai_depth.mp4"].is_file() else names["depth.mp4"]
        try:  # the review plate: reference, depth and pose side by side
            ffmpeg(["-i", str(names["reference.mp4"]), "-i", str(best_depth), "-i", str(names["pose_high_contrast.mp4"]), "-filter_complex",
                    "[0:v]scale=640:-2,setpts=PTS-STARTPTS[a];[1:v]scale=640:-2,setpts=PTS-STARTPTS[b];[2:v]scale=640:-2,setpts=PTS-STARTPTS[c];"
                    f"[a][b][c]hstack=inputs=3,fps={fps:g}[v]", "-map", "[v]", "-t", f"{pose['duration']}", "-r", f"{fps:g}", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(names["combined_reference_depth_pose.mp4"])], span["duration"],
                   lambda at: progress.step(done + at, of, "Rendering the review plate"))
        except RuntimeError as error:
            progress.note(f"The review plate was not made: {str(error)[-120:]}")

    # 6. pack: contact sheet, animatic, the JSON files, the prompt pack, the manifest, the ZIP
    progress.start("pack", "Writing the production pack")
    ffmpeg(["-i", str(names["reference.mp4"]), "-vf", "fps=1,scale=240:-2,tile=5x2", "-frames:v", "1", str(names["contact_sheet.jpg"])])
    progress.step(1, 4, "Contact sheet written")
    ffmpeg(["-i", str(names["reference.mp4"]), "-vf", "fps=2,scale=960:-2,setpts=1.4*PTS", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf",
            "20", "-pix_fmt", "yuv420p", str(names["animatic.mp4"])])
    progress.step(2, 4, "Animatic written")
    (folder / "pose_landmarks.json").write_text(json.dumps(pose_data, indent=2))
    if camera:
        (folder / "camera_motion.json").write_text(json.dumps(camera, indent=2))
    if keypoints is not None:
        (folder / "openpose_keypoints.json").write_text(json.dumps(keypoints, indent=2))
    said = describe_camera(camera)
    quality = quality_report(pose_data, camera, len(layers))
    planning = planning_data(job, span["duration"], quality, settings, bool(camera))
    write_texts(folder, job, planning, pose_data, said)
    created = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    (folder / "analysis_manifest.json").write_text(json.dumps({
        "schemaVersion": 1, "analysisId": job["id"], "createdAt": created, "sourceName": job["source_name"], "range": span, "sampleFps": fps,
        "outputDir": ".", "frameSize": {"width": reference["width"], "height": reference["height"]}, "status": "prepared",
        "files": {key: name for key, name in (("reference", "reference.mp4"), ("depth", "depth.mp4"), ("edges", "edges.mp4"),
                                              ("lineart", "lineart.mp4"), ("motionMask", "motion_mask.mp4"), ("normals", "normals_proxy.mp4"),
                                              ("animatic", "animatic.mp4"), ("contactSheet", "contact_sheet.jpg"), ("preview", "preview.jpg"))
                  if (folder / name).is_file()}}, indent=2))
    progress.step(3, 4, "Manifests written")
    progress.finish()
    manifest = {"schemaVersion": 1, "app": PORT_NAME, "version": job.get("studio_version", ""), "createdAt": created,
                "sourceName": job["source_name"], "range": span, "sampleFps": fps, "frameCount": len(pose_data["frames"]),
                "cameraMoveFrames": len(camera["frames"]) if camera else 0, "analysisSettings": settings,
                "poseDiagnostics": pose_data["summary"]["diagnostics"], "planning": planning,
                "files": {key: (name if name and (folder / name).is_file() else None) for key, name in BUNDLE_FILES.items()},
                # everything below is this port's own: the original's manifest ends at "files"
                "port": {"of": SOURCE, "modified": True, "cameraDescription": said, "notes": progress.notes,
                         "frameSize": {"width": reference["width"], "height": reference["height"]},
                         "frameRate": reference["frameRate"], "duration": reference["duration"],
                         "available": {key: can[key] for key in ("pose", "camera", "ai_depth")},
                         "packages": {key: can[key] for key in ("numpy", "cv2", "mediapipe", "onnxruntime")},
                         "timings": progress.timings, "peakMemoryMB": peak_memory_mb()}}
    (folder / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2))
    zip_folder(folder, folder.with_suffix(".zip"))
    emit(finished=True, timings=progress.timings, memory=manifest["port"]["peakMemoryMB"], notes=progress.notes)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--probe":
        print(json.dumps(probe()))
        sys.exit(0)
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        run(json.loads(Path(sys.argv[1]).read_text()))
    except Exception as error:  # noqa: BLE001 - the app shows the reason
        emit(failed=str(error)[-600:])
        sys.exit(1)
