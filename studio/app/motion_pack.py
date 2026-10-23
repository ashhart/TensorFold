"""The production pack (quality, texts, manifests, ZIP): port of Motion Previs Studio's quality.ts and main.cjs."""
# Copyright 2026 Sam Wasserman, Apache-2.0, modified: https://github.com/wassermanproductions/motion-previs-studio

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

from motion_media import clamp

HERE = Path(__file__).resolve().parent
SOURCE = {"name": "Motion Previs Studio", "version": "4.1.0", "author": "Sam Wasserman",
          "url": "https://github.com/wassermanproductions/motion-previs-studio",
          "commit": "95e7d0ff1d4cc546f7eb09a74ccbd084988a19bc", "license": "Apache-2.0"}


# ---------- quality (quality.ts / shared/quality.cjs) ----------

def quality_band(value: float) -> str:
    return "Excellent" if value >= 0.82 else "Good" if value >= 0.64 else "Review" if value > 0 else "Missing"


def quality_report(pose: dict, camera: dict | None, layer_count: int) -> dict:
    summary = pose["summary"]
    raw, filled, total = summary.get("rawDetectedFrames", 0), summary.get("filledFrames", 0), len(pose["frames"])
    tracking = clamp(raw / total + min(filled / total, 0.18), 0, 1) if total else 0.0
    confidence = clamp((camera or {}).get("summary", {}).get("averageConfidence", 0), 0, 1)
    layers = min(1.0, layer_count / 6)
    score = math.floor(tracking * 34 + confidence * 42 + layers * 24 + 0.5)
    return {"score": score, "tracking": quality_band(tracking), "camera": quality_band(confidence) if camera else "Missing",
            "layers": quality_band(layers), "readiness": "Ready" if score >= 80 else "Review" if score >= 58 else "Blocked",
            "notes": [f"Pose frames detected: {raw}/{total}", f"Filled pose gaps: {filled}", f"Camera confidence: {round(confidence * 100)}%",
                      "Control layers generated: depth, edges, lineart, motion mask, normals proxy, pose, camera."]}


# ---------- the pack (main.cjs savePoseArtifacts) ----------

def mode_instruction(mode: str) -> str:
    return {"camera-only": "Preserve only the camera move and timing. Replace the character, car, object, and environment freely.",
            "actor-motion": "Preserve the actor body motion and camera movement. Replace identity, wardrobe, styling, and environment.",
            "object-motion": "Preserve the motion path of the main object or vehicle plus camera movement. Replace design, surface, and world.",
            }.get(mode, "Preserve camera, blocking, subject motion, and depth rhythm as a full-scene reference.")


def planning_data(job: dict, duration: float, quality: dict, settings: dict, camera_on: bool) -> dict:
    plan = job.get("planning") or {}
    mode = job["mode"]
    scene, shot = (plan.get("sceneTitle") or "").strip() or "Scene 01", (plan.get("shotTitle") or "").strip() or "Shot 01A"
    intent = (plan.get("creativeIntent") or "").strip()
    return {"projectTitle": (plan.get("projectTitle") or "").strip() or "Motion Previs Project", "sceneTitle": scene, "shotTitle": shot,
            "creativeIntent": intent or "Preserve reference timing and camera language for a new AI-film shot.",
            "visualStyle": (plan.get("visualStyle") or "").strip() or "Professional cinematic previs.", "subjectMode": mode,
            "selectedLayers": job["layers"], "exportPresets": ["seedance", "comfyui", "blender"],
            "shotBible": [{"id": "shot-001", "scene": scene, "shot": shot, "description": intent or "Reference-derived previs shot.",
                           "duration": duration, "subjectMode": mode,
                           "cameraIntent": ("Recreate the solved camera pan, tilt, zoom, roll, and timing from the reference." if camera_on
                                            else "Camera solve disabled for this pass."), "selected": True}],
            "qualityReport": quality, "analysisSettings": settings}


def write_texts(folder: Path, job: dict, planning: dict, pose: dict, said: dict) -> None:
    source, span, mode, quality = job["source_name"], job["range"], job["mode"], planning["qualityReport"]
    dump = lambda name, data: (folder / name).write_text(json.dumps(data, indent=2))
    dump("comfyui_manifest.json", {
        "name": f"Motion Previs Studio v{SOURCE['version']} Control Reference Bundle", "source": source, "range": span,
        "recommendedInputs": [
            {"role": "reference_video", "file": "reference.mp4"},
            {"role": "depth_control", "file": "ai_depth.mp4", "note": "Use when present for AI monocular depth conditioning."},
            {"role": "fast_depth_proxy", "file": "depth.mp4", "note": "Always present fast local proxy depth pass."},
            {"role": "pose_control", "file": "pose_high_contrast.mp4", "note": "Use with OpenPose/DWPose-style control workflows."},
            {"role": "camera_move", "file": "camera_motion.json", "note": "Use to recreate shot camera pan, tilt, zoom, and roll without preserving the subject."},
            {"role": "edge_control", "file": "edges.mp4", "note": "Use for composition, silhouettes, and line fidelity."},
            {"role": "lineart_control", "file": "lineart.mp4", "note": "Use for strong graphic line guidance."},
            {"role": "mask_motion_proxy", "file": "motion_mask.mp4", "note": "Use as a motion/foreground proxy mask."},
            {"role": "normals_proxy", "file": "normals_proxy.mp4", "note": "Use as a depth-gradient normals proxy."},
            {"role": "combined_review", "file": "combined_reference_depth_pose.mp4"}],
        "comfyuiNotes": [
            "Load reference.mp4, ai_depth.mp4 or depth.mp4, and pose_high_contrast.mp4 as separate conditioning sources when possible.",
            "If a workflow only accepts one control movie, start with ai_depth.mp4 when present, otherwise depth.mp4, then test pose_high_contrast.mp4.",
            "pose_landmarks.json preserves MediaPipe landmark coordinates for custom nodes or retiming.",
            "camera_motion.json contains global camera move keyframes for subject-independent camera recreation.",
            "edges.mp4, lineart.mp4, motion_mask.mp4, and normals_proxy.mp4 provide extra control-layer options for precision workflows."]})
    pose_line = ("- `pose_high_contrast.mp4` only if you decide to reintroduce body-motion guidance; keep it optional for camera-only remixes."
                 if mode == "camera-only" else "- `pose_high_contrast.mp4` as the human/object motion control input.")
    (folder / "seedance_prompt.md").write_text("\n".join([
        "# Seedance / AI Video Reference Notes", "", f"Source: {source or 'Imported clip'}",
        f"Shot range: {span['start']:.2f}s to {span['end']:.2f}s", f"Subject mode: {mode}", "",
        "Use the attached reference assets according to the selected subject mode.", mode_instruction(mode), "", "Suggested inputs:",
        "- `reference.mp4` for composition, staging, timing, and camera feel.",
        "- `ai_depth.mp4` as the primary AI monocular depth control input when present.",
        "- `depth.mp4` as the fast local proxy depth control input.", pose_line,
        "- `camera_motion.json` to recreate the camera pan, tilt, zoom, and roll without reusing the original character, car, or object motion.",
        "- `edges.mp4`, `lineart.mp4`, `motion_mask.mp4`, and `normals_proxy.mp4` as optional precision control layers.",
        "- `blender_import_camera.py` to build an animated Blender camera from the solved camera move.",
        "- `blender_import_scene.py` to assemble a Blender scene with camera, pose, and reference plates.",
        "- `combined_reference_depth_pose.mp4` for a quick visual review plate.", "", "Prompt starter:",
        "Create a new shot that follows the same camera move, depth rhythm, and timing as the reference. Do not copy the original subject or objects unless requested; use the camera_motion.json / camera reference to preserve the movement of the camera independently from the scene content."]))
    (folder / "prompt_pack.md").write_text("\n".join([
        f"# Motion Previs Studio v{SOURCE['version']} Prompt Pack", "", f"Project: {planning['projectTitle']}", f"Scene: {planning['sceneTitle']}",
        f"Shot: {planning['shotTitle']}", f"Subject mode: {mode}", f"Quality score: {quality['score']}/100 ({quality['readiness']})", "",
        "## Creative Intent", planning["creativeIntent"], "", "## Visual Style", planning["visualStyle"], "",
        "## Universal Camera Instruction",
        "Use the supplied camera_motion.json and reference timing to preserve camera movement, lens rhythm, pan/tilt/zoom, and shot duration. Do not copy the original subject unless the subject mode explicitly asks for it.",
        "", "## Subject Handling", mode_instruction(mode), "", "## Seedance",
        "Use reference.mp4 for timing, ai_depth.mp4 or depth.mp4 for depth structure, pose_high_contrast.mp4 only when subject motion should be retained, and camera_motion.json as the camera-move authority.",
        "", "## ComfyUI",
        "Load separate control inputs where possible: depth/ai_depth, pose, edges, lineart, motion_mask, and camera_motion.json. Disable pose when using Camera only mode.",
        "", "## Runway / Kling",
        "Use the reference clip for camera/timing, but phrase the prompt around new subjects and environments. Use camera-only wording when replacing the source subject.",
        "", "## Negative Guidance",
        "Do not drift from the camera move. Do not inherit unwanted faces, wardrobe, props, cars, or background objects unless requested. Avoid inconsistent lens changes, sudden perspective jumps, and unmotivated speed ramps.",
        "", "## MiniMax H3 (added by the Python port in Wasserman Slate AI Studio)",
        "The camera move in words, from camera_motion.json: " + (said["sentence"] or "no camera solve in this pack.")
        + ("" if said.get("reliable", True) or not said["sentence"] else " (Low solve confidence: check it against the clip.)")]))
    dump("shot_bible.json", planning["shotBible"])
    dump("quality_report.json", quality)
    dump("model_presets.json", {"selected": planning["exportPresets"], "analysisSettings": pose["summary"].get("settings"),
                                "poseDiagnostics": pose["summary"].get("diagnostics", []), "presets": {
        "seedance": {"primaryInputs": ["reference.mp4", "ai_depth.mp4", "depth.mp4", "pose_high_contrast.mp4", "camera_motion.json"],
                     "guidance": "Use separate depth and pose controls when available. Use camera-only prompt language when replacing subjects."},
        "comfyui": {"primaryInputs": ["depth.mp4", "edges.mp4", "lineart.mp4", "motion_mask.mp4", "pose_high_contrast.mp4"],
                    "guidance": "Wire control videos as separate conditioning branches. Start with depth + camera, then add pose only if needed."},
        "blender": {"primaryInputs": ["blender_import_scene.py", "blender_import_camera.py", "blender_import_pose.py"],
                    "guidance": "Run blender_import_scene.py from Blender Text Editor or Python console with the bundle folder as the script location."},
        "runway": {"primaryInputs": ["reference.mp4", "prompt_pack.md", "camera_motion.json"],
                   "guidance": "Use the prompt pack language to preserve camera while replacing subject and world."},
        "kling": {"primaryInputs": ["reference.mp4", "depth.mp4", "prompt_pack.md"],
                  "guidance": "Use reference video for timing and depth/video controls for structure when available."}}})
    dump("control_layers_manifest.json", {"selectedLayers": planning["selectedLayers"], "layers": [
        {"key": "depth", "file": "depth.mp4", "purpose": "Fast structural/depth proxy."},
        {"key": "ai-depth", "file": "ai_depth.mp4", "purpose": "Optional AI monocular depth pass."},
        {"key": "pose", "file": "pose_high_contrast.mp4", "purpose": "Human pose and motion control."},
        {"key": "camera", "file": "camera_motion.json", "purpose": "Subject-independent camera move."},
        {"key": "edges", "file": "edges.mp4", "purpose": "Composition and silhouette edges."},
        {"key": "lineart", "file": "lineart.mp4", "purpose": "High-contrast line control."},
        {"key": "masks", "file": "motion_mask.mp4", "purpose": "Motion/foreground proxy mask (frame-difference)."},
        {"key": "normals", "file": "normals_proxy.mp4", "purpose": "Depth-gradient normals proxy."}]})
    for script in ("blender_import_pose.py", "blender_import_camera.py", "blender_import_scene.py"):
        shutil.copyfile(HERE / "motion_templates" / script, folder / script)


# bundle_manifest.json "files": the original's keys -> the file each names
BUNDLE_FILES = {"reference": "reference.mp4", "depth": "depth.mp4", "aiDepthWebm": None, "aiDepthMp4": "ai_depth.mp4", "edges": "edges.mp4",
                "lineart": "lineart.mp4", "motionMask": "motion_mask.mp4", "normalsProxy": "normals_proxy.mp4",
                "contactSheet": "contact_sheet.jpg", "animatic": "animatic.mp4", "poseJson": "pose_landmarks.json",
                "cameraMotionJson": "camera_motion.json", "poseWebm": None, "poseMp4": "pose_high_contrast.mp4",
                "openPosePose": "openpose_pose.mp4", "openPoseKeypoints": "openpose_keypoints.json",
                "combined": "combined_reference_depth_pose.mp4", "blenderImportScript": "blender_import_pose.py",
                "blenderCameraImportScript": "blender_import_camera.py", "blenderSceneImportScript": "blender_import_scene.py",
                "comfyuiManifest": "comfyui_manifest.json", "seedancePrompt": "seedance_prompt.md", "promptPack": "prompt_pack.md",
                "shotBible": "shot_bible.json", "qualityReport": "quality_report.json", "modelPresets": "model_presets.json",
                "controlLayersManifest": "control_layers_manifest.json"}


def zip_folder(folder: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and "handoff" not in path.relative_to(folder).parts and not path.name.startswith("."):
                archive.write(path, path.relative_to(folder).as_posix())
