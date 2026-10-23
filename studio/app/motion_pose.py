"""Pose track, OpenPose keypoints and skeleton videos: port of Motion Previs Studio's pose.ts and poseVideo.ts."""
# Copyright 2026 Sam Wasserman, Apache-2.0, modified: https://github.com/wassermanproductions/motion-previs-studio

from __future__ import annotations

import math
from pathlib import Path

from motion_media import Progress, clamp, models_dir, read_frames

POSE_MODELS = {"lite": "pose_landmarker_lite.task", "full": "pose_landmarker_full.task", "heavy": "pose_landmarker_heavy.task"}
# pose.ts DEFAULT_POSE_SETTINGS but Heavy, not Lite: fast enough outside a browser, and Lite misplaced shoulders
POSE_DEFAULTS = {"poseModel": "heavy", "depthModel": "depth-anything", "detectionConfidence": 0.35, "trackingConfidence": 0.35,
                 "smoothing": 0.7, "temporalWindow": 5, "maxPeople": 1, "fillGaps": True, "optimizeForExport": True}


# ---------- pose (pose.ts) ----------

def normalize_pose_settings(settings: dict | None) -> dict:
    s = {**POSE_DEFAULTS, **{k: v for k, v in (settings or {}).items() if v is not None and k in POSE_DEFAULTS}}
    return {"poseModel": s["poseModel"] if s["poseModel"] in POSE_MODELS else "lite",
            "depthModel": s["depthModel"] if s["depthModel"] in ("proxy", "depth-anything") else "depth-anything",
            "detectionConfidence": clamp(float(s["detectionConfidence"]), 0, 1), "trackingConfidence": clamp(float(s["trackingConfidence"]), 0, 1),
            "smoothing": clamp(float(s["smoothing"]), 0, 1), "temporalWindow": int(clamp(round(float(s["temporalWindow"])), 1, 30)),
            "maxPeople": int(clamp(round(float(s["maxPeople"])), 1, 4)), "fillGaps": bool(s["fillGaps"]),
            "optimizeForExport": bool(s["optimizeForExport"])}


def pose_model_label(key: str) -> str:
    return {"lite": "MediaPipe Pose (Lite)", "full": "MediaPipe Pose (Full)", "heavy": "MediaPipe Pose (Heavy)"}.get(key, key)


def landmark(point) -> dict:
    out = {"x": float(point.x or 0), "y": float(point.y or 0), "z": float(point.z or 0)}
    for key in ("visibility", "presence"):  # left out when the model gave none, as JSON.stringify drops undefined
        value = getattr(point, key, None)
        if value is not None:
            out[key] = float(value)
    return out


def score_landmarks(landmarks: list[dict]) -> float:
    if not landmarks:
        return 0.0
    return sum(p.get("visibility", p.get("presence", 0.75)) for p in landmarks) / len(landmarks)


def clone_landmarks(landmarks: list[dict]) -> list[dict]:
    return [dict(p) for p in landmarks]


def clone_frame(frame: dict) -> dict:
    out = {**frame, "landmarks": clone_landmarks(frame["landmarks"]), "worldLandmarks": clone_landmarks(frame["worldLandmarks"])}
    if frame.get("poses") is not None:
        out["poses"] = [{"id": p["id"], "landmarks": clone_landmarks(p["landmarks"]), "worldLandmarks": clone_landmarks(p["worldLandmarks"]),
                         "score": p["score"]} for p in frame["poses"]]
    return out


def blend_optional(previous, following, t: float):
    if previous is None:
        return following
    if following is None:
        return previous
    return previous + (following - previous) * t


def mix(previous: list[dict], following: list[dict], t: float, empty: list[dict]) -> list[dict]:
    """interpolateLandmarks and blendLandmarks are the same arithmetic: previous + (following - previous) * t."""

    count = min(len(previous), len(following))
    if not count:
        return clone_landmarks(empty)
    out = []
    for a, b in zip(previous[:count], following[:count]):
        point = {"x": a["x"] + (b["x"] - a["x"]) * t, "y": a["y"] + (b["y"] - a["y"]) * t, "z": a["z"] + (b["z"] - a["z"]) * t}
        for key in ("visibility", "presence"):
            value = blend_optional(a.get(key), b.get(key), t)
            if value is not None:
                point[key] = value
        out.append(point)
    return out


def fill_missing_frames(frames: list[dict], window: int) -> list[dict]:
    def neighbour(index: int, direction: int):
        for offset in range(1, window + 1):
            at = index + offset * direction
            if 0 <= at < len(frames) and frames[at]["landmarks"]:
                return frames[at]
        return None

    out = []
    for index, frame in enumerate(frames):
        if frame["landmarks"]:
            out.append(frame)
            continue
        previous, following = neighbour(index, -1), neighbour(index, 1)
        if not previous and not following:
            out.append(frame)
            continue
        if previous and following:
            t = clamp((frame["time"] - previous["time"]) / max(0.001, following["time"] - previous["time"]), 0, 1)
            landmarks = mix(previous["landmarks"], following["landmarks"], t, previous["landmarks"] or following["landmarks"])
            world = mix(previous["worldLandmarks"], following["worldLandmarks"], t, previous["worldLandmarks"] or following["worldLandmarks"])
            score = previous["score"] + (following["score"] - previous["score"]) * t
            filled = {"time": frame["time"], "landmarks": landmarks, "worldLandmarks": world, "score": score,
                      "poses": [{"id": 0, "landmarks": landmarks, "worldLandmarks": world, "score": score}]}
        else:
            filled = clone_frame(previous or following)
        out.append({**filled, "time": frame["time"], "source": "filled", "filled": True, "score": filled["score"] * 0.92})
    return out


def smooth_frames(frames: list[dict], smoothing: float) -> list[dict]:
    alpha = clamp(1 - smoothing, 0.08, 1)
    previous = previous_world = None
    out = []
    for frame in frames:
        if not frame["landmarks"]:
            out.append(frame)
            continue
        landmarks = mix(previous, frame["landmarks"], alpha, frame["landmarks"]) if previous else clone_landmarks(frame["landmarks"])
        world = (mix(previous_world, frame["worldLandmarks"], alpha, frame["worldLandmarks"])
                 if previous_world and frame["worldLandmarks"] else clone_landmarks(frame["worldLandmarks"]))
        previous, previous_world = landmarks, (world if world else previous_world)
        score = score_landmarks(landmarks) or frame["score"]
        out.append({**frame, "landmarks": landmarks, "worldLandmarks": world, "score": score, "smoothed": True,
                    "poses": [{"id": 0, "landmarks": landmarks, "worldLandmarks": world, "score": score}, *(frame.get("poses") or [])[1:]]})
    return out


def motion_energy(frames: list[dict]) -> float:
    previous, total, count = None, 0.0, 0
    for frame in frames:
        if not frame["landmarks"]:
            continue
        if previous:
            pairs = list(zip(previous, frame["landmarks"]))
            if pairs:
                total += sum(math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2) for a, b in pairs) / len(pairs)
            count += 1
        previous = frame["landmarks"]
    return total / count if count else 0.0


def summarize_pose(raw: list[dict], frames: list[dict], settings: dict, model: str, fallback: bool) -> dict:
    raw_detected = sum(1 for f in raw if f["landmarks"])
    detected = sum(1 for f in frames if f["landmarks"])
    filled = sum(1 for f in frames if f.get("filled"))
    average = sum(f["score"] for f in frames if f["landmarks"]) / detected if detected else 0.0
    diagnostics = []
    if fallback:
        diagnostics.append(f"{pose_model_label(settings['poseModel'])} was not available, so Lite was used for this run.")
    if not raw_detected:
        diagnostics.append("No body pose was detected. Use a clearer subject, lower confidence, or switch to Camera only mode.")
    elif raw_detected / max(len(raw), 1) < 0.45:
        diagnostics.append("Pose detection is intermittent. Lower detection confidence or trim to frames where the body is more visible.")
    if filled:
        diagnostics.append(f"Filled {filled} short pose gaps using a {settings['temporalWindow']}-frame temporal window.")
    if 0 < average < settings["detectionConfidence"]:
        diagnostics.append("Average landmark confidence is below the selected detection threshold.")
    if settings["maxPeople"] > 1 and not any(len(f.get("poses") or []) > 1 for f in raw):
        diagnostics.append("Multi-person tracking is enabled, but only one person was detected in this range.")
    if not diagnostics:
        diagnostics.append("Pose track is ready for export.")
    return {"totalFrames": len(frames), "detectedFrames": detected, "rawDetectedFrames": raw_detected, "filledFrames": filled,
            "missingFrames": len(frames) - detected,
            "lowConfidenceFrames": sum(1 for f in frames if f["landmarks"] and f["score"] < settings["detectionConfidence"]),
            "maxPeopleDetected": max([0, *(len(f.get("poses") or []) for f in raw)]), "averageScore": average,
            "motionEnergy": motion_energy(frames), "settings": settings, "runtimeModel": model, "runtimeDelegate": "CPU",
            "diagnostics": diagnostics}


def empty_pose(reference: dict, fps: float, settings: dict, why: str) -> dict:
    """pose_landmarks.json for a pack made without a pose track: the file other tools expect, with no frames."""

    return {"fps": fps, "duration": reference["duration"], "width": reference["width"], "height": reference["height"], "frames": [],
            "summary": {"totalFrames": 0, "detectedFrames": 0, "rawDetectedFrames": 0, "filledFrames": 0, "missingFrames": 0,
                        "lowConfidenceFrames": 0, "maxPeopleDetected": 0, "averageScore": 0, "motionEnergy": 0, "settings": settings,
                        "diagnostics": [why]}}


def analyze_pose(video: Path, reference: dict, fps: float, settings: dict, progress: Progress) -> dict:
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python as tasks
    from mediapipe.tasks.python import vision

    plan = [key for key in dict.fromkeys([settings["poseModel"], "lite"]) if (models_dir() / POSE_MODELS[key]).is_file()]
    if not plan:
        raise RuntimeError(f"no pose model in {models_dir()}")
    landmarker = None
    for model in plan:  # the model asked for, then Lite, on the CPU
        try:
            landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=tasks.BaseOptions(model_asset_path=str(models_dir() / POSE_MODELS[model]), delegate=tasks.BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO, num_poses=settings["maxPeople"],
                min_pose_detection_confidence=settings["detectionConfidence"], min_pose_presence_confidence=settings["detectionConfidence"],
                min_tracking_confidence=settings["trackingConfidence"]))
            break
        except Exception as error:  # noqa: BLE001 - try the next model
            last = error
    if landmarker is None:
        raise RuntimeError(f"could not load a MediaPipe pose model: {last}")
    duration = reference["duration"] or 1.0
    total = max(1, math.ceil(duration * fps))
    raw = []
    try:
        for index, frame in enumerate(read_frames(video, fps, reference["width"], reference["height"])):
            when = min(index / fps, max(duration - 0.001, 0))
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)),
                                                 int(round(index * 1000 / fps)))
            people, worlds = result.pose_landmarks or [], result.pose_world_landmarks or []
            poses = []
            for person, points in enumerate(people[:settings["maxPeople"]]):
                marks = [landmark(p) for p in points]
                poses.append({"id": person, "landmarks": marks,
                              "worldLandmarks": [landmark(p) for p in (worlds[person] if person < len(worlds) else [])],
                              "score": score_landmarks(marks)})
            first = poses[0] if poses else None
            raw.append({"time": when, "landmarks": first["landmarks"] if first else [], "worldLandmarks": first["worldLandmarks"] if first else [],
                        "score": first["score"] if first else 0, "poses": poses, "source": "detected" if first else "missing"})
            progress.step(index + 1, max(total, index + 1), f"Tracking pose {index + 1}/{max(total, index + 1)}")
    finally:
        landmarker.close()
    frames = [clone_frame(f) for f in raw]
    if settings["fillGaps"]:
        frames = fill_missing_frames(frames, settings["temporalWindow"])
    if settings["smoothing"] > 0:
        frames = smooth_frames(frames, settings["smoothing"])
    return {"fps": fps, "duration": duration, "width": reference["width"], "height": reference["height"], "frames": frames,
            "summary": summarize_pose(raw, frames, settings, model, model != settings["poseModel"])}


# ---------- OpenPose BODY_25 (openpose.ts) and the high-contrast skeleton (poseVideo.ts) ----------

BODY_25_FROM_MEDIAPIPE = [0, (11, 12), 12, 14, 16, 11, 13, 15, (23, 24), 24, 26, 28, 23, 25, 27, 5, 2, 8, 7, 31, 31, 29, 32, 32, 30]
BODY_25_PAIRS = [(1, 0), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13),
                 (13, 14), (1, 8), (0, 15), (0, 16), (15, 17), (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)]
BODY_25_COLORS = ["#ff0000", "#ff5500", "#ffaa00", "#ffff00", "#aaff00", "#55ff00", "#00ff00", "#00ff55", "#00ffaa", "#00ffff",
                  "#00aaff", "#0055ff", "#0000ff", "#5500ff", "#aa00ff", "#ff00ff", "#ff00aa", "#ff0055", "#ff0000", "#ff0033",
                  "#ff0066", "#ff0099", "#3300ff", "#6600ff", "#9900ff"]
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20),
                    (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                    (29, 31), (30, 32), (27, 31), (28, 32), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (3, 7), (6, 8), (9, 10)]


def bgr(colour: str, alpha: float = 1.0) -> tuple[int, int, int]:
    """A CSS hex colour as OpenCV's blue, green, red; `alpha` is the canvas globalAlpha over the black background."""

    value = colour.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return int(blue * alpha), int(green * alpha), int(red * alpha)


def confidence_of(point: dict) -> float:
    value = point.get("visibility", point.get("presence"))
    return 0.9 if value is None else clamp(value, 0, 1)


def body25(landmarks: list[dict], width: int, height: int) -> list[tuple[float, float, float]]:
    """One MediaPipe person as BODY_25 keypoints (x, y, confidence) in pixels; a joint with no source is (0, 0, 0)."""

    out = []
    for source in BODY_25_FROM_MEDIAPIPE:
        if isinstance(source, tuple):
            a, b = (landmarks[i] if i < len(landmarks) else None for i in source)
            out.append(((a["x"] + b["x"]) / 2 * width, (a["y"] + b["y"]) / 2 * height, min(confidence_of(a), confidence_of(b))) if a and b else (0.0, 0.0, 0.0))
        else:
            point = landmarks[source] if source < len(landmarks) else None
            out.append((point["x"] * width, point["y"] * height, confidence_of(point)) if point else (0.0, 0.0, 0.0))
    return out


def js_round(value: float, digits: int) -> float:
    scale = 10 ** digits
    return math.floor(value * scale + 0.5) / scale  # Math.round rounds halves up


def frame_people(frame: dict | None) -> list[dict]:
    if not frame:
        return []
    if frame.get("poses"):
        return frame["poses"]
    if frame.get("landmarks"):
        return [{"id": 0, "landmarks": frame["landmarks"], "worldLandmarks": frame["worldLandmarks"], "score": frame["score"]}]
    return []


def openpose_json(pose: dict, width: int, height: int) -> list[dict]:
    """One OpenPose object per pose frame, 75 numbers per person, in pixels of the reference clip."""

    out = []
    for frame in pose["frames"]:
        people = []
        for index, person in enumerate(frame_people(frame)):
            flat = []
            for x, y, c in body25(person["landmarks"], width, height):
                flat += [js_round(x, 2), js_round(y, 2), js_round(c, 4)]
            people.append({"person_id": [-1 if index == 0 else index], "pose_keypoints_2d": flat})
        out.append({"version": 1.3, "people": people})
    return out


def draw_openpose(frame: dict | None, width: int, height: int):
    import cv2
    import numpy as np

    canvas = np.zeros((height, width, 3), np.uint8)
    limb = max(3, round(min(width, height) * 0.012))
    joint = max(3, round(min(width, height) * 0.008))
    for person in frame_people(frame):
        points = body25(person["landmarks"], width, height)
        for a, b in BODY_25_PAIRS:
            (ax, ay, ac), (bx, by, bc) = points[a], points[b]
            if ac <= 0.05 or bc <= 0.05:
                continue
            cv2.line(canvas, (round(ax), round(ay)), (round(bx), round(by)), bgr(BODY_25_COLORS[a], min(1, min(ac, bc) + 0.25)), limb, cv2.LINE_AA)
        for index, (x, y, c) in enumerate(points):
            if c > 0.05:
                cv2.circle(canvas, (round(x), round(y)), joint, bgr(BODY_25_COLORS[index], min(1, c + 0.3)), -1, cv2.LINE_AA)
    return canvas


def connection_colour(a: int, b: int) -> str:
    if a <= 10 or b <= 10:
        return "#e646d8"
    for group, colour in (({11, 13, 15, 17, 19, 21}, "#46f083"), ({12, 14, 16, 18, 20, 22}, "#56e7ff"),
                          ({23, 25, 27, 29, 31}, "#f4e855"), ({24, 26, 28, 30, 32}, "#ff7a2f")):
        if a in group or b in group:
            return colour
    return "#f5d76e"


def draw_pose(frame: dict | None, width: int, height: int):
    """The high-contrast MediaPipe skeleton: 33 landmarks on a dark grid, limbs coloured by side."""

    import cv2
    import numpy as np

    canvas = np.full((height, width, 3), bgr("#030405"), np.uint8)
    for x in range(0, width, 80):
        cv2.line(canvas, (x, 0), (x, height), bgr("#172226"), 1)
    for y in range(0, height, 80):
        cv2.line(canvas, (0, y), (width, y), bgr("#172226"), 1)
    people = frame_people(frame)
    if not people:
        size = cv2.getTextSize("No pose detected", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(canvas, "No pose detected", ((width - size[0]) // 2, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr("#667176"), 2, cv2.LINE_AA)
        return canvas
    for index, person in enumerate(people):
        points, fade = person["landmarks"], (1.0 if index == 0 else 0.58)
        for a, b in POSE_CONNECTIONS:
            if a >= len(points) or b >= len(points):
                continue
            sure = min(points[a].get("visibility", 1), points[b].get("visibility", 1)) > 0.5
            cv2.line(canvas, (round(points[a]["x"] * width), round(points[a]["y"] * height)),
                     (round(points[b]["x"] * width), round(points[b]["y"] * height)),
                     bgr(connection_colour(a, b), fade if sure else fade * 0.38), 7 if sure else 3, cv2.LINE_AA)
        for at, point in enumerate(points):
            sure = point.get("visibility", 1) > 0.5
            centre, radius = (round(point["x"] * width), round(point["y"] * height)), (5 if at <= 10 else 7)
            cv2.circle(canvas, centre, radius, bgr("#e646d8" if at == 0 else "#8ff5ff", fade if sure else fade * 0.45), -1, cv2.LINE_AA)
            cv2.circle(canvas, centre, radius, bgr("#050708"), 2, cv2.LINE_AA)
    label = f"{(frame or {}).get('time', 0):.2f}s  {'filled gap' if (frame or {}).get('source') == 'filled' else 'tracked'}"
    cv2.putText(canvas, label, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr("#c8d2d2"), 1, cv2.LINE_AA)
    return canvas
