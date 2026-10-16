"""Camera solve and the camera move in words: port of Motion Previs Studio's cameraMotion.ts, on OpenCV."""
# Copyright 2026 Sam Wasserman, Apache-2.0, modified: https://github.com/wassermanproductions/motion-previs-studio

from __future__ import annotations

import math
from pathlib import Path

from motion_media import Progress, clamp, read_frames
from motion_pose import frame_people


# ---------- camera solve (cameraMotion.ts, on OpenCV) ----------

CORNER_TARGET = 150
LK_WINDOW = 21
PYRAMID_LEVELS = 3
ANALYSIS_EDGE = 384        # the original tracks on a 192 pixel canvas, a size chosen for JavaScript
RANSAC_INLIER_PX = 2.0     # the original's 1.6 at 192 pixels
MASK_SEGMENTS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


def analysis_size(width: int, height: int) -> tuple[int, int]:
    if not width or not height:
        return ANALYSIS_EDGE, round(ANALYSIS_EDGE * 0.5625)
    if width >= height:
        return ANALYSIS_EDGE, max(48, round(height / width * ANALYSIS_EDGE))
    return max(48, round(width / height * ANALYSIS_EDGE)), ANALYSIS_EDGE


def visible(point: dict) -> bool:
    value = point.get("visibility", point.get("presence"))
    return True if value is None else value > 0.3


def subject_mask(width: int, height: int, frame: dict | None):
    """255 where a corner may be seeded: everything but capsules round the limbs and a circle round the head."""

    import cv2
    import numpy as np

    people = [p["landmarks"] for p in frame_people(frame)]
    if not people:
        return None
    mask = np.full((height, width), 255, np.uint8)
    radius = max(4, round(min(width, height) * 0.06))
    drawn = False
    for marks in people:
        if marks and visible(marks[0]):
            cv2.circle(mask, (round(marks[0]["x"] * width), round(marks[0]["y"] * height)), round(radius * 1.3), 0, -1)
            drawn = True
        for a, b in MASK_SEGMENTS:
            if a < len(marks) and b < len(marks) and visible(marks[a]) and visible(marks[b]):
                cv2.line(mask, (round(marks[a]["x"] * width), round(marks[a]["y"] * height)),
                         (round(marks[b]["x"] * width), round(marks[b]["y"] * height)), 0, radius * 2)
                drawn = True
    return mask if drawn else None


def pose_frame_at(pose: dict | None, when: float) -> dict | None:
    frames = (pose or {}).get("frames") or []
    return min(frames, key=lambda f: abs(f["time"] - when)) if frames else None


def seed_corners(gray, mask):
    """Shi-Tomasi corners on the background; with no pose track the middle of the frame is the last place taken from."""

    import cv2
    import numpy as np

    height, width = gray.shape
    spacing = max(3, round(min(width, height) / 18))
    find = lambda limit, where: cv2.goodFeaturesToTrack(gray, maxCorners=limit, qualityLevel=0.01, minDistance=spacing, mask=where, blockSize=3)
    if mask is not None:
        found = find(CORNER_TARGET, mask)
        return found if found is not None else np.empty((0, 1, 2), np.float32)
    edge = np.full((height, width), 255, np.uint8)
    edge[round(height * 0.275):round(height * 0.725), round(width * 0.275):round(width * 0.725)] = 0  # edgeBias <= 0.45
    found = find(CORNER_TARGET, edge)
    found = found if found is not None else np.empty((0, 1, 2), np.float32)
    if len(found) < CORNER_TARGET * 0.6:
        more = find(CORNER_TARGET - len(found), 255 - edge)
        if more is not None:
            found = np.concatenate([found, more])
    return found


def solve_step(previous, current, mask) -> dict:
    """The similarity (shift, zoom, roll) that carries the background of `previous` onto `current`."""

    import cv2
    import numpy as np

    height, width = previous.shape
    corners = seed_corners(previous, mask)
    identity = {"tx": 0.0, "ty": 0.0, "scale": 1.0, "rotation": 0.0, "inlierRatio": 0.0}
    if len(corners) == 0:
        return identity
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
    window = dict(winSize=(LK_WINDOW, LK_WINDOW), maxLevel=PYRAMID_LEVELS - 1, criteria=criteria)
    moved, found, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None, **window)
    back, found_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None, **window)
    start, end = corners.reshape(-1, 2), moved.reshape(-1, 2)
    flow = end - start
    keep = ((found.ravel() == 1) & (found_back.ravel() == 1) & (np.abs(back.reshape(-1, 2) - start).max(axis=1) < 1.0)
            & (end[:, 0] >= 0) & (end[:, 1] >= 0) & (end[:, 0] < width) & (end[:, 1] < height)
            & (np.abs(flow).max(axis=1) <= min(width, height) * 0.5))
    start, end = start[keep], end[keep]
    if len(start) < 3:
        return {**identity, "inlierRatio": 0.15 if len(start) else 0.0}
    cv2.setRNGSeed(0x1e3779b9)  # the same clip gives the same solve every run
    matrix, inliers = cv2.estimateAffinePartial2D(start, end, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_INLIER_PX,
                                                  maxIters=2000, confidence=0.995, refineIters=10)
    count = int(inliers.sum()) if inliers is not None else 0
    if matrix is None or count < 3:
        shift = np.median(end - start, axis=0)  # a slow shot that starves RANSAC still gets its translation
        return {"tx": float(shift[0]), "ty": float(shift[1]), "scale": 1.0, "rotation": 0.0, "inlierRatio": 0.2}
    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    cx, cy = (width - 1) / 2, (height - 1) / 2
    # shift taken at the frame centre, not the top-left corner as in the original: a centred push-in is then no pan
    tx = a * cx - b * cy + float(matrix[0, 2]) - cx
    ty = b * cx + a * cy + float(matrix[1, 2]) - cy
    limit = max(width, height) * 0.6
    return {"tx": clamp(tx, -limit, limit), "ty": clamp(ty, -limit, limit), "scale": clamp(math.hypot(a, b), 0.8, 1.25),
            "rotation": clamp(true_roll(start[inliers.ravel() == 1], end[inliers.ravel() == 1], math.atan2(b, a)), -0.2, 0.2),
            "inlierRatio": count / len(start)}


def true_roll(start, end, fitted: float) -> float:
    """The roll in a step, only where both axes of a free 2x2 fit turn the same way (parallax shear is not a roll)."""

    import numpy as np

    if len(start) < 6:
        return fitted
    design = np.hstack([start, np.ones((len(start), 1), start.dtype)]).astype(np.float64)
    solved, *_ = np.linalg.lstsq(design, end.astype(np.float64), rcond=None)
    across, down = math.atan2(solved[0, 1], solved[0, 0]), math.atan2(-solved[1, 0], solved[1, 1])  # the x axis, the y axis
    if across * down <= 0:
        return 0.0
    return math.copysign(min(abs(across), abs(down), abs(fitted)), across)


def analyze_camera(video: Path, reference: dict, fps: float, pose: dict | None, progress: Progress) -> dict:
    sample = min(max(fps, 2), 12)
    duration = reference["duration"] or 1.0
    total = max(1, math.ceil(duration * sample))
    width, height = analysis_size(reference["width"], reference["height"])
    sx, sy = reference["width"] / width, reference["height"] / height
    previous, frames = None, []
    x = y = roll = 0.0
    zoom = 1.0
    for index, frame in enumerate(read_frames(video, sample, width, height, "gray")):
        gray = frame[:, :, 0]
        motion, confidence = {"xPixels": 0, "yPixels": 0, "scale": 1, "rollRadians": 0}, 1
        if previous is not None:
            when = min(index / sample, max(duration - 0.001, 0))
            step = solve_step(previous, gray, subject_mask(width, height, pose_frame_at(pose, when - 1 / sample)))
            motion = {"xPixels": step["tx"] * sx, "yPixels": step["ty"] * sy, "scale": step["scale"], "rollRadians": step["rotation"]}
            confidence = clamp(step["inlierRatio"], 0, 1)
            x, y, zoom, roll = x + motion["xPixels"], y + motion["yPixels"], zoom * step["scale"], roll + step["rotation"]
        frames.append({"time": min(index / sample, max(duration - 0.001, 0)), "frameIndex": index, "imageMotion": motion,
                       "cameraMove": {"pan": -x / max(reference["width"], 1), "tilt": -y / max(reference["height"], 1), "dollyZoom": zoom, "roll": -roll},
                       "confidence": confidence})
        previous = gray
        progress.step(index + 1, max(total, index + 1), f"Solving camera move {index + 1}/{max(total, index + 1)}")
    last = frames[-1]["cameraMove"] if frames else None
    return {"fps": sample, "duration": duration, "width": reference["width"], "height": reference["height"], "frames": frames,
            "summary": {"panPixels": -last["pan"] * reference["width"] if last else 0, "tiltPixels": -last["tilt"] * reference["height"] if last else 0,
                        "zoomRatio": (last["dollyZoom"] if last else 1) or 1, "rollDegrees": (last["roll"] if last else 0) * 180 / math.pi,
                        "averageConfidence": sum(f["confidence"] for f in frames) / max(len(frames), 1)}}

# ---------- the camera move in words ----------

SHAKE = 0.004  # of the frame per sample: above this the shot reads as handheld

def describe_camera(camera: dict | None, seconds: float | None = None) -> dict:
    """camera_motion.json in a cinematographer's words: {sentence, moves, confidence, reliable}, as seen on screen."""

    frames = (camera or {}).get("frames") or []
    if len(frames) < 2:
        return {"sentence": "", "moves": [], "confidence": 0.0, "reliable": False}
    length = float(seconds or camera.get("duration") or frames[-1]["time"] or 1.0)
    pan = [f["cameraMove"]["pan"] for f in frames]                       # of the frame width, + is right
    tilt = [-f["cameraMove"]["tilt"] for f in frames]                    # of the frame height, + is up
    zoom = [math.log(max(f["cameraMove"]["dollyZoom"], 1e-3)) for f in frames]
    roll = [math.degrees(f["cameraMove"]["roll"]) for f in frames]       # + is clockwise
    confidence = float(camera.get("summary", {}).get("averageConfidence", 0))

    def amount(track, floor):
        """(final value, whether it went out and came back, where in the shot most of it happens)."""
        end, peak = track[-1], max(track, key=abs)
        if abs(peak) >= floor and abs(end) < 0.4 * abs(peak):
            return peak, True, None
        if abs(end) < floor:
            return 0.0, False, None
        last = max(frames[-1]["time"], 1e-6)
        passed = lambda part: next((f["time"] for f, v in zip(frames, track) if abs(v) >= part * abs(end) and v * end > 0), 0.0)
        return end, False, (passed(0.2) / last, passed(0.8) / last)

    def pace(per_second, slow, fast):
        return "slowly " if per_second < slow else "quickly " if per_second > fast else ""

    def span(value):
        value = abs(value)
        return ("a little" if value < 0.08 else "across about a tenth of the frame" if value < 0.15
                else "across about a quarter of the frame" if value < 0.3 else "across about a third of the frame" if value < 0.42
                else "across about half the frame" if value < 0.65 else "across most of the frame" if value < 0.9
                else "across more than a full frame")

    moves, parts, timing = [], [], []
    for name, track, words in (("pan", pan, ("pans left", "pans right")), ("tilt", tilt, ("tilts down", "tilts up"))):
        value, returns, when = amount(track, 0.03)
        if not value:
            continue
        verb = words[value > 0]
        if returns:
            text = f"{verb} {span(value)} and comes back"
        elif abs(value) / length > 0.7:
            text = f"whip-{verb} {span(value)}"
        else:
            text = f"{pace(abs(value) / length, 0.05, 0.22)}{verb} {span(value)}"
        moves.append({"move": name, "direction": verb.split()[1], "amount": round(value, 4), "returns": returns})
        parts.append((abs(value), text))
        timing.append(when)
    value, returns, when = amount(zoom, 0.04)
    if value:
        ratio = math.exp(abs(value))
        verb = "pushes in" if value > 0 else "pulls back"
        size = " slightly" if ratio < 1.12 else "" if ratio < 1.4 else " a long way"
        text = f"{verb} and returns" if returns else f"{pace(abs(value) / length, 0.03, 0.15)}{verb}{size}"
        moves.append({"move": "zoom", "direction": "in" if value > 0 else "out", "amount": round(math.exp(value), 4), "returns": returns})
        parts.append((abs(value) * 1.5, text))
        timing.append(when)
    # under 4 degrees over a shot is not called a roll: what is left of parallax after true_roll is of that size
    value, returns, when = amount(roll, 4.0)
    if value:
        text = f"rolls {'clockwise' if value > 0 else 'counter-clockwise'} about {round(abs(value))} degrees" + (" and back" if returns else "")
        moves.append({"move": "roll", "direction": "clockwise" if value > 0 else "counter-clockwise", "amount": round(value, 2), "returns": returns})
        parts.append((abs(value) / 60, text))
        timing.append(when)

    # unsteadiness: what is left of the frame-to-frame shift once the move itself is taken out
    width, height = max(camera.get("width", 1), 1), max(camera.get("height", 1), 1)
    dx = [f["imageMotion"]["xPixels"] / width for f in frames[1:]]
    dy = [f["imageMotion"]["yPixels"] / height for f in frames[1:]]
    near = max(1, round(camera.get("fps", 12) * 0.25))

    def rough(track):
        smooth = [sum(track[max(0, i - near):i + near + 1]) / len(track[max(0, i - near):i + near + 1]) for i in range(len(track))]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(track, smooth)) / len(track))

    shake = math.hypot(rough(dx), rough(dy))
    parts.sort(key=lambda item: -item[0])
    if not parts:
        sentence = ("Handheld shot that holds its framing: the camera stays on one spot with a slight natural shake." if shake > SHAKE
                    else "Locked-off static shot on a still camera.")
    else:
        words = [text for _, text in parts]
        sentence = "The camera " + (words[0] if len(words) == 1 else ", ".join(words[:-1]) + " while it " + words[-1]) + "."
        when = next((w for w in timing if w), None)
        if shake > SHAKE:
            sentence += " The move is handheld, with a slight natural shake."
        elif when and when[0] > 0.4:
            sentence += " The move starts gently and gathers pace toward the end of the shot."
        elif when and when[1] < 0.6:
            sentence += " The move is quickest at the start and then eases to a stop."
        else:
            sentence += " The move is smooth and continuous through the whole shot."
    return {"sentence": sentence, "moves": moves, "confidence": round(confidence, 4), "reliable": confidence >= 0.3,
            "shake": round(shake, 5)}
