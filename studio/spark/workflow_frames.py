"""A clip's frames for the person remover (spark_workflows.py): prepare, read, write, cut, and mend left-over green."""
import math
import subprocess
import sys
import uuid
from pathlib import Path

import spark_generate as sg
from workflow_comfy import execute, remover_nodes, take

# frames played backwards lengthen the clip so the last window has frames after it (a frozen tail failed)
TAIL = 8


def prepared_source(path: str, folder: Path) -> tuple[Path, dict]:
    """The clip as the remover takes it: 24 frames a second, sides that are multiples of 32, at most 15 seconds."""

    probe = sg.probe_media(path)
    if probe["video"] is None:
        sys.exit(f"{path} has no picture")
    width, height = probe["video"]
    w32, h32 = max(32, width // 32 * 32), max(32, height // 32 * 32)
    if w32 * h32 > sg.MODEL_MAX_PIXELS:  # the model's largest picture: bring it down, sides kept in proportion
        scale = math.sqrt(sg.MODEL_MAX_PIXELS / (w32 * h32))
        w32, h32 = max(32, int(width * scale) // 32 * 32), max(32, int(height * scale) // 32 * 32)
    target = folder / "source.mp4"
    # scaled to cover, then cropped about the centre: nothing is stretched
    chain = (f"fps={sg.FPS},scale={w32}:{h32}:force_original_aspect_ratio=increase:flags=lanczos,crop={w32}:{h32},setsar=1,"
             f"trim=end_frame={15 * sg.FPS}")
    plain = folder / "plain.mp4"
    encode = ["-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"]
    done = subprocess.run([sg._tool("ffmpeg"), "-v", "error", "-y", "-i", path, "-vf", chain, *encode, str(plain)], capture_output=True, text=True)
    if done.returncode:
        sys.exit(f"the clip could not be read: {done.stderr.strip()[-300:]}")
    frames = int(subprocess.run([sg._tool("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                 "stream=nb_read_frames", "-of", "csv=p=0", str(plain)], capture_output=True, text=True).stdout.strip() or 0)
    if frames < 22:
        sys.exit(f"the clip is {frames} frames at 24 a second: the remover works in windows of 22, so it needs at least 22")
    # the clip, then its frames n-2 down to n-1-TAIL: the motion runs back the way it came
    back = (f"[0:v]split[a][b];[b]trim=start_frame={frames - 1 - TAIL}:end_frame={frames - 1},setpts=PTS-STARTPTS,reverse[r];"
            f"[a][r]concat=n=2:v=1[v]")
    done = subprocess.run([sg._tool("ffmpeg"), "-v", "error", "-y", "-i", str(plain), "-filter_complex", back, "-map", "[v]",
                           *encode, str(target)], capture_output=True, text=True)
    if done.returncode:
        sys.exit(f"the clip could not be prepared: {done.stderr.strip()[-300:]}")
    plain.unlink()
    return target, {"width": w32, "height": h32, "frames": frames, "sound": bool(probe["audio"])}


def green_of(frames):
    """Where each frame still shows the mask's green, and which frames do (more than one point in 500)."""

    import numpy as np

    wide = frames.astype(np.int16)
    green = (wide[..., 1] > 170) & (wide[..., 1] - wide[..., 0] > 70) & (wide[..., 1] - wide[..., 2] > 70)
    return green, [int(i) for i in np.nonzero(green.reshape(len(green), -1).mean(1) > 0.002)[0]]


def green_left(clip: Path, size: dict) -> list[int]:
    """The frames of a result that still show the mask's green: a window that did not take the person out."""

    return green_of(read_frames(clip, size["width"], size["height"]))[1]


def mend(clip: Path, size: dict, reach: int = 4) -> tuple[int, int]:
    """Paint green that flashes back for a few frames over from clean frames within `reach`; (mended, still green)."""

    import numpy as np

    frames = read_frames(clip, size["width"], size["height"])
    green, flagged = green_of(frames)
    # a frame from another scene differs from both its neighbours far more than one shot's frames: replace it whole
    small = frames[:, ::8, ::8].astype(np.int16)
    step = np.abs(small[1:] - small[:-1]).reshape(len(frames) - 1, -1).mean(1)
    jump = max(4 * float(np.median(step)), 25.0)          # measured: 60 against 6 to 8 between frames of the shot
    alien = [i for i in range(len(frames)) if (i == 0 or step[i - 1] > jump) and (i == len(frames) - 1 or step[i] > jump)
             and len(frames) > 2]
    for i in alien:
        green[i] = True
    # a frame whose greenish share jumps above the frames before it is mended where it is greenish and they are not
    wide = frames.astype(np.int16)
    tint = (wide[..., 1] - wide[..., 0] > 14) & (wide[..., 1] - wide[..., 2] > 14)
    level = tint.reshape(len(frames), -1).mean(1)
    for i in range(1, len(frames)):
        calm = [j for j in range(max(0, i - 8), i) if j not in flagged and not green[j].any()]
        if i in alien or not calm or level[i] - float(np.median(level[calm])) < 0.015:
            continue
        stain = tint[i] & ~grown(tint[calm[-1]], 6)
        if stain.mean() > 0.004:
            green[i] |= stain
    flagged = sorted(set(flagged) | set(alien) | {i for i in range(len(frames)) if green[i].mean() > 0.002})
    if not flagged:
        return 0, 0
    out, bad, mended = frames.copy(), set(flagged), 0
    for i in flagged:
        def wider(mask, by):
            for axis in (0, 1):
                wide = mask.copy()
                for step in range(1, by + 1):
                    wide |= np.roll(mask, step, axis) | np.roll(mask, -step, axis)
                mask = wide
            return mask

        # the green and its soft, half-clear fringe: every greenish point within 16 points of real green
        one = frames[i].astype(np.int16)
        tint = (one[..., 1] - one[..., 0] > 14) & (one[..., 1] - one[..., 2] > 14)
        hole = wider(wider(green[i], 16) & (tint | wider(green[i], 5)), 3)
        before = next((j for j in range(i - 1, max(-1, i - 1 - reach), -1) if j not in bad), None)
        after = next((j for j in range(i + 1, min(len(frames), i + 1 + reach)) if j not in bad), None)
        if before is None and after is None:
            continue
        if before is None or after is None:
            fill = frames[before if after is None else after].astype(np.float32)
        else:
            share = (i - before) / (after - before)
            fill = frames[before].astype(np.float32) * (1 - share) + frames[after].astype(np.float32) * share
        soft = hole.astype(np.float32)
        for _ in range(4):                                    # a feathered edge, so the patch has no outline
            soft = (soft + np.roll(soft, 1, 0) + np.roll(soft, -1, 0) + np.roll(soft, 1, 1) + np.roll(soft, -1, 1)) / 5
        soft = np.maximum(soft, hole * 1.0)[..., None]
        out[i] = (frames[i] * (1 - soft) + fill * soft).round().astype(np.uint8)
        mended += 1
    if mended:
        fresh = clip.with_name(clip.stem + ".mended.mp4")
        done = subprocess.run([sg._tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                               f"{size['width']}x{size['height']}", "-r", str(sg.FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
                               "-crf", "14", "-pix_fmt", "yuv420p", str(fresh)], input=out.tobytes(), capture_output=True)
        if done.returncode or not fresh.is_file():
            return 0, len(flagged)
        fresh.replace(clip)
    return mended, len(set(green_of(out)[1]) | {i for i in alien if (out[i] == frames[i]).all()})


def grown(mask, by: int):
    """`mask` made wider by `by` points each way."""

    import numpy as np

    for axis in (0, 1):
        wide = mask.copy()
        for step in range(1, by + 1):
            wide |= np.roll(mask, step, axis) | np.roll(mask, -step, axis)
        mask = wide
    return mask


def write_frames(frames, target: Path, size: dict, crf: int = 14) -> bool:
    done = subprocess.run([sg._tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{size['width']}x{size['height']}", "-r", str(sg.FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
                           "-crf", str(crf), "-pix_fmt", "yuv420p", str(target)], input=frames.tobytes(), capture_output=True)
    return done.returncode == 0 and target.is_file() and target.stat().st_size > 0


def again(clip: Path, masked: Path, size: dict, lora: tuple[str, float], args, scratch: Path, limit: int) -> tuple[int, int, float]:
    """Run the worst green-left stretches of 22 frames through the remover again; (stretches, frames fixed, seconds)."""

    import numpy as np

    frames = read_frames(clip, size["width"], size["height"]).copy()
    green, flagged = green_of(frames)
    count = len(frames)
    if not flagged or limit < 1 or count < 22:
        return 0, 0, 0.0
    share, bad = green.reshape(count, -1).mean(1), set(flagged)
    todo, starts = set(flagged), []
    while todo and len(starts) < limit:
        worst = max(todo, key=lambda i: share[i])
        first = [s for s in range(max(0, worst - 21), min(worst - 1, count - 22) + 1) if s not in bad]
        if not first:
            todo.discard(worst)
            continue
        start = max(first, key=lambda s: (len(todo & set(range(s, s + 22))), -abs(s + 11 - worst)))
        starts.append(start)
        todo -= set(range(start, start + 22))
    if not starts:
        return 0, 0, 0.0
    person = green_of(read_frames(masked, size["width"], size["height"]))[0]      # where the person is, frame by frame
    source = read_frames(masked, size["width"], size["height"])
    repaired, total = 0, 0.0
    for number, start in enumerate(starts):
        span = frames[start:start + 22].copy()
        for k in range(22):
            if start + k in bad:
                span[k][person[start + k]] = source[start + k][person[start + k]]
        stretch, anchor = scratch / f"again{number}.mp4", scratch / f"again{number}.png"
        if not write_frames(span, stretch, size, 10):
            continue
        from PIL import Image

        Image.fromarray(frames[start]).save(anchor)
        w = {"20": {"class_type": "LoadVideo", "inputs": {"file": sg.upload(str(stretch))}},
             "21": {"class_type": "GetVideoComponents", "inputs": {"video": ["20", 0]}}}
        remover_nodes(w, lora, sg.upload(str(anchor)), ["21", 0], args, 22, args.seed + 101 + number)
        w["37"] = {"class_type": "CreateVideo", "inputs": {"images": ["31", 0], "fps": ["21", 2]}}
        w["38"] = {"class_type": "SaveVideo", "inputs": {"video": ["37", 0], "filename_prefix": f"studio/again_{uuid.uuid4().hex[:8]}",
                                                         "format": "auto", "codec": "auto"}}
        saved, seconds = execute(w, "31")
        total += seconds
        if not saved.get("38"):
            continue
        take(saved["38"][0], scratch / "again.mp4")
        second = read_frames(scratch / "again.mp4", size["width"], size["height"])
        if len(second) < 22:
            continue
        left = green_of(second[:22])[0].reshape(22, -1).mean(1)
        for k in range(1, 22):
            i = start + k
            if i not in bad or left[k] > 0.002 or left[k] >= share[i]:
                continue                                      # the second look kept green there too
            hole = grown(person[i] | green[i], 8).astype(np.float32)
            for _ in range(6):                                # a feathered edge: the two passes' pictures meet softly
                hole = (hole + np.roll(hole, 1, 0) + np.roll(hole, -1, 0) + np.roll(hole, 1, 1) + np.roll(hole, -1, 1)) / 5
            soft = np.maximum(hole, (person[i] | green[i]) * 1.0)[..., None]
            frames[i] = (frames[i] * (1 - soft) + second[k].astype(np.float32) * soft).round().astype(np.uint8)
            bad.discard(i)
            repaired += 1
    if repaired:
        fresh = clip.with_name(clip.stem + ".again.mp4")
        if write_frames(frames, fresh, size):
            fresh.replace(clip)
        else:
            repaired = 0
    return len(starts), repaired, total


def cut(source: Path, target: Path, frames: int, sound: str | None = None) -> bool:
    """`source` cut to `frames` frames, re-encoded (a stream copy cut misorders frames), with `sound`'s audio."""

    command = [sg._tool("ffmpeg"), "-v", "error", "-y", "-i", str(source)] + (["-i", sound] if sound else [])
    command += ["-filter_complex", f"[0:v]trim=end_frame={frames},setpts=PTS-STARTPTS[v]", "-map", "[v]"]
    command += (["-map", "1:a:0", "-c:a", "aac", "-b:a", sg.SOUND_BITRATE] if sound else ["-an"])
    command += ["-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-t", f"{frames / sg.FPS:.6f}", "-movflags", "+faststart", str(target)]
    return subprocess.run(command, capture_output=True, text=True).returncode == 0 and target.is_file() and target.stat().st_size > 0


def read_frames(path: Path, width: int, height: int):
    import numpy as np

    raw = subprocess.run([sg._tool("ffmpeg"), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
