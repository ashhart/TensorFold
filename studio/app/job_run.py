"""The render queue's worker: runs each job's script, follows its progress, and checks, joins and exports clips."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import qa
import shot
import sparks
from engines import environment, plan
from job_build import build
from job_state import lock, running, wake
from settings import FPS, HOME, LEAD, LONGEST, PACK, PRESETS, QA_TAKES, SPARK, inside, relative

hub = None  # set by server.py: the multi-Spark hub (sparks.py)


def last_frame(video: Path, image: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.2", "-i", str(video), "-vsync", "0", "-update", "1",
                    "-q:v", "1", str(image)], check=True, env=environment({}), timeout=120)


def frame_at(video: Path, image: Path, at: float) -> None:
    """One frame of a clip at its own resolution: the frame shown ``at`` seconds in."""

    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(at, 0):.3f}", "-i", str(video), "-frames:v", "1",
                    "-q:v", "1", str(image)], check=True, env=environment({}), timeout=120)


def export(job: dict) -> None:
    """Cut each timeline clip to its in and out points and join them, with a short audio fade at each join."""

    params = job["params"]
    clips = params["clips"]
    if not clips:
        raise RuntimeError("the timeline is empty")
    width, height = int(params.get("width", 1920)), int(params.get("height", 1080))
    target = HOME / "exports" / f"{job['id']}.mp4"
    inputs, filters, labels = [], [], []
    for index, clip in enumerate(clips):
        source = inside(clip["path"])
        start = max(0.0, float(clip.get("in", 0.0)))
        end = float(clip.get("out", 0.0))
        inputs += ["-i", str(source)]
        span = f"trim=start={start}:end={end},setpts=PTS-STARTPTS," if end > start else ""
        aspan = f"atrim=start={start}:end={end},asetpts=PTS-STARTPTS," if end > start else ""
        length = (end - start) if end > start else float(clip.get("duration", 5.0))
        fade = min(0.08, length / 4)
        filters.append(f"[{index}:v]{span}scale={width}:{height}:force_original_aspect_ratio=decrease,"
                       f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}[v{index}]")
        filters.append(f"[{index}:a]{aspan}afade=t=in:d={fade},afade=t=out:st={max(length - fade, 0)}:d={fade},"
                       f"aresample=48000[a{index}]")
        labels.append(f"[v{index}][a{index}]")
    filters.append(f"{''.join(labels)}concat=n={len(clips)}:v=1:a=1[v][a]")
    command = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[v]", "-map",
               "[a]", "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac",
               "-b:a", "256k", str(target)]
    done = subprocess.run(command, capture_output=True, text=True, env=environment({}))
    job["log"] += done.stderr
    if done.returncode:
        raise RuntimeError("ffmpeg failed")
    job["outputs"] = [relative(target)]


STEP = re.compile(r"\[tensorfold\] (step|voice) (\d+)/(\d+)")
RAN_ON = re.compile(r"\[tensorfold\] engine: (\w+)")


def run(job: dict, command: list[str], env: dict, label: str = "") -> bool:
    """Run one script for ``job``, following its progress; False when the job was cancelled meanwhile."""

    process = subprocess.Popen(command, cwd=PACK, env=environment(env), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    running.update(process=process, id=job["id"])
    for line in process.stdout:
        job["log"] = (job["log"] + line)[-20000:]
        engine = RAN_ON.search(line)
        if engine:
            job["engine"] = engine.group(1)
        found = STEP.search(line)
        if found:
            kind, at, total = found.group(1), int(found.group(2)), int(found.group(3))
            job["progress"] = {"stage": label + ("picture" if kind == "step" else "sound"), "at": at, "of": total}
    code = process.wait()
    running.update(process=None, id=None)
    if job["status"] == "cancelled":
        return False
    if code:
        # the script's own last words say why: a file out of range, a model not installed, memory
        said = [line.strip() for line in job["log"].splitlines() if line.strip() and not STEP.search(line)]
        raise RuntimeError((said[-1][-400:] if said else "the script stopped") + f" (status {code})")
    return True


def take(job: dict, params: dict, label: str = "") -> tuple[bool, dict, list[str]]:
    """One render of a job's clip, here or on the Spark it was handed to: (not cancelled, environment, output paths)."""

    worker = job.get("remote") and next((one for one in hub.workers() if one["address"] == job["remote"]), {"address": job["remote"]})
    if worker:
        _, env, outputs = build(job["kind"], params, job["id"], room=False)
        return hub.render(job, worker, params, Path(outputs[0]), label), env, outputs
    command, env, outputs = build(job["kind"], params, job["id"])
    return run(job, command, env, label), env, outputs


def hush(clip: Path) -> None:
    """The clip with its sound track silenced (``silent``): asking the model for near silence is not enough."""

    quiet = clip.with_name(clip.stem + ".quiet" + clip.suffix)
    done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-map", "0:v:0", "-map", "0:a?", "-c:v", "copy", "-af", "volume=0",
                           "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(quiet)], capture_output=True, text=True, env=environment({}))
    if done.returncode or not quiet.is_file():
        quiet.unlink(missing_ok=True)
        raise RuntimeError("the clip's sound could not be silenced: " + (done.stderr or "").strip()[-200:])
    quiet.replace(clip)


def gated(job: dict, clip: Path) -> bool:
    """Sam Wasserman's QA gate on a clip made to his Sharp Spec; False when the job was cancelled meanwhile."""

    params, takes = job["params"], []
    can = qa.ready()
    if not can["runs"]:
        # said in the job's result: a machine that has the tools can run the gate on this clip (POST /api/qa)
        job["qa"] = qa.not_run("The QA gate did not run: this machine has no " + " and no ".join(can["missing"]) + ". The clip was made and is not "
                               "checked. A machine that has them can run the gate on it: POST /api/qa {\"from\": this app's address, "
                               f"\"path\": \"{relative(clip)}\", \"prompt\": the prompt}}, or import the take into its Slate shot.")
        return True
    for number in range(1, QA_TAKES + 1):
        if number > 1:
            shutil.copy(clip, HOME / "qa" / f"{job['id']}-take{number - 1}.mp4")  # the failed take is kept
            params["seed"] = int(params.get("seed", 0)) + 1
            if not take(job, params, f"take {number} · ")[0]:
                return False
        job["progress"] = {"stage": "QA gate, take", "at": number, "of": QA_TAKES}
        picture = HOME / "qa" / f"{job['id']}-take{number}.jpg"
        try:
            report = qa.check(clip, params["prompt"], picture, params.get("brain"))
        except Exception as error:  # noqa: BLE001 - the clip is made; a gate that cannot run does not lose it
            job["qa"] = {"passed": None, "takes": takes, "error": str(error)[-300:]}
            break
        takes.append({**report, "take": number, "seed": int(params.get("seed", 0)), "sheet": relative(picture)})
        job["qa"] = {"passed": report["passed"], "takes": takes}
        if report["passed"] or job["status"] == "cancelled":
            break
    return job["status"] != "cancelled"


SHOT_TAKES = 2  # like the QA gate: a clip that fails the shot check is rendered once more at the next seed, when asked


def canvas_of(env: dict, preset: str) -> tuple[int, int]:
    """The size a clip's keyframes were fitted to: the generation canvas, or one of its shape (for shot.py)."""

    if env.get("QHD") or env.get("CROP") == "2560x1440":
        return 2560, 1472
    if env.get("TWOK"):
        return 2048, 1152
    if env.get("WIDTH") and env.get("HEIGHT"):
        return int(env["WIDTH"]), int(env["HEIGHT"])
    return PRESETS[preset][1] if preset in PRESETS else (1344, 768)


def shot_checked(job: dict, clip: Path, env: dict) -> bool:
    """The shot check on a single clip (shot.py), with one retake at the next seed when ``retake`` asks for it."""

    params, takes = job["params"], []
    # the QA gate has its own retake: a clip it rendered twice, or will judge, is not rendered a third time here
    may = bool(params.get("retake")) and "qa" not in job
    for number in range(1, SHOT_TAKES + 1):
        if number > 1:
            kept = HOME / "qa" / f"{job['id']}-shot-take{number - 1}.mp4"  # the failed take is kept
            kept.parent.mkdir(exist_ok=True)
            shutil.copy(clip, kept)
            takes[-1]["kept"] = relative(kept)
            params["seed"] = int(params.get("seed", 0)) + 1
            done, env, _ = take(job, params, f"take {number} · ")
            if not done:
                return False
        job["progress"] = {"stage": "shot check, take", "at": number, "of": SHOT_TAKES if may else 1}
        first = env.get("IMAGE_FILE") or (str(clip.with_suffix(".png")) if job["kind"] == "studio" and clip.with_suffix(".png").is_file() else None)
        try:
            report = shot.check(clip, first, env.get("LAST_FRAME"), canvas_of(env, params.get("preset", "")), shot.one_shot(params.get("prompt") or ""), not SPARK)
        except Exception as error:  # noqa: BLE001 - the clip is made; a check that cannot run does not lose it
            job["shot"] = {"passed": None, "takes": takes, "error": str(error)[-300:]}
            break
        takes.append({**report, "take": number, "seed": int(params.get("seed", 0))})
        job["shot"] = {"passed": report["passed"], "takes": takes}
        if report["passed"] or job["status"] == "cancelled":
            break
        if not may:  # said beside the clip, with the seed a retake would use: the page offers it as a button
            job["shot"]["offer"] = int(params.get("seed", 0)) + 1
            break
    return job["status"] != "cancelled"


def long_take(job: dict) -> bool:
    """A video longer than one clip: pieces, each opening on the last frame of the one before, then joined."""

    params = job["params"]
    total = int(params["total_seconds"])
    pieces = plan(total, int(params.get("segment_seconds", LONGEST)))
    beats = [line.strip() for line in params.get("beats") or [] if line.strip()]
    folder = HOME / "renders" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    image, clips = params.get("image"), []
    for index, seconds in enumerate(pieces):
        prompt = params["prompt"]
        if beats:  # one line per piece says what happens in it; the last line holds for the rest
            beat = f"In this part of the video: {beats[min(index, len(beats) - 1)].rstrip('.')}."
            prompt = prompt.replace("[Shot 1] ", f"[Shot 1] {beat} ", 1) if "[Shot 1] " in prompt else f"{beat} {prompt}"
        if index:
            still = folder / f"seg{index:03d}_first.jpg"
            last_frame(clips[-1], still)
            image = relative(still)
        if image and not prompt.startswith("For the target video"):
            prompt = f"{LEAD}\n\n{prompt}"
        target = folder / f"seg{index + 1:03d}.mp4"
        piece = {**params, "prompt": prompt, "image": image, "seconds": seconds, "output": relative(target),
                 "seed": int(params.get("seed", 0)) + index}
        command, env, _ = build("video", piece, job["id"])
        if not run(job, command, env, f"part {index + 1}/{len(pieces)} · "):
            return False
        if not target.is_file():
            raise RuntimeError(f"part {index + 1} wrote no clip")
        clips.append(target)
        job["outputs"] = [relative(c) for c in clips]  # finished parts can be watched while the rest renders
    # join: every part after the first repeats the frame it started from, so that frame is dropped
    job["progress"] = {"stage": "joining", "at": len(clips), "of": len(clips)}
    listing = folder / "parts.txt"
    trimmed = []
    for index, clip in enumerate(clips):
        part = folder / f"join{index + 1:03d}.ts"
        cut = ["-vf", "trim=start_frame=1,setpts=PTS-STARTPTS", "-af", f"atrim=start={1 / FPS},asetpts=PTS-STARTPTS"] if index else []
        done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), *cut, "-c:v", "libx264", "-crf", "17",
                               "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a",
                               "256k", "-ar", "48000", "-f", "mpegts", str(part)], capture_output=True, text=True,
                              env=environment({}))
        if done.returncode:
            job["log"] += done.stderr
            raise RuntimeError("ffmpeg failed while preparing a part")
        trimmed.append(part)
    listing.write_text("".join(f"file '{part}'\n" for part in trimmed))
    final = HOME / "renders" / f"{job['id']}.mp4"
    done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-t",
                           str(total), "-c", "copy", "-movflags", "+faststart", str(final)], capture_output=True,
                          text=True, env=environment({}))
    for part in trimmed:
        part.unlink(missing_ok=True)
    if done.returncode:
        job["log"] += done.stderr
        raise RuntimeError("ffmpeg failed while joining the parts")
    job["outputs"] = [relative(final)] + [relative(c) for c in clips]
    return True


def perform(job: dict, worker: dict | None = None) -> None:
    """One claimed job to its end, here or with its clip on the Spark it was handed to (``worker``, sparks.py)."""

    params = job["params"]
    asked = dict(params)
    try:
        if params.get("first_frame_from"):  # continuity: open on the last frame of an earlier clip
            source = inside(params["first_frame_from"])
            if not source.is_file():
                raise RuntimeError("the clip this one continues from has not been rendered")
            still = HOME / "uploads" / f"{job['id']}_first.jpg"
            last_frame(source, still)
            params["image"] = relative(still)
        if params.get("last_frame_from"):  # end on the first frame of another clip, so the two join
            source = inside(params["last_frame_from"])
            if not source.is_file():
                raise RuntimeError("the clip this one leads into has not been rendered")
            still = HOME / "uploads" / f"{job['id']}_last.png"
            frame_at(source, still, 0)
            params["last_image"] = relative(still)
        if job["kind"] == "export":
            export(job)
        elif job["kind"] == "long":
            if not long_take(job):
                return
        else:
            done, env, outputs = take(job, params)
            if not done:
                return
            job["outputs"] = [relative(Path(p)) for p in outputs if Path(p).is_file()]
            if not job["outputs"]:
                raise RuntimeError("the script finished without writing its output")
            if job["kind"] == "video" and params.get("preset") == "sharp" and params.get("qa", True):
                if not gated(job, Path(outputs[0])):
                    return
            elif job["kind"] == "video" and params.get("preset") == "sharp":
                job["qa"] = qa.not_run("The QA gate was switched off for this job (qa: false): the clip is not checked.")
            if job["kind"] in ("video", "studio") and params.get("shot_check", True):
                if not shot_checked(job, Path(outputs[0]), env):
                    return
            if job["kind"] == "video" and params.get("silent"):
                hush(Path(outputs[0]))
        hub.record(job)  # the clip, its seed and settings, and the Spark that rendered it (sparks/takes.jsonl)
        if job["status"] != "cancelled":
            job["status"] = "done"
    except sparks.Moved:  # the Spark it was on refused it, failed it or stopped answering: it is in the queue again
        params.clear()  # as it was queued: a retake's seed and the frames cut for it are made again
        params.update(asked)
    except Exception as error:  # noqa: BLE001 - a failed job must not stop the queue
        if job["status"] != "cancelled":
            job["status"], job["error"] = "failed", str(getattr(error, "detail", error))
    finally:
        if job["status"] != "queued":
            job["finished"] = time.time()


def work() -> None:
    while True:
        wake.wait()
        with lock:
            job = hub.claim()  # the next job for this machine; on a DGX Spark a worker may have taken others (sparks.py)
            if job is None:
                wake.clear()
                continue
        perform(job)
        hub.kick.set()
