"""The queue's HTTP API: the machine's state, prompts and estimates, submitting and cancelling jobs, and the QA gate."""

from __future__ import annotations

import os
import signal
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

import actions
import engines
import lengths
import lora_flows
import negation
import picture_models
import qa
import remote
import sparks
import voice
from engines import (FFMPEG_NOTE, custom, estimate, plan, ref2va_ready, ref2va_turbo_ready,
                     text_encoders, turbo_ready, without_ffmpeg, zig_ready)
from job_build import build, clip_seconds, longest_frames, memory_of, references
from job_state import jobs, lock, order, public, running, wake
from prompt_text import compose, compose_references
from settings import (CUSTOM_PIXELS, CUSTOM_SIDES, CUSTOM_STEPS, DURATIONS, FPS, HOME, LONG, LONGEST, PACK, PRESETS,
                      QUALITIES, REFERENCE_AUDIO_LIMIT, REFERENCE_LIMIT, REFERENCE_TOTAL, REFERENCE_VIDEO_LIMIT, SHARP,
                      SPARK, SPARK_ENGINE_LIB, inside, machine, relative)

router = APIRouter()
# Set by server.py: the multi-Spark hub, and the Library actions' job titles and sources (actions.py).
hub = action_title = action_source = None


@router.get("/api/state")
def state() -> dict:
    return {"presets": {k: {"label": v[0], "size": v[1]} for k, v in PRESETS.items()},
            "qualities": {k: v[0] for k, v in QUALITIES.items()}, "durations": DURATIONS, "long": LONG,
            "home": str(HOME), "pack": str(PACK), "version": (PACK / "VERSION").read_text().strip(),
            "zig": zig_ready(), "turbo": turbo_ready() and not SPARK, "x2": not SPARK,
            "ref2va": ref2va_ready(), "ref2va_turbo": ref2va_turbo_ready(), "reference_limit": REFERENCE_LIMIT,
            # reference videos and audio: wherever Ref2VA is installed
            "ref_media": ref2va_ready(), "reference_video_limit": REFERENCE_VIDEO_LIMIT,
            "reference_audio_limit": REFERENCE_AUDIO_LIMIT, "reference_total": REFERENCE_TOTAL,
            "text_encoders": sorted(text_encoders()),
            # the picture models installed here, in the order the page offers them (IMAGE_MODELS)
            "image_models": picture_models.offered(), "sharp": SHARP if "sharp" in PRESETS and "fh8" in QUALITIES else None, "custom": {"sides": CUSTOM_SIDES, "pixels": CUSTOM_PIXELS, "steps": CUSTOM_STEPS},
            "platform": machine()[0], "platform_label": machine()[1], "last_frame": True,
            # every clip length the model takes (lengths.py), and the longest this machine renders at each size
            "lengths": {"fps": FPS, "step": lengths.STEP, "shortest": lengths.SHORTEST, "longest": lengths.LONGEST,
                        "presets": {k: longest_frames(v[2]) for k, v in PRESETS.items()}, "custom": longest_frames({"WIDTH": 1344, "HEIGHT": 768})},
            "ffmpeg": engines.media_tools()["ffmpeg"], "ffmpeg_note": "" if engines.media_tools()["ffmpeg"] else FFMPEG_NOTE,
            "shot_check": True, "negation_reason": negation.REASON,
            # how many DGX Sparks a job may use (sparks.py); a Mac renders on itself
            "sparks": {"supported": SPARK, "limit": sparks.LIMIT},
            # Sam Wasserman's QA gate: whether it can run here, and what is missing when it cannot (qa.py)
            "qa": qa.ready(),
            # Voice: Fish Audio Drama 3 text to speech on this machine (voice.py): MLX on a Mac, CUDA on a DGX Spark
            "voice": voice.state(),
            "engine": ("CUDA · TensorFold int8" if SPARK_ENGINE_LIB.exists() else "CUDA · ComfyUI") if SPARK else ("native Zig engine" if zig_ready() else "MLX engine")}


@router.post("/api/compose")
def compose_prompt(body: dict) -> dict:
    if body.get("references") or body.get("referenced"):
        # a picked LoRA with its own caption style (Character Swap, Lineart Anime, Keyframe Animation) writes the prompt
        own = lora_flows.compose(body.get("loras") or (body.get("fields") or {}).get("loras"), body.get("fields", {}))
        if own:
            return {"prompt": own, "flow": True}
        return {"prompt": compose_references(body.get("fields", {}), int(body.get("references") or 0)),
                **negation.check_fields(body.get("fields", {}))}
    try:
        frames = lengths.of({"frames": body.get("frames"), "seconds": body.get("seconds") or 5})
    except ValueError:
        frames = lengths.nearest(5)
    # the prompt is written as the fields say; what they forbid is listed beside it, never taken out (negation.py)
    return {"prompt": compose(body.get("fields", {}), bool(body.get("first_frame")), bool(body.get("last_frame")), frames=frames),
            **negation.check_fields(body.get("fields", {}))}


@router.post("/api/prompt/check")
def check_prompt(body: dict) -> dict:
    """What a finished video prompt forbids by name (negation.py), before it is queued; nothing is changed."""

    return negation.check(str(body.get("prompt") or ""))


@router.post("/api/estimate")
def estimate_time(body: dict) -> dict:
    """A clip's expected time (None where not measured) and, on a DGX Spark, whether it fits the memory."""

    room = None
    if SPARK and float(body.get("seconds") or 5) <= LONGEST + 1:
        try:
            chosen, _ = custom(body)
            sized = {**(PRESETS[body["preset"]][2] if body.get("preset") in PRESETS else {}), **chosen}
            count = int(body.get("references") or 0)
            room = memory_of(body, sized, lengths.of({"frames": body.get("frames"), "seconds": body.get("seconds") or 5}), bool(body.get("loras") or body.get("with_lora")),
                             {"pictures": [None] * count, "videos": []} if body.get("referenced") else None)
            room = {key: room.get(key) for key in ("fits", "reason", "need_mb", "error_mb", "left_mb", "floor_mb", "available_mb", "nearest")}
        except (HTTPException, ValueError, KeyError):
            room = None  # the settings are refused for another reason when the job is queued
    if body.get("preset") == "custom" or body.get("steps") not in (None, "") or body.get("loras"):
        return {"seconds": None, "memory": room}  # a LoRA changes which engine renders the clip: no time is measured for it
    return {"seconds": estimate(body.get("preset", "draft"), body.get("quality", "standard"),
                                clip_seconds({"frames": body.get("frames"), "seconds": body.get("seconds", 5)})
                                if float(body.get("seconds") or 5) <= LONGEST + 1 or body.get("frames") else int(body["seconds"]),
                                bool(body.get("text_only")), bool(body.get("referenced")),
                                bool(body.get("ref_turbo")), bool(body.get("ref_media"))), "memory": room}


def clip_estimate(kind: str, params: dict, referenced: bool = False, ref_media: bool = False) -> int | None:
    """A job's expected time on one machine (None where none was measured: a custom size or passes, a LoRA)."""

    if kind not in ("video", "studio", "long") or params.get("preset") == "custom" or params.get("steps") not in (None, "") or params.get("loras"):
        return None
    return estimate(params.get("preset", ""), params.get("quality", "standard"), clip_seconds(params),
                    kind == "video" and not params.get("image") and not params.get("first_frame_from"), referenced,
                    bool(params.get("ref_turbo")), ref_media)


@router.post("/api/jobs")
def submit(body: dict) -> dict:
    kind = body.get("kind")
    params = dict(body.get("params") or {})
    if kind not in ("scout", "video", "studio", "export", "long", "edit") + actions.KINDS:
        raise HTTPException(400, "unknown job kind")
    if kind == "long" and (params.get("last_image") or params.get("last_same") or params.get("last_frame_from")):
        raise HTTPException(400, "a last frame is for single clips: a long video is a chain of clips")
    if kind == "long":
        total = int(params.get("total_seconds", 0))
        if not LONGEST < total <= max(LONG):
            raise HTTPException(400, f"a long video is over {LONGEST} seconds and at most {max(LONG) // 60} minutes")
        pieces = plan(total, int(params.get("segment_seconds", LONGEST)))
        build("video", {**params, "seconds": pieces[0], "output": None}, "check")
    if kind not in ("export", "edit") + actions.KINDS and not (params.get("prompt") or "").strip():
        raise HTTPException(400, "a prompt is needed")
    if kind != "video" and params.get("references"):
        raise HTTPException(400, "reference pictures are for single clips")
    if kind != "video" and (params.get("reference_videos") or params.get("reference_audios")):
        raise HTTPException(400, "reference videos and audio are for single clips")
    missing = without_ffmpeg(kind, params)
    if missing:
        raise HTTPException(400, missing)
    found = references(params) if kind == "video" else {}
    referenced, ref_media = bool(found), bool(found and (found["videos"] or found["audios"]))
    if kind in ("video", "studio", "long") and not referenced and not str(params.get("quality", "standard")).startswith("fh") and not turbo_ready():
        raise HTTPException(400, "MiniMax H3 Turbo is not installed here (oneshot-setup.sh --turbo adds it): use FastH3, quality fh4, fh8 or fh20")
    job_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    if kind not in ("export", "long"):
        build(kind, params, job_id)  # reject bad settings now, not when the job's turn comes
    job = {"id": job_id, "kind": kind, "status": "queued", "title": body.get("title") or (action_title(kind, params) if kind in actions.KINDS else kind), "params": params,
           "source": action_source(kind, params) if kind in actions.KINDS else None,
           "outputs": [], "progress": None, "error": None, "log": "", "created": time.time(), "started": None,
           "finished": None, "estimate": clip_estimate(kind, params, referenced, ref_media)}
    if referenced:  # the passes a reference clip runs: the number set by hand, or the model's 20, or the adapter's 4
        job["passes"] = int(params["steps"]) if params.get("steps") not in (None, "") else 4 if params.get("ref_turbo") else 20
    if kind in ("video", "studio", "long"):
        # said beside the job too, for a prompt that did not come through the wizard; the prompt goes as written
        job["negations"] = negation.check(params.get("prompt") or "")["negations"] or None
    # how many Sparks the job may use, this one counted (sparks.py): the number given, or the one set on the Sparks page
    try:
        chosen = int(body.get("sparks") or hub.config()["sparks"])
    except (TypeError, ValueError):
        raise HTTPException(400, f"sparks is a whole number from 1 to {sparks.LIMIT}")
    if not 1 <= chosen <= sparks.LIMIT:
        raise HTTPException(400, f"sparks is a whole number from 1 to {sparks.LIMIT}")
    said = hub.plan({"sparks": chosen, "kind": kind, "params": params})
    job["sparks"] = {"chosen": chosen if SPARK else 1, "available": said["available"], "used": said["used"], "says": said["says"]}
    with lock:
        jobs[job_id] = job
        order.append(job_id)
    wake.set()
    hub.kick.set()
    return public(job)


@router.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [public(jobs[i]) for i in reversed(order)]


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    if job_id not in jobs:
        raise HTTPException(404, "no such job")
    return public(jobs[job_id])


@router.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job["status"] in ("queued", "running"):
        job["status"] = "cancelled"
        process = running["process"] if running["id"] == job_id else None
        if process is not None:
            os.killpg(process.pid, signal.SIGTERM)
        hub.kick.set()  # a clip on another Spark is cancelled there by the thread that follows it
    return public(job)


def qa_run(job: dict) -> None:
    """The QA gate as a job of its own, on a clip here or fetched from another app; no GPU, so it skips the queue."""

    params = job["params"]
    try:
        can = qa.ready()
        if not can["runs"]:
            raise RuntimeError("the QA gate cannot run on this machine: it has no " + " and no ".join(can["missing"]))
        prompt, seed, path = params.get("prompt") or "", params.get("seed"), params.get("path") or ""
        if params.get("from"):
            base = remote.address(params["from"])
            if params.get("job"):  # the clip, its prompt and its seed as that app's job has them
                theirs = remote.job(base, str(params["job"]))
                if theirs.get("status") != "done" or not theirs.get("outputs"):
                    raise RuntimeError(f"job {params['job']} on {base} is {theirs.get('status')}, with no clip to check")
                path = path or theirs["outputs"][0]
                prompt = prompt or (theirs.get("params") or {}).get("prompt") or ""
                seed = seed if seed is not None else (theirs.get("params") or {}).get("seed")
            job["progress"] = {"stage": "fetching the clip from " + base, "at": 0, "of": 1}
            clip = remote.fetch(base, path, HOME / "qa" / "fetched" / f"{job['id']}{Path(path).suffix.lower()}")
        else:
            clip = inside(path)
            if not clip.is_file():
                raise RuntimeError("no such clip in the studio's folder")
        job["outputs"] = [relative(clip)]
        job["progress"] = {"stage": "QA gate, take", "at": 1, "of": 1}
        picture = HOME / "qa" / f"{job['id']}-take1.jpg"
        report = qa.check(clip, prompt, picture, params.get("brain"))
        job["qa"] = {"passed": report["passed"], "takes": [{**report, "take": 1, "seed": int(seed or 0), "sheet": relative(picture)}],
                     "from": params.get("from") or None}
        if job["status"] != "cancelled":
            job["status"] = "done"
    except Exception as error:  # noqa: BLE001 - the reason goes beside the job
        if job["status"] != "cancelled":
            job["status"], job["error"] = "failed", str(getattr(error, "detail", error))[-400:]
    finally:
        job["finished"] = time.time()


@router.post("/api/qa")
def qa_gate(body: dict) -> dict:
    """Sam Wasserman's QA gate on a clip here, or from another app (``from``, ``path`` or ``job``): a job to poll."""

    if not (body.get("path") or (body.get("from") and body.get("job"))):
        raise HTTPException(400, "give the clip: a path, or another app's address (from) with a path or a job id")
    try:
        if body.get("from"):
            remote.address(str(body["from"]))
    except ValueError as error:
        raise HTTPException(400, str(error))
    can = qa.ready()
    if not can["runs"]:
        raise HTTPException(400, "the QA gate cannot run on this machine: it has no " + " and no ".join(can["missing"]))
    job_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    params = {key: body[key] for key in ("path", "from", "job", "prompt", "seed", "brain") if body.get(key) not in (None, "")}
    job = {"id": job_id, "kind": "qa", "status": "running", "title": body.get("title") or "QA gate: " + str(params.get("path") or params.get("job")),
           "params": params, "source": None, "outputs": [], "progress": None, "error": None, "log": "", "created": time.time(),
           "started": time.time(), "finished": None, "estimate": None}
    with lock:
        jobs[job_id] = job
        order.append(job_id)
    threading.Thread(target=qa_run, args=(job,), daemon=True).start()
    return public(job)
