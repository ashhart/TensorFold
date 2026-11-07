#!/usr/bin/env python3
"""Wasserman Slate AI Studio web app (scripts/app.sh, http://127.0.0.1:7870): wires the queue, tabs and pages."""

from __future__ import annotations

import os
import subprocess  # noqa: F401  (tests stub subprocess.run through this module)
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import api_files
import api_jobs
import job_build
import job_run
import memory_guard
from api_jobs import clip_estimate, state
from engines import FFMPEG_NOTE, custom, media_seconds, media_tools, ref2va_ready, ref2va_turbo_ready, without_ffmpeg  # noqa: F401
from job_build import build, clip_frames, clip_seconds, memory_of  # noqa: F401  (names the tests reach through server)
from job_run import gated, last_frame, perform, run, shot_checked, work  # noqa: F401
from job_state import jobs, lock, order, public, wake  # noqa: F401
from picture_models import image_model_env, image_models, offered  # noqa: F401
from prompt_text import compose, compose_references, lead  # noqa: F401
from settings import (DURATIONS, GPU_NAME, HERE, HOME, LEAD, PACK, PRESETS, QUALITIES, REFERENCE_LIMIT,  # noqa: F401
                      SPARK, SPARK_ENGINE_LIB, SPARK_REF2VA, inside, relative)

app = FastAPI(title="Wasserman Slate AI Studio, powered by TensorFold")
threading.Thread(target=work, daemon=True).start()
app.include_router(api_jobs.router)
app.include_router(api_files.router)


@app.exception_handler(HTTPException)
async def errors(_, error: HTTPException):
    return JSONResponse({"error": error.detail}, status_code=error.status_code)


import lineart  # noqa: E402  (a line-art copy of a clip, for a line-art LoRA)
import lora_flows  # noqa: E402,F401  (what a hub LoRA needs of a clip)
import loras  # noqa: E402  (Keys and Wasserman LoRA Hub, beside this file)
lora_router, lora_resolve, lora_lead = loras.attach(HOME, PACK, SPARK)
job_build.lora_resolve, job_build.lora_lead = lora_resolve, lora_lead
app.include_router(lora_router)
app.include_router(lineart.attach(HOME))
import picture_edit  # noqa: E402  (a picture re-drawn by the image model, with the hub's image LoRAs)
edit_router, edit_build = picture_edit.attach(HOME, PACK, SPARK, lora_resolve, inside, relative, offered)
job_build.edit_build = edit_build
app.include_router(edit_router)
import pace  # noqa: E402,F401  (whether a clip's sound and motion agree, beside this file)
import negation  # noqa: E402,F401  (what a prompt forbids by name, shown before a job is queued, beside this file)
import shot  # noqa: E402,F401  (a clip's keyframes held and no cut inside it, beside this file)
import qa  # noqa: E402  (Sam Wasserman's QA gate, beside this file)
import remote  # noqa: E402,F401  (another machine's Studio app, read over its own API, beside this file)
import slate  # noqa: E402  (Wasserman Slate AI Studio, beside this file)


def slate_room(plan: dict, lora: bool) -> dict:
    """Whether a Slate shot's render plan fits this machine's memory: the check a queued job gets (memory_of)."""

    chosen, _ = custom(plan) if plan.get("preset") == "custom" else ({}, False)
    sized = {**(PRESETS[plan["preset"]][2] if plan.get("preset") in PRESETS else {}), **chosen}
    return memory_of(plan, sized, int(plan["frames"]), lora, {"pictures": plan["references"], "videos": []} if plan.get("references") else None)


app.include_router(slate.attach(HOME, DURATIONS, compose, compose_references, REFERENCE_LIMIT,
                                lambda: {"ref2va": ref2va_ready(), "ref2va_turbo": ref2va_turbo_ready()}, slate_room))
import motion  # noqa: E402  (Motion Previs: a modified Python port of Sam Wasserman's Motion Previs Studio analysis)
app.include_router(motion.attach(HOME, PACK, SPARK))
import actions  # noqa: E402  (Turnaround sheet, Upscale and Remove a person: Library actions, beside this file)
action_router, action_build, action_title, action_source = actions.attach(HOME, PACK, SPARK, inside, relative)
job_build.action_build, api_jobs.action_title, api_jobs.action_source = action_build, action_title, action_source
app.include_router(action_router)
import clear  # noqa: E402  (finished jobs and their files into the trash folder, and back, beside this file)
app.include_router(clear.attach(HOME, jobs, order, lock)[0])
import sparks  # noqa: E402  (worker Sparks behind this queue: a job's "sparks" number, beside this file)


def about_this_spark() -> dict:
    """What this app is and has, for a Studio on another Spark that hands clips to it (/api/spark/self)."""

    listing = next(route.endpoint for route in lora_router.routes if route.path == "/api/loras" and "GET" in route.methods)
    info = memory_guard.meminfo()
    memory = {"available_mb": info.get("MemAvailable"), "total_mb": info.get("MemTotal"), "room_mb": memory_guard.room_mb(),
              "floor_mb": memory_guard.floor_mb(info.get("MemTotal"))} if SPARK and info else {}
    with lock:
        active = sum(1 for job in jobs.values() if job["status"] in ("queued", "running"))
    return sparks.about(HOME, hub.code(), state(), [one["id"] for one in listing() if one.get("installed")], active, memory, SPARK_REF2VA.parent)


def clip_time(kind: str, params: dict) -> int | None:
    """One clip's time for the Sparks estimate; None for settings that would be refused or have no measured time."""

    try:
        return clip_estimate(kind, params, bool(params.get("references") or params.get("reference_videos")),
                             bool(params.get("reference_videos") or params.get("reference_audios")))
    except (HTTPException, ValueError, TypeError, KeyError):
        return None


hub = sparks.Hub(HOME, PACK, SPARK, jobs, order, lock, wake, about_this_spark, inside, perform, clip_time)
job_run.hub = api_jobs.hub = api_files.hub = hub
app.include_router(hub.router())
if hasattr(app.router, "on_shutdown"):
    app.router.on_shutdown.append(hub.close)  # the tunnels to the workers are this app's: closed with it (and at exit, sparks.py)
app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")

if not media_tools()["ffmpeg"]:
    print("[studio] " + FFMPEG_NOTE, flush=True)

if __name__ == "__main__":
    import uvicorn

    if os.environ.get("STUDIO_PIDFILE"):  # a worker Spark's app, started by another Spark's Studio: how that one finds it again
        Path(os.environ["STUDIO_PIDFILE"]).write_text(str(os.getpid()))
    hub.start()

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "7870")),
                log_level="warning")
