"""Sparks: one queue rendered on several DGX Sparks; workers run their own app on 127.0.0.1, reached by ssh tunnel."""

from __future__ import annotations

import json
import socket
import shutil
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from sparks_clip import LOST, POLL, ClipMixin, Moved  # noqa: F401
from sparks_queue import QueueMixin
from sparks_rules import (DISK_GB, KINDS, LIMIT, MAC, MEASURED, MEASURED_CLIPS, MOVES, ONE_CLIP,  # noqa: F401
                          capable, here_only, speedup)
from sparks_wire import (ADDRESS, CALL, CODE, FOLDER, PORT, SSH, Refused, Tunnel, address_of, call, copy,  # noqa: F401
                         differs, digest, folder_of, send, shell)
from sparks_workers import PREFIX, STATES, THEIR_HOME, WorkersMixin  # noqa: F401


# A job's ``sparks`` (1 to LIMIT) is how many Sparks it may use; only single clips go to workers (docs/multi-spark.md).
class Hub(WorkersMixin, QueueMixin, ClipMixin):
    """Workers and their clips; jobs/order/lock/wake are the server's queue, work(job, worker) runs a claimed job."""

    def __init__(self, home: Path, pack: Path, spark: bool, jobs: dict, order: list, lock, wake, info, inside, work, timing=None):
        self.home, self.pack, self.spark = home, pack, spark
        self.jobs, self.order, self.lock, self.wake = jobs, order, lock, wake
        self.info, self.inside, self.work, self.timing = info, inside, work, timing or (lambda kind, params: None)
        # sparks/: sparks.json (workers, default number), takes.jsonl (one line per clip), tunnels.json (open tunnels)
        self.folder = home / "sparks"
        self.book = self.folder / "sparks.json"
        self.name = socket.gethostname()
        self.live: dict[str, dict] = {}   # address -> state, why, info, base, tunnel, job, working, checked
        self.mutex = threading.RLock()    # over self.live; taken after the queue's lock, never before it
        self.kick = threading.Event()
        self.mine: dict | None = None
        self.shell, self.copy, self.dial = shell, copy, lambda address: Tunnel(address)  # stood in for by the tests
        self.started = False

    # ---- what a clip was rendered on ----

    def record(self, job: dict) -> None:
        """A finished clip written into the takes list: its path, seed, settings and the Spark that rendered it."""

        if not self.spark or job["kind"] not in ("video", "studio", "long") or not job.get("outputs"):
            return
        params = job["params"]
        line = {"path": job["outputs"][0], "job": job["id"], "title": job.get("title"), "kind": job["kind"], "seed": params.get("seed"),
                "settings": {key: params[key] for key in ("preset", "quality", "frames", "seconds", "width", "height", "steps", "loras", "total_seconds")
                             if params.get(key) not in (None, "", [])},
                "spark": job.get("spark") or self.name, "address": job.get("remote"), "sparks": self.asked(job),
                "moved": job.get("moved") or None, "elapsed": round(time.time() - job["started"], 1) if job.get("started") else None,
                "finished": time.time()}
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            with open(self.folder / "takes.jsonl", "a") as out:
                out.write(json.dumps(line) + "\n")
        except OSError:
            pass

    def takes(self) -> dict:
        """path -> the last line written for it."""

        found = {}
        try:
            with open(self.folder / "takes.jsonl") as lines:
                for line in lines:
                    try:
                        one = json.loads(line)
                        found[one["path"]] = one
                    except (ValueError, KeyError, TypeError):
                        continue
        except OSError:
            pass
        return found

    # ---- what a number means, said before a job is queued ----

    def plan(self, body: dict) -> dict:
        try:
            chosen = max(1, min(int(body.get("sparks") or 1), LIMIT))
            clips = max(1, int(body.get("clips") or 1))
        except (TypeError, ValueError):
            raise HTTPException(400, "sparks and clips are whole numbers")
        kind, params = str(body.get("kind") or "video"), body.get("params") if isinstance(body.get("params"), dict) else {}
        if not self.spark:
            return {"supported": False, "limit": LIMIT, "chosen": 1, "available": 1, "listed": 1, "used": 1, "mode": "one", "says": MAC, "without": [], "estimate": None}
        able, without = [], []
        for worker in self.workers():
            why = self.usable(worker) or capable(kind, params, self.state(worker["address"])["info"])
            (without.append({"name": worker.get("name") or worker["address"], "why": why}) if why else able.append(worker["address"]))
        listed, available = 1 + len(able) + len(without), 1 + len(able)
        here = here_only(kind, params)
        used = 1 if here else min(chosen, available)
        count = f"{available} of {chosen} available" if chosen > available else f"{used} Spark{'s' if used != 1 else ''}"
        if here:
            mode, says = "one", here
        elif used < 2:
            mode, says = "one", ("This Spark alone." if chosen < 2 else "No other Spark can take this job now, so it renders on this one.")
        elif clips > 1:
            mode, says = "parallel", f"One clip per Spark, {min(used, clips)} at a time: {clips} clips on {min(used, clips)} Sparks."
        else:
            mode, says = "one", f"This clip renders on one Spark, the first that is free of the {used}; other queued clips render beside it. " + ONE_CLIP
        estimate = None
        try:  # one clip's time on one Spark: given, or the app's own measured figure for these settings
            seconds = float(body.get("seconds") or self.timing(kind, params) or 0)
        except (TypeError, ValueError):
            seconds = 0
        if seconds > 0:
            faster, measured = speedup(used, clips) if mode == "parallel" else (1.0, True)
            estimate = {"seconds": int(seconds * clips / faster), "one_spark": int(seconds * clips), "faster": round(faster, 2), "measured": measured,
                        "note": "an estimate from one measured run of 8 clips" + ("" if measured else f"; {used} Sparks were not measured (five exist)")}
        return {"supported": True, "limit": LIMIT, "chosen": chosen, "available": available, "listed": listed, "used": used, "mode": mode,
                "count": count, "says": says, "without": without, "estimate": estimate}

    # ---- the API ----

    def router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/api/spark/self")
        def about_this_spark() -> dict:
            """What this app is and has, for a Studio on another Spark that hands clips to it."""

            return self.info()

        @router.get("/api/sparks")
        def sparks() -> dict:
            return self.overview()

        @router.post("/api/sparks")
        def settings(body: dict) -> dict:
            """``sparks``: the number new jobs start with."""

            found = self.config()
            try:
                found["sparks"] = max(1, min(int(body.get("sparks") or 1), LIMIT))
            except (TypeError, ValueError):
                raise HTTPException(400, "sparks is a whole number")
            self.keep(found)
            return self.overview()

        @router.post("/api/sparks/workers")
        def add(body: dict) -> dict:
            if not self.spark:
                raise HTTPException(400, MAC)
            try:
                address = address_of(str(body.get("address") or ""))
                folder = folder_of(str(body["folder"])) if body.get("folder") else None
            except ValueError as error:
                raise HTTPException(400, str(error))
            found = self.config()
            if any(worker["address"] == address for worker in found["workers"]):
                raise HTTPException(400, f"{address} is already in the list")
            if len(found["workers"]) >= LIMIT - 1:
                raise HTTPException(400, f"at most {LIMIT} Sparks, this one included")
            found["workers"].append({"address": address, "name": str(body.get("name") or "").strip()[:60] or address, "folder": folder, "enabled": True})
            self.keep(found)
            self.act(address, "check")
            return self.overview()

        @router.post("/api/sparks/workers/{address}")
        def change(address: str, body: dict) -> dict:
            found = self.config()
            worker = next((one for one in found["workers"] if one["address"] == address), None)
            if worker is None:
                raise HTTPException(404, "no such Spark in the list")
            if "enabled" in body:
                worker["enabled"] = bool(body["enabled"])
            if body.get("name"):
                worker["name"] = str(body["name"]).strip()[:60]
            if "folder" in body:
                try:
                    worker["folder"] = folder_of(str(body["folder"])) if body["folder"] else None
                except ValueError as error:
                    raise HTTPException(400, str(error))
            self.keep(found)
            self.kick.set()
            return self.overview()

        @router.delete("/api/sparks/workers/{address}")
        def remove(address: str) -> dict:
            found = self.config()
            if not any(one["address"] == address for one in found["workers"]):
                raise HTTPException(404, "no such Spark in the list")
            if self.state(address)["job"]:
                raise HTTPException(400, "that Spark is rendering one of this Spark's clips: cancel the job first")
            found["workers"] = [one for one in found["workers"] if one["address"] != address]
            self.keep(found)
            self.hang_up(address)  # its app, if a head started it, keeps running there: "Stop" first to free its memory
            with self.mutex:
                self.live.pop(address, None)
            return self.overview()

        @router.post("/api/sparks/workers/{address}/{what}")
        def action(address: str, what: str) -> dict:
            if what not in ("check", "update", "stop"):
                raise HTTPException(404, "check, update or stop")
            self.act(address, what)
            return self.overview()

        @router.post("/api/sparks/check")
        def check_all() -> dict:
            for worker in self.workers():
                self.act(worker["address"], "check")
            return self.overview()

        @router.post("/api/sparks/plan")
        def what_a_number_means(body: dict) -> dict:
            """What ``sparks`` does for a job or ``clips`` like it: Sparks able, those not and why, the time."""

            return self.plan(body)

        @router.get("/api/sparks/takes")
        def rendered_on(path: str = "") -> dict:
            found = self.takes()
            return found.get(path) or {} if path else found

        return router


def about(home: Path, code: dict, state: dict, loras: list, active: int, memory: dict, models: Path) -> dict:
    """What an app says of itself on /api/spark/self."""

    try:
        disk = round(shutil.disk_usage(home).free / 1e9, 1)
    except OSError:
        disk = None
    return {"name": socket.gethostname(), "version": state["version"], "platform": state["platform"], "engine": state["engine"],
            "guard": bool(memory), "code": code, "active": active, "memory": memory, "disk_gb": disk,
            "fasth3": any((models / "FastH3-Comfy/diffusion_models").glob("*fasth3*")),
            "image": any((models / "Qwen-Image-2.1-Comfy/diffusion_models").glob("qwen_image_2.1*.safetensors")),
            "ref2va": state["ref2va"], "ref2va_turbo": state["ref2va_turbo"], "ffmpeg": state["ffmpeg"],
            "loras": loras, "text_encoders": state["text_encoders"]}
