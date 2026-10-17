"""Sparks: one clip rendered on a worker through its job API, fetched back, or moved back to the queue."""

from __future__ import annotations

import os
import time
from pathlib import Path

import remote
from sparks_wire import CALL, Refused, call, send

POLL = float(os.environ.get("STUDIO_SPARK_POLL", "2"))     # seconds between looks at a worker's job
LOST = float(os.environ.get("STUDIO_SPARK_LOST", "60"))    # a worker silent this long while it renders is down


class Moved(Exception):
    """The clip was taken back from its worker and is in the queue again."""


class ClipMixin:
    """The Hub's clips on workers: sent, followed, fetched, or taken back."""

    def move(self, job: dict, worker: dict, why: str, down: bool) -> None:
        """A clip taken back from a worker and requeued, never for it again; ``down`` marks it down. Raises Moved."""

        address, name = worker["address"], worker.get("name") or worker["address"]
        live = self.state(address)
        if down:
            self.hang_up(address)
            live.update(state="down", why=why, checked=time.time(), info=None)
        with self.lock:
            if job["status"] != "running":
                return
            sparks = job.setdefault("sparks", {"chosen": 1})
            sparks.setdefault("not", {})[address] = why
            sparks["moves"] = sparks.get("moves", 0) + 1
            job.setdefault("moved", []).append({"from": name, "why": why, "at": time.time()})
            job["log"] = (job["log"] + f"[sparks] {name}: {why}. The clip is back in the queue.\n")[-20000:]
            job.update(status="queued", started=None, progress=None, spark=None, remote=None, their_job=None)
        raise Moved(why)

    def outbound(self, base: str, params: dict) -> dict:
        """A clip's settings for the worker: its media uploaded there, what this Spark does itself left out."""

        theirs = {key: value for key, value in params.items()
                  if key not in ("output", "first_frame_from", "last_frame_from", "retake", "silent", "slate", "brain")}
        for key in ("image", "last_image"):
            if theirs.get(key):
                theirs[key] = send(base, self.inside(theirs[key]))
        for key in ("references", "reference_videos", "reference_audios"):
            if theirs.get(key):
                theirs[key] = [send(base, self.inside(path)) for path in theirs[key] if path]
        theirs.update(qa=False, shot_check=False)  # both run here, on the fetched clip, where their tools are
        return theirs

    def render(self, job: dict, worker: dict, params: dict, target: Path, label: str = "") -> bool:
        """One take rendered on ``worker`` and fetched to ``target``; False if cancelled, Moved if refused or lost."""

        address, name = worker["address"], worker.get("name") or worker["address"]
        say = lambda text: job.update(log=(job["log"] + f"[sparks] {text}\n")[-20000:])
        try:
            base = self.connect(address)
            # the worker's own memory guard, asked before anything is sent (and again by its app when queued)
            room = (call(base, "/api/estimate", {**{key: params.get(key) for key in ("preset", "quality", "frames", "seconds", "width", "height", "steps", "text_encoder", "ref_turbo")},
                                                 "with_lora": bool(params.get("loras")), "referenced": bool(params.get("references") or params.get("reference_videos")),
                                                 "references": len(params.get("references") or [])}) or {}).get("memory") or {}
            if room.get("fits") is False:
                raise Refused(400, room.get("reason") or "its memory guard refused the clip")
            made = call(base, "/api/jobs", {"kind": job["kind"], "title": f"{job['title']} · for {self.name}", "params": self.outbound(base, params)}, timeout=60)
        except Refused as error:
            self.move(job, worker, f"it refused the clip: {error.said}", down=False)
            return False
        except OSError as error:
            self.move(job, worker, f"it stopped answering before the clip was queued ({str(error)[-120:]})", down=True)
            return False
        job["their_job"] = made["id"]
        say(f"{name} took the clip as its job {made['id']}" + (f"; its memory guard: about {room['need_mb'] / 1000:.0f} GB needed, {room.get('available_mb', 0) / 1000:.0f} GB there" if room.get("need_mb") else ""))
        silent, theirs = None, {}
        while True:
            if job["status"] != "running":  # cancelled here: cancelled there
                try:
                    call(base, f"/api/jobs/{made['id']}/cancel", {}, timeout=CALL)
                except (OSError, Refused):
                    pass
                return False
            try:
                theirs = call(base, f"/api/jobs/{made['id']}", timeout=CALL)
                silent = None
            except Refused as error:
                self.move(job, worker, "its app no longer has the job (it restarted)" if error.status == 404 else f"its app answered {error.status}", down=True)
                return False
            except OSError as error:
                silent = silent or time.time()
                if time.time() - silent > LOST:
                    self.move(job, worker, f"it stopped answering for {LOST:g} s while rendering ({str(error)[-100:]})", down=True)
                    return False
                time.sleep(POLL)
                continue
            if theirs.get("progress"):
                job["progress"] = {**theirs["progress"], "stage": f"{label}{name} · {theirs['progress'].get('stage', '')}"}
            if theirs.get("engine"):
                job["engine"] = theirs["engine"]
            if theirs.get("status") == "done":
                break
            if theirs.get("status") in ("failed", "cancelled"):
                self.move(job, worker, "its render failed: " + str(theirs.get("error") or "cancelled on that Spark")[-300:], down=False)
                return False
            time.sleep(POLL)
        try:
            if not theirs.get("outputs"):
                raise LookupError("its job finished without a clip")
            remote.fetch(base, theirs["outputs"][0], target, timeout=300)
        except (OSError, LookupError, ValueError) as error:
            self.move(job, worker, f"its clip could not be fetched ({str(error)[-160:]})", down=isinstance(error, OSError))
            return False
        said = [line for line in (theirs.get("log") or "").splitlines() if "[tensorfold]" in line or "[studio]" in line]
        job["log"] = (job["log"] + "\n".join(said[-12:]) + ("\n" if said else ""))[-20000:]
        say(f"{name} rendered it in {theirs.get('elapsed') or 0:.0f} s; the clip is in this Spark's folder")
        return job["status"] == "running"
