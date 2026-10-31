"""Sparks: which Spark renders which queued clip, and the loop that hands clips to idle workers."""

from __future__ import annotations

import atexit
import threading
import time

from sparks_rules import LIMIT, MOVES, capable, here_only
from sparks_wire import Refused, call, differs


class QueueMixin:
    """The Hub's scheduler over the server's queue (``jobs``, ``order``, ``lock``, ``wake``)."""

    def asked(self, job: dict) -> int:
        try:
            return max(1, min(int((job.get("sparks") or {}).get("chosen") or 1), LIMIT))
        except (TypeError, ValueError):
            return 1

    def team(self, job: dict) -> list[str]:
        """The workers a job may render on, in the list's order: the first N - 1 that are up and can render it."""

        sparks = job.get("sparks") or {}
        if not self.spark or self.asked(job) < 2 or here_only(job["kind"], job["params"]) or sparks.get("moves", 0) >= MOVES:
            return []
        found = []
        for worker in self.workers():
            if worker["address"] in (sparks.get("not") or {}) or self.usable(worker):
                continue
            if capable(job["kind"], job["params"], self.state(worker["address"])["info"]):
                continue
            found.append(worker["address"])
        return found[:self.asked(job) - 1]

    def waits(self, job: dict) -> bool:
        """Whether a clip opens or ends on a clip that is still to be made: it waits for that one."""

        wanted = {str(job["params"].get(key)) for key in ("first_frame_from", "last_frame_from") if job["params"].get(key)}
        if not wanted:
            return False
        for other in self.jobs.values():
            if other is not job and other["status"] in ("queued", "running") and other["kind"] in ("video", "studio", "long"):
                if (other["params"].get("output") or f"renders/{other['id']}.mp4") in wanted:
                    return True
        return False

    def claim(self, address: str | None = None) -> dict | None:
        """Next queued job for a machine (None: this Spark), marked running; queue lock held; leaves workers' clips."""

        ready = [self.jobs[i] for i in self.order if self.jobs[i]["status"] == "queued" and not self.waits(self.jobs[i])]
        if address is None:
            idle = {worker["address"] for worker in self.workers() if not self.usable(worker) and not self.state(worker["address"])["job"]}
            job = next((job for job in ready if not idle.intersection(self.team(job))), None) or (ready[0] if ready else None)
        else:
            job = next((job for job in ready if address in self.team(job)), None)
        if job is None:
            return None
        job["status"], job["started"] = "running", time.time()
        job["remote"] = address
        job["spark"] = self.name if address is None else next((w.get("name") or address for w in self.workers() if w["address"] == address), address)
        if address is not None:
            self.state(address).update(job=job["id"], state="busy")
        return job

    def glance(self, worker: dict) -> bool:
        """A quick look at a connected worker before a clip is offered to it: is its app there and idle."""

        address, live = worker["address"], self.state(worker["address"])
        try:
            info = call(self.connect(address), "/api/spark/self", timeout=5)
        except (OSError, Refused) as error:
            live["fails"] += 1
            if live["fails"] >= 3:
                self.hang_up(address)
                live.update(state="down", why="It stopped answering: " + str(error)[-120:], checked=time.time(), info=None)
            return False
        live.update(info=info, fails=0, checked=time.time())
        if differs(self.code(), info.get("code")):
            live.update(state="update", why="Its Studio code no longer matches this Spark's: “Update and start” brings it level.")
            return False
        if info.get("active"):
            live.update(state="busy", why="It is rendering a job that is not this Spark's.")
            return False
        live.update(state="ready", why="")
        return True

    def loop(self) -> None:
        """Hands queued clips to idle workers, one each; looks at every connected worker now and then."""

        looked = 0.0
        while True:
            self.kick.wait(timeout=1.0)
            self.kick.clear()
            try:
                with self.lock:
                    waiting = any(job["status"] == "queued" and self.asked(job) > 1 for job in self.jobs.values())
                refresh = time.time() - looked > 20
                if refresh:
                    looked = time.time()
                for worker in self.workers():
                    live = self.state(worker["address"])
                    if live["job"] and refresh and live["base"]:  # rendering for us: only its memory and disk are read again
                        try:
                            live["info"] = call(live["base"], "/api/spark/self", timeout=5)
                        except (OSError, Refused):
                            pass  # the thread that follows its job decides when it is lost
                    if live["job"] or live["working"] or not worker.get("enabled", True) or live["state"] not in ("ready", "busy"):
                        continue
                    if not (waiting or refresh) or not self.glance(worker) or not waiting:
                        continue
                    with self.lock:
                        job = self.claim(worker["address"])
                    if job is not None:
                        threading.Thread(target=self.run, args=(job, worker), daemon=True).start()
            except Exception as error:  # noqa: BLE001 - the loop outlives a bad moment
                print(f"[sparks] {error}", flush=True)

    def run(self, job: dict, worker: dict) -> None:
        try:
            self.work(job, worker)
        finally:
            live = self.state(worker["address"])
            with self.mutex:
                if live["job"] == job["id"]:
                    live["job"] = None
                    if live["state"] == "busy":
                        live["state"] = "ready"
            self.kick.set()
            self.wake.set()

    def start(self) -> None:
        """At the app's start on a Spark: last run's tunnels closed, the loop started, the listed workers looked at."""

        if self.started or not self.spark:
            return
        self.started = True
        self.sweep()
        atexit.register(self.close)
        threading.Thread(target=self.loop, daemon=True).start()
        for worker in self.workers():
            if worker.get("enabled", True):
                self.act(worker["address"], "check")
