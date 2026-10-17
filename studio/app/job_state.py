"""The job queue's shared state: every job by id, their order, and the process running now."""

from __future__ import annotations

import threading
import time

jobs: dict[str, dict] = {}
order: list[str] = []
lock = threading.Lock()
wake = threading.Event()
running: dict = {"process": None, "id": None}


def public(job: dict) -> dict:
    out = {k: job.get(k) for k in ("id", "kind", "status", "title", "outputs", "progress", "error", "created",
                                    "started", "finished", "estimate", "engine", "qa", "passes", "pace", "source", "shot",
                                    "negations", "sparks", "spark", "moved")}
    # by this machine's clock: the page may be open on another one
    out["elapsed"] = (job.get("finished") or time.time()) - job["started"] if job.get("started") else None
    out["log"] = job["log"][-3000:]
    out["params"] = {k: v for k, v in job["params"].items() if k != "clips"}
    return out
