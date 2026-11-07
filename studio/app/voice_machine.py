"""This machine for Drama 3: free memory, the studio's ComfyUI and the GPU lock shared with the scripts."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


SPARK = sys.platform.startswith("linux")
LOCK = Path(os.environ.get("STUDIO_GPU_LOCK", Path.home() / "studio-gpu.lock.d"))


def mem_available_gb() -> float:
    """Free memory in GB: MemAvailable (one pool on a Spark), or vm_stat's reclaimable pages on a Mac; -1 unknown."""

    if sys.platform == "darwin":
        try:
            said = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2).stdout
            size = int(re.search(r"page size of (\d+) bytes", said).group(1))
            pages = sum(int(found) for found in re.findall(r"Pages (?:free|inactive|speculative|purgeable):\s+(\d+)", said))
            return pages * size / 1024 ** 3
        except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired):
            return -1.0
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
    except (OSError, ValueError, IndexError):
        pass
    return -1.0


def comfy_up(port: int | None = None) -> bool:
    """Whether the studio's ComfyUI server (spark/comfy.sh) answers on its port: it may hold the video models."""

    import socket

    try:
        with socket.create_connection(("127.0.0.1", port or int(os.environ.get("SPARK_COMFY_PORT", 8190))), timeout=0.3):
            return True
    except OSError:
        return False



def settle() -> None:
    """Waits until MemAvailable stops climbing after a process is gone, 30 s at most."""

    before = -1.0
    for _ in range(30):
        free = mem_available_gb()
        if free - before < 0.2:
            return
        before = free
        time.sleep(1)


# ---------- the GPU lock (shared with the scripts that run outside the app) ----------

def lock_owner() -> str:
    try:
        return (LOCK / "owner").read_text().strip()
    except OSError:
        return ""


def stale() -> bool:
    """A lock whose owner is a process of this machine that is gone, or one left without an owner for a minute."""

    owner = lock_owner()
    pid = owner.split()[-1] if owner else ""
    if pid.isdigit():
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False
    try:
        return not owner and time.time() - LOCK.stat().st_mtime > 60
    except OSError:
        return False


def take_lock(job: dict, mine: str = "") -> bool:
    """Waits for the GPU lock (a job that is cancelled meanwhile stops waiting): True once it is this job's."""

    mine = mine or f"studio-voice {job['id']} {os.getpid()}"
    while True:
        try:
            LOCK.mkdir()
            (LOCK / "owner").write_text(mine + "\n")
            job["lock"] = mine
            return True
        except FileExistsError:
            if stale():
                shutil.rmtree(LOCK, ignore_errors=True)
                continue
        if job["status"] == "cancelled":
            return False
        job["progress"] = {"stage": "waiting for the GPU lock (" + (lock_owner() or "held") + ")", "at": 0, "of": 1}
        time.sleep(2)


def drop_lock(job: dict) -> None:
    if job.get("lock") and lock_owner() == job["lock"]:
        shutil.rmtree(LOCK, ignore_errors=True)
    job.pop("lock", None)
