"""Download checkpoints into the Hugging Face cache with progress the page can follow.

Progress is measured on disk: the bytes of the repository's blobs in the cache, finished or still `.incomplete`,
against the repository's size. That is what the user sees, and it does not depend on how the hub library draws
its own progress bars. The download itself runs in a process of its own, so that a cancel can end it; what reached
the disk stays, and the hub resumes an `.incomplete` blob at the next start. A download the disk cannot hold is
refused before it starts: a full disk breaks TensorFold Web's own files, not only the download.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MARGIN_BYTES = 2 * 1024**3        # left free beside a download: TensorFold Web's own files, the engine's log, a copy


@dataclass
class Progress:
    repo: str
    state: str = "idle"          # "idle", "running", "done", "failed", "cancelled"
    done_bytes: int = 0
    total_bytes: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    path: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fraction"] = min(1.0, self.done_bytes / self.total_bytes) if self.total_bytes else 0.0
        return d


class NoRoom(RuntimeError):
    """The disk cannot hold the download."""


def cache_root() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def blobs_dir(repo: str) -> Path:
    return cache_root() / f"models--{repo.replace('/', '--')}" / "blobs"


def bytes_on_disk(repo: str) -> int:
    """Bytes of the repository's blobs in the cache, counting partial downloads."""

    d = blobs_dir(repo)
    if not d.is_dir():
        return 0
    total = 0
    for p in d.iterdir():
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def free_bytes() -> int:
    """Free space where the cache lives, or where it will (its nearest existing parent before the first download)."""

    p = cache_root()
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def command(repo: str) -> list[str]:
    """The download as its own process: the hub library in this interpreter, the path printed at the end."""

    return [sys.executable, "-c",
            "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))", repo]


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


class Downloads:
    def __init__(self) -> None:
        self.jobs: dict[str, Progress] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def get(self, repo: str) -> Progress:
        with self._lock:
            job = self.jobs.setdefault(repo, Progress(repo))
        if job.state != "running":
            job.done_bytes = bytes_on_disk(repo)
        return job

    def list(self) -> list[dict]:
        with self._lock:
            jobs = list(self.jobs.values())
        return [j.to_dict() for j in jobs]

    def start(self, repo: str, *, total_bytes: int | None = None) -> Progress:
        """Start one job for ``repo``; a second call while it runs returns the running job (checked under the lock).
        Refused, with both figures, when the disk cannot hold what is left to download."""

        with self._lock:
            old = self.jobs.get(repo)
            if old is not None and old.state == "running":
                return old
            # each attempt its own record: a stopped attempt's worker, finishing late, writes only to its own
            job = Progress(repo)
            on_disk = bytes_on_disk(repo)
            if total_bytes:
                need, free = max(0, int(total_bytes) - on_disk) + MARGIN_BYTES, free_bytes()
                if free < need:
                    raise NoRoom(f"not enough room on disk: this download needs {_gb(need)} more "
                                 f"(a {_gb(MARGIN_BYTES)} margin included) and {_gb(free)} are free")
            self.jobs[repo] = job
            job.state, job.error, job.started_at, job.finished_at = "running", None, time.time(), None
            job.total_bytes, job.done_bytes = int(total_bytes or 0), on_disk
            env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
            try:
                proc = subprocess.Popen(command(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                        env=env, start_new_session=True)
            except OSError as exc:
                job.state, job.error, job.finished_at = "failed", str(exc)[:300], time.time()
                return job
            self._procs[repo] = proc
        threading.Thread(target=self._run, args=(job, proc), name=f"pull-{repo}", daemon=True).start()
        threading.Thread(target=self._watch, args=(job,), name=f"pull-watch-{repo}", daemon=True).start()
        return job

    def cancel(self, repo: str) -> bool:
        """Ends a running download; what reached the disk stays, and the next start resumes it."""

        with self._lock:
            job, proc = self.jobs.get(repo), self._procs.get(repo)
            if job is None or job.state != "running" or proc is None:
                return False
            job.state, job.error = "cancelled", None
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
        return True

    def _run(self, job: Progress, proc: subprocess.Popen) -> None:
        try:
            out, err = proc.communicate()
            if job.state == "cancelled":
                return
            if proc.returncode == 0:
                said = [x for x in (out or "").strip().splitlines() if x.strip()]
                job.path, job.state = (said[-1].strip() if said else None), "done"
                job.done_bytes = max(job.done_bytes, bytes_on_disk(job.repo))
                if not job.total_bytes:
                    job.total_bytes = job.done_bytes
            else:
                said = [x for x in (err or out or "").strip().splitlines() if x.strip()]
                job.state = "failed"
                job.error = (said[-1].strip() if said else f"exit code {proc.returncode}")[:300]
        finally:
            job.finished_at = time.time()
            with self._lock:
                if self._procs.get(job.repo) is proc:
                    del self._procs[job.repo]

    def _watch(self, job: Progress) -> None:
        while job.state == "running":
            job.done_bytes = bytes_on_disk(job.repo)
            time.sleep(1.0)
