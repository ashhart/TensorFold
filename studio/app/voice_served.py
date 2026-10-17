"""A Drama 3 engine as a persistent server (``Served``), and a running take's live WAV."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import voice_machine
from voice_machine import SPARK, comfy_up, lock_owner, mem_available_gb, settle, take_lock


class Served:
    """Drama 3 as a ``--serve`` server (JSON on stdin and stderr): holds the GPU lock while loaded, idles out."""

    load_s = 10.0
    NEED_GB = 50.0

    def __init__(self) -> None:
        self.idle_s = float(os.environ.get("DRAMA3_IDLE_S", 300))
        self.guard_gb = float(os.environ.get("DRAMA3_GUARD_GB", 8))
        self.known: tuple[float, str] | None = None
        self.mutex = threading.RLock()  # a request, a load and an unload never run at once
        self.process: subprocess.Popen | None = None
        self.answers: "queue.Queue[dict]" = queue.Queue()
        self.log: list[str] = []
        self.owner = ""  # what this server wrote in the lock folder's owner file
        self.loaded_at = self.last_used = 0.0
        self.before_gb = -1.0  # MemAvailable just before the server started: what a job has once it is unloaded
        self.load_s_measured: float | None = None
        self.device_gb: float | None = None
        self.killed = ""  # why the guard killed the engine, if it did
        self.events: list[dict] = []  # loads and unloads, newest last (shown in /api/voice/state)
        self.idler: threading.Thread | None = None
        self.ready: dict = {}  # the engine's ready line

    # -- the server --

    def loaded(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def note(self, what: str, **more) -> None:
        self.events = (self.events + [{"at": time.time(), "what": what, **more}])[-20:]

    def status(self) -> dict:
        return {"loaded": self.loaded(), "idle_unload_s": self.idle_s, "load_s": self.load_s_measured,
                "idle_for_s": round(time.time() - self.last_used, 1) if self.loaded() and self.last_used else None,
                "events": self.events[-5:]}

    def read(self, process: subprocess.Popen) -> None:
        """The engine's stderr: its JSON answers into ``answers``, the rest into ``log``."""

        for line in process.stderr:
            line = line.rstrip("\n")
            if line.startswith("{"):
                try:
                    self.answers.put(json.loads(line))
                    continue
                except ValueError:
                    pass
            self.log = (self.log + [line])[-200:]
            found = re.search(r"weights on the device: ([\d.]+) GB", line)
            if found:
                self.device_gb = float(found.group(1))
        self.answers.put({"ok": False, "error": "the engine stopped", "gone": True})

    def guard(self, process: subprocess.Popen) -> None:
        """MemAvailable four times a second; the engine is killed by its PID under the floor."""

        while process.poll() is None:
            free = mem_available_gb()
            if 0 <= free < self.guard_gb:
                self.killed = f"MemAvailable {free:.1f} GB < {self.guard_gb:g} GB: the memory guard killed Drama 3 (pid {process.pid})"
                try:
                    os.kill(process.pid, 9)
                except OSError:
                    pass
                return
            time.sleep(0.25 if SPARK else 1.0)  # a Mac's count is a vm_stat process: once a second

    def idle(self) -> None:
        """Unloads the engine after ``idle_s`` without a take, and gives its memory and the GPU lock back."""

        while True:
            time.sleep(5)
            if not self.loaded():
                if self.process is not None and self.mutex.acquire(blocking=False):
                    try:  # it ended by itself (a crash, the guard): the lock goes back
                        if self.process is not None and not self.loaded():
                            self.stop("the engine ended (" + (self.killed or f"status {self.process.returncode}") + ")")
                    finally:
                        self.mutex.release()
                continue
            if time.time() - self.last_used < self.idle_s or not self.mutex.acquire(blocking=False):
                continue
            try:
                if self.loaded() and time.time() - self.last_used >= self.idle_s:
                    self.stop(f"idle for {self.idle_s:g} s")
            finally:
                self.mutex.release()

    def make_room(self, job: dict) -> None:
        """Enough free memory to load (ComfyUI stopped first when it holds it), or a RuntimeError saying why not."""

        free = mem_available_gb()
        if free < 0 or free >= self.NEED_GB:
            return
        if SPARK and comfy_up():
            job["progress"] = {"stage": "stopping ComfyUI to free the video models' memory", "at": 0, "of": 1}
            subprocess.run(["bash", str(Path(__file__).resolve().parent.parent / "spark" / "comfy.sh"), "stop"],
                           capture_output=True, text=True, timeout=30)
            for _ in range(60):
                if not comfy_up():
                    break
                time.sleep(1)
            settle()
            free = mem_available_gb()
            self.note("stopped ComfyUI before loading", available_gb=round(free, 1))
        if 0 <= free < self.NEED_GB:
            raise RuntimeError(f"Drama 3 needs about {self.NEED_GB:.0f} GB of free memory to load and this machine has "
                               f"{free:.0f} GB available")

    def start(self, job: dict) -> bool:
        """The server loaded and ready, under the GPU lock; False when the job was cancelled meanwhile."""

        if self.loaded():
            return True
        if self.process is not None:
            self.stop("the engine had ended (" + (self.killed or f"status {self.process.returncode}") + ")")
        mine = f"studio-voice drama3 {os.getpid()}"
        if not take_lock(job, mine):
            return False
        job.pop("lock", None)  # the server's, not the take's: it is given back when the server stops
        self.owner = mine
        try:
            self.make_room(job)
            job["progress"] = {"stage": f"loading Drama 3 (about {self.load_s:.0f} s)", "at": 0, "of": 1}
            env = self.environment()
            self.answers, self.log, self.killed, self.device_gb = queue.Queue(), [], "", None
            began, self.before_gb = time.time(), mem_available_gb()
            self.process = subprocess.Popen(self.argv(), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
            threading.Thread(target=self.read, args=(self.process,), daemon=True).start()
            threading.Thread(target=self.guard, args=(self.process,), daemon=True).start()
            if self.idler is None:
                self.idler = threading.Thread(target=self.idle, daemon=True)
                self.idler.start()
            while True:
                if job["status"] == "cancelled":
                    self.stop("the take was cancelled while loading")
                    return False
                try:
                    answer = self.answers.get(timeout=0.5)
                except queue.Empty:
                    if time.time() - began > 600:
                        raise RuntimeError("Drama 3 did not load in 10 minutes")
                    continue
                if answer.get("ready"):
                    self.ready = answer
                    break
                raise RuntimeError("Drama 3 did not load: " + self.why(answer))
            self.loaded_at = self.last_used = time.time()
            self.load_s_measured = round(self.loaded_at - began, 1)
            self.note("loaded", seconds=self.load_s_measured, device_gb=self.device_gb)
            job["load_s"] = self.load_s_measured
            return True
        except BaseException:
            self.stop("it failed to load")
            raise

    def argv(self) -> list[str]:
        raise NotImplementedError

    def environment(self) -> dict:
        return dict(os.environ)

    def why(self, answer: dict) -> str:
        said = self.killed or answer.get("error") or "no answer"
        tail = [line for line in self.log if line.strip()][-2:]
        return said + (" (" + " / ".join(tail)[-300:] + ")" if tail and answer.get("gone") else "")

    def ask(self, job: dict, request: dict, seconds: float) -> dict | None:
        """One request to the loaded server: its answer, or None when cancelled (the server is stopped then)."""

        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        began = time.time()
        while True:
            if job["status"] == "cancelled":
                self.stop("the take was cancelled")
                return None
            try:
                answer = self.answers.get(timeout=0.25)
            except queue.Empty:
                if time.time() - began > seconds:
                    self.stop("a request took longer than %d s" % seconds)
                    raise RuntimeError(f"Drama 3 did not answer in {seconds:.0f} s")
                continue
            if answer.get("ready"):
                continue
            if not answer.get("ok"):
                if answer.get("gone"):
                    self.stop("the engine ended during a take")
                raise RuntimeError("Drama 3: " + self.why(answer))
            return answer

    def stop(self, why: str) -> None:
        """The server ended (stdin closed, killed after 20 s), its memory back, the GPU lock given back."""

        with self.mutex:
            process, self.process = self.process, None
            if process is not None:
                if process.poll() is None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                settle()
                self.note("unloaded", why=why, available_gb=round(mem_available_gb(), 1))
            if self.owner and lock_owner() == self.owner:
                shutil.rmtree(voice_machine.LOCK, ignore_errors=True)
            self.owner = ""

    def release(self, why: str) -> bool:
        """Unload now if loaded (another job needs the GPU): True when it was loaded."""

        with self.mutex:
            if self.process is None:
                return False
            self.stop(why)
            return True



LIVE: dict[str, dict] = {}  # a running take's raw PCM parts, as the engine writes them (the live WAV, /live)


def forget_live(job_id: str, folder: Path) -> None:
    LIVE.pop(job_id, None)
    shutil.rmtree(folder, ignore_errors=True)


def live_wav(entry: dict, rate: int = 44100, poll: float = 0.05):
    """A running take as one open-length WAV: each part's PCM as written; ends when done or 10 min idle."""

    import struct

    open_length = 0xFFFFFFFF - 36
    yield (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
           + b"data" + struct.pack("<I", open_length))
    quiet = time.time()
    for index, part in enumerate(entry["parts"]):
        sent = 0
        if entry.get("wav"):  # a growing WAV (the MLX engine's --stream): its samples start after its "data" header
            sent = -1
            while sent < 0:
                try:
                    with open(part, "rb") as handle:
                        head = handle.read(4096)
                except OSError:
                    head = b""
                at = head.find(b"data")
                if at >= 0 and len(head) >= at + 8:
                    sent = at + 8
                elif entry["done"] or time.time() - quiet > 600:
                    return
                else:
                    time.sleep(poll)
        start = sent
        while True:
            try:
                with open(part, "rb") as handle:
                    handle.seek(sent)
                    data = handle.read()
            except OSError:
                data = b""
            data = data[: len(data) - (len(data) % 2)] if entry["finished"] <= index else data
            if data:
                if index and sent == start:
                    yield b"\0" * (int(rate * entry.get("gap_s", 0)) * 2)
                sent += len(data)
                quiet = time.time()
                yield data
                continue
            if entry["finished"] > index:
                break
            if entry["done"] or time.time() - quiet > 600:
                return
            time.sleep(poll)


def join_wavs(files: list[Path], target: Path, gap_s: float) -> None:
    import wave

    with wave.open(str(files[0])) as first:
        params = first.getparams()
    with wave.open(str(target), "wb") as out:
        out.setparams(params)
        silence = b"\0" * (int(params.framerate * gap_s) * params.sampwidth * params.nchannels)
        for index, one in enumerate(files):
            with wave.open(str(one)) as handle:
                if index:
                    out.writeframes(silence)
                out.writeframes(handle.readframes(handle.getnframes()))
