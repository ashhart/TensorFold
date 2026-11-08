"""Run and watch the TensorFold server: a child process with its log, or a server TensorFold Web only polls."""


from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import httpx
import psutil

from . import engines, native
from . import settings as settings_mod
from .engine_child import (
    _Child,
)
from .engine_metrics import (
    COUNTERS,
    KV_FILL,
    METRICS,
    Health,
    _num,
    kv_fill,
    parse_health,
    parse_load,
    parse_metrics,
    prompt_cache_of,
    rates_from_counters,
    streams_of,
    totals_of,
)
from .settings import Settings

LOG_LINES = 6000                      # a lesson writes about 1,500 lines; its reason must still be readable

LOG_MAX_BYTES = 20 * 1024**2          # the engine's log file starts over past this size

LINE_MAX = 8000                       # characters of one log line kept; the rest is cut

GIB = 1024**3

LIMIT_ENV = {"mlx": "TENSORFOLD_MEMORY_LIMIT_GB", "cuda": "TENSORFOLD_CUDA_MEMORY_LIMIT_GB"}

_STARTUP = {
    "budget_gib": re.compile(r"memory budget ([\d.]+) GiB"),
    "resident_gib": re.compile(r"([\d.]+) GiB of weights kept resident"),
    "context_tokens": re.compile(r"context window ([\d,]+) tokens"),
    "loaded_s": re.compile(r"loaded in ([\d.]+)s"),
    "bytes_per_token": re.compile(r"then ([\d.]+) KB a token"),
}

def tensorfold_command() -> list[str] | None:
    """How to launch the frozen Python line: its `tensorfold` on PATH, or this interpreter's module beside us."""

    exe = shutil.which("tensorfold")
    if exe and not native.version_of(exe):         # Homebrew's `tensorfold` is the native engine, not this one
        return [exe]
    try:
        import tensorfold  # noqa: F401
    except ImportError:
        return None
    return [sys.executable, "-m", "tensorfold"]

def memory_limit_env(fraction: float) -> dict[str, str]:
    """The memory limit for this machine's backend: the fraction of what that backend can give an engine."""

    from .catalog import capacity_of
    from .machine import detect

    m = detect()
    cap = capacity_of(m)
    key = LIMIT_ENV.get(m.backend)
    if cap is None or key is None:
        return {}
    return {key: f"{cap.bytes * float(fraction) / GIB:.1f}"}

def build_command(model: str, s: Settings, *, host: str = "127.0.0.1") -> tuple[list[str], dict]:
    """The `tensorfold serve` command and environment for the settings; the port is the one TensorFold Web watches."""

    base = tensorfold_command()
    if base is None:
        raise RuntimeError("TensorFold is not installed in this environment")
    cmd = base + ["serve", model, "--host", "0.0.0.0" if s.share else host, "--port", str(s.engine_port())]
    if s.context:
        cmd += ["--context", str(int(s.context))]
    if s.thinking == "on":
        cmd.append("--thinking")
    elif s.thinking == "off":
        cmd.append("--no-thinking")
    kept, _ = native.safe_extra(list(s.extra_args))         # saved before the check existed, or by hand
    cmd += kept
    env = dict(os.environ)
    if s.budget_fraction:
        env.update(memory_limit_env(s.budget_fraction))
    if s.api_key:
        env["TENSORFOLD_API_KEY"] = s.api_key        # the environment, never the command line or the log
    return cmd, env

def _redrawn(line: str) -> str:
    """A progress bar redraws its line with carriage returns: what shows is the text after the last one."""

    line = line.rstrip("\r")
    return line.rsplit("\r", 1)[-1][:LINE_MAX]

class Engine:
    """One engine: a process TensorFold Web owns, or a server it only watches (its own or a remote one)."""

    def __init__(self, settings: Settings, state_dir: Path | None = None) -> None:
        self.settings = settings
        self.state_dir = state_dir
        self.proc: _Child | None = None
        self.model: str | None = None            # the repository id an owned process was started with
        self.active_key: str | None = None       # the key an owned process was started with; settings hold the next one
        self.started_at: float | None = None
        self.exit_code: int | None = None
        self.log: deque[str] = deque(maxlen=LOG_LINES)
        self._reader: threading.Thread | None = None   # follows the current engine's log
        self._subscribers: list[Any] = []
        self._lock = threading.Lock()
        self._client = httpx.Client(trust_env=False, timeout=2.0)
        self.last_health: Health | None = None
        self.command: list[str] | None = None
        self.startup: dict[str, float] = {}      # numbers read from the engine's startup lines
        # Each poll reader owns its counter baseline so independent polling rates remain meaningful.
        self._counters: dict[str, dict[str, float]] = {}
        self.mock_version: str | None = None
        self.kind: str | None = None             # "native" or "python": the engine an owned process runs
        self._stopping = False                   # a stop was asked: its exit code is not a failure
        self._rates_lock = threading.Lock()      # the counters' baseline, moved by one caller at a time
        self._window_asked: int | None = None    # the generation /tokenize was asked under, once per target
        # Target generations prevent a reading begun before an address or key change from escaping afterward.
        self.target_lock = threading.Lock()
        self.generation = 0

    # -- lifecycle ---------------------------------------------------------------------
    @property
    def owned(self) -> bool:
        return self.proc is not None and self.proc.alive()

    # -- the engine across TensorFold Web restarts --------------------------------------------------
    def _state_path(self) -> Path | None:
        return self.state_dir / "engine.json" if self.state_dir else None

    def _log_path(self) -> Path:
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            return self.state_dir / "engine.log"
        fd, name = tempfile.mkstemp(prefix="tensorfold-engine-", suffix=".log")
        os.close(fd)
        return Path(name)

    def _save_state(self, log: Path) -> None:
        path = self._state_path()
        if path is None or self.proc is None:
            return
        try:
            created = psutil.Process(self.proc.pid).create_time()
        except psutil.Error:
            return
        doc = {"pid": self.proc.pid, "create_time": created, "model": self.model, "kind": self.kind,
               "command": self.command, "active_key": self.active_key, "started_at": self.started_at, "log": str(log)}
        settings_mod.write_private(path, json.dumps(doc))      # it holds the engine's key, like the settings file

    def _forget_state(self, pid: int) -> None:
        path = self._state_path()
        if path is None:
            return
        with contextlib.suppress(OSError, ValueError):
            if json.loads(path.read_text()).get("pid") == pid:
                path.unlink()

    def adopt(self) -> bool:
        """Finds the engine a previous TensorFold Web started, if it still runs; returns whether one was found."""

        path = self._state_path()
        if path is None or self.owned:
            return False
        try:
            doc = json.loads(path.read_text())
            found = psutil.Process(int(doc["pid"]))
            same = abs(found.create_time() - float(doc["create_time"])) < 1.0 and found.is_running()
        except (OSError, ValueError, KeyError, TypeError, psutil.Error):
            same = False
        if not same:
            with contextlib.suppress(OSError):
                path.unlink()
            return False
        with self._lock:
            self.proc = _Child(found=found)
            self.model, self.kind, self.command = doc.get("model"), doc.get("kind"), doc.get("command")
            self.active_key, self.started_at = doc.get("active_key"), doc.get("started_at")
            self.exit_code = None
            log = Path(doc.get("log") or self._log_path())
            self.log.clear()
            self._reader = threading.Thread(target=self._pump, args=(self.proc, log, True), name="engine-log", daemon=True)
            self._reader.start()
        return True

    def key_in_use(self) -> str | None:
        """The key that opens the engine now: an owned process keeps its own until it restarts."""

        return self.active_key if self.owned else self.settings.api_key

    def key_pending_restart(self) -> bool:
        return self.owned and self.active_key != self.settings.api_key

    def start(self, model: str, *, native_model: bool = False, drafter: str | None = None, slide: str | None = None,
              serve: str | None = None) -> dict:
        """Start ``model``: on the native engine when it is qualified there and installed, else on the Python line."""

        # the address probed before the lock (up to a few seconds over HTTP): the log reader needs the lock meanwhile
        answers = not self.owned and self.health().ok
        with self._lock:
            self._stopping = False
            if self.owned:
                raise RuntimeError(f"an engine is already running ({self.model}); stop it first")
            if answers:
                raise RuntimeError(f"a server already answers at {self.settings.engine_url}; stop it or change the address")
            launch = engines.choose(native_model).launch(serve or model, self.settings, drafter, slide)
            cmd, env, notes = launch.cmd, launch.env, launch.notes
            self.command = cmd
            self.kind = launch.kind
            self.log.clear()
            self.startup = {}
            self._counters = {}
            self._window_asked = None
            self.exit_code = None
            self._emit(f"[web] $ {' '.join(cmd)}")
            for note in notes:
                self._emit(f"[web] {note}")
            for key in LIMIT_ENV.values():
                if key in env:
                    self._emit(f"[web] {key}={env[key]}")
            if "TENSORFOLD_API_KEY" in env:
                self._emit("[web] TENSORFOLD_API_KEY set (not shown)")
            # the log goes to a file, not a pipe: a pipe would break, and stop the engine, when TensorFold Web stops
            log = self._log_path()
            flags = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
                     if os.name == "nt" else 0)
            # Append mode permits log rotation without leaving a sparse hole at the engine's old offset.
            log.write_bytes(b"")
            with contextlib.suppress(OSError):
                os.chmod(log, 0o600)                    # what the engine logs (learned facts too) is this user's alone
            with open(log, "ab") as out:
                popen = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                         start_new_session=(os.name != "nt"), creationflags=flags)
            self.proc = _Child(popen=popen)
            self.model = model
            self.active_key = env.get("TENSORFOLD_API_KEY")
            self.started_at = time.time()
            self._save_state(log)
            self._reader = threading.Thread(target=self._pump, args=(self.proc, log, False), name="engine-log",
                                            daemon=True)
            self._reader.start()
        return self.status().to_dict()

    def stop(self, *, timeout: float = 20.0) -> dict:
        with self._lock:
            proc = self.proc
        if proc and proc.alive():
            self._emit("[web] stopping the engine")
            self._stopping = True
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._emit("[web] the engine did not stop in time; killing it")
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:    # stuck in the kernel (a driver, a hung mount): said, not a 500
                    self._emit("[web] the engine does not die; it may be stuck in the kernel (a GPU driver, a mount)")
        return self.status().to_dict()

    def target(self) -> tuple[int, str, str | None]:
        """The generation, address and key, read together."""

        with self.target_lock:
            return self.generation, self.settings.engine_url.rstrip("/"), self.key_in_use()

    def retargeted(self) -> None:
        """The address or the key changed (the caller holds ``target_lock``): readings begun before are not used."""

        self.generation += 1

    def reset_target(self) -> None:
        """The engine's address or mode changed: forget what was read from the previous target."""

        self.startup = {} if not self.owned else self.startup
        self.last_health = None
        self._counters = {}
        self._window_asked = None

    def _pump(self, child: _Child, log: Path, replay: bool) -> None:

        def say(line: str, **kw: bool) -> None:
            with self._lock:                         # the check and the write as one step against a replacement
                if child is self.proc:
                    self._emit(line, **kw)

        try:
            f = open(log, encoding="utf-8", errors="replace", newline="")  # noqa: SIM115 - kept open while following
        except OSError:
            f = None
        buf = ""
        if f is not None and replay:
            size = os.path.getsize(log)
            if size > 1024**2:                       # only the tail: the file may be up to LOG_MAX_BYTES
                f.seek(size - 1024**2)
                f.readline()                         # from the next whole line
            for line in f.read().split("\n")[:-1][-LOG_LINES:]:
                say(_redrawn(line), earlier=True)
            say(f"[web] found the engine a previous TensorFold Web started (pid {child.pid}); it kept running")
        elif f is not None:
            f.seek(0)
        while f is not None:
            chunk = f.read(65536)
            if chunk:
                buf += chunk
                *lines, buf = buf.split("\n")
                for line in lines:
                    say(_redrawn(line))
                if len(buf) > LINE_MAX:                  # a line that never ends, such as a progress bar
                    buf = _redrawn(buf)[-LINE_MAX:]
                continue
            if not child.alive():
                rest = f.read()
                for line in (buf + rest).split("\n"):
                    if line.strip():
                        say(_redrawn(line))
                break
            with contextlib.suppress(OSError):
                # POSIX O_APPEND permits truncation; review inherited offsets before supporting Windows engines.
                if os.path.getsize(log) > LOG_MAX_BYTES:
                    os.truncate(log, 0)
                    f.seek(0)
            time.sleep(0.2)
        if f is not None:
            f.close()
        if child.popen is not None:
            child.popen.wait()
        with self._lock:
            if child is not self.proc:
                return
            # SIGTERM and a kill requested by stop() are deliberate stops, not engine failures.
            self.exit_code = 0 if self._stopping and child.returncode in (-15, 143, -9, 137) else child.returncode
            self._stopping = False
            self._forget_state(child.pid)
            self._emit(f"[web] engine exited with code {self.exit_code}" if self.exit_code is not None
                       else "[web] the engine stopped")

    def note(self, text: str) -> None:
        """A line of TensorFold Web's own in the engine log."""

        self._emit(f"[web] {text}")

    def wait_stopped(self, seconds: float) -> bool:
        """Wait for a process that is exiting by itself to be gone and its log read to the end; True once both are."""

        end = time.time() + seconds
        while self.owned and time.time() < end:
            time.sleep(0.1)
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(max(0.0, end - time.time()))
        return not self.owned and not (reader and reader.is_alive())

    def _emit(self, line: str, *, earlier: bool = False) -> None:
        # a line read back from before this TensorFold Web has no time of its own
        stamped = f"{'--:--:--' if earlier else time.strftime('%H:%M:%S')} {line}"
        self.log.append(stamped)
        for key, rx in _STARTUP.items():
            m = rx.search(line)
            if m:
                value = float(m.group(1).replace(",", ""))
                self.startup[key] = value * 1000 if key == "bytes_per_token" else value
        self.startup.update(native.parse_startup(line))
        for q in list(self._subscribers):
            with contextlib.suppress(Exception):      # a full or closed queue drops the line for that subscriber
                q.put_nowait(stamped)

    def subscribe(self, q: Any) -> None:
        self._subscribers.append(q)

    def unsubscribe(self, q: Any) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    # -- health ------------------------------------------------------------------------
    def health(self, _depth: int = 0, reader: str = "page") -> Health:
        # Read the address and key together so target changes cannot leak a key or publish the wrong reply.
        gen, base, key = self.target()
        hdrs = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            r = self._client.get(base + "/health", headers=hdrs)
            r.raise_for_status()
            h = parse_health(r.json())
        except Exception as exc:  # noqa: BLE001
            h = Health(ok=False, status="unreachable", error=str(exc)[:200])
        bare = h.ok and not h.model and h.raw is not None and "live" not in h.raw and h.counters is None
        if h.ok:
            # the served model and its load from the server itself: /health is bare behind a key, /v1/load is not
            with contextlib.suppress(Exception):
                r = self._client.get(base + "/v1/load", headers=hdrs)
                if r.status_code == 200 and (ld := parse_load(r.json())) is not None:
                    h.load, h.parked = ld, ld["parked"]
                    if not h.model and ld["model"]:
                        h.model = ld["model"]
                        h.model_ids = h.model_ids or [ld["model"]]
                elif r.status_code == 401:
                    h.status = "unauthorized"           # the server answers; TensorFold Web's key does not open it
                    h.error = "the engine refuses the API key TensorFold Web holds" if key else \
                        "the engine requires an API key TensorFold Web does not have"
        if h.ok and not h.model and h.status != "unauthorized":
            with contextlib.suppress(Exception):      # an engine without /v1/load: /v1/models, a miss keeps None
                r = self._client.get(base + "/v1/models", headers=hdrs)
                if r.status_code == 200:
                    ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
                    h.model_ids = ids
                    h.model = ids[0] if ids else None
                elif r.status_code == 401:
                    h.status = "unauthorized"           # the server answers; TensorFold Web's key does not open it
                    h.error = "the engine refuses the API key TensorFold Web holds" if key else \
                        "the engine requires an API key TensorFold Web does not have"
        if bare and h.status != "unauthorized" and key:
            with contextlib.suppress(Exception):      # a bare /health behind a key: the figures live on /metrics
                r = self._client.get(base + "/metrics", headers=hdrs)
                if r.status_code == 200:
                    self._read_metrics(h, parse_metrics(r.text))
        if bare and h.load and "running" in h.load:   # the ledger's queue, when it gives one, over the scrape's
            h.connections = h.load["running"] + h.load.get("waiting", 0)
            h.waiting = h.load.get("waiting", 0)
        if h.ok and not bare and h.status != "unauthorized" and h.source != "metrics":
            # what /health leaves out: the Mac's memory and drafts, and on both engines the prompt cache (1.0.5)
            with contextlib.suppress(Exception):
                r = self._client.get(base + "/metrics", headers=hdrs)
                if r.status_code == 200:
                    m = parse_metrics(r.text)
                    if "process_footprint_bytes" in m and not h.memory:
                        h.memory = {"footprint": int(m["process_footprint_bytes"])}
                    h.kv_fill = h.kv_fill if h.kv_fill is not None else kv_fill({}, m)
                    h.preemptions = h.preemptions if h.preemptions is not None else m.get("preemptions_total")
                    h.prompt_cache = prompt_cache_of(m)
                    h.totals = totals_of(m)
                    if m.get("mtp_drafted_total") and h.draft_acceptance is None:
                        h.draft_acceptance = min(1.0, max(0.0, m.get("mtp_accepted_total", 0.0) / m["mtp_drafted_total"]))
        if h.ok and not h.context_length and "context_tokens" not in self.startup and self._window_asked != gen:
            self._window_asked = gen                  # the served window, once per target, when nothing else said it
            with contextlib.suppress(Exception):
                r = self._client.post(base + "/tokenize", json={"model": h.model or "", "prompt": ""},
                                      headers=hdrs)
                if r.status_code == 200 and isinstance(r.json().get("max_model_len"), int):
                    h.context_length = int(r.json()["max_model_len"])
        # Publish readings and rate baselines only while their target generation remains current.
        with self.target_lock:
            current = self.generation == gen
            if current:
                if h.ok and h.counters:
                    with self._rates_lock:           # the baseline is read and moved by one caller at a time
                        h.decode_tps, h.prefill_tps = rates_from_counters(self._counters.get(reader), h.counters)
                        self._counters[reader] = h.counters
                    drafted, accepted = h.counters.get("drafted_total"), h.counters.get("accepted_total")
                    if drafted:
                        h.draft_acceptance = min(1.0, max(0.0, (accepted or 0.0) / drafted))
                h.generation = gen
                self.last_health = h
        if not current:                               # the target changed meanwhile: this answer is the old engine's
            return self.health(_depth + 1, reader) if _depth < 3 else Health(ok=False, status="unreachable",
                                                           error="the engine's address keeps changing")
        return h

    @staticmethod
    def _read_metrics(h: Health, m: dict[str, float]) -> None:
        h.source = "metrics"
        found = {METRICS[k]: v for k, v in m.items() if k in METRICS}
        h.counters = found or None
        h.connections = int(m.get("num_requests_running", 0))
        h.waiting = int(m.get("num_requests_waiting", 0))
        if "process_footprint_bytes" in m:
            h.memory = {"footprint": int(m["process_footprint_bytes"])}
        h.kv_fill = kv_fill({}, m)
        h.preemptions = m.get("preemptions_total")
        h.prompt_cache = prompt_cache_of(m)
        h.totals = totals_of(m)

    def status(self, reader: str = "page") -> Health:
        """What TensorFold Web tells the UI: the process state first, then what /health says."""

        h = self.health(reader=reader)
        if self.settings.remote:
            return h
        if self.owned and not h.ok:
            h.status = "starting"
        elif not self.owned and self.proc is not None and self.exit_code not in (None, 0) and not h.ok:
            h.status = "failed"
            h.error = f"exit code {self.exit_code}"
        elif not self.owned and not h.ok and h.status != "error":
            h.status = "stopped"
        if self.owned and h.model is None:
            h.model = self.model
        return h

    def info(self, _depth: int = 0, reader: str = "page") -> dict:
        st = self.status(reader)
        tf_available = native.find_binary() is not None or tensorfold_command() is not None   # file checks, no lock
        # Snapshot settings with the reading so a replaced engine cannot inherit its predecessor's model or window.
        with self.target_lock:
            stale = getattr(st, "generation", self.generation) != self.generation
            if not stale:
                if st.context_length:                 # the CUDA server says it on every poll; keep the latest
                    self.startup["context_tokens"] = float(st.context_length)
                if st.load and "lanes" in st.load:    # the server's own count, over any log line's
                    self.startup["streams"] = float(st.load["lanes"])
                h = st.to_dict()
                h.update({
                    "managed": not self.settings.remote, "owned": self.owned, "engine_url": self.settings.engine_url,
                    "model_repo": self.model if self.owned else None, "started_at": self.started_at,
                    "uptime_s": (time.time() - self.started_at) if self.started_at and self.owned else None,
                    "command": self.command, "tensorfold_available": tf_available,
                    "engine_kind": self.kind if self.owned else None, "startup": dict(self.startup),
                    "api_key_set": bool(self.settings.api_key), "key_pending_restart": self.key_pending_restart()})
        if stale:
            if _depth < 3:
                return self.info(_depth + 1, reader)
            h = Health(ok=False, status="unreachable", error="the engine's address keeps changing").to_dict()
            h.update({"managed": not self.settings.remote, "owned": self.owned, "engine_url": self.settings.engine_url})
        if not self.settings.remote:
            h["machine_memory"] = self.machine_memory(h.get("memory"))
        return h

    def machine_memory(self, mem: dict | None) -> dict:
        """This machine's memory by holder: the engine, everything else, and what is left, with the headroom level."""

        vm = psutil.virtual_memory()
        engine = None
        for k in ("active", "footprint"):
            if mem and isinstance(mem.get(k), (int, float)) and mem[k] > 0:
                engine = int(mem[k])
                break
        if engine is None and self.owned and self.proc is not None:
            with contextlib.suppress(psutil.Error):
                engine = psutil.Process(self.proc.pid).memory_info().rss
        free = int(vm.available)
        warn, bad = max(4 * GIB, 0.10 * vm.total), max(2 * GIB, 0.05 * vm.total)
        return {"total": int(vm.total), "available": free, "engine": engine,
                "headroom": "low" if free < bad else "tight" if free < warn else "ok"}


__all__ = ['COUNTERS', 'GIB', 'KV_FILL', 'LIMIT_ENV', 'LINE_MAX', 'LOG_LINES', 'LOG_MAX_BYTES', 'METRICS', 'Engine', 'Health', '_Child', '_num', 'build_command', 'kv_fill', 'memory_limit_env', 'parse_health', 'parse_load', 'parse_metrics', 'prompt_cache_of', 'rates_from_counters', 'streams_of', 'tensorfold_command', 'totals_of']
