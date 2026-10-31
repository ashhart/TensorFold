"""Run and watch the TensorFold server: a child process with its log, or a server TensorFold Web only polls.

Two engines can be launched. The native one (TensorFold 1.0, `native.py`) serves the models qualified on it; the
frozen Python line (0.6.6) still serves the others until they are ported, and goes away then.
"""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import psutil

from . import engines, native
from . import settings as settings_mod
from .settings import Settings

LOG_LINES = 6000                      # a lesson writes about 1,500 lines; its reason must still be readable
LOG_MAX_BYTES = 20 * 1024**2          # the engine's log file starts over past this size
LINE_MAX = 8000                       # characters of one log line kept; the rest is cut
GIB = 1024**3
# the memory limit each backend reads, in GB (TensorFold's cuda/capacity.py and server/memory_budget.py)
LIMIT_ENV = {"mlx": "TENSORFOLD_MEMORY_LIMIT_GB", "cuda": "TENSORFOLD_CUDA_MEMORY_LIMIT_GB"}
# the engine's startup lines the page wants as numbers
_STARTUP = {
    "budget_gib": re.compile(r"memory budget ([\d.]+) GiB"),
    "resident_gib": re.compile(r"([\d.]+) GiB of weights kept resident"),
    "context_tokens": re.compile(r"context window ([\d,]+) tokens"),
    "loaded_s": re.compile(r"loaded in ([\d.]+)s"),
    "bytes_per_token": re.compile(r"then ([\d.]+) KB a token"),
}
# the CUDA server's cumulative counters that give rates between two polls
COUNTERS = ("completion_tokens_total", "decode_seconds_total", "prompt_tokens_total", "prefill_seconds_total",
            "drafted_total", "accepted_total", "prompt_tokens_cached_total")
# the KV cache fill on /metrics, as a ratio or (vLLM's name) a "perc" that is also a ratio on most builds
KV_FILL = ("kv_cache_usage_ratio", "kv_cache_usage_perc", "gpu_cache_usage_perc")
# the same figures under their Prometheus names on /metrics, which both servers keep behind the API key
METRICS = {"generation_tokens_total": "completion_tokens_total", "request_decode_seconds_sum": "decode_seconds_total",
           "prompt_tokens_total": "prompt_tokens_total", "request_prefill_seconds_sum": "prefill_seconds_total",
           "mtp_drafted_total": "drafted_total", "mtp_accepted_total": "accepted_total",
           "prompt_tokens_cached_total": "prompt_tokens_cached_total"}


@dataclass
class Health:
    ok: bool
    status: str                     # "stopped", "starting", "running", "unauthorized", "failed", "unreachable", "error"
    model: str | None = None
    model_ids: list[str] | None = None
    warming: bool = False
    connections: int = 0
    waiting: int = 0
    decode_tps: float | None = None     # None: this engine gives no live figure (the page shows a dash, not 0)
    prefill_tps: float | None = None
    memory: dict | None = None
    context_length: int | None = None   # the CUDA server reports it in /health
    counters: dict[str, float] | None = None   # cumulative totals (CUDA /health, or /metrics), for rates between polls
    draft_acceptance: float | None = None      # drafted tokens kept since the engine started, when it counts them
    kv_fill: float | None = None               # share of the KV cache held by requests and kept prompts, 0 to 1
    streams: dict[str, int] | None = None      # requests decoding, prefilling and paused, when the engine says it
    kept_prompts: int | None = None            # prompt states kept for reuse (the CUDA server)
    preemptions: float | None = None           # requests set aside to free memory, since the engine started
    prompt_cache: dict | None = None           # prompt tokens restored from kept states, of all prompt tokens (1.0.5)
    totals: dict | None = None                 # tokens and requests since the engine started, for the usage history
    source: str = "health"                     # "health", or "metrics" when /health is bare behind an API key
    load: dict | None = None                   # TensorFold's GET /v1/load: model, parked, lanes and the queue
    parked: bool = False                       # the idle unload has closed the engine; the next request reloads it
    raw: dict | None = None
    error: str | None = None
    generation: int | None = None              # the engine target it was read under (Engine.target)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "generation"}


def _num(body: dict, *keys: str) -> float | None:
    for k in keys:
        v = body.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def parse_health(body: dict) -> Health:
    """The MLX server's /health (status, model, live, memory) and the CUDA server's counters, read the same way.

    A status other than "ok" is an error state, whatever else the body says. A rate is a number only when the body
    carries a live reading; the CUDA server's totals are kept for the caller to turn into rates; anything else stays
    None. A missing model stays None and is fetched from /v1/models by the caller (the MLX server answers a bare
    {"status": "ok"} once an API key is on).
    """

    status = body.get("status", "ok")
    if status != "ok" and body.get("ok") is not True:
        return Health(ok=False, status="error", raw=body, error=f"engine reports status {status!r}")
    live = body.get("live") if isinstance(body.get("live"), dict) else None
    ctx = _num(body, "context_length", "max_model_len")
    counters = {k: float(body[k]) for k in COUNTERS if isinstance(body.get(k), (int, float))} or None
    return Health(ok=True, status="starting" if body.get("warming") else "running", model=body.get("model"),
                  model_ids=body.get("model_ids"), warming=bool(body.get("warming")),
                  connections=int(_num(live or {}, "connections") or _num(body, "requests_running") or 0),
                  waiting=int(_num(live or {}, "waiting") or _num(body, "requests_waiting") or 0),
                  decode_tps=_num(live, "decode_tokens_per_second") if live else None,
                  prefill_tps=_num(live, "prefill_tokens_per_second") if live else None,
                  memory=body.get("memory") or None, context_length=int(ctx) if ctx else None,
                  counters=counters, kv_fill=kv_fill(body), streams=streams_of(body),
                  kept_prompts=int(body["kept_prompts"]) if isinstance(body.get("kept_prompts"), int) else None,
                  raw=body)


def parse_load(body: object) -> dict | None:
    """TensorFold's `GET /v1/load` (zig/src/server/load_metrics.zig): the served model, whether the idle unload has
    parked the engine, its lanes and, when the engine reports them, its running and waiting requests. None when the
    body is not that object."""

    if not isinstance(body, dict) or not isinstance(body.get("model"), str) or not isinstance(body.get("parked"), bool):
        return None
    out: dict = {"model": body["model"], "parked": body["parked"]}
    for k in ("lanes", "running", "waiting"):
        v = body.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            out[k] = v
    return out


def parse_metrics(text: str) -> dict[str, float]:
    """TensorFold's Prometheus page as numbers: the unlabelled `tensorfold:` series, the prefix dropped."""

    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("tensorfold:"):
            continue
        name, _, value = line[len("tensorfold:"):].partition(" ")
        if "{" in name:
            # a labelled series (per key, per stream, histogram buckets): only the KV fill is wanted, its largest pool
            base = name.split("{", 1)[0]
            if base in KV_FILL:
                with contextlib.suppress(ValueError):
                    out[base] = max(out.get(base, 0.0), float(value.strip()))
            continue
        with contextlib.suppress(ValueError):
            out[name] = float(value.strip())
    return out


def kv_fill(body: dict, m: dict[str, float] | None = None) -> float | None:
    """The share of the KV cache held, 0 to 1: the CUDA server's pool in /health, else the /metrics gauge."""

    pool, free = _num(body, "pool_tokens"), _num(body, "pool_free_tokens")
    if pool and free is not None:
        return min(1.0, max(0.0, 1 - free / pool))
    for k in KV_FILL:
        if m and k in m:
            v = m[k]
            return min(1.0, max(0.0, v / 100 if v > 1 else v))
    return None


def streams_of(body: dict) -> dict[str, int] | None:
    """Requests by phase, when the engine says it: writing, reading their prompt, set aside."""

    st = body.get("streams") if isinstance(body.get("streams"), dict) else None
    if not st:
        return None
    pick = {"decoding": ("decoding",), "prefilling": ("prefilling", "filling"), "paused": ("paused",)}
    out = {k: int(next((st[n] for n in names if isinstance(st.get(n), (int, float))), 0)) for k, names in pick.items()}
    return out


def rates_from_counters(previous: dict[str, float] | None, current: dict[str, float]) -> tuple[float | None, float | None]:
    """Decode and prefill rates between two readings of the CUDA server's totals: tokens over the engine's own seconds.

    No previous reading gives no rate. Nothing new since the last reading is a true zero. Tokens without time (a
    counter the server does not keep) stays unknown.
    """

    if previous is None:
        return None, None

    def rate(tokens: str, seconds: str) -> float | None:
        if tokens not in current or tokens not in previous:
            return None
        dtok = current[tokens] - previous[tokens]
        dsec = current.get(seconds, 0.0) - previous.get(seconds, 0.0)
        if dtok <= 0:
            return 0.0
        return dtok / dsec if dsec > 0 else None

    def read(tokens: str, seconds: str, restored: str) -> float | None:
        # tokens restored from a kept prompt state take no prefill time: counting them would inflate the rate
        r = rate(tokens, seconds)
        if not r or restored not in current or restored not in previous:
            return r
        dtok = current[tokens] - previous[tokens] - (current[restored] - previous[restored])
        dsec = current.get(seconds, 0.0) - previous.get(seconds, 0.0)
        return max(0.0, dtok) / dsec if dsec > 0 else None

    return rate("completion_tokens_total", "decode_seconds_total"), \
        read("prompt_tokens_total", "prefill_seconds_total", "prompt_tokens_cached_total")


def totals_of(m: dict[str, float]) -> dict | None:
    """Prompt and written tokens, prompt tokens restored, and requests answered since the engine started."""

    if "generation_tokens_total" not in m and "prompt_tokens_total" not in m:
        return None
    return {"prompt": int(m.get("prompt_tokens_total", 0)), "completion": int(m.get("generation_tokens_total", 0)),
            "cached": int(m.get("prompt_tokens_cached_total", 0)), "requests": int(m.get("request_latency_seconds_count", 0))}


def prompt_cache_of(m: dict[str, float]) -> dict | None:
    """Prompt tokens restored from kept prompt states since the engine started, of all its prompt tokens."""

    if "prompt_tokens_cached_total" not in m:
        return None
    return {"restored": int(m["prompt_tokens_cached_total"]), "prompt": int(m.get("prompt_tokens_total", 0))}


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


class _Child:
    """The engine process: one this TensorFold Web started, whose exit code it learns, or one a previous TensorFold Web left running,
    found again by its pid and start time (a pid alone may have been given to another process since)."""

    def __init__(self, popen: subprocess.Popen | None = None, found: psutil.Process | None = None) -> None:
        self.popen, self.found = popen, found
        self.pid = popen.pid if popen else found.pid if found else 0

    def alive(self) -> bool:
        if self.popen is not None:
            return self.popen.poll() is None
        try:
            return self.found is not None and self.found.is_running() and self.found.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    @property
    def returncode(self) -> int | None:
        return self.popen.returncode if self.popen is not None else None     # unknown for a process found again

    def terminate(self) -> None:
        with contextlib.suppress(ProcessLookupError, psutil.Error):
            (self.popen or self.found).terminate()

    def kill(self) -> None:
        with contextlib.suppress(ProcessLookupError, psutil.Error):
            (self.popen or self.found).kill()

    def wait(self, timeout: float) -> None:
        """Waits for the end; raises subprocess.TimeoutExpired as Popen does."""

        if self.popen is not None:
            self.popen.wait(timeout=timeout)
            return
        try:
            self.found.wait(timeout=timeout)
        except psutil.TimeoutExpired as exc:
            raise subprocess.TimeoutExpired("engine", timeout) from exc
        except psutil.Error:
            pass


def _redrawn(line: str) -> str:
    """A progress bar redraws its line with carriage returns: what shows is the text after the last one."""

    line = line.rstrip("\r")
    return line.rsplit("\r", 1)[-1][:LINE_MAX]


class Engine:
    """One engine: a process TensorFold Web owns, or a server it only watches (its own or a remote one).

    An owned engine writes its log to a file and runs in its own session, so it outlives TensorFold Web: a TensorFold Web that
    restarts finds it again from ``engine.json`` in ``state_dir`` and reads its log on from the file.
    """

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
        self._client = httpx.Client(timeout=2.0)
        self.last_health: Health | None = None
        self.command: list[str] | None = None
        self.startup: dict[str, float] = {}      # numbers read from the engine's startup lines
        # the CUDA server's totals at each reader's previous poll: the page and the history poll at their own pace, and
        # a rate covers the time since the same reader last asked
        self._counters: dict[str, dict[str, float]] = {}
        self.kind: str | None = None             # "native" or "python": the engine an owned process runs
        self._stopping = False                   # a stop was asked: its exit code is not a failure
        self._rates_lock = threading.Lock()      # the counters' baseline, moved by one caller at a time
        self._window_asked: int | None = None    # the generation /tokenize was asked under, once per target
        # the engine's address and key change together, under this lock; each change is a new generation, and a
        # reading begun under an older one is read again rather than let out (see health())
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
        """Start ``model``: on the native engine when it is qualified there and installed, else on the Python line.

        ``serve``: what the engine is given to serve when it is not ``model`` itself (a living copy's folder);
        ``slide``: the fact graph file of a copy that learns."""

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
            # appended to, so the file can start over under the engine (see _pump): with "wb" the engine would go on
            # writing at its old offset into a sparse file
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
        """Follows the log file while the engine runs; ``replay`` first reads what it wrote before this TensorFold Web.

        Once another engine has been started in its place, this one's lines and exit go nowhere: a reader that ends
        late never marks the new engine as exited."""

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
                # the engine appends (O_APPEND on POSIX), so the file can start over under it; on Windows an inherited
                # handle keeps its offset, which no engine there needs today: to revisit with a Windows engine
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
            # a binary without a SIGTERM handler exits -15 on the stop we asked for, and one that took too long to
            # stop was killed by stop() itself (-9): either is a stop, not a failure
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
        # the target read once, address and key together: a change while this poll runs must not send the new key to
        # the old engine, nor let the old engine's answer stand for the new one
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
        # published under the target lock, and only if the target is still the one read: a change of engine at any
        # moment of this poll leaves nothing of the old engine behind (rates, the latest reading)
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
        # the reading and the settings described together, under the target lock: an engine replaced at any moment
        # after the reading is never described with the old engine's model or window
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
        """This machine's memory by holder: the engine, everything else, and what is left, with the headroom level.

        The engine's share is what it reports (MLX's active memory, or its process footprint), else the resident size
        of the process this TensorFold Web started. Headroom is judged in GB rather than as a share: 3 GB free is tight on
        any machine, whatever its size.
        """

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
