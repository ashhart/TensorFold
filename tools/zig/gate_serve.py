"""Start, watch and stop one gate server: a process under the caller's GPU lock, or a container on a CUDA box."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import qual_run

KNOBS = ("TENSORFOLD_", "TF_")   # switch.KNOBS: the runner may lack the install, and a stray one blocks --engine zig
STOP_WAIT_S = 90                 # past the grace, the qualification's own wait before SIGKILL
GIB = 1024**3


def serve_argv(plan: dict, engine: str) -> list[str]:
    """`tensorfold serve` for one arm: the plan's flags, the same for both engines, and --engine."""

    return [plan["python"], "-B", "-m", "tensorfold", "serve", plan["model_dir"], "--name", plan["served"],
            "--host", "127.0.0.1", "--port", str(plan["port"]), "--engine", engine, *plan["args"]]


def variables(plan: dict) -> dict[str, str]:
    """What a plan sets in the server's environment: PYTHONPATH for a source tree, then the plan's own."""

    found = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    if plan.get("src"):
        found["PYTHONPATH"] = plan["src"]
    return {**found, **{key: str(value) for key, value in plan["env"].items()}}


def ready(port: int, alive, timeout: float) -> float | None:
    """Seconds until /v1/models answers; None when the server exits or the time runs out."""

    started = time.perf_counter()
    while time.perf_counter() - started < timeout and alive():
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2):
                return time.perf_counter() - started
        except Exception:  # noqa: BLE001 - not listening yet
            time.sleep(0.5)
    return None


def launcher(plan: dict) -> "Local | Docker":
    return Docker(plan) if plan["launch"]["kind"] == "docker" else Local(plan)


class Local:
    """A server in this process group, so the caller's gpu_run counts its footprint against the cap."""

    def __init__(self, plan: dict) -> None:
        self.plan, self.proc, self.tripped = plan, None, None

    def environment(self) -> dict[str, str]:
        kept = {key: value for key, value in os.environ.items() if not key.startswith(KNOBS)}
        return {**kept, **variables(self.plan)}

    def start(self, engine: str, log) -> None:
        self.proc = subprocess.Popen(serve_argv(self.plan, engine), env=self.environment(), stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def pid(self) -> int | None:
        return self.proc.pid if self.proc is not None else None

    def state(self) -> dict | None:
        """The qualification's snapshot of heavy processes and memory pressure (macOS), to explain a slow visit."""

        return qual_run.system_state() if sys.platform == "darwin" else None

    def owns_port(self) -> bool:
        return bool(qual_run.listeners(self.plan["port"]) & {self.proc.pid, *qual_run.descendants(self.proc.pid)})

    def stop(self, grace: float) -> tuple[int | None, float]:
        """(exit status, seconds from SIGTERM to exit); SIGKILL only for a server still running past the wait."""

        if self.proc is None:
            return None, 0.0
        began = time.perf_counter()
        _signal([self.proc.pid, *qual_run.descendants(self.proc.pid)], signal.SIGTERM)
        try:
            self.proc.wait(timeout=grace + STOP_WAIT_S)
        except subprocess.TimeoutExpired:
            _signal([self.proc.pid, *qual_run.descendants(self.proc.pid)], signal.SIGKILL)
        took = time.perf_counter() - began
        return self.proc.wait(timeout=30), took

    def tool(self, argv: list[str]) -> subprocess.CompletedProcess:
        """A short command in this arm's install and environment (the cell's identity, its gate entry)."""

        return subprocess.run([self.plan["python"], "-B", *argv], env=self.environment(), capture_output=True,
                              text=True, timeout=300, stdin=subprocess.DEVNULL)


def _signal(pids: list[int], number: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, number)
        except ProcessLookupError:
            pass


class Docker:
    """A server container with the CUDA box harness's limits and checks; the caller holds the box's GPU lock."""

    def __init__(self, plan: dict) -> None:
        self.plan, self.spec, self.proc, self.tripped = plan, plan["launch"], None, None
        self.name = self.spec["name"]

    def _argv(self, inner: list[str], served: bool) -> list[str]:
        spec, argv = self.spec, ["docker", "run", "--rm", "--gpus", "all", "--network", "host", "--ipc", "host"]
        if served:
            argv += ["--name", self.name, "--label", f"tensorfold.gate={self.name}", "--log-driver", "none",
                     "--oom-score-adj", "1000", "--memory", spec["memory"], "--memory-swap", spec["memory"]]
        argv += ["--read-only", "--tmpfs", "/tmp:size=4g", "-w", spec.get("workdir", "/")]
        for mount in spec.get("mounts", []):
            argv += ["-v", mount]
        for key, value in {**variables(self.plan), **spec.get("env", {})}.items():
            argv += ["-e", f"{key}={value}"]
        return [*argv, *spec.get("docker_args", []), "--entrypoint", "timeout", spec["image"],
                str(spec.get("timeout", 3600) if served else 300), *inner]

    def preflight(self) -> None:
        """The box harness's rule: no container, no compute app but the allowed ones, and an idle GPU."""

        if _run(["docker", "ps", "-q"]).stdout.split():
            raise RuntimeError("preflight: containers are running; nothing started")
        apps = _run(["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"]).stdout.split("\n")
        foreign = [app.strip() for app in apps if app.strip() and app.strip() not in self.spec.get("allow_apps", [])]
        if foreign:
            raise RuntimeError(f"preflight: compute apps present ({', '.join(foreign)}); nothing started")
        for _ in range(300):
            busy = _run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).stdout
            if max((float(x) for x in busy.split()), default=0.0) <= 10:
                return
            time.sleep(1)
        raise RuntimeError("preflight: GPU busy for 300 s")

    def start(self, engine: str, log) -> None:
        self.preflight()
        self.proc = subprocess.Popen(self._argv(serve_argv(self.plan, engine), True), stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        threading.Thread(target=self._guard, daemon=True).start()

    def _guard(self) -> None:
        floor = float(self.spec.get("min_available_gib", 6.0)) * GIB
        while self.alive():
            if _available() < floor:
                self.tripped = f"memory guard: MemAvailable under {floor / GIB:.1f} GiB"
                _run(["docker", "stop", "--time", "15", self.name], timeout=60)
                return
            time.sleep(0.5)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def pid(self) -> None:
        return None                     # the server runs in the container: no footprint from here

    def state(self) -> dict:
        """MemAvailable and GPU use when the visit began or ended, as the box harness samples them."""

        busy = _run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,clocks.sm,power.draw",
                     "--format=csv,noheader,nounits"]).stdout.strip()
        return {"time": time.time(), "mem_available_gib": _available() / GIB, "gpu": busy}

    def owns_port(self) -> bool:
        state = _run(["docker", "inspect", "--format", "{{.State.Running}}", self.name])
        return state.returncode == 0 and state.stdout.strip() == "true"

    def stop(self, grace: float) -> tuple[int | None, float]:
        if self.proc is None:
            return None, 0.0
        began = time.perf_counter()
        owner = _run(["docker", "inspect", "--format", '{{index .Config.Labels "tensorfold.gate"}}', self.name])
        if owner.returncode == 0 and owner.stdout.strip() == self.name:
            _run(["docker", "stop", "--time", str(int(grace)), self.name], timeout=grace + STOP_WAIT_S)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        took = time.perf_counter() - began
        if _run(["docker", "inspect", self.name]).returncode == 0:
            raise RuntimeError(f"container {self.name} is still present after its stop")
        return self.proc.wait(), took

    def tool(self, argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(self._argv([self.plan["python"], "-B", *argv], False), capture_output=True, text=True,
                              timeout=600, stdin=subprocess.DEVNULL)


def _run(argv: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def _available() -> int:
    """MemAvailable in bytes (Linux)."""

    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0
