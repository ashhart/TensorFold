"""Tool calls, measured by `tool-eval-bench` (MIT, SeraphimSerapis), which TensorFold Web runs rather than rewrites.

It plays scripted conversations with mock tools against an OpenAI-compatible endpoint and scores each scenario as
pass, partial or fail; it already knows TensorFold. TensorFold Web finds it, runs it on the served engine (or another),
follows its progress events, and keeps its scenarios as ordinary Bench results: a pass counts as right, so two runs
compare scenario by scenario with McNemar's test like the quality suite. Install: `uv tool install tool-eval-bench==2.8.0`,
or the Bench's Install button, which runs the same with the version TensorFold Web's reading was checked with.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

CANDIDATES = ("~/.local/bin/tool-eval-bench", "~/peers/tool-eval-bench/bin/tool-eval-bench")
# where uv lives when a service's PATH leaves it out: its own installer, cargo, Homebrew
UV_CANDIDATES = ("~/.local/bin/uv", "~/.cargo/bin/uv", "/opt/homebrew/bin/uv", "/usr/local/bin/uv")
VERSION = "2.8.0"                  # the version TensorFold Web installs: its output read and checked on the three machines
INSTALL_TIMEOUT = 300
_installing = threading.Lock()
SEED = 42


def find() -> str | None:
    """`TOOL_EVAL_BENCH`, then PATH, then where `uv tool install` puts it."""

    for p in (os.environ.get("TOOL_EVAL_BENCH"), shutil.which("tool-eval-bench"),
              *(os.path.expanduser(c) for c in CANDIDATES)):
        if p and os.access(p, os.X_OK):
            return p
    return None


def install(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict:
    """`uv tool install tool-eval-bench==VERSION` for this user, no administrator rights; where it landed and its
    version, or a RuntimeError the page can show."""

    uv = shutil.which("uv") or next((p for p in (os.path.expanduser(c) for c in UV_CANDIDATES) if os.access(p, os.X_OK)), None)
    if uv is None:
        raise RuntimeError("uv is needed to install it and is not on this machine: see https://docs.astral.sh/uv/, "
                           "then try again")
    if not _installing.acquire(blocking=False):
        raise RuntimeError("an installation is already going")
    try:
        proc = run([uv, "tool", "install", f"tool-eval-bench=={VERSION}"], capture_output=True, text=True,
                   timeout=INSTALL_TIMEOUT, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"the installation took more than {INSTALL_TIMEOUT // 60} minutes and was stopped") from exc
    finally:
        _installing.release()
    if proc.returncode != 0:
        tail = " ".join((proc.stderr or proc.stdout or "").strip().splitlines()[-3:])
        raise RuntimeError(f"uv could not install it: {tail or f'exit code {proc.returncode}'}")
    exe = find()
    if exe is None:
        raise RuntimeError("installed, but not found where uv puts its tools (~/.local/bin): set TOOL_EVAL_BENCH")
    return {"path": exe, "version": version(exe)}


def version(exe: str) -> str | None:
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip().split()[-1] if out.strip() else None


def command(exe: str, base: str, model: str, *, full: bool) -> list[str]:
    """The run, thinking off and greedy like the rest of the Bench; the key goes through the environment."""

    cmd = [exe, "run", "--base-url", base.rstrip("/") + "/v1", "--model", model, "--no-think", "--temperature", "0",
           "--seed", str(SEED), "--no-live", "--redact-url"]
    return cmd if full else [*cmd, "--short"]


class Run:
    """One tool-eval-bench run: the process, its progress events, and its result file."""

    def __init__(self, exe: str, base: str, key: str | None, model: str, *, full: bool,
                 on_event: Callable[[dict], None]) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="tool-eval-"))
        self.result = self.dir / "result.json"
        cmd = [*command(exe, base, model, full=full), "--json-file", str(self.result), "--output-dir", str(self.dir)]
        env = dict(os.environ)
        if key:
            env["TOOL_EVAL_API_KEY"] = key
        self.proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, text=True, cwd=self.dir)
        self.lines: list[str] = []
        self.categories: dict[str, str] = {}             # scenario id -> its category letter, from the events
        self._reader = threading.Thread(target=self._read, args=(on_event,), daemon=True)
        self._reader.start()

    def _read(self, on_event: Callable[[dict], None]) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            line = line.strip()
            try:
                event = json.loads(line)
            except ValueError:
                self.lines = [*self.lines[-40:], line]      # its own messages, kept for the error said if it fails
                continue
            if isinstance(event, dict):
                if event.get("event") == "scenario_start" and event.get("scenario_id"):
                    self.categories[event["scenario_id"]] = event.get("category") or ""
                on_event(event)

    def wait(self, cancel: threading.Event) -> dict:
        while self.proc.poll() is None:
            if cancel.wait(0.5):
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self.proc.wait(timeout=5)      # reaped before its folder is removed under it
                raise InterruptedError
        self._reader.join(timeout=5)
        if self.proc.returncode != 0 or not self.result.exists():
            raise RuntimeError(f"tool-eval-bench stopped ({self.proc.returncode}): "
                               + (" ".join(self.lines[-3:]) or "no message")[:300])
        return json.loads(self.result.read_text())

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def results(doc: dict, letters: dict[str, str] | None = None) -> list[dict]:
    """Its scenarios as Bench results: right only when passed; partial and fail say why."""

    cats = {c["category"]: c.get("label", c["category"]) for c in doc["scores"].get("category_scores", [])}
    out = []
    for i, s in enumerate(doc["scores"].get("scenario_results", [])):
        cat = cats.get((letters or {}).get(s["scenario_id"], ""))
        out.append({"suite": "tools", "prompt": s["scenario_id"], "draft": True, "passed": s.get("status") == "pass",
                    "status": s.get("status"), "points": s.get("points"), "category": cat,
                    "read": f"{s.get('status')}: {s.get('summary') or ''}"[:300],
                    "expected": (s.get("expected_behavior") or "")[:300], "text": (s.get("summary") or "")[:1000],
                    "error": None, "index": i,
                    "m": {"tokens": s.get("completion_tokens"), "tps": None,
                          "ttft": (s["ttft_ms"] / 1000) if isinstance(s.get("ttft_ms"), (int, float)) else None,
                          "prefill_s": None, "wall_s": s.get("duration_seconds"),
                          "gap_p50": None, "gap_p95": None, "runtime": {}, "sha": None, "finish": None, "timing": "studio"}})
    return out


def summary(doc: dict) -> dict:
    """The tool's own scores, without its host name or address."""

    s = doc["scores"]
    return {"version": doc.get("tool_eval_bench_version"), "final_score": s.get("final_score"),
            "rating": s.get("rating"), "points": s.get("total_points"), "max_points": s.get("max_points"),
            "deployability": s.get("deployability"), "responsiveness": s.get("responsiveness"),
            "median_turn_ms": s.get("median_turn_ms"), "worst_category": s.get("worst_category"),
            "categories": [{k: c.get(k) for k in ("category", "label", "earned", "max", "percent", "pass_count",
                                                   "partial_count", "fail_count")}
                           for c in s.get("category_scores", [])],
            "scenarios": doc.get("total_scenarios"), "seed": SEED,
            "safety_warnings": len(doc.get("safety_warnings") or [])}
