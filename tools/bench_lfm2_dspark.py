"""Same-head full-matrix qualification of the LFM2 DSpark drafter against plain, cell by cell (see --help)."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_concurrent as bc  # noqa: E402
from bench_lfm2_dspark_verdict import (ARMS, EXIT_USAGE, SCHEMA, activity, compare, problems, render,  # noqa: E402
                                       seeds_for, startup_modes, summarize, total_activity)

LOCK = "/tmp/tensorfold-model.lock"
ARM_FLAGS = {"plain": [], "bf16": ["--drafter", "{drafter}"], "bits8": ["--drafter", "{drafter}", "--drafter-bits", "8"]}
PARALLEL = 16
DEFAULT_WIDTHS = (1, 4, 8, 16)
DEFAULT_TEMPERATURES = (0.0, 1.0)
DEFAULT_TOKENS = 512
SERVED_NAME = "dspark-qual"
SCRUBBED_PREFIXES = ("TF_", "TENSORFOLD_", "MTL_", "METAL_")
SCRUBBED_NAMES = ("GEMMA_DRAFT_TRACE", "FZ_PRICE_BATCHED")
GRACE_S = 90


class Fatal(Exception):
    """The run cannot continue (or its data is invalid); the server is stopped and the summary still written."""


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def pressure_level() -> int | None:
    """macOS kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical; None where unavailable."""
    value = sysctl("kern.memorystatus_vm_pressure_level")
    return int(value) if value is not None and value.isdigit() else None


def competing_processes(binary_name: str) -> list[str]:
    """Names of running model processes (the repo's native binaries and this binary's own name)."""
    ps = subprocess.run(["ps", "-Ao", "pid=,comm="], capture_output=True, text=True, check=True).stdout
    busy = []
    for line in ps.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            name = Path(parts[1].strip()).name
            if name == binary_name or name.startswith(("tensorfold", "tf-")):
                busy.append(f"{parts[0]}:{name}")
    return busy


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def clean_env() -> tuple[dict, list[str]]:
    env, removed = {}, []
    for key, value in os.environ.items():
        if key.startswith(SCRUBBED_PREFIXES) or key in SCRUBBED_NAMES:
            removed.append(key)
        else:
            env[key] = value
    return env, sorted(removed)


class Context:
    """Paths, settings and the append-only evidence files of one run."""

    def __init__(self, args, out: Path):
        self.out, self.args = out, args
        self.binary = Path(args.binary).resolve()
        self.binary_sha256 = sha256_file(self.binary)
        self.target = Path(args.target).resolve()
        self.drafter = Path(args.drafter).resolve()
        self.widths = [int(w) for w in args.widths.split(",")]
        self.temps = [float(t) for t in args.temperatures.split(",")]
        self.tokens, self.reps, self.seed = args.tokens, args.reps, args.seed
        self.base = f"http://127.0.0.1:{args.port}"
        self.logs = out / "private-logs"
        self.cells_done: list[dict] = []
        self.env, self.env_removed = clean_env()

    def scrub(self, text: str) -> str:
        for path, name in ((self.out, "<output>"), (self.binary, "<binary>"), (self.target, "<target>"),
                           (self.drafter, "<drafter>"), (Path.home(), "<home>")):
            text = text.replace(str(path), name)
        return text

    def append(self, name: str, record: dict) -> None:
        with open(self.out / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def event(self, kind: str, **data) -> None:
        record = {"event": kind, "at": now_utc(), **data}
        self.append("events.jsonl", record)
        print(json.dumps(record), flush=True)

    def raw(self, cell: str, phase: str, runs: list[dict]) -> None:
        for r in runs:
            self.append("requests.jsonl", {"cell": cell, "phase": phase, "run": r})

    def record(self, record: dict) -> None:
        self.append("cells.jsonl", record)
        self.cells_done.append(record)


class Server:
    """One native server process: started on an unused port, gracefully stopped, never leaked."""

    def __init__(self, ctx: Context, label: str, arm: str):
        self.ctx, self.label, self.arm = ctx, label, arm
        self.proc = None
        self.log = None
        self.log_path = ctx.logs / f"{label}.server.log"

    def argv(self) -> list[str]:
        c = self.ctx
        flags = [f.format(drafter=c.drafter) for f in ARM_FLAGS[self.arm]]
        return [str(c.binary), "serve", str(c.target), "--name", SERVED_NAME, "--host", "127.0.0.1",
                "--port", str(c.args.port), "--parallel", str(PARALLEL), "--context", str(c.args.context),
                "--prompt-cache-gib", "0", "--snapshot-dir", "none", "--max-snapshots", "0", "--no-update-check",
                *flags]

    def __enter__(self):
        busy = competing_processes(self.ctx.binary.name)
        if busy:
            raise Fatal(f"another model process is running: {busy}")
        if port_in_use(self.ctx.args.port):
            raise Fatal(f"port {self.ctx.args.port} is already in use")
        launch_sha256 = sha256_file(self.ctx.binary)
        if launch_sha256 != self.ctx.binary_sha256:
            raise Fatal(f"{self.label}: binary changed since the run began")
        self.log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(self.argv(), stdout=self.log, stderr=subprocess.STDOUT, env=self.ctx.env,
                                     start_new_session=True)
        started = time.monotonic()
        try:
            while True:
                if self.proc.poll() is not None:
                    raise Fatal(f"{self.label}: server exited with {self.proc.returncode} while loading")
                try:
                    with urllib.request.urlopen(self.ctx.base + "/v1/models", timeout=2) as r:
                        json.load(r)
                    break
                except Exception:  # noqa: BLE001 - not listening yet
                    if time.monotonic() - started > self.ctx.args.startup_timeout:
                        raise Fatal(f"{self.label}: server not ready after {self.ctx.args.startup_timeout}s") from None
                    time.sleep(0.5)
            log = self.log_path.read_text(errors="replace")
            modes = startup_modes(log)
            if modes != ([] if self.arm == "plain" else [self.arm]):
                raise Fatal(f"{self.label}: the startup log states DSpark modes {modes} for arm {self.arm}")
            weights = re.findall(r"lfm2 dspark: weights ([0-9.]+) MiB", log)  # optional metadata
            self.ctx.event("ready", label=self.label, arm=self.arm, seconds=round(time.monotonic() - started, 1),
                           flags=[f for f in ARM_FLAGS[self.arm]], dspark_mode=modes[0] if modes else None,
                           binary_sha256=launch_sha256,
                           dspark_weights_mib=float(weights[0]) if weights else None)
            self.check()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        forced = self.stop()
        if forced and exc_type is None:
            raise Fatal(f"{self.label}: server needed a forced kill")
        return False

    def check(self) -> int | None:
        """Raise Fatal when the server died or memory pressure appeared (its data would be invalid)."""
        if self.proc.poll() is not None:
            raise Fatal(f"{self.label}: server exited with {self.proc.returncode}")
        self.log.flush()
        if "memory pressure:" in self.log_path.read_text(errors="replace").lower():
            raise Fatal(f"{self.label}: memory pressure reported in the server log")
        level = pressure_level()
        if level not in (None, 1):
            raise Fatal(f"{self.label}: OS memory pressure level {level}")
        return level

    def stop(self) -> bool:
        forced, started = False, time.monotonic()
        if self.proc is None:
            return False
        interrupted = False

        def defer_interrupt(signum, frame):
            nonlocal interrupted
            interrupted = True

        handlers = {s: signal.signal(s, defer_interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            if self.proc.poll() is None:
                self.proc.send_signal(signal.SIGTERM)
                try:
                    self.proc.wait(timeout=GRACE_S)
                except subprocess.TimeoutExpired:
                    forced = True
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.proc.wait()
            self.log.close()
            self.ctx.event("shutdown", label=self.label, forced_kill=forced,
                           seconds=round(time.monotonic() - started, 1), returncode=self.proc.returncode)
            self.proc = None
            return forced
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if interrupted and sys.exc_info()[0] is None:
                raise KeyboardInterrupt


REQUEST_FIELDS = ("seed", "tokens", "complete", "error", "unmeasured", "decode_tps", "ttft_s", "token_sha256",
                  "output_sha256", "prompt_tokens", "cached_tokens", "cache_state")


def oracles(ctx: Context, cell_id: str, rep: int, arm: str, item: dict, temp: float, seeds: list[int]) -> dict:
    """Each seed alone, drafting as served and with ``draft: false``; their hashes must agree."""
    out = {}
    for seed in seeds:
        solo = bc.stream(ctx.base, SERVED_NAME, item, ctx.tokens, temp, seed)
        serial = bc.stream(ctx.base, SERVED_NAME, item, ctx.tokens, temp, seed, draft=False)
        ctx.raw(cell_id, "solo", [solo])
        ctx.raw(cell_id, "draft_false", [serial])
        failures = problems(solo, ctx.tokens, False) + problems(serial, ctx.tokens, False)
        diff = compare(solo, serial, "solo_vs_draft_false", seed)
        failures += [diff] if diff else []
        out[seed] = {"solo": solo, "draft_false": serial}
        ctx.record({"kind": "oracle", "rep": rep, "arm": arm, "prompt": item["name"], "temperature": temp,
                    "seed": seed, "solo_token_sha256": solo.get("token_sha256"),
                    "draft_false_token_sha256": serial.get("token_sha256"), "activity": activity(solo),
                    "failures": failures, "ok": not failures})
    return out


def scenario(ctx: Context, srv: Server, rep: int, arm: str, item: dict, temp: float) -> None:
    specs = [(item, sd) for sd in seeds_for(ctx.seed, ctx.widths, temp)]
    stem = f"r{rep}-{arm}-{item['name']}-T{temp:g}"
    ctx.raw(stem, "warmup", bc.together(ctx.base, SERVED_NAME, specs[:1], 16, temp))
    seeds = list(dict.fromkeys(s for _, s in specs[:max(ctx.widths)]))
    refs = oracles(ctx, stem, rep, arm, item, temp, seeds)
    srv.check()
    for n in ctx.widths:
        cell_id = f"{stem}-w{n}"
        ctx.raw(cell_id, "warmup", bc.together(ctx.base, SERVED_NAME, specs[:n], 16, temp))
        before = srv.check()
        runs = bc.together(ctx.base, SERVED_NAME, specs[:n], ctx.tokens, temp)
        ctx.raw(cell_id, "measured", runs)
        after = srv.check()
        agg = bc.aggregates(runs)
        failures = []
        for r in runs:
            failures += problems(r, ctx.tokens, True)
            ref = refs[r["seed"]]
            for name, oracle in (("alone", ref["solo"]), ("draft_false", ref["draft_false"])):
                diff = compare(oracle, r, f"concurrent_vs_{name}", r["seed"])
                failures += [diff] if diff else []
        ctx.record({"kind": "cell", "rep": rep, "arm": arm, "prompt": item["name"], "temperature": temp,
                    "streams": n, "tokens": ctx.tokens, "aggregate_tps": agg.get("aggregate_tps"),
                    "per_stream_tps": [r.get("decode_tps") for r in runs], "pressure_before": before,
                    "pressure_after": after, "activity": total_activity(runs),
                    "requests": [{**{k: r.get(k) for k in REQUEST_FIELDS}, "activity": activity(r)} for r in runs],
                    "failures": failures, "ok": not failures and agg.get("aggregate_tps", 0) > 0})
        last = ctx.cells_done[-1]
        ctx.event("cell", rep=rep, arm=arm, prompt=item["name"], temperature=temp, streams=n,
                  aggregate_tps=None if last["aggregate_tps"] is None else round(last["aggregate_tps"], 1),
                  ok=last["ok"], failures=len(failures))


def launch(ctx: Context, rep: int, arm: str) -> None:
    with Server(ctx, f"r{rep}-{arm}", arm) as srv:
        for item in bc.PROMPTS:
            for temp in ctx.temps:
                scenario(ctx, srv, rep, arm, item, temp)


def order(rep: int) -> list[str]:
    k = rep % len(ARMS)
    return list(ARMS[k:] + ARMS[:k])


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_dir(out: Path) -> dict:
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    summary = summarize(meta, load_jsonl(out / "cells.jsonl"), load_jsonl(out / "events.jsonl"))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(render(summary), flush=True)
    return summary


def metadata(ctx: Context) -> dict:
    a = ctx.args
    smoke = a.smoke

    def model(directory: Path, revision: str | None) -> dict:
        cfg = directory / "config.json"
        return {"revision": revision or "unknown", "config_json_sha256": sha256_file(cfg) if cfg.is_file() else None}

    return {
        "schema": SCHEMA, "label": "SMOKE" if smoke else "QUALIFICATION", "smoke": smoke, "started_at": now_utc(),
        "same_head": f"every arm runs this one binary ({ctx.binary.name}); arms differ only in the drafter flags",
        "binary": {"name": ctx.binary.name, "sha256": ctx.binary_sha256},
        "target": model(ctx.target, a.target_revision), "drafter": model(ctx.drafter, a.drafter_revision),
        "settings": {
            "context": a.context, "parallel": PARALLEL, "prompt_cache_gib": 0, "tokens": ctx.tokens, "reps": ctx.reps,
            "rotation": [order(r) for r in range(ctx.reps)], "rotation_balanced": ctx.reps % len(ARMS) == 0,
            "seed": ctx.seed, "seed_rule": "sampled: seed + stream index; greedy: seed (bench_concurrent)",
            "widths": ctx.widths, "temperatures": ctx.temps, "prompts": [p["name"] for p in bc.PROMPTS],
            "sampling": {"temperature>0": {"top_k": 20, "top_p": 0.95}, "ignore_eos": True},
            "arms": {k: [f for f in v] for k, v in ARM_FLAGS.items()},
            "serve_flags": ["--parallel 16", f"--context {a.context}", "--prompt-cache-gib 0", "--snapshot-dir none",
                            "--max-snapshots 0", "--no-update-check"],
            "speed_metric": "bench_concurrent.aggregates()['aggregate_tps'] per launch cell; median over reps",
            "diagnostic_logging": "off", "env_prefixes_removed": list(SCRUBBED_PREFIXES) + list(SCRUBBED_NAMES),
            "env_variables_removed": ctx.env_removed},
        "environment": {
            "os": platform.platform(), "os_build": sysctl("kern.osversion"), "machine": platform.machine(),
            "hw_model": sysctl("hw.model"), "cpu": sysctl("machdep.cpu.brand_string"),
            "memory_bytes": sysctl("hw.memsize"), "python": platform.python_version()},
        "publishable": ["metadata.json", "requests.jsonl", "cells.jsonl", "events.jsonl", "summary.json"],
        "not_publishable": ["private-logs/"]}


def validate(p: argparse.ArgumentParser, a: argparse.Namespace) -> None:
    try:
        widths = [int(w) for w in a.widths.split(",")]
        temps = [float(t) for t in a.temperatures.split(",")]
    except ValueError:
        p.error("--widths and --temperatures take comma lists of numbers")
    if not widths or any(not 1 <= w <= PARALLEL for w in widths) or len(set(widths)) != len(widths):
        p.error(f"--widths: distinct widths from 1 to {PARALLEL}")
    if not temps or any(t < 0 for t in temps):
        p.error("--temperatures: nonnegative numbers")
    if a.tokens < 16 or a.reps < 1:
        p.error("--tokens needs at least 16 and --reps at least 1")
    full = (a.tokens == DEFAULT_TOKENS and a.reps >= len(ARMS) and a.reps % len(ARMS) == 0
            and tuple(widths) == DEFAULT_WIDTHS and tuple(temps) == DEFAULT_TEMPERATURES)
    if not full and not a.smoke:
        p.error(f"anything short of the full matrix (--tokens {DEFAULT_TOKENS}, --reps a positive multiple of 3, "
                f"--widths {','.join(map(str, DEFAULT_WIDTHS))}, --temperatures 0,1) needs --smoke")
    if not Path(a.binary).is_file() or not os.access(a.binary, os.X_OK):
        p.error("--binary: not an executable file")
    for name, path in (("--target", a.target), ("--drafter", a.drafter)):
        if not Path(path).is_dir():
            p.error(f"{name}: not a directory")
    for name, path, revision in (("target", a.target, a.target_revision), ("drafter", a.drafter, a.drafter_revision)):
        if revision is None and not (Path(path) / "config.json").is_file():
            p.error(f"--{name}-revision: needed, since {name} has no config.json to identify it")


def interrupt(signum, frame):
    """SIGTERM unwinds like Ctrl-C, so every server is stopped and the summary still written."""
    raise KeyboardInterrupt


def run(p: argparse.ArgumentParser, a: argparse.Namespace) -> int:
    validate(p, a)
    out = Path(a.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        p.error("--output-dir must not exist or be empty (raw results are never overwritten)")
    ctx = Context(a, out)
    lock = open(LOCK, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"refusing to run: {LOCK} is held by another model run", file=sys.stderr)
        return EXIT_USAGE
    try:
        busy = competing_processes(ctx.binary.name)
        if busy:
            print(f"refusing to run: model processes already running: {busy}", file=sys.stderr)
            return EXIT_USAGE
        level = pressure_level()
        if level not in (None, 1):
            print(f"refusing to run: OS memory pressure level {level}", file=sys.stderr)
            return EXIT_USAGE
        if port_in_use(a.port):
            print(f"refusing to run: port {a.port} is in use", file=sys.stderr)
            return EXIT_USAGE
        out.mkdir(parents=True, exist_ok=True)
        ctx.logs.mkdir()
        meta = metadata(ctx)
        (out / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        signal.signal(signal.SIGTERM, interrupt)
        try:
            for rep in range(ctx.reps):
                for arm in order(rep):
                    launch(ctx, rep, arm)
        except BaseException as exc:  # noqa: BLE001 - recorded, the server already stopped, summary still written
            detail = ctx.scrub(f"{type(exc).__name__}: {exc}")
            ctx.event("abort", detail=detail)
        summary = summarize_dir(out)
        return summary["exit_code"]
    finally:
        lock.close()


HELP = """\
One native binary, one server at a time (parallel 16, prompt cache 0), arms plain / bf16 (--drafter DIR) / bits8
(--drafter DIR --drafter-bits 8); arm order rotates per repetition. Per launch, prompt (code, chat), temperature (0, 1)
and width (1, 4, 8, 16): each seed runs alone, with and without "draft": false, then the width's streams together.
Every hash must be complete, --tokens long and equal across solo, draft:false, concurrent AND the plain arm (a missing
or unverifiable one fails). Cell speed = bench_concurrent aggregate tps; each cell's median over reps is compared with
plain's, every cell below plain is reported (no tolerance, no pooling). Plain = no external drafter (core proposals such
as copies may still run there); speculative counters are recorded per arm as activity, never as proof of DSpark drafts,
and drafter arms must show some. Each startup log must state its mode ("lfm2: DSpark drafts from --drafter ... bf16 matrices
as released" / "8-bit codes quantized at load (--drafter-bits 8)"), and plain none; weights MiB is optional metadata.
Holds /tmp/tensorfold-model.lock, refuses competing model processes, aborts on
memory pressure, strips TF_/TENSORFOLD_/MTL_/METAL_ variables. --smoke (any reduced matrix) labels all output SMOKE.
Output: metadata.json, requests.jsonl (raw), cells.jsonl, events.jsonl, summary.json, private-logs/ (local paths).
Model revisions are only as supplied (else "unknown") plus config.json sha256; no local names are recorded.
Exit: 0 pass; 1 exactness/scenario failure or interruption; 2 unusable arguments/environment; 3 a cell below plain.
  tools/bench_lfm2_dspark.py run --binary BIN --target DIR --drafter DIR --output-dir OUT; ... summarize OUT"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog=HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="start servers and run the matrix", epilog=HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    r.add_argument("--binary", required=True, help="native binary; every arm runs this same one")
    r.add_argument("--target", required=True, help="target model directory")
    r.add_argument("--drafter", required=True, help="DSpark drafter directory")
    r.add_argument("--output-dir", required=True)
    r.add_argument("--context", type=int, default=4096)
    r.add_argument("--port", type=int, default=18194)
    r.add_argument("--reps", type=int, default=3, help="launches per arm; arm order rotates (default 3; >= 2)")
    r.add_argument("--tokens", type=int, default=DEFAULT_TOKENS, help="tokens per reply (default 512; others need --smoke)")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--widths", default=",".join(map(str, DEFAULT_WIDTHS)))
    r.add_argument("--temperatures", default="0,1")
    r.add_argument("--target-revision", help="public revision id of the target (default: unknown)")
    r.add_argument("--drafter-revision", help="public revision id of the drafter (default: unknown)")
    r.add_argument("--startup-timeout", type=float, default=900)
    r.add_argument("--smoke", action="store_true", help="allow a reduced matrix; everything is labelled SMOKE")
    s = sub.add_parser("summarize", help="rebuild summary.json from an output directory and print it")
    s.add_argument("output_dir")
    a = p.parse_args()
    if a.command == "summarize":
        return summarize_dir(Path(a.output_dir))["exit_code"]
    return run(p, a)


if __name__ == "__main__":
    sys.exit(main())
