#!/usr/bin/env python3
"""Interleaved A/B of main vs a route-tap head on Metal, with the tap unset.

Cells match the M5 Ultra LFM2 report: chat T0 at 1 and 2 streams, code T0 at
2 and 4 streams, code T1 at 1 and 4 streams. Two jobs in opposite orders
(base-new-new-base, then new-base-base-new). Then tap-on vs tap-off token
equality and a shutdown leak scan for LFM2 and, when a checkpoint is set,
Nemotron.

Run on a Metal host. Paths and binaries come from the environment (see `plan`).
Host tests do not start a GPU server.

  python3 tools/run_metal_ab.py plan
  python3 tools/run_metal_ab.py ab
  python3 tools/run_metal_ab.py proofs
  python3 tools/run_metal_ab.py all

Standard library plus tools/bench_concurrent.py. Python 3.11+.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]})")

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import bench_concurrent  # noqa: E402

REPO = TOOLS.parent

# Same six cells as the M5 Ultra LFM2 tap-unset sweep.
CELLS = (
    ("chat", 0.0, 1),
    ("chat", 0.0, 2),
    ("code", 0.0, 2),
    ("code", 0.0, 4),
    ("code", 1.0, 1),
    ("code", 1.0, 4),
)
JOBS = (
    ("base-new-new-base", ("base", "new", "new", "base")),
    ("new-base-base-new", ("new", "base", "base", "new")),
)
PROMPTS = {p["name"]: p for p in bench_concurrent.PROMPTS}
LEAK_MARK = "leaked"


def env_path(name: str, default: str | None = None) -> Path | None:
    got = os.environ.get(name, default)
    return Path(got) if got else None


def native_bin(which: str) -> Path:
    key = "TF_BIN_MAIN" if which == "base" else "TF_BIN_NEW"
    override = os.environ.get(key)
    if override:
        return Path(override)
    root = env_path("TF_SRC_MAIN" if which == "base" else "TF_SRC_NEW") or REPO
    return root / "zig-out" / "native" / "bin" / "tensorfold-native"


def print_plan() -> None:
    print("Metal tap-unset A/B (main vs new head), then tap-on proofs")
    print(f"base binary: {native_bin('base')}")
    print(f"new binary:  {native_bin('new')}")
    print(f"LFM2:        {os.environ.get('LFM2_CHECKPOINT', '(set LFM2_CHECKPOINT)')}")
    print(f"Nemotron:    {os.environ.get('NEMOTRON_CHECKPOINT', '(optional NEMOTRON_CHECKPOINT)')}")
    print("cells:")
    for name, temp, streams in CELLS:
        print(f"  {name}, T {temp:g}, {streams} stream{'s' if streams != 1 else ''}")
    print("jobs: base-new-new-base, new-base-base-new")
    print("TF_ROUTE_TAP is unset for A/B. proofs set it only for the on pass.")


def wait_ready(base: str, timeout: float = 900.0) -> None:
    url = base.rstrip("/") + "/v1/models"
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            last = err
        time.sleep(1)
    raise SystemExit(f"server at {base} did not become ready ({last})")


def serve(
    bin_path: Path,
    checkpoint: str,
    port: int,
    extra: list[str],
    tap: str | None,
    log_path: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    if tap is None:
        env.pop("TF_ROUTE_TAP", None)
    else:
        env["TF_ROUTE_TAP"] = tap
    cmd = [
        str(bin_path),
        "serve",
        checkpoint,
        "--name",
        "local-model",
        "--port",
        str(port),
        "--prompt-cache-gib",
        "0",
        *extra,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    proc._tap_log = log  # type: ignore[attr-defined]
    return proc


def stop(proc: subprocess.Popen) -> str:
    log = getattr(proc, "_tap_log", None)
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
    if log is not None:
        log.flush()
        path = Path(log.name)
        log.close()
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def leak_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if LEAK_MARK in ln and ln.startswith("error(")]


def chat(base: str, model: str, tokens: int, prompt: str) -> str:
    url = base.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "max_tokens": tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def run_cell(base: str, model: str, prompt: str, temp: float, streams: int, tokens: int, reps: int, seed: int) -> dict:
    item = PROMPTS[prompt]
    specs = [(item, seed + i if temp > 0 else seed) for i in range(streams)]
    bench_concurrent.together(base, model, specs[:1], 16, temp)
    aggs = []
    for _ in range(reps):
        runs = bench_concurrent.together(base, model, specs, tokens, temp)
        aggs.append(bench_concurrent.aggregates(runs))
    tps = [a["aggregate_tps"] for a in aggs if a["aggregate_tps"]]
    return {
        "prompt": prompt,
        "temperature": temp,
        "streams": streams,
        "failed": sum(a["failed"] for a in aggs),
        "unmeasured": sum(a["unmeasured"] for a in aggs),
        "aggregate_tps": round(statistics.median(tps), 1) if tps else None,
    }


def run_ab(args: argparse.Namespace) -> dict:
    ckpt = os.environ.get("LFM2_CHECKPOINT")
    if not ckpt:
        raise SystemExit("set LFM2_CHECKPOINT")
    bins = {"base": native_bin("base"), "new": native_bin("new")}
    for name, path in bins.items():
        if not path.is_file():
            raise SystemExit(f"missing {name} binary {path}")
    extra = args.extra.split() if args.extra else ["--parallel", "4"]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"cells": [list(c) for c in CELLS], "jobs": []}
    port = args.port
    for job_name, order in JOBS:
        job: dict = {"order": job_name, "slots": []}
        for i, which in enumerate(order):
            log = out_dir / f"{job_name}-{i}-{which}.log"
            proc = serve(bins[which], ckpt, port, extra, None, log)
            slot = {"bin": which, "cells": [], "leaks": []}
            try:
                wait_ready(f"http://127.0.0.1:{port}")
                for prompt, temp, streams in CELLS:
                    cell = run_cell(
                        f"http://127.0.0.1:{port}",
                        "local-model",
                        prompt,
                        temp,
                        streams,
                        args.tokens,
                        args.reps,
                        args.seed,
                    )
                    slot["cells"].append(cell)
                    print(json.dumps({"job": job_name, "bin": which, **cell}))
            finally:
                text = stop(proc)
                slot["leaks"] = leak_lines(text)
                time.sleep(2)
            job["slots"].append(slot)
            port += 1
        report["jobs"].append(job)
    path = out_dir / "ab.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path)
    return report


def run_proof(family: str, checkpoint: str, bin_path: Path, port: int, extra: list[str], tokens: int, out_dir: Path) -> dict:
    if not bin_path.is_file():
        raise SystemExit(f"missing {bin_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = out_dir / f"{family}.tfr1"
    base = f"http://127.0.0.1:{port}"
    prompt = bench_concurrent.PROMPTS[0]["prompt"]
    off_log = out_dir / f"{family}-off.log"
    on_log = out_dir / f"{family}-on.log"
    result: dict = {"family": family, "equal": False, "leaks_off": [], "leaks_on": []}

    off_proc = serve(bin_path, checkpoint, port, extra, None, off_log)
    try:
        wait_ready(base)
        off_text = chat(base, "local-model", tokens, prompt)
    finally:
        result["leaks_off"] = leak_lines(stop(off_proc))
    time.sleep(1)

    on_proc = serve(bin_path, checkpoint, port, extra, str(trace), on_log)
    try:
        wait_ready(base)
        on_text = chat(base, "local-model", tokens, prompt)
    finally:
        result["leaks_on"] = leak_lines(stop(on_proc))

    result["equal"] = off_text == on_text
    result["trace"] = str(trace) if trace.is_file() else None
    if result["leaks_off"] or result["leaks_on"]:
        raise SystemExit(f"{family}: leak on shutdown: off={result['leaks_off']} on={result['leaks_on']}")
    if not result["equal"]:
        raise SystemExit(f"{family}: tap-on tokens differ from tap-off")
    if trace.is_file():
        decode = subprocess.run(
            [sys.executable, str(TOOLS / "route_tap_bench.py"), "decode", str(trace)],
            check=False,
            capture_output=True,
            text=True,
        )
        if decode.returncode != 0:
            raise SystemExit(f"{family}: TFR1 decode failed\n{decode.stderr}")
        result["records"] = len(json.loads(decode.stdout).get("records") or [])
    (out_dir / f"{family}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("family", "equal", "trace", "records", "leaks_off", "leaks_on") if k in result}))
    return result


def run_proofs(args: argparse.Namespace) -> list[dict]:
    extra = args.extra.split() if args.extra else ["--parallel", "1"]
    out_dir = Path(args.output)
    bin_path = native_bin("new")
    ran = []
    port = args.port
    lfm2 = os.environ.get("LFM2_CHECKPOINT")
    if lfm2:
        ran.append(run_proof("lfm2_moe", lfm2, bin_path, port, extra, args.tokens, out_dir))
        port += 1
    nem = os.environ.get("NEMOTRON_CHECKPOINT")
    if nem:
        ran.append(run_proof("nemotron_h", nem, bin_path, port, extra, args.tokens, out_dir))
    if not ran:
        raise SystemExit("set LFM2_CHECKPOINT and/or NEMOTRON_CHECKPOINT")
    return ran


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="print binaries, checkpoints and cells")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tokens", type=int, default=256)
    common.add_argument("--reps", type=int, default=1)
    common.add_argument("--seed", type=int, default=1234)
    common.add_argument("--port", type=int, default=8096)
    common.add_argument("--output", default="metal-ab")
    common.add_argument("--extra", default="", help="extra serve flags, space-separated (A/B default --parallel 4)")
    sub.add_parser("ab", parents=[common], help="tap-unset interleaved A/B of main vs new")
    sub.add_parser("proofs", parents=[common], help="tap-on token equality and leak scan")
    sub.add_parser("all", parents=[common], help="A/B then proofs")
    args = p.parse_args()
    if args.cmd == "plan":
        print_plan()
        return 0
    if args.cmd in ("ab", "all"):
        run_ab(args)
        args.port += 20
    if args.cmd in ("proofs", "all"):
        run_proofs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
