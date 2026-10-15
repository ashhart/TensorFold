"""Capture one owned child process per context; Linux RSS/PSS and native allocator scopes stay distinct."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def proc_sample(pid):
    result = {"monotonic_ns": time.monotonic_ns(), "status": "unmeasured"}
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        for line in status.splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM"}:
                words = value.split()
                if len(words) == 2 and words[1] == "kB":
                    result[key + "_bytes"] = int(words[0]) * 1024
        result["status"] = "recorded" if "VmRSS_bytes" in result else "unsupported"
        try:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in {"Pss", "Locked"}:
                    words = value.split()
                    if len(words) == 2 and words[1] == "kB":
                        result[key + "_bytes"] = int(words[0]) * 1024
        except (OSError, ValueError):
            result["rollup_status"] = "unmeasured"
    except (OSError, ValueError):
        pass
    return result


def native_receipts(text, rank, pid=None):
    rows, errors = [], []
    phases = {"load_begin", "load_end", "cleanup_begin", "cleanup_end", "reset"}
    for line in text.splitlines():
        if not line.startswith("TF_MEMORY "):
            continue
        try:
            row = json.loads(line[len("TF_MEMORY "):])
            if type(row["schema"]) is not int or row["schema"] != 1 or type(row["rank"]) is not int or row["rank"] != rank or row["phase"] not in phases:
                raise ValueError("schema, rank or phase")
            if pid is not None and row["pid"] != pid:
                raise ValueError("unexpected process")
            if row["phase"] == "reset" and not row["reset"]:
                raise ValueError("reset marker without reset")
            if row["scope"] != "process_owned_cuda_buffers" or type(row["reset"]) is not bool:
                raise ValueError("scope or reset")
            for key in ["device", "host", "peak", "host_peak"]:
                if type(row["usage"][key]) is not int or row["usage"][key] < 0:
                    raise ValueError("invalid allocator bytes")
            if row["usage"]["peak"] < row["usage"]["device"] or row["usage"]["host_peak"] < row["usage"]["host"]:
                raise ValueError("peak below current")
            if type(row["monotonic_ns"]) is not int or row["monotonic_ns"] < 0:
                raise ValueError("missing clock")
            if rows and row["monotonic_ns"] < rows[-1]["monotonic_ns"]:
                raise ValueError("clock order")
            rows.append(row)
        except (KeyError, TypeError, ValueError):
            errors.append("invalid native receipt")
    lifecycle = [row["phase"] for row in rows if row["phase"] != "reset"]
    complete = lifecycle == ["load_begin", "load_end", "cleanup_begin", "cleanup_end"]
    complete = complete and rows[0]["reset"] and not errors if rows else False
    return {"status": "recorded" if complete else "unmeasured", "rows": rows, "errors": errors,
            "reason": None if complete else "requires one complete load/cleanup lifecycle and valid clock/rank",
            "scope": "owned buffers only; excludes driver, library and other untracked allocations"}


def capture(command, rank, context_tokens, output, interval):
    if sys.platform != "linux":
        raise ValueError("capture currently requires Linux wait4/procfs; other platforms unsupported")
    if not command or (not math.isfinite(interval) or interval <= 0) or context_tokens <= 0 or rank < 0:
        raise ValueError("command, positive interval and context required")
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, TF_CUDA_MEMORY_RECEIPTS="1", TF_TP_RANK=str(rank))
    samples, stop = [], threading.Event()
    started = time.monotonic_ns()
    with (output / "process.log").open("xb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        def sample():
            while not stop.is_set():
                samples.append(proc_sample(child.pid))
                stop.wait(interval)
        thread = threading.Thread(target=sample)
        thread.start()
        previous = {}
        def forward(signum, _frame):
            try:
                os.kill(child.pid, signum)
            except ProcessLookupError:
                pass
        for signum in [signal.SIGTERM, signal.SIGINT]:
            previous[signum] = signal.signal(signum, forward)
        try:
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
        finally:
            stop.set()
            thread.join()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    ended = time.monotonic_ns()
    report = {"schema": 1, "rank": rank, "context_tokens_operator_label": context_tokens, "pid": child.pid,
              "process_start_ns": started, "process_exit_ns": ended, "returncode": child.returncode,
              "resident_peak": {"status": "recorded", "bytes": usage.ru_maxrss * 1024,
                                "scope": "Linux wait4 lifetime peak RSS, fresh process per context; not resettable"},
              "physical_footprint": {"status": "unsupported", "bytes": None,
                                     "reason": "Linux RSS/PSS is not a total unified-memory physical-footprint peak"},
              "samples": samples,
              "native": native_receipts((output / "process.log").read_text(errors="replace"), rank, child.pid)}
    if child.returncode != 0:
        report["native"]["status"] = "unmeasured"
        report["native"]["reason"] = "child did not exit successfully; inspect partial lifecycle"
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return child.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return capture(command, args.rank, args.context_tokens, args.output, args.interval)


if __name__ == "__main__":
    sys.exit(main())
