"""Measure the first generation burst after a fresh native server launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
from typing import Any

import bench_concurrent as bench
import run_panel


SCHEMA = "pr836_ttft_startup_client_v1"
RECEIPT_SCHEMA = "native_ttft_startup_v1"
FILE_CACHE_POLICY = "warm_or_uncontrolled_os_cache"
OUTPUT_TOKENS = run_panel.OUTPUT_TOKENS
SEED = run_panel.SEED


def require(condition: bool, message: str) -> None:
    """Reject a measurement with ambiguous identity or clocks."""
    if not condition:
        raise ValueError(message)


def read_startup(path: Path) -> dict[str, Any]:
    """Validate the controller's same-host monotonic startup boundaries."""
    receipt = json.loads(path.read_text())
    fields = (
        "launch_before_ns", "launch_after_ns", "ready_probe_begin_ns",
        "ready_probe_end_ns",
    )
    values = [receipt.get(field) for field in fields]
    require(receipt.get("schema") == RECEIPT_SCHEMA
            and receipt.get("complete") is True,
            "startup receipt is not final")
    require(all(type(value) is int and value > 0 for value in values)
            and values == sorted(values) and len(set(values)) == len(values),
            "startup monotonic clocks are missing or out of order")
    require(type(receipt.get("server_pid")) is int
            and receipt["server_pid"] > 0,
            "startup server PID is invalid")
    digest = receipt.get("binary_sha256")
    require(isinstance(digest, str) and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            "startup binary identity is invalid")
    require(receipt.get("file_cache_policy") == FILE_CACHE_POLICY,
            "OS file-cache policy is not explicit")
    probes = receipt.get("readiness_probes")
    require(isinstance(probes, list) and bool(probes)
            and receipt.get("readiness_probe_count") == len(probes),
            "readiness probe list or count is invalid")
    for index, probe in enumerate(probes):
        require(isinstance(probe, dict)
                and type(probe.get("begin_ns")) is int
                and type(probe.get("end_ns")) is int
                and probe["begin_ns"] < probe["end_ns"]
                and type(probe.get("ready")) is bool
                and probe["ready"] is (index == len(probes) - 1),
                "readiness probe clock or status is invalid")
        if index:
            require(probes[index - 1]["end_ns"] <= probe["begin_ns"],
                    "readiness probes overlap")
    require(probes[0]["begin_ns"] >= receipt["launch_after_ns"],
            "readiness probe began before process launch completed")
    require(probes[-1]["begin_ns"] == receipt["ready_probe_begin_ns"]
            and probes[-1]["end_ns"] == receipt["ready_probe_end_ns"],
            "final readiness probe does not match receipt")
    binary = receipt.get("binary")
    require(receipt.get("clock") == "time.perf_counter_ns on the Spark host"
            and isinstance(binary, dict) and binary.get("sha256") == digest,
            "startup clock or binary receipt differs")
    return receipt


def selected_cases(corpus_path: Path, name: str,
                   streams: int) -> list[dict[str, Any]]:
    """Pick frozen literal prompts for exactly one process-first-use cell."""
    require(name in ("code", "chat") and streams in run_panel.LEVELS,
            "unsupported startup cell")
    if streams == 1:
        cases = [case for case in run_panel.anchors(corpus_path)
                 if case["name"] == name]
    else:
        cases = [case for case in run_panel.corpus(corpus_path)
                 if case["name"] == name and case["streams"] == streams]
        cases.sort(key=lambda case: case["stream_index"])
    require(len(cases) == streams, "startup cell has missing prompts")
    return cases


def timing(receipt: dict[str, Any],
           rows: list[dict[str, Any]]) -> dict[str, float]:
    """Join same-host process and first text-SSE clocks without headers."""
    require(bool(rows), "startup burst has no requests")
    sent = [row.get("sent") for row in rows]
    first = [row.get("first") for row in rows]
    last = [row.get("last") for row in rows]
    done = [row.get("done") for row in rows]
    require(all(type(value) in (int, float) and math.isfinite(value)
                for value in sent + first + last + done),
            "startup SSE clocks are invalid")
    ready = receipt["ready_probe_end_ns"] / 1e9
    launch = receipt["launch_before_ns"] / 1e9
    require(launch < ready <= min(sent)
            and all(s < f <= l <= d
                    for s, f, l, d in zip(sent, first, last, done)),
            "client request began before readiness or clocks reversed")
    return {
        "launch_to_ready_s": ready - launch,
        "ready_to_first_send_s": min(sent) - ready,
        "earliest_send_to_earliest_sse_s": min(first) - min(sent),
        "launch_to_first_sse_s": min(first) - launch,
        "launch_to_slowest_first_sse_s": max(first) - launch,
        "launch_to_last_done_s": max(done) - launch,
        "median_client_ttft_s": statistics.median(
            f - s for s, f in zip(sent, first)
        ),
        "slowest_client_ttft_s": max(f - s for s, f in zip(sent, first)),
        "batch_last_text_sse_s": max(last) - min(sent),
        "batch_wall_s": max(done) - min(sent),
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Run the first burst before any tokenization or primer HTTP call."""
    require(args.label in ("base", "candidate"), "unknown treatment label")
    require(args.repeat_cache == "zero",
            "PR836 Qwen3.5 CUDA has no prompt-state cache")
    selected = selected_cases(args.corpus, args.mode, args.streams)
    receipt = read_startup(args.startup_receipt)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "complete": False,
        "label": args.label,
        "mode": args.mode,
        "streams": args.streams,
        "base": args.base,
        "model": args.model,
        "repeat_cache_rule": args.repeat_cache,
        "corpus_sha256": run_panel.digest(args.corpus),
        "startup_receipt_sha256": run_panel.digest(args.startup_receipt),
        "source_sha256": {
            name: run_panel.digest(Path(__file__).with_name(name))
            for name in ("run_startup.py", "run_panel.py",
                         "bench_concurrent.py", "bench_evidence.py")
        },
        "case_ids": [case["case_id"] for case in selected],
        "startup": receipt,
        "first_burst": [],
        "prime": None,
        "repeat_burst": [],
    }
    run_panel.write(args.output, report)

    first = bench.together(
        args.base, args.model, [(case, SEED) for case in selected],
        OUTPUT_TOKENS, 0.0,
    )
    report["first_burst"] = first
    run_panel.write(args.output, report)
    require(len(first) == args.streams, "incomplete first burst")
    for row, case in zip(first, selected):
        run_panel.validate(row, case, cache_rule="zero")
        require(row["request_body"]["model"] == args.model,
                "first request model differs")
    report["first_burst_timing"] = timing(receipt, first)
    run_panel.write(args.output, report)

    anchor = selected[0]
    prime = bench.stream(
        args.base, args.model, anchor, OUTPUT_TOKENS, 0.0, SEED,
    )
    report["prime"] = prime
    run_panel.write(args.output, report)
    run_panel.validate(prime, anchor, cache_rule="zero")
    repeated = bench.together(
        args.base, args.model, [(anchor, SEED)] * args.streams,
        OUTPUT_TOKENS, 0.0,
    )
    report["repeat_burst"] = repeated
    run_panel.write(args.output, report)
    require(len(repeated) == args.streams, "incomplete repeat burst")
    expected_output = (first[0]["output_sha256"], first[0]["token_sha"])
    for row in [prime] + repeated:
        run_panel.validate(row, anchor, cache_rule="zero")
        require((row["output_sha256"], row["token_sha"]) == expected_output,
                "repeated output differs from fresh reference")
    report["complete"] = True
    run_panel.write(args.output, report)
    return report


def main() -> None:
    """Collect a bounded fresh-process cell and retain partial failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("code", "chat"), required=True)
    parser.add_argument("--streams", type=int, choices=run_panel.LEVELS, required=True)
    parser.add_argument("--label", choices=("base", "candidate"), required=True)
    parser.add_argument("--repeat-cache", choices=("zero",), required=True)
    parser.add_argument("--startup-receipt", type=Path, required=True)
    parser.add_argument("--corpus", type=Path,
                        default=Path(__file__).with_name("corpus.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("startup cell deadline")
    ))
    signal.alarm(args.max_seconds)
    try:
        collect(args)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
