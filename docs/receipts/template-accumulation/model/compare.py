"""Independently validate and compare four private model-client receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

LABELS = ("A1", "B1", "A2", "B2")
FIXTURES = ("multiturn", "prose", "code")
TOKENS = 64
REPETITIONS = 3
HASH = re.compile(r"[0-9a-f]{64}\Z")
CellKey = tuple[str, float, int]
RequestKey = tuple[str, float, int, bool]


def require(condition: bool, message: str) -> None:
    """Stop at the first invalid or unequal piece of evidence."""
    if not condition:
        raise ValueError(message)


def close(actual: Any, expected: float, tag: str) -> None:
    """Check finite unrounded arithmetic with small floating-point tolerance."""
    require(type(actual) in (int, float) and math.isfinite(actual), tag + ": invalid number")
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8), tag + ": arithmetic differs")


def ids_hash(ids: list[int]) -> str:
    """Hash comma-separated decimal IDs, independently of the benchmark helpers."""
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def valid_ids(ids: Any, tag: str, length: int | None = None) -> None:
    """Require actual unsigned token integers and the expected nonempty length."""
    require(isinstance(ids, list) and bool(ids), tag + ": missing IDs")
    require(length is None or len(ids) == length, tag + ": ID count")
    require(all(type(n) is int and 0 <= n <= 0xffffffff for n in ids), tag + ": malformed IDs")


def fingerprint(run: dict[str, Any]) -> tuple[Any, ...]:
    """Compare full tokens and independent content/reasoning hashes."""
    return tuple(run["token_ids"]), run["token_sha256"], run["output_sha256"]


def validate_request(
    run: dict[str, Any], expected: RequestKey, prompt_count: int, tag: str,
) -> None:
    """Recompute hashes and observed request timing without trusting qualification flags."""
    name, temperature, seed, draft = expected
    require(type(run.get("temperature")) in (int, float), tag + ": temperature type")
    require(run["fixture"] == name and run["temperature"] == temperature, tag + ": request identity")
    require(type(run["seed"]) is int and run["seed"] == seed, tag + ": seed")
    require(run["draft"] is draft, tag + ": draft mode")
    require(run.get("complete") is True and run.get("error") is None, tag + ": incomplete/error")
    require(run.get("valid") is True and run.get("validation_errors") == [], tag + ": client invalid")
    require(type(run.get("tokens")) is int and run["tokens"] == TOKENS, tag + ": reply count")
    require(type(run.get("cached_tokens")) is int and run["cached_tokens"] == 0, tag + ": cache reused")
    require(run.get("cache_state") == "cold", tag + ": unverified cache")
    require(type(run.get("prompt_tokens")) is int and run["prompt_tokens"] == prompt_count,
            tag + ": tokenize/prompt count")
    require(0 < prompt_count <= 8192 - TOKENS, tag + ": context bound")
    valid_ids(run.get("token_ids"), tag, TOKENS)
    require(run.get("token_sha256") == ids_hash(run["token_ids"]), tag + ": token hash")
    for field in ("token_sha256", "output_sha256"):
        require(isinstance(run.get(field), str) and HASH.fullmatch(run[field]) is not None,
                tag + ": invalid " + field)
    require(run.get("token_sha") == run["token_sha256"][:12], tag + ": short token hash")
    content, reasoning = run.get("content"), run.get("reasoning_content")
    require(isinstance(content, str) and isinstance(reasoning, str), tag + ": missing output text")
    digest = hashlib.sha256(
        hashlib.sha256(content.encode()).digest() + hashlib.sha256(reasoning.encode()).digest()
    ).hexdigest()
    require(run["output_sha256"] == digest, tag + ": output hash")
    for field in ("sent", "first", "last", "completed"):
        value = run.get(field)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                tag + ": invalid timestamp " + field)
    require(run["sent"] < run["first"] < run["last"] <= run["completed"], tag + ": event order")
    close(run["ttft_s"], run["first"] - run["sent"], tag + " TTFT")
    close(run["decode_tps"], (TOKENS - 1) / (run["last"] - run["first"]), tag + " decode")
    close(run["elapsed_s"], run["completed"] - run["sent"], tag + " elapsed")
    pieces = run.get("pieces")
    require(isinstance(pieces, list) and len(pieces) >= 4, tag + ": missing streamed chunks")
    require(all(isinstance(p, list) and len(p) == 2 for p in pieces), tag + ": chunk shape")
    require(all(type(p[0]) in (int, float) and math.isfinite(p[0]) for p in pieces),
            tag + ": chunk timestamps")
    require(all(type(p[1]) is int and p[1] > 0 for p in pieces), tag + ": chunk character counts")
    times = [p[0] for p in pieces]
    require(times == sorted(times), tag + ": chunk order")
    close(times[0], run["first"], tag + " first chunk")
    close(times[-1], run["last"], tag + " last chunk")
    offsets = run.get("chunk_offsets_s")
    require(isinstance(offsets, list) and len(offsets) == len(times), tag + ": chunk offsets")
    for actual, at in zip(offsets, times):
        close(actual, at - run["sent"], tag + " chunk offset")


def validate_metadata(data: dict[str, Any], label: str) -> dict[str, int]:
    """Recompute fixture and tokenization fingerprints and require the complete protocol."""
    require(data.get("schema_version") == 1 and data.get("label") == label, label + ": header")
    require(data.get("qualified") is True and data.get("errors") == [], label + ": not qualified")
    require(data.get("requested_reply_tokens") == TOKENS and data.get("repetitions") == REPETITIONS,
            label + ": protocol size")
    require(data.get("levels") == [1, 2] and data.get("temperatures") == [0.0, 1.0], label + ": grid")
    require(data.get("resume_test") == "not_run: prompt cache is required to be disabled",
            label + ": resume scope")
    require(set(data["fixtures"]) == set(FIXTURES), label + ": fixtures")
    require(set(data["tokenize"]) == set(FIXTURES), label + ": tokenization fixtures")
    helpers = data.get("helper_sha256")
    expected_helpers = {"bench_evidence.py", "bench_timing.py", "bench_metrics.py"}
    require(isinstance(helpers, dict) and set(helpers) == expected_helpers, label + ": helper hashes")
    require(all(isinstance(h, str) and HASH.fullmatch(h) for h in helpers.values()),
            label + ": helper hash shape")
    prompt_counts = {}
    for name in FIXTURES:
        record = data["fixtures"][name]
        messages = record.get("messages")
        require(isinstance(messages, list) and len(messages) == (34 if name == "multiturn" else 1),
                label + ": fixture message count " + name)
        require(record.get("message_count") == len(messages), label + ": fixture count metadata")
        raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
        require(record.get("sha256") == hashlib.sha256(raw).hexdigest(), label + ": fixture hash " + name)
        snapshots = data["tokenize"][name]
        require(set(snapshots) == {"generation", "history"}, label + ": snapshot phases")
        for phase, snapshot in snapshots.items():
            ids = snapshot.get("token_ids")
            valid_ids(ids, label + ": tokenize " + name + "/" + phase)
            require(type(snapshot.get("count")) is int and snapshot["count"] == len(ids),
                    label + ": tokenize count")
            require(len(ids) + TOKENS <= 8192, label + ": tokenizer context bound")
            require(snapshot.get("token_sha256") == ids_hash(ids), label + ": tokenizer hash")
        prompt_counts[name] = snapshots["generation"]["count"]
    return prompt_counts


def validate_run(
    data: dict[str, Any], label: str,
) -> tuple[dict[CellKey, dict[str, float]], dict[RequestKey, tuple[Any, ...]], int]:
    """Reconstruct all 73 inference responses and the 12 cell medians."""
    prompt_counts = validate_metadata(data, label)
    seen: dict[RequestKey, tuple[Any, ...]] = {}
    count = 0

    def observe(run: dict[str, Any], key: RequestKey, tag: str) -> None:
        """Validate each observation and reject any within-run repetition drift."""
        nonlocal count
        validate_request(run, key, prompt_counts[key[0]], label + ": " + tag)
        value = fingerprint(run)
        require(key not in seen or seen[key] == value, label + ": repeated output differs " + str(key))
        seen[key] = value
        count += 1

    observe(data["first_cold"], ("multiturn", 0.0, 1234, True), "first cold")
    references = data.get("references")
    require(isinstance(references, list) and len(references) == 9, label + ": reference count")
    reference_index = {}
    for ref in references:
        key = ref["fixture"], ref["temperature"], ref["seed"]
        require(key not in reference_index, label + ": duplicate reference")
        observe(ref["alone"], (*key, True), "alone " + str(key))
        observe(ref["plain"], (*key, False), "plain " + str(key))
        require(fingerprint(ref["alone"]) == fingerprint(ref["plain"]), label + ": draft/plain differs")
        require(ref.get("equal") is True, label + ": reported plain mismatch")
        reference_index[key] = fingerprint(ref["alone"])
    expected_references = {
        (name, temp, seed) for name in FIXTURES for temp in (0.0, 1.0)
        for seed in ([1234] if temp == 0 else [1234, 1235])
    }
    require(set(reference_index) == expected_references, label + ": reference grid")
    cells = data.get("cells")
    require(isinstance(cells, list) and len(cells) == 12, label + ": expected 12 cells")
    cell_values = {}
    for cell in cells:
        key = cell["fixture"], cell["temperature"], cell["streams"]
        require(key not in cell_values and cell.get("qualified") is True, label + ": cell identity/flag")
        name, temp, n = key
        require(type(temp) in (int, float) and type(n) is int, label + ": cell numeric types")
        require(name in FIXTURES and temp in (0.0, 1.0) and n in (1, 2), label + ": unexpected cell")
        reps = cell.get("reps")
        require(isinstance(reps, list) and len(reps) == REPETITIONS, label + ": repetition count")
        aggregate, e2e, ttft = [], [], []
        for ri, rep in enumerate(reps):
            require(rep.get("repetition") == ri, label + ": repetition index")
            runs = rep.get("runs")
            require(isinstance(runs, list) and len(runs) == n, label + ": stream count")
            require(rep.get("comparisons_to_alone") == [True] * n, label + ": reported alone mismatch")
            for si, run in enumerate(runs):
                seed = 1234 + si if temp else 1234
                observe(run, (name, temp, seed, True), "cell " + str(key))
                require(fingerprint(run) == reference_index[name, temp, seed], label + ": alone/concurrent differs")
                ttft.append(run["first"] - run["sent"])
            makespan = max(r["completed"] for r in runs) - min(r["sent"] for r in runs)
            decode_span = max(r["last"] for r in runs) - min(r["first"] for r in runs)
            timing = rep.get("timing")
            require(isinstance(timing, dict) and timing.get("timing_valid") is True, label + ": timing flag")
            require(timing.get("requested_requests") == timing.get("timed_requests") == n,
                    label + ": timing stream count")
            close(timing["makespan_s"], makespan, label + " makespan")
            close(timing["end_to_end_tps"], n * TOKENS / makespan, label + " E2E")
            close(rep["aggregate_tps"], n * (TOKENS - 1) / decode_span, label + " aggregate")
            aggregate.append(n * (TOKENS - 1) / decode_span)
            e2e.append(n * TOKENS / makespan)
        values = {
            "ttft_s": statistics.median(ttft),
            "aggregate_decode_tps": statistics.median(aggregate),
            "end_to_end_tps": statistics.median(e2e),
        }
        close(cell["median_request_ttft_s"], values["ttft_s"], label + " cell TTFT")
        close(cell["median_aggregate_tps"], values["aggregate_decode_tps"], label + " cell aggregate")
        close(cell["median_end_to_end_tps"], values["end_to_end_tps"], label + " cell E2E")
        cell_values[key] = values
    expected_cells = {(name, temp, n) for name in FIXTURES for temp in (0.0, 1.0) for n in (1, 2)}
    require(set(cell_values) == expected_cells and count == 73, label + ": incomplete grid/responses")
    return cell_values, seen, count


def comparison(values: dict[str, float]) -> dict[str, Any]:
    """Compare arithmetic means of the two per-run medians on each side."""
    a = statistics.mean(values[label] for label in ("A1", "A2"))
    b = statistics.mean(values[label] for label in ("B1", "B2"))
    return {
        "per_run": values, "base_mean": a, "candidate_mean": b,
        "candidate_change_pct": 100 * (b / a - 1),
        "base_A2_vs_A1_pct": 100 * (values["A2"] / values["A1"] - 1),
    }


def verify(directory: Path) -> dict[str, Any]:
    """Check immutable inputs before producing the complete paired comparison."""
    raw = {label: (directory / (label + ".json")).read_bytes() for label in LABELS}
    data = {label: json.loads(raw[label]) for label in LABELS}
    validated = {label: validate_run(data[label], label) for label in LABELS}
    for label in LABELS[1:]:
        for field in ("fixtures", "helper_sha256", "tokenize"):
            require(data[label][field] == data["A1"][field], label + ": cross-build " + field + " differs")
        require(validated[label][1] == validated["A1"][1], label + ": cross-build inference differs")
    cells = []
    for key in sorted(validated["A1"][0]):
        name, temp, n = key
        metrics = {
            metric: comparison({label: validated[label][0][key][metric] for label in LABELS})
            for metric in ("ttft_s", "aggregate_decode_tps", "end_to_end_tps")
        }
        cells.append({"fixture": name, "temperature": temp, "streams": n, "metrics": metrics})
    return {
        "qualified": True, "responses_per_run": {label: validated[label][2] for label in LABELS},
        "total_inference_responses": sum(validated[label][2] for label in LABELS),
        "cells_per_run": 12, "cells": cells,
        "first_cold_ttft_s": comparison({label: data[label]["first_cold"]["ttft_s"] for label in LABELS}),
        "fixture_helper_tokenize_and_output_equality": True,
        "input_sha256": {
            label: hashlib.sha256(raw[label]).hexdigest() for label in LABELS
        },
        "method": "Median of raw repetitions per run; arithmetic mean of A1/A2 and B1/B2 medians.",
        "scope": data["A1"].get("scope"), "resume_test": data["A1"]["resume_test"],
    }


def main() -> int:
    """Write a passing or explicit failed summary without modifying receipt inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.directory / "model-summary.json"
    try:
        summary = verify(args.directory)
    except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        summary = {"qualified": False, "error": str(error), "error_type": type(error).__name__}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "cells"}, indent=2))
    return int(not summary["qualified"])


if __name__ == "__main__":
    raise SystemExit(main())
