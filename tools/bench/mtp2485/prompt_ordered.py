#!/usr/bin/env python3
"""Paired fresh-server cold-prompt A/B measurements for dense MTP."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_receipt as rr  # noqa: E402
OUT = rr.OUT
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TTFT_REGRESSION_TOLERANCE_PERCENT = 0.1
DONE_RE = re.compile(
    r"tokens=(?P<tokens>\d+).*?rounds=(?P<rounds>\d+)\s+"
    r"accepted=(?P<accepted>\d+)/(?P<drafted>\d+).*?"
    r"ttft=(?P<ttft>[0-9.]+)\s+decode=(?P<decode>[0-9.]+).*?tok/s=(?P<tok_s>[0-9.]+)"
)
def write_prompt(source: Path, count: int, output: Path) -> str:
    base = [int(x) for x in source.read_text().strip().split(",") if x]
    if not base:
        raise ValueError(f"empty prompt token fixture: {source}")
    ids = [base[i % len(base)] for i in range(count)]
    encoded = json.dumps(ids, separators=(",", ":")) + "\n"
    output.write_text(encoded)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def context_for(prompt_tokens: int) -> int:
    needed = prompt_tokens + 64
    return max(4096, ((needed + 1023) // 1024) * 1024)


def clean_env(solo_chunk_rows: int = 4096) -> dict:
    return rr.benchmark_env(solo_chunk_rows)[0]


def parse_server_completions(server_log: Path) -> list[dict]:
    text = server_log.read_text(errors="replace") if server_log.exists() else ""
    completions = []
    for line in text.splitlines():
        match = DONE_RE.search(line)
        if not match:
            continue
        item = {name: int(match[name]) for name in ("tokens", "rounds", "accepted", "drafted")}
        item.update({name: float(match[name]) for name in ("ttft", "decode", "tok_s")})
        item["acceptance_rate"] = item["accepted"] / item["drafted"] if item["drafted"] else None
        completions.append(item)
    return completions


def context_is_unsupported(server_log: Path) -> bool:
    text = server_log.read_text(errors="replace").lower() if server_log.exists() else ""
    return ("memory budget fits no stream" in text or "out of memory" in text
            or ("context" in text and "exceed" in text))


def baseline_capacity_refusal(name: str, result: dict, server_log: Path) -> bool:
    return name == "main" and bool(result.get("error")) and context_is_unsupported(server_log)


def measure(label: str, binary: Path, mtp: bool, context: int,
            prompt_path: Path, prompt_tokens: int, drafter_path: Path | None = None,
            solo_chunk_rows: int = 4096) -> dict:
    env = clean_env(solo_chunk_rows)
    proc, log_file, monitor, url, startup_s, server_log = rr.start_server(
        label, binary, mtp, context, 1, env, drafter_path=drafter_path)
    output = OUT / f"{label}.json"
    bench_log = OUT / f"{label}.log"
    result = None
    try:
        cmd = [sys.executable, str(rr.PROJECT / "tools/bench_cache.py"), url, rr.MODEL,
               "--tokens", "64", "--repeats", "1", "--streams", "1", "--require-cold",
               "--prompt-file", str(prompt_path), "--expected-prompt-tokens", str(prompt_tokens),
               "--output", str(output)]
        with bench_log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=rr.PROJECT, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        if code:
            raise RuntimeError(f"bench_cache exited {code}; see {bench_log}")
        report = json.loads(output.read_text())
        rows = report.get("rows") or []
        if not rows:
            raise ValueError(f"benchmark returned no request rows: {output}")
        first = rows[0]
        all_rows = [{k: row.get(k) for k in (
            "phase", "prompt_tokens", "ttft_s", "cache_state", "cached_tokens", "token_sha256")}
                    for row in rows]
        result = {
            "qualified": bool(report.get("qualified")),
            "startup_s": startup_s,
            "server_log": str(server_log),
            "report": str(output),
            "bench_log": str(bench_log),
            "primary_first_request": {k: first.get(k) for k in (
                "phase", "prompt_tokens", "ttft_s", "cache_state", "cached_tokens", "token_sha256")},
            "first_request_cold": first.get("phase") == "first" and first.get("cache_state") == "cold"
                and first.get("cached_tokens") == 0 and first.get("prompt_tokens") == prompt_tokens
                and isinstance(first.get("ttft_s"), (int, float)) and math.isfinite(first["ttft_s"])
                and first["ttft_s"] > 0,
            "secondary_requests": all_rows[1:],
            "all_rows": all_rows,
        }
    finally:
        memory = rr.stop_server(proc, log_file, monitor)
        if result is not None:
            result["memory"] = memory
        # Attach memory even when the client command failed.
        if output.exists():
            try:
                detail = json.loads(output.read_text())
                detail["server_memory"] = memory
                output.write_text(json.dumps(detail, indent=2) + "\n")
            except (OSError, ValueError):
                pass
    if result is not None:
        completions = parse_server_completions(server_log)
        result["server_completions"] = completions
        result["server_completion"] = completions[0] if completions else None
    return result


def binary_specs() -> list[tuple[str, Path, bool]]:
    base = rr.BASE / "zig-out/native/bin/tensorfold-native"
    candidate = rr.PROJECT / "zig-out/native/bin/tensorfold-native"
    specs = [("main", base, False), ("candidate-no-drafts", candidate, False)]
    specs.append(("candidate-mtp-after", candidate, True))
    specs.append(("candidate-dflash2", candidate, False))
    missing = [str(binary) for _, binary, _ in specs if not binary.is_file()]
    if missing:
        raise FileNotFoundError("missing benchmark binaries; build these checkouts first: " + ", ".join(missing))
    return specs


def summarize_server_metrics(observations: list[dict]) -> dict:
    completions = [item.get("server_completion") for item in observations
                   if isinstance(item.get("server_completion"), dict)]
    decode_rates = [item["tok_s"] for item in completions]
    acceptance = [item["acceptance_rate"] for item in completions
                  if item["acceptance_rate"] is not None]
    return {
        "server_tok_s_samples": decode_rates,
        "median_server_tok_s": statistics.median(decode_rates) if decode_rates else None,
        "draft_acceptance_samples": acceptance,
        "median_draft_acceptance": statistics.median(acceptance) if acceptance else None,
    }


def summarize_case(case: dict, specs: list[tuple[str, Path, bool]], reps: int) -> None:
    results = case["results"]
    if case.get("supported") is False:
        case["summary"] = {}
        case["status"] = "skipped_unsupported"
        case["qualified"] = None
        case["comparisons_vs_main"] = {}
        return

    def primary_request(item: dict) -> dict:
        value = item.get("primary_first_request")
        return value if isinstance(value, dict) else {}

    metrics = {}
    for name, _, _ in specs:
        observations = [results[f"rep-{rep:02d}-{name}"] for rep in range(reps)
                        if f"rep-{rep:02d}-{name}" in results]
        first_ttft = [primary_request(item).get("ttft_s") for item in observations]
        valid_ttft = [value for value in first_ttft
                      if isinstance(value, (int, float)) and math.isfinite(value) and value > 0]
        metrics[name] = {
            "samples": first_ttft,
            "median_first_ttft_s": statistics.median(valid_ttft) if valid_ttft else None,
            "token_hashes": [primary_request(item).get("token_sha256") for item in observations],
            "qualified_samples": sum(
                bool(item.get("qualified") and item.get("first_request_cold")
                     and isinstance(primary_request(item).get("ttft_s"), (int, float))
                     and math.isfinite(primary_request(item)["ttft_s"])
                     and primary_request(item)["ttft_s"] > 0)
                for item in observations
            ),
        }
        metrics[name].update(summarize_server_metrics(observations))
    case["summary"] = metrics
    baseline = metrics["main"]["median_first_ttft_s"]
    if baseline is None:
        case["qualified"] = False
        case["status"] = "failed"
        case["error"] = "no main baseline samples"
        return
    comparisons = {}

    def compare(name: str) -> dict:
        current = metrics[name]["median_first_ttft_s"]
        candidate_hashes = metrics[name]["token_hashes"]
        delta = (100 * (current / baseline - 1)) if current is not None else None
        hashes_valid = bool(candidate_hashes) and all(
            isinstance(value, str) and SHA256_RE.fullmatch(value) for value in candidate_hashes
        ) and all(isinstance(value, str) and SHA256_RE.fullmatch(value)
                 for value in metrics["main"]["token_hashes"])
        return {
            "median_ttft_s": current,
            "delta_percent_vs_main": delta,
            "not_slower": current is not None and current <= baseline,
            "within_regression_tolerance": delta is not None and delta <= TTFT_REGRESSION_TOLERANCE_PERCENT,
            "token_hashes_match_main": hashes_valid and candidate_hashes == metrics["main"]["token_hashes"],
        }
    for name in ("candidate-no-drafts", "candidate-mtp-after"):
        comparisons[name] = compare(name)
    dflash2_comparison = None
    if "candidate-dflash2" in metrics:
        dflash2_comparison = compare("candidate-dflash2")
        comparisons["candidate-dflash2-diagnostic-vs-main"] = {
            "median_ttft_s": dflash2_comparison["median_ttft_s"],
            "delta_percent_vs_main": dflash2_comparison["delta_percent_vs_main"],
            "token_hashes_match_main": dflash2_comparison["token_hashes_match_main"],
            "diagnostic_only": True,
        }
    mtp_ttft = metrics["candidate-mtp-after"]["median_first_ttft_s"]
    plain_ttft = metrics["candidate-no-drafts"]["median_first_ttft_s"]
    mtp_hashes = metrics["candidate-mtp-after"]["token_hashes"]
    plain_hashes = metrics["candidate-no-drafts"]["token_hashes"]
    mtp_plain_hashes_valid = bool(mtp_hashes) and len(mtp_hashes) == len(plain_hashes) and all(
        isinstance(value, str) and SHA256_RE.fullmatch(value) for value in mtp_hashes + plain_hashes
    )
    comparisons["candidate-mtp-after-vs-candidate-no-drafts"] = {
        "mtp_median_ttft_s": mtp_ttft,
        "no_drafts_median_ttft_s": plain_ttft,
        "delta_percent": (100 * (mtp_ttft / plain_ttft - 1))
            if mtp_ttft is not None and plain_ttft else None,
        "not_slower": mtp_ttft is not None and plain_ttft is not None and mtp_ttft <= plain_ttft,
        "within_regression_tolerance": (
            mtp_ttft is not None and plain_ttft is not None and
            100 * (mtp_ttft / plain_ttft - 1) <= TTFT_REGRESSION_TOLERANCE_PERCENT
        ),
        "token_hashes_match": mtp_plain_hashes_valid and mtp_hashes == plain_hashes,
    }
    case["comparisons_vs_main"] = comparisons
    required = ("candidate-no-drafts", "candidate-mtp-after")
    def sample_valid(name: str, rep: int) -> bool:
        item = results.get(f"rep-{rep:02d}-{name}")
        first = primary_request(item) if isinstance(item, dict) else {}
        ttft = first.get("ttft_s")
        return bool(item and item.get("qualified") and item.get("first_request_cold")
                    and isinstance(ttft, (int, float)) and math.isfinite(ttft) and ttft > 0)

    dflash2_valid = None
    if dflash2_comparison is not None:
        dflash2_valid = bool(
            dflash2_comparison["token_hashes_match_main"]
            and all(sample_valid("candidate-dflash2", rep) for rep in range(reps))
        )
        case["dflash2_diagnostic_valid"] = dflash2_valid
        if dflash2_valid:
            case["dflash2_diagnostic_status"] = "measured"
        else:
            capacity_failures = []
            for rep in range(reps):
                item = results.get(f"rep-{rep:02d}-candidate-dflash2")
                if item and item.get("error"):
                    label = f"prompt-{case['prompt_tokens_expected']}-rep-{rep + 1:02d}-candidate-dflash2"
                    capacity_failures.append(context_is_unsupported(OUT / f"{label}.server.log"))
            case["dflash2_diagnostic_status"] = (
                "capacity_limited" if capacity_failures and all(capacity_failures) else "incomplete"
            )

    samples_valid = all(
        comparisons[name]["within_regression_tolerance"]
        and comparisons[name]["token_hashes_match_main"]
        and all(sample_valid(name, rep) for rep in range(reps))
        for name in required
    ) and all(sample_valid("main", rep) for rep in range(reps))
    case["qualified"] = bool(samples_valid)
    case["status"] = "passed" if case["qualified"] else "failed"


def summarize_overall(cases: dict[str, dict], requested: list[int]) -> dict:
    required = (2048, 8192)
    missing = sorted(set(required) - set(requested))
    required_cases = {int(length): case for length, case in cases.items() if int(length) in required}
    skipped = sorted(length for length, case in required_cases.items()
                     if case.get("status") == "skipped_unsupported")
    measured = sorted(length for length, case in required_cases.items()
                      if case.get("status") != "skipped_unsupported")
    passed = (not missing and not skipped and measured == sorted(required)
              and all(required_cases[length].get("qualified") is True for length in measured))
    return {"qualified": passed, "required_contexts": list(required),
            "missing_required_contexts": missing, "skipped_required_contexts": skipped,
            "measured_required_contexts": measured,
            "status": "passed" if passed else "incomplete_or_failed"}


def main(argv=None) -> int:
    global OUT
    parser = rr.add_config_arguments(argparse.ArgumentParser(description=__doc__), "prompt-ordered")
    parser.add_argument("--dflash2-drafter", default=os.environ.get("TF2485_DFLASH2_DRAFTER"),
                        help="local DFlash2 checkpoint directory for the required diagnostic arm")
    parser.add_argument("--prompt-tokens", default="2048,8192",
                        help="comma-separated exact token lengths; required PR gate contexts are 2048 and 8192")
    parser.add_argument("--reps", type=int, default=5,
                        help="fresh-server samples per variant and prompt length")
    parser.add_argument("--solo-chunk-rows", type=int, choices=(4096, 8192), default=4096,
                        help="explicit solo prompt chunk size; run 8192 separately to qualify the 8K-solo path")
    args = parser.parse_args(argv)
    rr.configure(args, "prompt-ordered")
    OUT = rr.OUT
    if not args.dflash2_drafter:
        parser.error("pass --dflash2-drafter or set TF2485_DFLASH2_DRAFTER")
    dflash2_drafter = Path(args.dflash2_drafter).expanduser().resolve()
    if not dflash2_drafter.is_dir():
        parser.error(f"DFlash2 drafter must be an existing local checkpoint directory: {dflash2_drafter}")
    if args.reps < 1:
        parser.error("--reps must be positive")
    try:
        prompt_lengths = [int(value) for value in args.prompt_tokens.split(",")]
        if not prompt_lengths or any(value < 1 for value in prompt_lengths) or len(set(prompt_lengths)) != len(prompt_lengths):
            raise ValueError
    except ValueError:
        parser.error("--prompt-tokens must be unique positive comma-separated integers")
    preflight = rr.gpu_preflight()
    specs = binary_specs()
    runtime_environment = rr.benchmark_env(args.solo_chunk_rows)[1]
    source = rr.PROJECT / "tools/zig/prompts/p3.ids"
    summary = {
        "gpu": preflight,
        "baseline_commit": rr.git(rr.BASE, "rev-parse", "HEAD"),
        "candidate_commit": rr.git(rr.PROJECT, "rev-parse", "HEAD"),
        "baseline_git_state": rr.git_state(rr.BASE),
        "candidate_git_state": rr.git_state(rr.PROJECT),
        "binary_sha256": {name: rr.sha256(binary) for name, binary, _ in specs},
        "checkpoint": str(rr.TARGET),
        "settings": {"prompt_cache_gib": 0, "streams": 1, "reply_tokens": 64,
                     "repetitions": args.reps, "order": "alternates forward/reverse by repetition",
                     "primary_metric": "first request only", "sampling": "greedy", "thinking": "off",
                     "nvfp4_tensor_cores": True,
                     "mtp_batch": 1,
                     "solo_chunk_rows": args.solo_chunk_rows,
                     "mtp_initial_prompt_rows": 256,
                     "dflash2_diagnostic_only": True},
        "runtime_environment": runtime_environment,
        "dflash2_drafter": {
            "path": str(dflash2_drafter),
            "metadata_sha256": {
                name: rr.sha256(dflash2_drafter / name)
                if (dflash2_drafter / name).is_file() else None
                for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")
            },
        },
        "cases": {},
    }
    required_contexts = (2048, 8192)
    for count in prompt_lengths:
        context = context_for(count)
        prompt_path = OUT / f"prompt-{count}.ids.json"
        prompt_sha = write_prompt(source, count, prompt_path)
        case = {"prompt_tokens_expected": count, "context": context,
                "prompt_ids_sha256": prompt_sha, "results": {}, "supported": True,
                "required_for_pr_gate": count in required_contexts}
        print(f"CASE prompt={count} context={context} reps={args.reps}", flush=True)
        order = list(specs)
        for rep in range(args.reps):
            trial_order = order if rep % 2 == 0 else list(reversed(order))
            for name, binary, mtp in trial_order:
                label = f"prompt-{count}-rep-{rep + 1:02d}-{name}"
                print(f"BEGIN {label} mtp={mtp}", flush=True)
                try:
                    drafter_path = dflash2_drafter if name == "candidate-dflash2" else None
                    result = measure(label, binary, mtp, context, prompt_path, count,
                                     drafter_path, args.solo_chunk_rows)
                except Exception as exc:
                    result = {"qualified": False, "error": f"{type(exc).__name__}: {exc}"}
                case["results"][f"rep-{rep:02d}-{name}"] = result
                first = result.get("primary_first_request", {})
                if first.get("ttft_s") is not None:
                    print(f"END {label} first_ttft={first['ttft_s']:.6f}s cache={first.get('cache_state')} cached={first.get('cached_tokens')}", flush=True)
                else:
                    print(f"END {label} failed={result.get('error')}", flush=True)
                if result.get("error"):
                    server_log = OUT / f"{label}.server.log"
                    if baseline_capacity_refusal(name, result, server_log):
                        case["supported"] = False
                        case["unsupported_reason"] = "main target-only model does not fit this context on the reserved GPU"
                        break
            if case.get("supported") is False:
                break
        summarize_case(case, specs, args.reps)
        summary["cases"][str(count)] = case
        print(f"CASE_RESULT prompt={count} qualified={case['qualified']} comparisons={case.get('comparisons_vs_main')}", flush=True)
    summary.update(summarize_overall(summary["cases"], prompt_lengths))
    (OUT / "prompt_ordered.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Paired cold-prompt A/B measurements", "",
             "- Each sample uses a fresh server and a cold first request with prompt cache disabled.",
             "- Primary TTFT is the first request only; bench_cache repeated and concurrent rows remain secondary evidence.",
             "- Forward and reverse variant order alternate by repetition.", "",
             f"- Runtime environment explicitly sets MTP batch=1 and solo chunk rows={args.solo_chunk_rows}; inherited overrides are recorded in JSON.", "",
             "| Prompt | Context | Variant | First-request TTFT samples (s) | Median (s) |", "|---:|---:|---|---|---:|"]
    for count, case in summary["cases"].items():
        for name, metric in case.get("summary", {}).items():
            lines.append(f"| {count} | {case['context']} | {name} | {metric['samples']} | {metric['median_first_ttft_s']} |")
    lines += ["", "## Gate comparisons", "", "| Prompt | Comparison | Δ median | Within 0.1% | Hashes match |", "|---:|---|---:|---|---|"]
    for count, case in summary["cases"].items():
        for name, comp in case.get("comparisons_vs_main", {}).items():
            delta = comp.get("delta_percent_vs_main", comp.get("delta_percent"))
            hash_match = comp.get("token_hashes_match_main", comp.get("token_hashes_match"))
            within_tolerance = comp.get("within_regression_tolerance", "diagnostic")
            lines.append(f"| {count} | {name} | {delta}% | {within_tolerance} | {hash_match} |")
    lines += ["", "## Draft diagnostics", "", "| Prompt | Variant | Median server tok/s | Median draft acceptance |", "|---:|---|---:|---:|"]
    for count, case in summary["cases"].items():
        for name, metric in case.get("summary", {}).items():
            lines.append(f"| {count} | {name} | {metric['median_server_tok_s']} | {metric['median_draft_acceptance']} |")
    diagnostic_cases = [(count, case) for count, case in summary["cases"].items()
                        if "dflash2_diagnostic_status" in case]
    if diagnostic_cases:
        lines += ["", "## DFlash2 diagnostic availability", "",
                  "| Prompt | Status | Matching cold samples |", "|---:|---|---:|"]
        for count, case in diagnostic_cases:
            metric = case.get("summary", {}).get("candidate-dflash2", {})
            lines.append(f"| {count} | {case['dflash2_diagnostic_status']} | {metric.get('qualified_samples', 0)} |")
    unsupported = [(count, case.get("unsupported_reason")) for count, case in summary["cases"].items()
                   if case.get("supported") is False]
    if unsupported:
        lines += ["", "## Unsupported contexts", ""]
        lines.extend(f"- {count} prompt tokens: {reason}." for count, reason in unsupported)
    lines += ["", f"Overall qualification: `{summary['qualified']}` ({summary['status']})."]
    if summary["missing_required_contexts"]:
        lines.append(f"- Required contexts omitted: {summary['missing_required_contexts']}.")
    if summary["skipped_required_contexts"]:
        lines.append(f"- Required contexts explicitly skipped as unsupported: {summary['skipped_required_contexts']}.")
    lines.append("")
    (OUT / "prompt_ordered.md").write_text("\n".join(lines))
    return 0 if summary["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
