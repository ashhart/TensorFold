"""Privately verify a synthetic chat model gate without measuring accumulator hot-path speed."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request

REPLY_TOKENS = 64
REPETITIONS = 3
LEVELS = (1, 2)
SEED = 1234
DEADLINE_SECONDS = 600.0
HASH = re.compile(r"[0-9a-f]{64}\Z")


def token_hash(tokens: list[int]) -> str:
    """Reproduce the server's decimal-token-ID SHA256 convention."""
    return hashlib.sha256(",".join(map(str, tokens)).encode("ascii")).hexdigest()


def fixture_hash(messages: list[dict[str, str]]) -> str:
    """Fingerprint the exact public synthetic message array."""
    raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def fixtures() -> dict[str, list[dict[str, str]]]:
    """Use only synthetic prose, code and a 34-message conversation."""
    history = [{"role": "system", "content": "You are a concise assistant. Treat the turns as a test dialogue."}]
    for i in range(16):
        history.extend([
            {"role": "user", "content": f"Turn {i:02d}: explain alpha and beta labels for a small test."},
            {"role": "assistant", "content": f"Turn {i:02d}: alpha marks the first group and beta marks the second."},
        ])
    history.append({"role": "user", "content": "Summarise this test dialogue and explain why consistent labels help."})
    return {
        "multiturn": history,
        "prose": [{"role": "user", "content": "Explain matrix multiplication on a GPU simply, with an example."}],
        "code": [{"role": "user", "content": "Write a short Python Fibonacci function and explain it."}],
    }


def equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Require independently validated token IDs and both full fingerprints."""
    return bool(
        left.get("valid") and right.get("valid")
        and left["token_ids"] == right["token_ids"]
        and left["token_sha256"] == right["token_sha256"]
        and left["output_sha256"] == right["output_sha256"]
    )


class Client:
    """Collect complete evidence while enforcing a single bounded client deadline."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.url = args.url.rstrip("/")
        self.output = args.output
        self.started = time.monotonic()
        self.deadline = self.started + DEADLINE_SECONDS
        self.key = os.environ.get(args.api_key_env) if args.api_key_env else None
        if args.api_key_env and not self.key:
            raise ValueError("the explicitly selected API-key environment variable is empty")
        helpers = [args.tools / (name + ".py") for name in ("bench_evidence", "bench_timing", "bench_metrics")]
        if not all(path.is_file() for path in helpers):
            raise ValueError("the selected repository benchmark helpers are unavailable")
        sys.path.insert(0, str(args.tools.resolve()))
        self.evidence = importlib.import_module("bench_evidence")
        self.timing = importlib.import_module("bench_timing")
        self.metrics = importlib.import_module("bench_metrics")
        self.data = fixtures()
        self.model = args.model
        self.prompt_counts: dict[str, int] = {}
        self.report: dict[str, Any] = {
            "schema_version": 1, "label": args.output.stem,
            "requested_reply_tokens": REPLY_TOKENS, "repetitions": REPETITIONS,
            "levels": list(LEVELS), "temperatures": [0.0, 1.0],
            "scope": "Ordinary checkpoint template model no-regression gate; not accumulator hot-path speed evidence.",
            "resume_test": "not_run: prompt cache is required to be disabled",
            "helper_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in helpers},
            "fixtures": {
                name: {"messages": value, "message_count": len(value), "sha256": fixture_hash(value)}
                for name, value in self.data.items()
            },
            "tokenize": {}, "references": [], "cells": [], "errors": [], "qualified": False,
        }

    def remaining(self) -> float:
        """Reject new work after the shared deadline."""
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("whole_client_deadline")
        return value

    def request(self, path: str, body: dict[str, Any] | None = None) -> urllib.request.Request:
        """Construct one request without recording the origin or credentials."""
        raw = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=raw, headers={"Content-Type": "application/json"})
        if self.key:
            request.add_header("Authorization", "Bearer " + self.key)
        return request

    def json_request(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Bound metadata and tokenizer calls by the same deadline."""
        with urllib.request.urlopen(self.request(path, body), timeout=min(120.0, self.remaining())) as response:
            result = json.load(response)
        if not isinstance(result, dict) or "error" in result:
            raise ValueError("invalid_metadata_response")
        return result

    def save(self) -> None:
        """Checkpoint evidence atomically after each complete cell."""
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.report, indent=2) + "\n")
        temporary.replace(self.output)

    def stream(
        self, name: str, temperature: float, seed: int, draft: bool = True,
        gate: threading.Barrier | None = None,
    ) -> dict[str, Any]:
        """Stream a real message array and regenerate the complete token fingerprint."""
        result: dict[str, Any] = {
            "fixture": name, "temperature": temperature, "seed": seed, "draft": draft,
            "sent": None, "first": None, "last": None, "completed": None,
            "complete": False, "valid": False, "error": None, "pieces": [],
        }
        evidence = self.evidence.Evidence()
        usage: dict[str, Any] = {}
        runtime: dict[str, Any] = {}
        content: list[str] = []
        reasoning: list[str] = []
        try:
            body: dict[str, Any] = {
                "model": self.model, "messages": self.data[name], "max_tokens": REPLY_TOKENS,
                "temperature": temperature, "seed": seed, "draft": draft,
                "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
                "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False},
            }
            if temperature > 0:
                body.update(top_k=20, top_p=0.95)
            request = self.request("/v1/chat/completions", body)
            if gate is not None:
                gate.wait(timeout=min(30.0, self.remaining()))
            result["sent"] = time.perf_counter()
            with urllib.request.urlopen(request, timeout=min(180.0, self.remaining())) as response:
                while True:
                    remaining = self.remaining()
                    # Refresh a stalled read's timeout so streaming cannot exceed the whole-client budget.
                    try:
                        response.fp.raw._sock.settimeout(min(180.0, remaining))
                    except AttributeError:
                        pass
                    raw = response.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8").strip()
                    if line == "data: [DONE]":
                        result["complete"] = True
                        break
                    if not line.startswith("data:"):
                        continue
                    chunk = json.loads(line[5:])
                    if "error" in chunk:
                        raise ValueError("server_stream_error")
                    usage = chunk.get("usage") or usage
                    runtime.update(chunk.get("tensorfold") or {})
                    for choice in chunk.get("choices", []):
                        evidence.add(choice)
                        delta = choice.get("delta") or {}
                        text = choice.get("text") or delta.get("content") or ""
                        thought = delta.get("reasoning_content") or ""
                        content.append(text)
                        reasoning.append(thought)
                        if text or thought:
                            result["pieces"].append((time.perf_counter(), len(text or thought)))
            result["completed"] = time.perf_counter()
        except Exception as error:
            result["error"] = type(error).__name__
            result["completed"] = time.perf_counter()
            if gate is not None:
                gate.abort()
        result.update(evidence.finish(usage, runtime, result["complete"]))
        result["token_ids"] = runtime.get("token_ids")
        result["content"] = "".join(content)
        result["reasoning_content"] = "".join(reasoning)
        result["ttft_s"] = result["decode_tps"] = None
        if result["pieces"]:
            result["first"], result["last"] = result["pieces"][0][0], result["pieces"][-1][0]
            result["ttft_s"] = result["first"] - result["sent"]
            if len(result["pieces"]) >= 4 and result["last"] > result["first"]:
                result["decode_tps"] = (REPLY_TOKENS - 1) / (result["last"] - result["first"])
        result.update(self.timing.request(result))
        result.update(self.metrics.request(result))
        result["server_timing"] = {
            key: runtime.get(key) for key in ("time_to_first_token", "prefill_seconds", "seconds")
        }
        result["validation_errors"] = self.validate(result)
        result["valid"] = not result["validation_errors"]
        return result

    def validate(self, run: dict[str, Any]) -> list[str]:
        """Fail closed on incomplete, reused, missing, malformed or unmeasurable evidence."""
        errors = []
        if not run["complete"] or run["error"]:
            errors.append("incomplete_or_error")
        if run.get("tokens") != REPLY_TOKENS:
            errors.append("reply_token_count")
        if run.get("cached_tokens") != 0 or run.get("cache_state") != "cold":
            errors.append("not_verified_cold")
        count = run.get("prompt_tokens")
        if type(count) is not int or not 0 < count < 8192:
            errors.append("prompt_token_bound")
        expected = self.prompt_counts.get(run["fixture"])
        if expected is not None and count != expected:
            errors.append("tokenize_prompt_count_differs")
        ids = run.get("token_ids")
        valid_ids = isinstance(ids, list) and len(ids) == REPLY_TOKENS and all(
            type(value) is int and 0 <= value <= 0xffffffff for value in ids
        )
        if not valid_ids:
            errors.append("missing_or_invalid_full_token_ids")
        for field in ("token_sha256", "output_sha256"):
            if not isinstance(run.get(field), str) or HASH.fullmatch(run[field]) is None:
                errors.append("invalid_" + field)
        if valid_ids and run.get("token_sha256") != token_hash(ids):
            errors.append("recomputed_token_hash_differs")
        for field in ("ttft_s", "decode_tps", "elapsed_s"):
            value = run.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                errors.append("invalid_" + field)
        return errors

    def snapshots(self) -> None:
        """Snapshot generation-on and history-only rendered IDs after the first cold inference."""
        for name, messages in self.data.items():
            snapshots = {}
            for generation in (True, False):
                result = self.json_request("/tokenize", {
                    "model": self.model, "messages": messages, "add_generation_prompt": generation,
                    "chat_template_kwargs": {"enable_thinking": False},
                })
                ids = result.get("tokens")
                if not isinstance(ids, list) or not ids or any(
                    type(value) is not int or not 0 <= value <= 0xffffffff for value in ids
                ):
                    raise ValueError("invalid_tokenize_ids")
                if type(result.get("count")) is not int or result["count"] != len(ids):
                    raise ValueError("tokenize_count_or_context_bound")
                if len(ids) + REPLY_TOKENS > 8192:
                    raise ValueError("tokenize_count_or_context_bound")
                snapshots["generation" if generation else "history"] = {
                    "count": len(ids), "token_ids": ids, "token_sha256": token_hash(ids),
                }
                if generation:
                    self.prompt_counts[name] = len(ids)
            self.report["tokenize"][name] = snapshots
        first = self.report["first_cold"]
        first["validation_errors"] = self.validate(first)
        first["valid"] = not first["validation_errors"]

    def run(self) -> int:
        """Capture each full cell before stopping at any exactness failure."""
        if not self.model:
            models = self.json_request("/v1/models").get("data")
            if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0].get("id"), str):
                raise ValueError("expected_one_served_model")
            self.model = models[0]["id"]
        self.report["first_cold"] = self.stream("multiturn", 0.0, SEED)
        self.save()
        if not self.report["first_cold"]["valid"]:
            raise ValueError("first_cold_request_invalid")
        self.snapshots()
        self.save()
        if not self.report["first_cold"]["valid"]:
            raise ValueError("first_cold_tokenize_mismatch")
        for name in self.data:
            for temperature in (0.0, 1.0):
                seeds = [SEED] if temperature == 0 else [SEED, SEED + 1]
                references = {}
                for seed in seeds:
                    alone = self.stream(name, temperature, seed)
                    plain = self.stream(name, temperature, seed, draft=False)
                    references[seed] = alone
                    self.report["references"].append({
                        "fixture": name, "temperature": temperature, "seed": seed,
                        "alone": alone, "plain": plain, "equal": equal(alone, plain),
                    })
                for streams in LEVELS:
                    cell: dict[str, Any] = {
                        "fixture": name, "temperature": temperature, "streams": streams, "reps": [],
                    }
                    for repetition in range(REPETITIONS):
                        gate = threading.Barrier(streams)
                        requested_seeds = [SEED + i if temperature else SEED for i in range(streams)]
                        with ThreadPoolExecutor(max_workers=streams) as pool:
                            futures = [pool.submit(self.stream, name, temperature, seed, True, gate)
                                       for seed in requested_seeds]
                            runs = [future.result() for future in futures]
                        timing = self.timing.workload(runs)
                        aggregate_tps = None
                        if all(run["valid"] for run in runs):
                            span = max(run["last"] for run in runs) - min(run["first"] for run in runs)
                            aggregate_tps = streams * (REPLY_TOKENS - 1) / span
                        comparisons = [equal(references[seed], run) for seed, run in zip(requested_seeds, runs)]
                        cell["reps"].append({
                            "repetition": repetition, "runs": runs, "comparisons_to_alone": comparisons,
                            "timing": timing, "aggregate_tps": aggregate_tps,
                        })
                    valid_runs = all(run["valid"] for rep in cell["reps"] for run in rep["runs"])
                    matches = all(all(rep["comparisons_to_alone"]) for rep in cell["reps"])
                    plain_matches = all(
                        ref["equal"] for ref in self.report["references"]
                        if ref["fixture"] == name and ref["temperature"] == temperature
                    )
                    cell["qualified"] = valid_runs and matches and plain_matches
                    if valid_runs:
                        cell["median_aggregate_tps"] = statistics.median(
                            rep["aggregate_tps"] for rep in cell["reps"]
                        )
                        cell["median_end_to_end_tps"] = statistics.median(
                            rep["timing"]["end_to_end_tps"] for rep in cell["reps"]
                        )
                        cell["median_request_ttft_s"] = statistics.median(
                            run["ttft_s"] for rep in cell["reps"] for run in rep["runs"]
                        )
                    self.report["cells"].append(cell)
                    self.save()
                    if not cell["qualified"]:
                        raise ValueError("current_cell_failed_validation_or_exactness")
        self.report["qualified"] = True
        self.save()
        return 0


def main() -> int:
    """Run the private controller client and always retain partial evidence on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--api-key-env")
    args = parser.parse_args()
    client = Client(args)
    try:
        result = client.run()
    except Exception as error:
        client.report["errors"].append(type(error).__name__ + ": " + str(error))
        result = 1
    client.report["elapsed_s"] = time.monotonic() - client.started
    client.save()
    print(json.dumps({
        "qualified": client.report["qualified"], "cells": len(client.report["cells"]),
        "elapsed_s": client.report["elapsed_s"], "errors": client.report["errors"],
    }))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
