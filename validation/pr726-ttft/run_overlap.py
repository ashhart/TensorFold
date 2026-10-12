"""Collect a bounded BF16 text overlap panel from one already-running server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import signal
from typing import Any
import urllib.request

import bench_concurrent as bench


SCHEMA = "pr726_pr608_overlap_text_r3"
SHORT_NAMES = ("code", "chat")
PROMPT_TOKENS = {"code": 14, "chat": 31}
PROMPTS = {
    "code": ("completion", "Write a short Python function that computes the Fibonacci sequence and explain it."),
    "chat": ("chat", "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."),
}
SEED = 1234
SHORT_OUTPUT = 256
LONG_OUTPUT = 32


def digest(path: Path) -> str:
    """Hash exact source or input bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_cache(arm: str, case: str, stage: str) -> int:
    """Return the source-derived cached depth for this admitted request."""
    if stage == "initial" or arm in ("A", "B") and case in SHORT_NAMES:
        return 0
    if case in SHORT_NAMES:
        return PROMPT_TOKENS[case]
    return 4096 if arm in ("B", "D") else 0


def long_inputs(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require rendered chat history exposing PR726's planned 4096 mark."""
    data = json.loads(path.read_text())
    if data.get("schema") != "pr726_chat_history_fixture_v1":
        raise ValueError("unrecognized long-input manifest")
    messages, rendered = data.get("messages"), data.get("rendered")
    if not isinstance(messages, dict) or not isinstance(rendered, dict):
        raise ValueError("missing literal messages or rendered IDs")
    first, next_ = rendered["initial"], rendered["followup"]
    histories = (first["history"], next_["history"])
    full = (first["generation"], next_["generation"])
    vocab_size = data.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("missing tokenizer vocabulary size")
    if any(not isinstance(ids, list) or any(type(x) is not int or x < 0 or x >= vocab_size
                                            for x in ids) for ids in (*histories, *full)):
        raise ValueError("rendered token IDs are invalid")
    if (not 4096 <= len(histories[0]) < 8192 or len(full[1]) >= 8192
            or histories[0] != full[0][:len(histories[0])]
            or histories[1] != full[1][:len(histories[1])]
            or histories[0] != histories[1][:len(histories[0])]
            or full[0][:4096] != full[1][:4096]
            or data.get("planned_mark") != 4096 or data.get("lookahead") != 0
            or data.get("history_len") != len(histories[0])
            or data.get("followup_history_len") != len(histories[1])):
        raise ValueError("chat history does not expose the 4096 mark")
    for name in ("initial", "followup"):
        if not isinstance(messages[name], list) or not messages[name]:
            raise ValueError("missing chat messages")
        for kind in ("history", "generation"):
            digest = hashlib.sha256(",".join(map(str, rendered[name][kind])).encode()).hexdigest()
            if data["id_sha256"][name][kind] != digest:
                raise ValueError("rendered token hash drift")
    if messages["followup"][:len(messages["initial"])] != messages["initial"]:
        raise ValueError("follow-up changed the prior chat messages")
    if any(not isinstance(data.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", data[key])
           for key in ("tokenizer_sha256", "tokenizer_config_sha256", "chat_template_sha256")):
        raise ValueError("missing pinned tokenizer/template identity")
    return (
        {"name": "long_initial", "kind": "chat", "messages": messages["initial"],
         "prompt_tokens": len(full[0])},
        {"name": "long_followup", "kind": "chat", "messages": messages["followup"],
         "prompt_tokens": len(full[1])},
    )


def tokenize_preflight(base: str, manifest: Path) -> dict[str, Any]:
    """Require the live server's exact chat rendering to match the manifest."""
    data = json.loads(manifest.read_text())
    result: dict[str, Any] = {}
    for name in ("initial", "followup"):
        result[name] = {}
        for generation, kind in ((False, "history"), (True, "generation")):
            body = {
                "messages": data["messages"][name],
                "add_generation_prompt": generation,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            req = urllib.request.Request(
                base + "/tokenize", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                actual = json.load(response)
            expected = data["rendered"][name][kind]
            if actual.get("tokens") != expected or actual.get("count") != len(expected):
                raise ValueError(f"{name}/{kind}: live /tokenize IDs differ from pinned fixture")
            result[name][kind] = {
                "count": len(expected),
                "sha256": hashlib.sha256(",".join(map(str, expected)).encode()).hexdigest(),
            }
    return result


def validate_row(
    row: dict[str, Any], *, item: dict[str, Any], tokens: int, cached: int,
) -> None:
    """Fail on incomplete work, wrong cache depth, changed body or SSE clocks."""
    if row.get("error") or row.get("complete") is not True:
        raise ValueError(f"{item['name']}: incomplete SSE response")
    # The 32-token long-prefix check is a correctness cell: a speculative
    # response may arrive in fewer than four text pieces. Short 256-token
    # cells need multiple pieces to support their decode-timing diagnostic.
    if tokens == SHORT_OUTPUT and row.get("unmeasured"):
        raise ValueError(f"{item['name']}: unmeasured short response")
    if row.get("tokens") != tokens or row.get("finish_reason") != "length":
        raise ValueError(f"{item['name']}: completion count or finish reason drift")
    body = row.get("request_body") or {}
    expected = {
        "model": body.get("model"), "max_tokens": tokens, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
        "ignore_eos": True, "seed": SEED,
    }
    if item["kind"] == "chat":
        expected["messages"] = item.get("messages") or [{"role": "user", "content": item["prompt"]}]
        expected["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        expected["prompt"] = item["prompt"]
    if body != expected or row.get("request_body_sha256") != hashlib.sha256(
        json.dumps(expected).encode()
    ).hexdigest():
        raise ValueError(f"{item['name']}: literal body drift")
    route = "/v1/chat/completions" if item["kind"] == "chat" else "/v1/completions"
    if not isinstance(row.get("request_url"), str) or not row["request_url"].endswith(route):
        raise ValueError(f"{item['name']}: request route drift")
    if not isinstance(row.get("response_id"), str) or not row["response_id"]:
        raise ValueError(f"{item['name']}: missing response ID")
    usage = row.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    prompt_tokens = item.get("prompt_tokens", PROMPT_TOKENS.get(item["name"]))
    if (usage.get("prompt_tokens"), usage.get("completion_tokens"),
        details.get("cached_tokens"), row.get("cached_tokens")) != (
        prompt_tokens, tokens, cached, cached
    ):
        raise ValueError(f"{item['name']}: prompt/output/cache usage drift")
    if row.get("cache_state") != ("reused" if cached else "cold"):
        raise ValueError(f"{item['name']}: cache-state label drift")
    runtime = row.get("runtime") or {}
    if runtime.get("token_sha") != row.get("token_sha"):
        raise ValueError(f"{item['name']}: token fingerprint drift")
    for field, widths in (("output_sha256", {64}), ("token_sha", {12, 64})):
        value = row.get(field)
        if not isinstance(value, str) or len(value) not in widths or not re.fullmatch(r"[0-9a-f]+", value):
            raise ValueError(f"{item['name']}: missing {field}")
    clocks = [row.get(key) for key in ("sent", "first", "last", "done")]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in clocks):
        raise ValueError(f"{item['name']}: missing clocks")
    sent, first, last, done = clocks
    if not sent < first <= last <= done or (tokens == SHORT_OUTPUT and first == last):
        raise ValueError(f"{item['name']}: clock order drift")
    if not math.isclose(row.get("ttft_s", -1), first - sent, abs_tol=1e-7):
        raise ValueError(f"{item['name']}: TTFT drift")
    if not math.isclose(row.get("full_response_s", -1), done - sent, abs_tol=1e-7):
        raise ValueError(f"{item['name']}: full-response drift")


def save(path: Path, report: dict[str, Any]) -> None:
    """Keep partial evidence atomically after each completed stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Run one arm's short and optional grid-prefix panels."""
    root = Path(__file__).resolve().parent
    report: dict[str, Any] = {
        "schema": SCHEMA, "complete": False, "arm": args.arm, "suite": args.suite,
        "model": args.model, "base": args.base,
        "source_sha256": {name: digest(root / name) for name in (
            "run_overlap.py", "bench_concurrent.py", "bench_evidence.py")},
        "short": {}, "long": {},
    }
    if args.suite in ("short", "both"):
        items = {item["name"]: item for item in bench.PROMPTS}
        if tuple(items) != SHORT_NAMES or any(
            (items[name].get("kind"), items[name].get("prompt")) != PROMPTS[name]
            for name in SHORT_NAMES
        ):
            raise ValueError("upstream short-prompt order changed")
        for name in SHORT_NAMES:
            item = items[name]
            stages: dict[str, Any] = {}
            for stage in ("initial", "repeat"):
                row = bench.stream(args.base, args.model, item, SHORT_OUTPUT, 0.0, SEED)
                stages[stage] = row
                report["short"][name] = stages
                save(args.output, report)
                validate_row(row, item=item, tokens=SHORT_OUTPUT,
                             cached=expected_cache(args.arm, name, stage))
            burst = bench.together(
                args.base, args.model, [(item, SEED)] * 16, SHORT_OUTPUT, 0.0,
            )
            stages["burst"] = burst
            save(args.output, report)
            if len(burst) != 16:
                raise ValueError(f"{name}: incomplete concurrent burst")
            for row in burst:
                validate_row(row, item=item, tokens=SHORT_OUTPUT,
                             cached=expected_cache(args.arm, name, "burst"))
                if (row["output_sha256"], row["token_sha"]) != (
                    stages["repeat"]["output_sha256"], stages["repeat"]["token_sha"]
                ):
                    raise ValueError(f"{name}: concurrent reply differs from solo")
    if args.suite in ("long", "both"):
        if args.long_manifest is None:
            raise ValueError("long panel requires --long-manifest")
        initial, followup = long_inputs(args.long_manifest)
        report["long_manifest_sha256"] = digest(args.long_manifest)
        report["tokenize_preflight"] = tokenize_preflight(args.base, args.long_manifest)
        save(args.output, report)
        for stage, item in (("initial", initial), ("followup", followup)):
            row = bench.stream(args.base, args.model, item, LONG_OUTPUT, 0.0, SEED)
            report["long"][stage] = row
            save(args.output, report)
            validate_row(row, item=item, tokens=LONG_OUTPUT,
                         cached=expected_cache(args.arm, "long", stage))
    report["complete"] = True
    save(args.output, report)
    return report


def main() -> None:
    """Parse one arm and report a complete or partial private receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--suite", choices=("short", "long", "both"), default="both")
    parser.add_argument("--long-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("arm deadline")))
    signal.alarm(args.max_seconds)
    try:
        collect(args)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
