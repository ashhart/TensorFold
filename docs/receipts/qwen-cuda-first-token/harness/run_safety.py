"""Check seeded sampling and stream cancellation outside TTFT measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import bench_concurrent as bench


SCHEMA = "pr836_ttft_safety_v1"


def require_uncached(row: dict) -> None:
    """Reject hidden prompt-state reuse in the Qwen3.5 CUDA safety controls."""
    usage = row.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    if details.get("cached_tokens") != 0 or row.get("cached_tokens") != 0:
        raise ValueError("Qwen3.5 CUDA safety control unexpectedly used prompt cache")


def health(base: str) -> dict:
    """Read live connection count from the native server."""
    with urllib.request.urlopen(base + "/health", timeout=10) as response:
        value = json.load(response)
    if value.get("status") != "ok" or not isinstance(value.get("live"), dict):
        raise ValueError("server health unavailable")
    return value


def cancel_after_first(base: str, model: str) -> dict:
    """Close a new stream on its first nonempty SSE text piece."""
    body = {
        "model": model, "prompt": "Cancellation probe 62917. Write an extended technical essay.",
        "max_tokens": 1024, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True}, "ignore_eos": True,
        "seed": 1234,
    }
    request = urllib.request.Request(
        base + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    before = health(base)
    response_id = None
    saw_text = False
    saw_done = False
    with urllib.request.urlopen(request, timeout=90) as response:
        for raw in response:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                saw_done = True
                break
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            if chunk.get("error"):
                raise ValueError("cancellation probe received a server error")
            response_id = chunk.get("id") or response_id
            if any((choice.get("text") or (choice.get("delta") or {}).get("content"))
                   for choice in chunk.get("choices", [])):
                saw_text = True
                break
    if not saw_text or saw_done:
        raise ValueError("probe did not disconnect before completion")
    deadline = time.monotonic() + 30
    after = health(base)
    while after["live"].get("connections") != 0 and time.monotonic() < deadline:
        time.sleep(.25)
        after = health(base)
    if after["live"].get("connections") != 0:
        raise ValueError("cancelled stream remained live")
    return {
        "request_body_sha256": hashlib.sha256(json.dumps(body).encode()).hexdigest(),
        "response_id": response_id, "saw_nonempty_piece": saw_text,
        "saw_done_before_close": saw_done,
        "before_live": before["live"], "after_live": after["live"],
    }


def save(path: Path, report: dict) -> None:
    """Preserve partial evidence before each validation step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def run(base: str, model: str, label: str, corpus: Path, output: Path) -> dict:
    """Run two seeded samples, a cancellation, then one healthy control."""
    data = json.loads(corpus.read_text())
    anchors = data["anchors"]
    report = {"schema": SCHEMA, "complete": False, "label": label,
              "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
              "sampled": {}, "undrafted": {}, "cancel": {}}
    save(output, report)
    for case in anchors:
        row = bench.stream(base, model, case, 64, .7, 1234)
        report["sampled"][case["name"]] = row
        save(output, report)
        if (row.get("error") or not row.get("complete") or row.get("tokens") != 64
                or row.get("finish_reason") != "length"
                or not row.get("output_sha256") or not row.get("token_sha")):
            raise ValueError(f"{case['name']}: sampled control failed")
        require_uncached(row)
        plain = bench.stream(base, model, case, 64, .7, 1234, draft=False)
        report["undrafted"][case["name"]] = plain
        save(output, report)
        if (plain.get("error") or plain.get("complete") is not True
                or plain.get("tokens") != 64
                or (plain.get("output_sha256"), plain.get("token_sha"))
                != (row["output_sha256"], row["token_sha"])):
            raise ValueError(f"{case['name']}: drafted and undrafted replies differ")
        require_uncached(plain)
    report["cancel"] = cancel_after_first(base, model)
    save(output, report)
    control = bench.stream(base, model, anchors[0], 16, 0.0, 1234)
    report["post_cancel_control"] = control
    save(output, report)
    if control.get("error") or not control.get("complete") or control.get("tokens") != 16:
        raise ValueError("post-cancellation control failed")
    require_uncached(control)
    report["complete"] = True
    save(output, report)
    return report


def main() -> None:
    """Write a separate, non-timing safety receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.model, args.label, args.corpus, args.output)


if __name__ == "__main__":
    main()
