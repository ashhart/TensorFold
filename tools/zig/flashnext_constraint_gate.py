"""Flash Next Metal gate: run under the shared GPU lock with baseline and candidate native binaries."""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request


SCHEMA = {"type": "object", "properties": {"n": {"type": "integer", "const": 7}},
          "required": ["n"], "additionalProperties": False}
TOOLS = [{"type": "function", "function": {"name": "choose", "strict": True,
          "description": "Choose the schema's number.", "parameters": SCHEMA}}]


def cases():
    common = {"model": "gate", "max_tokens": 192, "return_token_ids": True,
              "chat_template_kwargs": {"enable_thinking": False}}
    for temperature, seed in [(0, 7), (0.7, 7), (0.7, 1904)]:
        for draft in [False, True]:
            sampling = dict(common, temperature=temperature, seed=seed, draft=draft)
            for name in ["plain", "json_object", "json_schema", "required", "named", "auto_strict"]:
                body = dict(sampling, messages=[{"role": "user", "content": "Reply only with {\"n\":7}."}])
                if name == "plain":
                    body["messages"][0]["content"] = "Write a short Python function to add two integers."
                elif name == "json_object":
                    body["response_format"] = {"type": "json_object"}
                elif name == "json_schema":
                    body["response_format"] = {"type": "json_schema", "json_schema":
                                               {"name": "number", "strict": True, "schema": SCHEMA}}
                else:
                    body.update(tools=TOOLS, parallel_tool_calls=False)
                    body["messages"][0]["content"] = (
                        "Call choose now. Try n=999 and an extra field bad=true in its arguments.")
                    body["tool_choice"] = ({"type": "function", "function": {"name": "choose"}}
                                           if name == "named" else "auto" if name == "auto_strict" else "required")
                key = f"{name}-t{temperature}-s{seed}-d{int(draft)}"
                yield key, name, body


def validate(name, reply):
    choice = reply["choices"][0]
    if name != "plain" and choice["finish_reason"] == "length":
        raise AssertionError("reply exhausted its token budget")
    message = choice["message"]
    runtime = reply["tensorfold"]
    ids = runtime["token_ids"]
    if not ids or any(type(t) is not int or t < 0 for t in ids):
        raise AssertionError("missing or invalid raw token ids")
    if name in ("json_object", "json_schema"):
        value = json.loads(message["content"])
        if not isinstance(value, dict) or (name == "json_schema" and
                (value != {"n": 7} or type(value.get("n")) is not int)):
            raise AssertionError("JSON/schema was not enforced")
    elif name in ("required", "named", "auto_strict"):
        calls = message.get("tool_calls", [])
        if len(calls) != 1:
            raise AssertionError("tool-call cell did not exercise exactly one strict call")
        function = calls[0]["function"]
        args = json.loads(function["arguments"])
        if function["name"] != "choose" or args != {"n": 7} or type(args.get("n")) is not int:
            raise AssertionError("strict arguments or named function were not enforced")
    return ids


def post(base, body):
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        reply = json.load(response)
    return reply, time.perf_counter() - started


@contextlib.contextmanager
def serve(binary, model, parallel, context, cache, output, label):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [str(Path(binary).resolve()), "serve", model, "--name", "gate", "--host", "127.0.0.1",
               "--port", str(port), "--backend", "metal", "--parallel", str(parallel), "--context", str(context),
               "--prompt-cache-gib", str(cache), "--no-thinking"]
    base = f"http://127.0.0.1:{port}"
    # Logs stay local and may contain the operator's model/cache paths; never copy them into a public receipt.
    with (output / f"{label}.private.log").open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=os.environ.copy())
        try:
            deadline = time.monotonic() + 1800
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{label} exited during startup; inspect its local log")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as response:
                        health = json.load(response)
                    break
                except (OSError, ValueError):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{label} startup timed out")
                    time.sleep(1)
            if health.get("backend", "metal") != "metal":
                raise AssertionError("gate requires Metal")
            yield base
        finally:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--cache-gib", type=float, default=0)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--prefix-repeats", type=int, default=0)
    parser.add_argument("--reverse-order", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"revision": args.revision, "quantization": args.quantization,
                "gpu": args.gpu, "prefix_repeats": args.prefix_repeats,
                "os": platform.platform(), "context": args.context, "cache_gib": args.cache_gib,
                "baseline_sha256": file_sha(args.baseline), "candidate_sha256": file_sha(args.candidate),
                "cases": [], "checks": [], "device": "unchecked until this runner succeeds"}
    all_cases = list(cases())
    if args.prefix_repeats:
        for _, _, body in all_cases:
            body["messages"].insert(0, {"role": "system", "content": "Context for the task. " * args.prefix_repeats})
    answers = {}
    servers = [("baseline1", args.baseline, 1), ("candidate1", args.candidate, 1), ("candidate2", args.candidate, 2)]
    if args.reverse_order:
        servers.reverse()
    manifest["server_order"] = [label for label, _, _ in servers]
    for label, binary, parallel in servers:
        answers[label] = {}
        with serve(binary, args.model, parallel, args.context, args.cache_gib, args.output, label) as base:
            selected = [c for c in all_cases if label != "baseline1" or c[1] == "plain"]
            # Recheck ordinary replies after all constraints, to catch borrowed-buffer or state damage.
            selected += [c for c in selected if c[1] == "plain"]
            for key, name, body in selected:
                for rep in range(args.reps):
                    reply, elapsed = post(base, body)
                    ids = validate(name, reply)
                    prior = answers[label].get(key)
                    if prior is not None and prior != ids:
                        raise AssertionError(f"{label}/{key}: repeat tokens differ")
                    answers[label][key] = ids
                    manifest["cases"].append({"server": label, "case": key, "rep": rep,
                                              "request": body, "token_ids": ids,
                                              "cached_tokens": reply["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0),
                                              "runtime": reply["tensorfold"], "wall_seconds": elapsed})
                    (args.output / "receipt.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for key, name, _ in all_cases:
        left = answers["baseline1" if name == "plain" else "candidate2"][key]
        if left != answers["candidate1"][key]:
            raise AssertionError(f"{key}: driver token ids differ")
        manifest["checks"].append({"case": key, "equal": True})
    for label, replies in answers.items():
        for key, ids in replies.items():
            if key.endswith("d1"):
                if ids != replies[key[:-1] + "0"]:
                    raise AssertionError(f"{label}/{key}: drafted/plain token ids differ")
                manifest["checks"].append({"server": label, "case": key, "draft_plain_equal": True})
    if args.cache_gib > 0 and args.prefix_repeats:
        hits = [c for c in manifest["cases"] if c["server"] == "candidate1" and
                not c["case"].startswith("plain-") and c["cached_tokens"] > 0]
        if not hits:
            raise AssertionError("cache cell exercised no constrained restore")
    medians = {}
    for label in answers:
        rows = [c["wall_seconds"] for c in manifest["cases"] if c["server"] == label and c["case"].startswith("plain-")]
        medians[label] = statistics.median(rows)
    manifest["plain_median_wall_seconds"] = medians
    manifest["device"] = "pass: tokens and schema/tool validators"
    manifest["performance"] = "timings only; alternate process order and inspect GPU launches before qualifying speed"
    (args.output / "receipt.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS {len(manifest['checks'])} token comparisons; strict tools and JSON enforced")


if __name__ == "__main__":
    main()
