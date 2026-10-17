#!/usr/bin/env python3
"""Compare serial and shared native HTTP replies on identical public fixtures, with one server at a time."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request


BASE = "http://127.0.0.1:52417"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=3) as response:
        return response.read().decode()


def request(case, tokens, gate):
    payload = {
        "model": "batch-gate", "messages": [{"role": "user", "content": case["text"]}],
        "max_tokens": tokens, "temperature": 0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": case["thinking"]},
    }
    gate.wait()
    start = time.monotonic()
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    choice = result["choices"][0]
    usage = result["usage"]
    token_sha = result["tensorfold"].get("token_sha")
    if not isinstance(token_sha, str) or len(token_sha) != 12 or any(c not in "0123456789abcdef" for c in token_sha):
        raise RuntimeError("missing native token fingerprint")
    prompt = usage["prompt_tokens"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    tokens = usage["completion_tokens"]
    if type(prompt) is not int or prompt < 1 or type(cached) is not int or cached != 0 or type(tokens) is not int or tokens < 1:
        raise RuntimeError("invalid token counts or unexpected prefix reuse")
    return {
        "reply": {"message": choice["message"], "finish_reason": choice["finish_reason"]},
        "tokens": tokens, "prompt_tokens": prompt, "cached_tokens": cached, "token_sha": token_sha,
        "seconds": time.monotonic() - start,
    }


def run(args, slots, cases, block):
    log_path = args.output / f"serve-{slots}-block-{block}.log"
    command = [str(args.binary), "serve", str(args.model), "--name", "batch-gate", "--host", "127.0.0.1",
               "--port", "52417", "--context", str(getattr(args, "context", 16384)),
               "--temperature", "0", "--prompt-cache-gib", "0", "--no-update-check"]
    if str(slots) != "default":
        command += ["--parallel", str(slots)]
    env = os.environ.copy()
    if getattr(args, "require_shared_rounds", False):
        env["FZ_BATCH_LOG"] = "1"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        (args.output / "server.pid").write_text(str(process.pid) + "\n")
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"native server exited {process.returncode}; inspect {log_path.name}")
                try:
                    health = json.loads(get("/health"))
                    if health.get("status") == "ok" and not health.get("warming", False):
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("native server startup")
            batches = []
            for iteration in range(args.rounds):
                log_start = log_path.stat().st_size
                stop = threading.Event()
                peaks = [0, 0]

                def monitor():
                    while not stop.is_set():
                        try:
                            for line in get("/v1/metrics").splitlines():
                                for index, key in enumerate(("tensorfold:requests_running", "tensorfold:requests_waiting")):
                                    if line.startswith(key + " "):
                                        peaks[index] = max(peaks[index], int(float(line.split()[1])))
                        except (OSError, ValueError):
                            pass
                        stop.wait(0.05)

                watcher = threading.Thread(target=monitor, daemon=True)
                watcher.start()
                start = time.monotonic()
                try:
                    gate = threading.Barrier(len(cases))
                    with ThreadPoolExecutor(max_workers=len(cases)) as pool:
                        futures = [pool.submit(request, case, args.tokens, gate) for case in cases]
                        results = [future.result() for future in futures]
                    seconds = time.monotonic() - start
                    log_end = log_path.stat().st_size
                finally:
                    stop.set()
                    watcher.join(timeout=4)
                generated = sum(result["tokens"] for result in results)
                with log_path.open("rb") as observed:
                    observed.seek(log_start)
                    rounds = re.findall(rb"flashnext batch: streams (\d+), rows \d+, shared rounds \d+",
                                        observed.read(log_end - log_start))
                batch = {"iteration": iteration, "seconds": seconds, "tokens": generated,
                         "tokens_per_second": generated / seconds, "peak_running": peaks[0],
                         "peak_waiting": peaks[1], "shared_rounds": sum(int(n) > 1 for n in rounds), "results": results}
                batches.append(batch)
                print(json.dumps({"parallel": slots, **{key: batch[key] for key in
                      ("iteration", "seconds", "tokens", "tokens_per_second", "peak_running", "peak_waiting")}}), flush=True)
            return batches
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            (args.output / "server.pid").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--parallel", default="1,default,auto,4",
                        help="comma-separated blocks starting with 1; default omits --parallel")
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--require-shared-rounds", action="store_true",
                        help="enable batch logging and require multi-stream GPU rounds (timings include logging)")
    args = parser.parse_args()
    if args.tokens < 1 or args.rounds < 1 or args.context < 1:
        parser.error("tokens, rounds and context must be positive")
    slots = args.parallel.split(",")
    if slots[0] != "1" or any(value not in {"default", "auto", *(str(n) for n in range(1, 17))} for value in slots):
        parser.error("parallel must start with 1 and contain default, auto or 1..16")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"reply_token_limit": args.tokens, "rounds": args.rounds, "context": args.context,
               "exact": False, "shared_rounds_verified": False,
               "batch_logging": args.require_shared_rounds or "FZ_BATCH_LOG" in os.environ, "runs": {}}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    cases = [json.loads((args.fixtures / f"{name}.json").read_text()) for name in
             ("short", "thinking", "sparse", "sparse-thinking")]
    reference = None
    for block, count in enumerate(slots):
        batches = run(args, count, cases, block)
        for batch in batches:
            batch["block"] = block
        receipt["runs"].setdefault(str(count), []).extend(batches)
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        for batch in batches:
            replies = [{key: result[key] for key in ("reply", "tokens", "prompt_tokens", "token_sha")}
                       for result in batch["results"]]
            if reference is None:
                reference = replies
            if replies != reference:
                receipt["exact"] = False
                (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
                raise RuntimeError("HTTP replies differ from serial reference")
        if count != "1" and not any(batch["peak_running"] > 1 for batch in batches):
            raise RuntimeError("no simultaneous HTTP sessions observed")
        if args.require_shared_rounds and count != "1" and not any(batch["shared_rounds"] > 0 for batch in batches):
            raise RuntimeError("no shared GPU rounds observed; auto may have fallen back to one session")
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["exact"] = True
    receipt["shared_rounds_verified"] = any(key != "1" for key in receipt["runs"]) and all(
        any(batch["shared_rounds"] > 0 for batch in batches)
        for key, batches in receipt["runs"].items() if key != "1")
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
