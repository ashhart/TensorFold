"""Run a fixed Qwen3.8 NVFP4 CUDA receipt with the upstream benchmark clients."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen


MODEL = "qwen-fp8-pair"
PROMPT = "Write a Python function that returns the prime numbers up to a given limit."
CORPUS = "A graph stores named vertices and edges. Visit each neighbor in order, keeping a queue and a set. "
TOOLS = Path(__file__).resolve().parent


def post(base, route, data):
    request = Request(base + route, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def wait_ready(server, base, started, timeout):
    while time.monotonic() - started < timeout:
        if server.poll() is not None:
            raise RuntimeError(f"TensorFold exited during startup: {server.returncode}")
        try:
            with urlopen(base + "/v1/models", timeout=2) as response:
                if any(row["id"] == MODEL for row in json.load(response)["data"]):
                    return time.monotonic() - started
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"TensorFold did not become ready in {timeout} seconds")


def smoke(base, startup, output):
    reply = post(base, "/v1/completions", {
        "model": MODEL, "prompt": PROMPT, "temperature": 0, "seed": 1234, "max_tokens": 32,
    })
    choice = reply["choices"][0]
    tokens = reply["usage"]["completion_tokens"]
    runtime = reply.get("tensorfold", {})
    if not tokens or not choice["text"] or not runtime.get("token_sha256"):
        raise RuntimeError("TensorFold returned incomplete smoke evidence")
    report = {"startup_s": startup, "completion_tokens": tokens,
              "output_sha256": hashlib.sha256(choice["text"].encode()).hexdigest(),
              "finish_reason": choice["finish_reason"], "token_sha256": runtime["token_sha256"],
              "min_rows": runtime.get("min_rows")}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def benchmark(args, base, tool, flags):
    command = [sys.executable, str(TOOLS / tool), base, MODEL, *flags]
    subprocess.run(command, check=True, timeout=args.bench_timeout)


def cold_prefill(args, base):
    data = {"prompt": CORPUS * (max(args.prefill_lengths) // 10 + 1), "add_special_tokens": False}
    ids = post(base, "/tokenize", data)["tokens"]
    if len(ids) < max(args.prefill_lengths):
        raise RuntimeError(f"Fixture tokenizer returned only {len(ids)} token IDs")
    for length in args.prefill_lengths:
        path = args.output / f"prompt-{length}.json"
        path.write_text(json.dumps(ids[:length], separators=(",", ":")) + "\n", encoding="utf-8")
        benchmark(args, base, "bench_cache.py", ["--tokens", "32", "--repeats", "1", "--streams", "1",
                  "--prompt-file", str(path), "--expected-prompt-tokens", str(length), "--require-cold",
                  "--output", str(args.output / f"prefill-{length}.json")])


def run_suite(args, base):
    fixed = ["--tokens", str(args.tokens), "--reps", str(args.reps), "--temperatures", "0"]
    if args.suite in {"decode", "decode-prefill"}:
        benchmark(args, base, "bench_openai.py", [*fixed, "--label", args.output.name,
                                                 "--output", str(args.output / "decode.json")])
    if args.suite == "concurrent":
        benchmark(args, base, "bench_concurrent.py", [*fixed, "--levels", f"1,{args.parallel}", "--alone", "--serial",
                                                     "--label", args.output.name,
                                                     "--output", str(args.output / "concurrent.json")])
    if args.suite in {"prefill", "decode-prefill"}:
        cold_prefill(args, base)


def stop_on_signal(number, _frame):
    raise SystemExit(128 + number)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path, help="path to tensorfold-native")
    parser.add_argument("model", type=Path, help="local Qwen3.8-27B NVFP4 checkpoint")
    parser.add_argument("drafter", type=Path, help="local DFlash2 checkpoint")
    parser.add_argument("output", type=Path, help="new directory for benchmark receipts")
    parser.add_argument("--suite", choices=("smoke", "decode", "prefill", "decode-prefill", "concurrent"),
                        default="decode-prefill")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--prefill-lengths", type=lambda v: [int(n) for n in v.split(",")],
                        default=[2048, 8192, 32768, 65536, 130816])
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--bench-timeout", type=int, default=900)
    args = parser.parse_args()
    if not args.server.is_file() or not args.model.is_dir() or not args.drafter.is_dir():
        parser.error("server file, model directory, and drafter directory must exist")
    if args.suite in {"prefill", "decode-prefill"} and any(n < 1 or n + 32 > args.context for n in args.prefill_lengths):
        parser.error("prefill lengths must leave room for 32 reply tokens")
    if args.suite == "concurrent" and args.parallel < 2:
        parser.error("the concurrent suite needs --parallel 2 or wider")
    with socket.socket() as port_check:
        if os.name != "nt":
            port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            port_check.bind(("127.0.0.1", args.port))
        except OSError as exc:
            parser.error(f"loopback port unavailable: {exc}")
    args.output.mkdir(parents=True)
    signal.signal(signal.SIGTERM, stop_on_signal)
    base = f"http://127.0.0.1:{args.port}"
    command = [str(args.server.resolve()), "serve", str(args.model.resolve()), "--backend", "cuda",
               "--name", MODEL, "--host", "127.0.0.1", "--port", str(args.port),
               "--context", str(args.context), "--parallel", str(args.parallel), "--kv-cache", "fp8",
               "--prompt-cache-gib", "0", "--temperature", "0", "--thinking", "--reasoning-effort", "low",
               "--drafter", str(args.drafter.resolve()), "--drafter-bits", "4", "--no-update-check"]
    env = {**os.environ, "TF_DRAFT_NODES": "15", "TF_NO_TREES": "0"}
    with tempfile.TemporaryFile() as logs:
        started = time.monotonic()
        server = subprocess.Popen(command, stdout=logs, stderr=subprocess.STDOUT, env=env)
        try:
            startup = wait_ready(server, base, started, args.startup_timeout)
            smoke(base, startup, args.output / "smoke.json")
            run_suite(args, base)
        except Exception:
            logs.seek(max(0, logs.tell() - 4096))
            print(logs.read().decode(errors="replace"), file=sys.stderr)
            raise
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=2)


if __name__ == "__main__":
    main()
