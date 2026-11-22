"""Bit-identical token checks across restarts, parallel levels and drafting, against a served model.

Each scenario launches the server fresh (or reuses a running one with --base), asks every
prompt in tools/bench_prompts.json at temperature 0 with return_token_ids, and compares
the token id lists — a flip is reported at its first differing position, never by text.

  python3 tools/exact_restarts.py --bin zig-out/native/bin/tensorfold-native --model /path/to/model \
      [--drafter /path/to/draft] [--port 8080] [--tokens 64] [--base http://127.0.0.1:8080]

Scenarios: (a) two fresh restarts of the same binary, (b) --parallel 1 vs --parallel 4,
each request alone and beside three others in flight, (c) --drafter on vs off. Standard
library only; loopback only; no weights in the repo.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROMPTS = Path(__file__).with_name("bench_prompts.json")
READY_S = 300


class ServerFailed(Exception):
    """The server did not come up: the harness refuses to compare anything."""


def load_prompts(path=PROMPTS):
    return [p for p in json.loads(path.read_text()) if isinstance(p.get("name"), str)]


def request_ids(base, item, tokens, api_key=None, timeout=600):
    """One request's reply tokens: the ids under `tensorfold`, not the text."""
    body = {"model": "served", "max_tokens": tokens, "temperature": 0, "seed": 1,
            "ignore_eos": True, "return_token_ids": True}
    if item["kind"] == "chat":
        url, body["messages"] = base + "/v1/chat/completions", [{"role": "user", "content": item["prompt"]}]
    else:
        url, body["prompt"] = base + "/v1/completions", item["prompt"]
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        reply = json.load(resp)
    ids = reply.get("tensorfold", {}).get("token_ids")
    if not isinstance(ids, list):
        raise RuntimeError(f"{item['name']}: reply carried no tensorfold.token_ids")
    return ids


def collect(base, prompts, tokens, busy=0, api_key=None):
    """Every prompt's ids; `busy` extra copies of each request in flight beside it."""
    out = {}
    def one(item):
        with ThreadPoolExecutor(max_workers=busy + 1) as pool:
            runs = [pool.submit(request_ids, base, item, tokens, api_key) for _ in range(busy + 1)]
            return runs[0].result()
    for item in prompts:
        out[item["name"]] = one(item)
    return out


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def compare(named, runs):
    """`runs`: {label: {name: ids}} — first-diff position per prompt per pair, plus flip counts."""
    labels = list(runs)
    rows, flips = [], 0
    for name in named:
        base = runs[labels[0]].get(name)
        for label in labels[1:]:
            pos = None if base is None else first_diff(base, runs[label].get(name, []))
            rows.append({"prompt": name, "pair": f"{labels[0]} vs {label}", "first_diff": pos})
            flips += pos is not None
    return {"rows": rows, "flips": flips, "total": len(rows)}


class Server:
    """A launched server: argv once per scenario, stopped by terminate."""

    def __init__(self, argv, port):
        self.argv, self.port, self.process = argv, port, None

    def __enter__(self):
        self.process = subprocess.Popen(self.argv, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + READY_S
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = (self.process.stderr.read() or "")[-2000:]
                raise ServerFailed(f"server exited {self.process.returncode}: {tail}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=1).close()
                return self
            except (urllib.error.URLError, OSError):
                time.sleep(0.25)
        self.stop()
        raise ServerFailed(f"server not ready within {READY_S}s")

    def __exit__(self, *exc):
        self.stop()
        return False

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def serve_argv(bin_path, model, port, parallel=None, drafter=None):
    argv = [bin_path, "serve", model, "--host", "127.0.0.1", "--port", str(port)]
    if parallel:
        argv += ["--parallel", str(parallel)]
    if drafter:
        argv += ["--drafter", drafter]
    return argv


def scenario(args, label, parallel=None, drafter=None, busy=0):
    """One fresh server (or the --base one), all prompts collected once."""
    prompts = load_prompts(Path(args.prompts))
    base = args.base or f"http://127.0.0.1:{args.port}"
    if args.base:
        return {label: collect(base, prompts, args.tokens, busy, args.api_key)}
    with Server(serve_argv(args.bin, args.model, args.port, parallel, drafter), args.port):
        return {label: collect(base, prompts, args.tokens, busy, args.api_key)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin", default="zig-out/native/bin/tensorfold-native")
    ap.add_argument("--model", help="served model directory (required without --base)")
    ap.add_argument("--base", help="reuse this running server instead of launching (no restarts then)")
    ap.add_argument("--drafter", help="draft model: adds the drafting on/off comparison")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--prompts", default=str(PROMPTS))
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--skip", action="append", default=[], choices=["restart", "parallel", "busy", "draft"])
    args = ap.parse_args()
    if not args.base and not args.model:
        ap.error("--model is required without --base")

    named = [p["name"] for p in load_prompts(Path(args.prompts))]
    report = {"scenarios": [], "rows": [], "flips": 0, "total": 0}

    def fold(part):
        report["rows"] += part["rows"]
        report["flips"] += part["flips"]
        report["total"] += part["total"]

    if "restart" not in args.skip:
        runs = {}
        for i in (1, 2):
            runs.update(scenario(args, f"restart-{i}"))
        report["scenarios"].append("restart")
        fold(compare(named, runs))

    if "parallel" not in args.skip:
        runs = {}
        runs.update(scenario(args, "parallel-1", parallel=1))
        runs.update(scenario(args, "parallel-4", parallel=4))
        report["scenarios"].append("parallel")
        fold(compare(named, runs))

    if "busy" not in args.skip:
        runs = {}
        runs.update(scenario(args, "alone-4", parallel=4))
        runs.update(scenario(args, "busy-4", parallel=4, busy=3))
        report["scenarios"].append("busy")
        fold(compare(named, runs))

    if args.drafter and "draft" not in args.skip:
        runs = {}
        runs.update(scenario(args, "draft-off"))
        runs.update(scenario(args, "draft-on", drafter=args.drafter))
        report["scenarios"].append("draft")
        fold(compare(named, runs))

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    for row in report["rows"]:
        mark = "identical" if row["first_diff"] is None else f"FLIP at {row['first_diff']}"
        print(f"{row['prompt']:28s} {row['pair']:24s} {mark}")
    print(f"flips {report['flips']}/{report['total']} across {report['scenarios']}")
    return 1 if report["flips"] else 0


if __name__ == "__main__":
    sys.exit(main())
