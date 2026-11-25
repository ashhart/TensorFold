"""Public exact receipts for restart, serial/tree, stream, sampling, stop and fresh/resumed conformance."""

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
    """Every prompt's ids; `busy` extra copies of each request in flight beside it, each checked."""
    out = {}
    def one(item):
        with ThreadPoolExecutor(max_workers=busy + 1) as pool:
            runs = [pool.submit(request_ids, base, item, tokens, api_key) for _ in range(busy + 1)]
            ids = [f.result() for f in runs]  # a peer's error fails the run too
        head = ids[0]
        for peer in ids[1:]:
            pos = first_diff(head, peer)
            if pos is not None:
                raise RuntimeError(f"{item['name']}: a busy peer's reply differed at token {pos}")
        return head
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
        if self.process is not None and self.process.stderr is not None:
            self.process.stderr.close()


def serve_argv(bin_path, model, port, parallel=None, drafter=None, cache_disabled=False):
    argv = [bin_path, "serve", model, "--host", "127.0.0.1", "--port", str(port)]
    if parallel:
        argv += ["--parallel", str(parallel)]
    if drafter:
        argv += ["--drafter", drafter]
    if cache_disabled:
        argv += ["--prompt-cache-gib", "0"]
    return argv


def scenario(args, label, parallel=None, drafter=None, busy=0):
    """One fresh server (or the --base one), all prompts collected once."""
    prompts = load_prompts(Path(args.prompts))
    base = args.base or f"http://127.0.0.1:{args.port}"
    if args.base:
        return {label: collect(base, prompts, args.tokens, busy, args.api_key)}
    with Server(serve_argv(args.bin, args.model, args.port, parallel, drafter), args.port):
        return {label: collect(base, prompts, args.tokens, busy, args.api_key)}


def empty_matrix(args):
    return {"results": {}, "comparisons": [], "failures": [], "rounds": [],
            "coverage": {"streams": sorted(set(args.streams)), "sampling": args.sampling, "stops": args.stops}}


def main():
    import exact_receipt as rc

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bin", default="zig-out/native/bin/tensorfold-native")
    ap.add_argument("--candidate-bin", help="second binary: compare identical requests across binaries")
    ap.add_argument("--model", help="served model directory (required without --base)")
    ap.add_argument("--base", help="reuse this running server instead of launching (no restarts then)")
    ap.add_argument("--drafter", help="draft model used by launched servers")
    ap.add_argument("--fresh-base", help="attached cache-disabled endpoint for fresh follow-up references")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fresh-port", type=int, default=8081)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--eos-tokens", type=int, default=256, help="EOS cell budget; a length stop fails that cell")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--sampling", nargs="+", choices=["greedy", "seeded"], default=["greedy", "seeded"])
    ap.add_argument("--stops", nargs="+", choices=["budget", "eos"], default=["budget", "eos"])
    ap.add_argument("--prompts", help="qualified prompts from an earlier near-tie selection")
    ap.add_argument("--recording", help="raw-logit recording; select near-tie token prefixes in this command")
    ap.add_argument("--near-tie-cases", type=int, default=8)
    ap.add_argument("--lookback", type=int, default=8)
    ap.add_argument("--near-tie-tokens", type=int, default=32)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--output", default="exact-receipt.json")
    ap.add_argument("--build-id", help="public attached-server build id; launched binaries default to their SHA-256")
    ap.add_argument("--candidate-build-id", help="public candidate build id override")
    ap.add_argument("--checkpoint-id", required=True, help="public checkpoint id, never a model path")
    ap.add_argument("--checkpoint-sha256", required=True, help="public aggregate checkpoint content SHA-256")
    ap.add_argument("--drafter-id", help="public draft checkpoint id")
    ap.add_argument("--drafter-sha256", help="public aggregate draft checkpoint content SHA-256")
    ap.add_argument("--family", required=True, help="public model family id")
    ap.add_argument("--quant", required=True, help="public quantization id")
    ap.add_argument("--skip", action="append", default=[], choices=["restart", "parallel", "busy", "draft"])
    args = ap.parse_args()
    if not args.base and not args.model:
        ap.error("--model is required without --base")
    if args.base and not args.build_id:
        ap.error("--build-id is required with --base")
    if args.base and args.candidate_bin:
        ap.error("--candidate-bin cannot be combined with --base")
    if bool(args.prompts) == bool(args.recording):
        ap.error("exactly one of --prompts and --recording is required")
    if min(args.tokens, args.eos_tokens, args.timeout, args.near_tie_cases, args.near_tie_tokens,
           *args.streams) <= 0 or args.lookback < 0:
        ap.error("case/token budgets, timeout and stream counts must be positive; lookback nonnegative")
    if not 1 <= args.port <= 65535 or not 1 <= args.fresh_port <= 65535 or args.port == args.fresh_port:
        ap.error("base and fresh ports must be distinct valid TCP ports")
    if "parallel" in args.skip or "busy" in args.skip:
        args.streams = [1]

    def digest(value, field):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            ap.error(f"--{field} must be 64 lowercase hexadecimal characters")

    try:
        for field in ("checkpoint", "family", "quant"):
            rc.public_id(getattr(args, field + "_id") if field == "checkpoint" else getattr(args, field), field)
        for field in ("build", "candidate-build", "drafter"):
            value = getattr(args, field.replace("-", "_") + "_id")
            if value:
                rc.public_id(value, field)
    except ValueError as e:
        ap.error(str(e))
    digest(args.checkpoint_sha256, "checkpoint-sha256")
    if bool(args.drafter_id) != bool(args.drafter_sha256):
        ap.error("--drafter-id and --drafter-sha256 must be supplied together")
    if args.drafter_sha256:
        digest(args.drafter_sha256, "drafter-sha256")

    import hashlib
    import draft_divergence as dd

    try:
        if args.recording:
            raw = Path(args.recording).read_bytes()
            recording = json.loads(raw)
            if not isinstance(recording, dict):
                raise ValueError("recording must be a JSON object")
            recording_sha = hashlib.sha256(raw).hexdigest()
            if recording.get("identity", {}).get("checkpoint") not in (None, args.checkpoint_id):
                ap.error("recording checkpoint identity differs from --checkpoint-id")
            selected = dd.select_prompts(recording, args.near_tie_cases, args.lookback, args.near_tie_tokens)
            prompts = [{"name": case["id"], "kind": "completion", "prompt": case["prompt"],
                        "expected_token_ids": case["tokens"], "near_tie_offset": case["near_tie_offset"],
                        "near_tie": case["top2"][case["near_tie_offset"]], "recording_sha256": recording_sha,
                        "source_position": case["source_position"], "recorded_margin": case["margin"],
                        "bf16_spacing": case["bf16_spacing"], "followup_token_ids": [case["tokens"][-1]]}
                       for case in selected]
        else:
            prompts = json.loads(Path(args.prompts).read_text())
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("the prompt set must be a nonempty array")
        names = set()
        for item in prompts:
            if not isinstance(item, dict) or item.get("name") in names:
                raise ValueError("every prompt needs a unique public name")
            rc.public_id(item["name"], "prompt")
            names.add(item["name"])
            if item.get("kind") not in ("chat", "completion"):
                raise ValueError("each prompt needs kind chat/completion")
            if not isinstance(item.get("prompt"), (str, list)) or not item["prompt"]:
                raise ValueError("each prompt needs text or token IDs")
            if isinstance(item["prompt"], list):
                dd.ids(item["prompt"])
                item["followup_token_ids"] = item.get("followup_token_ids", [item["expected_token_ids"][-1]])
                dd.ids(item["followup_token_ids"])
            expected = dd.ids(item.get("expected_token_ids"))
            offset = item.get("near_tie_offset")
            if type(offset) is not int or not 0 <= offset < len(expected):
                raise ValueError("invalid near-tie offset")
            digest(item.get("recording_sha256", ""), "prompt recording_sha256")
            tie = dd.top2(item.get("near_tie"), item["near_tie"].get("position", 0))
            if not dd.near(tie) or tie["ids"][0] != expected[offset]:
                raise ValueError("near-tie evidence is not aligned with the expected greedy window")
    except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        ap.error(f"invalid near-tie input: {e}")

    def execute(label, bin_path=None, focused=False):
        run_args = args
        if focused:
            run_args = argparse.Namespace(**vars(args))
            run_args.streams, run_args.sampling, run_args.stops = [1], ["greedy"], ["budget"]
            run_args.skip = [*args.skip, "draft"]
        try:
            if run_args.base:
                return rc.run_matrix(run_args.base, prompts, run_args, run_args.fresh_base)
            argv = serve_argv(bin_path, run_args.model, run_args.port, max(run_args.streams), run_args.drafter)
            fresh_argv = serve_argv(bin_path, run_args.model, run_args.fresh_port, max(run_args.streams),
                                    run_args.drafter, cache_disabled=True)
            with Server(argv, run_args.port), Server(fresh_argv, run_args.fresh_port):
                base = f"http://127.0.0.1:{run_args.port}"
                fresh = f"http://127.0.0.1:{run_args.fresh_port}"
                return rc.run_matrix(base, prompts, run_args, fresh)
        except (ServerFailed, OSError):
            matrix = empty_matrix(run_args)
            matrix["failures"].append({"request": f"runtime:{label}", "reason": "server launch or readiness failed"})
            return matrix

    report = {"schema": 1, "status": "fail", "family": args.family, "quant": args.quant,
              "checkpoint_id": args.checkpoint_id, "checkpoint_sha256": args.checkpoint_sha256,
              "identity_source": "operator public IDs and content hashes; not server attested",
              "token_hash_encoding": "sha256(compact JSON token ID array)", "builds": {}, "runs": {},
              "comparisons": [], "failures": [], "not_run": [], "flips": 0, "total": 0}
    if args.drafter_id:
        report["drafter"] = {"id": args.drafter_id, "sha256": args.drafter_sha256}

    def fold(label, matrix, internal=True):
        report["runs"][label] = rc.public_matrix(matrix)
        if internal:
            report["comparisons"] += [{**r, "runtime": label} for r in matrix["comparisons"]]
        report["failures"] += [{**r, "runtime": label} for r in matrix["failures"]]

    def build_identity(matrix, path, supplied):
        if supplied:
            return supplied
        try:
            return rc.binary_sha256(path)
        except OSError:
            if any(f["reason"] == "server launch or readiness failed" for f in matrix["failures"]):
                return None
            raise

    if args.base:
        fold("attached", execute("attached"))
        report["builds"]["attached"] = args.build_id
        if "restart" not in args.skip:
            report["not_run"].append("restart: attached server cannot be restarted")
    else:
        reference = execute("reference", args.bin)
        fold("reference", reference)
        report["builds"]["reference"] = build_identity(reference, args.bin, args.build_id)
        if args.candidate_bin:
            candidate = execute("candidate", args.candidate_bin)
            fold("candidate", candidate)
            report["builds"]["candidate"] = build_identity(candidate, args.candidate_bin, args.candidate_build_id)
            report["comparisons"] += [{**r, "runtime": "reference_vs_candidate"}
                                      for r in rc.compare_results(reference, candidate, "cross_binary")]
        if "restart" not in args.skip:
            restarted = execute("restart", args.bin, focused=True)
            fold("restart", restarted, internal=False)
            report["comparisons"] += [{**r, "runtime": "restart"}
                                      for r in rc.compare_results(reference, restarted, "restart")]

    primary = "attached" if args.base else "reference"
    report["coverage"] = report["runs"][primary]["coverage"]

    report["rows"] = report["comparisons"]
    report["flips"] = sum(r["first_diff"] is not None for r in report["comparisons"])
    report["total"] = len(report["comparisons"])
    report["status"] = "pass" if report["total"] and not report["flips"] and not report["failures"] else "fail"
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for row in report["comparisons"]:
        mark = "identical" if row["first_diff"] is None else f"FLIP at {row['first_diff']}"
        print(f"{row['kind']:24s} {row['candidate_request']} {mark}")
    for failure in report["failures"]:
        print(f"FAIL {failure['runtime']} {failure['request']}: {failure['reason']}")
    tail = f"; not run: {report['not_run']}" if report["not_run"] else ""
    print(f"receipt {report['status']}: flips {report['flips']}/{report['total']}{tail}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
