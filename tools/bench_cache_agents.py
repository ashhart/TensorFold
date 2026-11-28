"""Sequential served agent-cache receipts and exact A/B comparison; standard library only."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

import bench_agent_compare as comparison
from bench_agent_trace import SHAPES, Texts, digest, prepare, requests
from bench_auth import authorize, read_api_key, redact
from bench_evidence import Evidence
import bench_stream
import bench_thinking


class History(Evidence):
    """Collect the next assistant message during the same parse that provides the shared evidence."""

    def __init__(self):
        super().__init__()
        self.content, self.reasoning = [], []

    def add(self, choice):
        kinds = super().add(choice)
        if choice.get("index", 0) != 0:
            raise ValueError("unexpected_choice")
        delta = choice.get("delta") or {}
        self.content.append(choice.get("text") or delta.get("content") or "")
        self.reasoning.append(delta.get("reasoning_content") or "")
        return kinds

    def message(self):
        out = {"role": "assistant", "content": "".join(self.content)}
        if any(self.reasoning):
            out["reasoning_content"] = "".join(self.reasoning)
        return out


def stream(base, body, timeout, api_key):
    request = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    authorize(request, api_key)
    sent = time.perf_counter()
    reply = bench_stream.Collector(sent)
    reply.evidence = History()
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply.read(response)
        if reply.end != "done":
            error = "server_stream_error" if reply.end == "error" else "incomplete_stream"
    except (Exception, KeyboardInterrupt) as exc:
        error = type(exc).__name__
        if isinstance(exc, urllib.error.HTTPError):
            exc.close()
    complete = reply.end == "done" and error is None
    evidence = reply.evidence.finish(reply.usage, reply.runtime, complete)
    if evidence["tool_calls"]:
        error = error or "unexpected_tool_reply"
    row = {**evidence, "error": error, "completion": reply.completion(),
           "ttft_s": reply.pieces[0][0] - sent if complete and reply.pieces else None}
    return row, reply.evidence.message()


def run(args, api_key):
    settings = {key: getattr(args, key) for key in ("trace", "conversations", "rounds", "prefix_tokens", "seed")}
    settings.update(reply_tokens=args.tokens, temperature=0, one_offs_per_turn=2, history_mode="measured_replies")
    capabilities = bench_thinking.discover(args.base, args.model, api_key)
    item = {"kind": "chat", **capabilities}
    report = {"schema_version": 1, "benchmark": "cache_agents", "settings": settings,
              "thinking": bench_thinking.receipt(item), "system_prefixes": [], "rows": [],
              "expected_requests": args.conversations * args.rounds * 3, "completed_schedule": False,
              "setup_error": None}
    try:
        fixtures = prepare(Texts(args.base, args.model, args.timeout, api_key), settings)
        report["system_prefixes"] = [proof for _, proof in fixtures[0]]
        for index, (identity, messages) in enumerate(requests(settings, fixtures)):
            body = {"model": args.model, "messages": messages, "temperature": 0, "seed": args.seed + index,
                    "max_tokens": args.tokens, "stream": True, "stream_options": {"include_usage": True},
                    "ignore_eos": True}
            if bench_thinking.request_off(item):
                body["chat_template_kwargs"] = {"enable_thinking": False}
            request_sha = digest({key: value for key, value in body.items() if key != "model"})
            observed, assistant = stream(args.base, body, args.timeout, api_key)
            if api_key and any(api_key in value for key, value in assistant.items() if key != "role"):
                observed["error"] = "credential_in_reply"
            row = {"index": index, "trace": args.trace, **identity, "request_seed": body["seed"],
                   "request_sha256": request_sha, **observed}
            report["rows"].append(row)
            if row["error"]:
                break
            if identity["kind"] == "agent":
                messages.append(assistant)
        report["completed_schedule"] = len(report["rows"]) == report["expected_requests"] and not any(r["error"] for r in report["rows"])
    except (Exception, KeyboardInterrupt) as exc:
        report["setup_error"] = type(exc).__name__
        if isinstance(exc, urllib.error.HTTPError):
            exc.close()
    report["fixture_sha256"] = digest([r["request_sha256"] for r in report["rows"]])
    report["summary"] = comparison.summarize(report["rows"])
    report["qualified"] = comparison.qualify(report)
    public = redact(report, api_key)
    if public != report:
        report["qualified"] = False
        public = redact(report, api_key)
    return public, report["qualified"]


def load(path):
    def reject(value):
        raise ValueError("nonfinite_json")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="one trace, one request at a time; start a fresh server for each receipt")
    runner.add_argument("base")
    runner.add_argument("model")
    runner.add_argument("--trace", choices=SHAPES, required=True)
    runner.add_argument("--conversations", type=int, default=6)
    runner.add_argument("--rounds", type=int, default=48)
    runner.add_argument("--prefix-tokens", type=int, default=8192,
                        help="standalone system-text token count, checked with the server's tokenizer")
    runner.add_argument("--tokens", type=int, default=16, help="reply token limit")
    runner.add_argument("--seed", type=int, default=7)
    runner.add_argument("--timeout", type=float, default=600)
    runner.add_argument("--api-key-env", metavar="NAME")
    runner.add_argument("--output")
    comparator = sub.add_parser("compare", help="verify every reply's token fingerprint and report per-trace sums")
    comparator.add_argument("a")
    comparator.add_argument("b")
    comparator.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "run":
        if min(args.conversations, args.rounds, args.prefix_tokens, args.tokens) < 1:
            runner.error("conversations, rounds, prefix tokens and reply tokens must be positive")
        if args.trace == "agent_cohorts" and args.rounds < 2:
            runner.error("agent_cohorts requires at least two rounds")
        if args.seed < 0 or args.seed + args.conversations * args.rounds * 3 - 1 > 0xffffffffffffffff:
            runner.error("request seeds must fit unsigned 64-bit integers")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            runner.error("timeout must be finite and positive")
        report, qualified = run(args, read_api_key(runner, args.api_key_env))
    else:
        try:
            report = comparison.compare(load(args.a), load(args.b))
        except (OSError, ValueError, TypeError, KeyError):
            report = {"qualified": False, "error": "invalid_comparison_receipt"}
            code = 2
        else:
            code = 0 if report["qualified"] else 1
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        try:
            Path(args.output).write_text(encoded, encoding="utf-8")
        except OSError:
            parser.error("cannot write receipt")
    print(encoded, end="", flush=True)
    return (0 if qualified else 1) if args.command == "run" else code


if __name__ == "__main__":
    sys.exit(main())
