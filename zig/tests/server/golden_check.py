"""The Zig server's fake engine against the frozen Python answers: golden/<group>/<name>.json per corpus case.

Walks the same cases in the same order as parity.py and freeze_golden.py, exchanges them with fake_serve
(the scripted engine), normalizes the replies the way parity compares, and diffs them against the golden
files. A mismatch on a case parity.py's EXPECTED table already names is reported as known; any other
mismatch fails the run - each one is a real Zig/Python difference to list, never a golden to edit.
`python3 zig/tests/server/golden_check.py [--binary zig-out/server/fake_serve]`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "src")]

import corpus  # noqa: E402
import parity  # noqa: E402
from fake_text import FakeTokenizer  # noqa: E402
from wire import exchange, framed, normal, request  # noqa: E402

FIXTURES = HERE / "fixtures"
GOLDEN = HERE / "golden"
CONTEXT = 1024


def load(group: str, name: str) -> dict[str, Any] | None:
    path = GOLDEN / group / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(ROOT / "zig-out" / "server" / "fake_serve"))
    args = parser.parse_args()
    binary = Path(args.binary)
    if not binary.exists():
        subprocess.run(["bash", str(HERE / "build.sh")], check=True, env={**os.environ})
    tokenizer = FakeTokenizer(json.loads((FIXTURES / "vocab.json").read_text())["pieces"])

    class Run:
        result: list[dict[str, Any]] = []

    run = Run()
    logs = Path(tempfile.mkdtemp())
    os.environ.pop("TENSORFOLD_REQUEST_LOG", None)
    cases: list[tuple[str, str, list[Any], dict[str, Any]]] = []
    for group, name, raws in (corpus.chat_cases() + corpus.tool_cases() + corpus.error_cases()
                              + corpus.completion_cases(tokenizer.encode) + corpus.response_cases()
                              + corpus.anthropic_cases() + corpus.token_cases() + corpus.status_cases()):
        cases.append((group, name, raws, {}))
    for group, name, raws, opts in corpus.framing_cases():
        cases.append((group, name, raws, opts))
    leave = corpus.post("/v1/chat/completions", corpus.chat("slow", stream=True))
    cases.append(("cancel", "client-leaves", [leave], {"leave_after_events": 3}))
    cases.append(("cancel", "metrics-after", [request("GET", "/metrics")], {}))

    proc, port = parity.zig_server(binary, [], {"TENSORFOLD_REQUEST_LOG": str(logs / "zig.jsonl")})
    try:
        check(run, port, cases) if False else None
        # the cancel pair needs its second case a second after the first, as parity runs it
        first = [c for c in cases if c[0] != "cancel"]
        check_run(run, port, first)
        check_run(run, port, [("cancel", "client-leaves", [leave], {"leave_after_events": 3})])
        time.sleep(1.0)
        check_run(run, port, [("cancel", "metrics-after", [request("GET", "/metrics")], {})])
    finally:
        proc.terminate()
        stopped = proc.wait(10)
    verdict = "equal" if stopped == 0 else "differ"
    run.result.append({"group": "lifecycle", "name": "sigterm-exit-0", "verdict": verdict,
                       "where": [] if stopped == 0 else [f"fake_serve exited {stopped} on SIGTERM"]})
    same = (logs / "zig.jsonl").read_bytes() == (GOLDEN / "lifecycle" / "request-log.json").read_bytes()
    run.result.append({"group": "lifecycle", "name": "request-log", "verdict": "equal" if same else "differ",
                       "where": [] if same else ["TENSORFOLD_REQUEST_LOG lines differ from the frozen ones"]})

    proc, port = parity.zig_server(binary, ["--api-key", "sk-cli", "--metrics-open"], {})
    try:
        for name, path in (("metrics", "/metrics"), ("v1-metrics", "/v1/metrics/"), ("models", "/v1/models"), ("health", "/health")):
            check_run(run, port, [("metrics-open", name, [request("GET", path)], {})])
    finally:
        proc.terminate()
        proc.wait(10)
    with tempfile.TemporaryDirectory() as tmp:
        key_file = Path(tmp) / "keys"
        key_file.write_text("# keys\nclient: sk-file\nsk-bare\n")
        key_file.chmod(0o600)
        proc, port = parity.zig_server(binary, ["--api-key", "sk-cli", "--api-key-file", str(key_file)],
                                       {"TENSORFOLD_API_KEY": "sk-env"})
        try:
            chat = corpus.chat("plain")
            gets = {"health": ("/health", {}), "models-none": ("/v1/models", {}), "models-cli": ("/v1/models", {"Authorization": "Bearer sk-cli"}),
                    "models-file": ("/v1/models", {"x-api-key": "sk-file"}), "models-bare": ("/v1/models", {"Authorization": "bearer   sk-bare "}),
                    "models-env": ("/v1/models", {"x-api-key": "sk-env"}), "models-wrong": ("/v1/models", {"Authorization": "Bearer sk-nope"}),
                    "models-basic": ("/v1/models", {"Authorization": "Basic sk-cli"}), "metrics-none": ("/metrics", {}), "messages-none": ("/v1/messages", {}),
                    "tokenize-none": ("/tokenize", {}), "alt-models": ("/alt/models", {}), "options": ("/v1/models", {})}
            key_cases: list[tuple[str, str, list[Any], dict[str, Any]]] = []
            for name, (path, headers) in gets.items():
                method = "OPTIONS" if name == "options" else "POST" if name in ("messages-none", "tokenize-none") else "GET"
                key_cases.append(("keys", name, [request(method, path, b"{}" if method == "POST" else None, headers)], {}))
            key_cases += [("keys", "twice-x-api-key", [b"GET /v1/models HTTP/1.1\r\nx-api-key: sk-cli\r\nx-api-key: sk-cli\r\n\r\n"], {}),
                          ("keys", "chat-with-key", [request("POST", "/v1/chat/completions", chat, {"x-api-key": "sk-cli"})], {}),
                          ("keys", "expect-without-key", [request("POST", "/v1/chat/completions", chat, {"Expect": "100-continue"})], {}),
                          ("keys", "pooled-identity", [request("GET", "/v1/models", None, {"x-api-key": "sk-cli"}), request("GET", "/v1/models")], {})]
            check_run(run, port, key_cases)
            key_file.write_text("next: sk-next\n")
            proc.send_signal(signal.SIGHUP)
            time.sleep(0.3)
            check_run(run, port, [("keys", "rotated-old", [request("GET", "/v1/models", None, {"x-api-key": "sk-file"})], {}),
                                  ("keys", "rotated-new", [request("GET", "/v1/models", None, {"x-api-key": "sk-next"})], {}),
                                  ("keys", "metrics-counted", [request("GET", "/metrics", None, {"x-api-key": "sk-cli"})], {})])
        finally:
            proc.terminate()
            proc.wait(10)

    groups: dict[str, list[int]] = {}
    failed = 0
    for r in run.result:
        g = groups.setdefault(r["group"], [0, 0, 0])
        g[0 if r["verdict"] == "equal" else 2 if (r["group"], r["name"]) in parity.EXPECTED else 1] += 1
        if r["verdict"] == "differ":
            known = (r["group"], r["name"]) in parity.EXPECTED
            print(f"{'KNOWN-DIFFER' if known else 'MISMATCH'} {r['group']}/{r['name']}: {parity.EXPECTED.get((r['group'], r['name']), '')}")
            for d in r.get("where", [])[:3]:
                print("  " + d)
            if not known:
                failed += 1
        elif r["verdict"] == "no-golden":
            print(f"NO-GOLDEN {r['group']}/{r['name']}")
            failed += 1
    for name, (ok, bad, known) in sorted(groups.items()):
        print(f"{name:12} {ok:4} equal  {bad:3} differ  {known:2} known")
    total = sum(1 for r in run.result if r["verdict"] == "equal")
    print(f"{total}/{len(run.result)} equal to the frozen Python answers, {failed} unexpected differences")
    return 1 if failed else 0


def check_run(run: Run, port: int, cases: list[tuple[str, str, list[Any], dict[str, Any]]]) -> None:
    for group, name, raws, opts in cases:
        got = exchange(port, raws, **opts)
        replies = [{"status": n["status"], "headers": [[k, v] for k, v in n["headers"]],
                    "body": n["body"], "interim": n["interim"]}
                   for n in (normal(r) for r in got["replies"])]
        want = load(group, name)
        if want is None:
            run.result.append({"group": group, "name": name, "verdict": "no-golden"})
            continue
        golden = want["replies"]
        framing = [f"reply {i}: zig Content-Length does not match its body"
                   for i, r in enumerate(got["replies"])
                   if not framed(r) and not r["status"].startswith("HTTP/1.1 501")]
        if got["closed"] == want["closed"] and golden == replies and not framing:
            run.result.append({"group": group, "name": name, "verdict": "equal"})
            continue
        where = list(framing)
        if got["closed"] != want["closed"]:
            where.append(f"closed: golden {want['closed']}, zig {got['closed']}")
        if len(want["replies"]) != len(got["replies"]):
            where.append(f"reply count: golden {len(want['replies'])}, zig {len(got['replies'])}")
        for i, (a, b) in enumerate(zip(golden, replies)):
            for key in ("interim", "status", "headers", "body"):
                if a[key] != b[key]:
                    where.append(f"reply {i} {key}:\n  golden {json.dumps(a[key])[:900]}\n  zig    {json.dumps(b[key])[:900]}")
        run.result.append({"group": group, "name": name, "verdict": "differ", "where": where})


if __name__ == "__main__":
    sys.exit(main())
