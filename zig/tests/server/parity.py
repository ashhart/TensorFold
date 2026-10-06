"""The Zig server against the Python server on one corpus: every reply equal but for ids, timestamps and timings."""

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
import corpus_real  # noqa: E402
import python_server  # noqa: E402
from fake_text import FakeTokenizer  # noqa: E402
from wire import exchange, framed, normal, request  # noqa: E402

FIXTURES = HERE / "fixtures"
CONTEXT = 1024
# where the two servers differ on purpose, and why
EXPECTED = {
    ("errors", "format-json"): "structured output: Python's test app has no grammar compiler, the Zig engine reports none",
    ("errors", "stream-structured"): "structured output: Python's test app has no grammar compiler, the Zig engine reports none",
    ("anthropic", "refuse-format"): "structured output: Python's test app has no grammar compiler, the Zig engine reports none",
    ("framing", "decisions"): "/v1/decisions scores labels in the engine; the Zig engine has no scoring yet, so the route is unknown",
    ("cancel", "metrics-after"): "#367/#407's families (TPOT, live tokens, decode rounds, prefill time) are Zig-only; the Python server is frozen",
    ("metrics-open", "metrics"): "the same #367/#407 Zig-only families",
    ("metrics-open", "v1-metrics"): "the same #367/#407 Zig-only families",
    ("keys", "metrics-counted"): "the same #367/#407 Zig-only families",
    ("tools", "bad_tool"): "a call to a tool the request did not offer goes out under its own name (PR #415); Python keeps the old rule",
    ("tools", "bad_tool-stream"): "a call to a tool the request did not offer goes out under its own name (PR #415); Python keeps the old rule",
    ("tools", "bad_tool-single"): "a call to a tool the request did not offer goes out under its own name (PR #415); Python keeps the old rule",
    ("tools", "bad_tool-single-stream"): "a call to a tool the request did not offer goes out under its own name (PR #415); Python keeps the old rule",
    ("errors", "stream-context"): "a refused streamed request answers 400 before its stream opens (#388); Python answers 200 and an error event",
}


def zig_server(binary: Path, extra: list[str], env: dict[str, str]) -> tuple[subprocess.Popen, int]:
    """fake_serve on a free port, with the flags the Python app is built with."""
    args = [str(binary), "serve", str(FIXTURES), "--name", "fake-model", "--alias", "fake-alias", "--port", "0",
            "--max-tokens", "64", "--parallel", "4", *extra]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "TENSORFOLD_NO_LIVE": "1", "TF_FAKE_CONTEXT": str(CONTEXT), **env})
    for line in proc.stdout:
        if line.startswith("PORT "):
            return proc, int(line.split()[1])
    raise RuntimeError("fake_serve did not start")


def compare(py: dict[str, Any], zg: dict[str, Any]) -> list[str]:
    """What differs between two exchanges, normalised; empty when equal."""
    diffs = []
    if py["closed"] != zg["closed"]:
        diffs.append(f"connection after: python {py['closed']}, zig {zg['closed']}")
    if len(py["replies"]) != len(zg["replies"]):
        diffs.append(f"replies: python {len(py['replies'])}, zig {len(zg['replies'])}")
    for i, (a, b) in enumerate(zip(py["replies"], zg["replies"])):
        if not framed(b) and not b["status"].startswith("HTTP/1.1 501"):
            diffs.append(f"reply {i}: zig Content-Length does not match its body")
        na, nb = normal(a), normal(b)
        for key in ("interim", "status", "headers", "body"):
            if na[key] != nb[key]:
                diffs.append(f"reply {i} {key}:\n  python {json.dumps(na[key])[:900]}\n  zig    {json.dumps(nb[key])[:900]}")
    return diffs


class Run:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def case(self, group: str, name: str, py_port: int, zig_port: int, raws: list[Any], **opts: Any) -> None:
        py = exchange(py_port, raws, **opts)
        zg = exchange(zig_port, raws, **opts)
        diffs = compare(py, zg)
        expected = EXPECTED.get((group, name))
        self.results.append({"group": group, "name": name, "equal": not diffs, "expected": expected, "diffs": diffs})

    def report(self) -> int:
        groups: dict[str, list[int]] = {}
        failed = 0
        for r in self.results:
            g = groups.setdefault(r["group"], [0, 0, 0])
            g[0 if r["equal"] else 2 if r["expected"] else 1] += 1
            if not r["equal"] and not r["expected"]:
                failed += 1
                print(f"MISMATCH {r['group']}/{r['name']}")
                for d in r["diffs"][:4]:
                    print("  " + d)
        for name, (ok, bad, known) in groups.items():
            print(f"{name:12} {ok:4} equal  {bad:3} differ  {known:2} expected")
        total = sum(1 for r in self.results if r["equal"])
        print(f"{total}/{len(self.results)} equal, {failed} unexpected differences")
        return 1 if failed else 0


def key_cases(run: Run, py_port: int, zig_port: int, proc: subprocess.Popen, key_file: Path, store: Any) -> None:
    """Keys from the flag, the file (labelled and bare) and the environment; the 401 shapes; SIGHUP rotation."""
    chat = corpus.chat("plain")
    gets = {"health": ("/health", {}), "models-none": ("/v1/models", {}), "models-cli": ("/v1/models", {"Authorization": "Bearer sk-cli"}),
            "models-file": ("/v1/models", {"x-api-key": "sk-file"}), "models-bare": ("/v1/models", {"Authorization": "bearer   sk-bare "}),
            "models-env": ("/v1/models", {"x-api-key": "sk-env"}), "models-wrong": ("/v1/models", {"Authorization": "Bearer sk-nope"}),
            "models-basic": ("/v1/models", {"Authorization": "Basic sk-cli"}), "metrics-none": ("/metrics", {}), "messages-none": ("/v1/messages", {}),
            "tokenize-none": ("/tokenize", {}), "alt-models": ("/alt/models", {}), "options": ("/v1/models", {})}
    for name, (path, headers) in gets.items():
        method = "OPTIONS" if name == "options" else "POST" if name in ("messages-none", "tokenize-none") else "GET"
        run.case("keys", name, py_port, zig_port, [request(method, path, b"{}" if method == "POST" else None, headers)])
    run.case("keys", "twice-x-api-key", py_port, zig_port, [b"GET /v1/models HTTP/1.1\r\nx-api-key: sk-cli\r\nx-api-key: sk-cli\r\n\r\n"])
    run.case("keys", "chat-with-key", py_port, zig_port, [request("POST", "/v1/chat/completions", chat, {"x-api-key": "sk-cli"})])
    run.case("keys", "expect-without-key", py_port, zig_port, [request("POST", "/v1/chat/completions", chat, {"Expect": "100-continue"})])
    run.case("keys", "pooled-identity", py_port, zig_port, [request("GET", "/v1/models", None, {"x-api-key": "sk-cli"}), get_models()])
    key_file.write_text("next: sk-next\n")
    os.kill(os.getpid(), signal.SIGHUP)
    proc.send_signal(signal.SIGHUP)
    time.sleep(0.3)
    run.case("keys", "rotated-old", py_port, zig_port, [request("GET", "/v1/models", None, {"x-api-key": "sk-file"})])
    run.case("keys", "rotated-new", py_port, zig_port, [request("GET", "/v1/models", None, {"x-api-key": "sk-next"})])
    run.case("keys", "metrics-counted", py_port, zig_port, [request("GET", "/metrics", None, {"x-api-key": "sk-cli"})])


def get_models() -> bytes:
    return request("GET", "/v1/models")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(ROOT / "zig-out" / "server" / "fake_serve"))
    parser.add_argument("--report", default="")
    parser.add_argument("--tokenizer", action="append", default=[], help="also run the checkpoint corpus on this model directory")
    args = parser.parse_args()
    tokenizer = FakeTokenizer(json.loads((FIXTURES / "vocab.json").read_text())["pieces"])
    run = Run()
    logs = Path(tempfile.mkdtemp())
    os.environ["TENSORFOLD_REQUEST_LOG"] = str(logs / "python.jsonl")       # the Python server reads it at import
    _, py_port = python_server.start(FIXTURES, context=CONTEXT)
    os.environ.pop("TENSORFOLD_REQUEST_LOG")
    proc, zig_port = zig_server(Path(args.binary), [], {"TENSORFOLD_REQUEST_LOG": str(logs / "zig.jsonl")})
    try:
        cases = (corpus.chat_cases() + corpus.tool_cases() + corpus.error_cases() + corpus.completion_cases(tokenizer.encode)
                 + corpus.response_cases() + corpus.anthropic_cases() + corpus.token_cases() + corpus.status_cases())
        for group, name, raws in cases:
            run.case(group, name, py_port, zig_port, raws)
        for group, name, raws, opts in corpus.framing_cases():
            run.case(group, name, py_port, zig_port, raws, **opts)
        leave = corpus.post("/v1/chat/completions", corpus.chat("slow", stream=True))
        run.case("cancel", "client-leaves", py_port, zig_port, [leave], leave_after_events=3)   # mid-stream: the role and two deltas
        time.sleep(1.0)
        run.case("cancel", "metrics-after", py_port, zig_port, [request("GET", "/metrics")])
    finally:
        proc.terminate()
        stopped = proc.wait(10)
    run.results.append({"group": "lifecycle", "name": "sigterm-exit-0", "equal": stopped == 0, "expected": None,
                        "diffs": [] if stopped == 0 else [f"fake_serve exited {stopped} on SIGTERM"]})
    same_log = (logs / "python.jsonl").read_text() == (logs / "zig.jsonl").read_text()
    run.results.append({"group": "lifecycle", "name": "request-log", "equal": same_log, "expected": None,
                        "diffs": [] if same_log else ["TENSORFOLD_REQUEST_LOG lines differ"]})
    _, py_port = python_server.start(FIXTURES, context=CONTEXT, keys=["sk-cli"], metrics_open=True)
    proc, zig_port = zig_server(Path(args.binary), ["--api-key", "sk-cli", "--metrics-open"], {})
    try:
        for name, path in (("metrics", "/metrics"), ("v1-metrics", "/v1/metrics/"), ("models", "/v1/models"), ("health", "/health")):
            run.case("metrics-open", name, py_port, zig_port, [request("GET", path)])
    finally:
        proc.terminate()
        proc.wait(10)
    with tempfile.TemporaryDirectory() as tmp:
        key_file = Path(tmp) / "keys"
        key_file.write_text("# keys\nclient: sk-file\nsk-bare\n")
        key_file.chmod(0o600)
        server, py_port = python_server.start(FIXTURES, context=CONTEXT, keys=["sk-cli"], key_file=str(key_file), environment="sk-env")
        store = server.RequestHandlerClass.auth
        signal.signal(signal.SIGHUP, store.request_reload)
        proc, zig_port = zig_server(Path(args.binary), ["--api-key", "sk-cli", "--api-key-file", str(key_file)], {"TENSORFOLD_API_KEY": "sk-env"})
        try:
            key_cases(run, py_port, zig_port, proc, key_file, store)
        finally:
            proc.terminate()
            proc.wait(10)
    for directory in args.tokenizer:
        real_cases(run, Path(args.binary), Path(directory))
    if args.report:
        Path(args.report).write_text(json.dumps(run.results, indent=1))
    return run.report()


def real_cases(run: Run, binary: Path, directory: Path) -> None:
    """The checkpoint corpus, both servers on that checkpoint's tokenizer and chat template."""
    _, py_port = python_server.start(FIXTURES, context=0, tokenizer_dir=directory, scripts_file="scripts_real.json")
    proc, zig_port = zig_server(binary, [], {"TF_FAKE_TOKENIZER": str(directory), "TF_FAKE_SCRIPTS": str(FIXTURES / "scripts_real.json"), "TF_FAKE_CONTEXT": "0"})
    label = directory.parent.parent.name.removeprefix("models--") if directory.parent.name == "snapshots" else directory.name
    try:
        tok = python_server.HfTokenizer(directory)
        for group, name, raws in corpus_real.cases(lambda text: tok.encode(text, add_special_tokens=False)):
            run.case(f"{group}", f"{label}/{name}", py_port, zig_port, raws)
    finally:
        proc.terminate()
        proc.wait(10)


if __name__ == "__main__":
    sys.exit(main())
