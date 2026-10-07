"""Freeze the server-parity corpus for the Zig golden check: every case's raw bytes, per phase, as golden/cases.json the Zig checker walks without Python."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import corpus  # noqa: E402
import parity  # noqa: E402
from fake_text import FakeTokenizer  # noqa: E402
from wire import request  # noqa: E402

GOLDEN = HERE / "golden"


def raw_bytes(raws: list) -> list:
    """Raw requests as {"b64": ...} bytes or {"tmpl_b64": ..., "ids": [...]} for the id-following templates."""
    out = []
    for raw in raws:
        if callable(raw):
            raise RuntimeError("an unresolved callable: build it as a template with ids")
        out.append({"b64": base64.b64encode(raw).decode()})
    return out


def response_cases_frozen(tokenizer: FakeTokenizer) -> list[dict]:
    """corpus.response_cases with stored-chain's id-following callables as marker templates."""
    out = []
    for group, name, raws in corpus.response_cases():
        if name != "stored-chain":
            out.append({"group": group, "name": name, "raws": raw_bytes(raws), "opts": {}})
            continue
        marker0, marker1 = "ZZID0ZZ", "ZZID1ZZ"
        chain = [
            corpus.post("/v1/responses", {"input": "one @script=plain"}),
            corpus.post("/v1/responses", {"input": "two @script=think", "previous_response_id": marker0}),
            corpus.get(f"/v1/responses/{marker1}"),
            request("DELETE", f"/v1/responses/{marker0}"),
            corpus.post("/v1/responses", {"input": "three", "previous_response_id": marker1}),
        ]
        out.append({"group": group, "name": name, "raws": [
            {"b64": base64.b64encode(chain[0]).decode()},
            {"tmpl_b64": base64.b64encode(chain[1]).decode(), "ids": [0]},
            {"tmpl_b64": base64.b64encode(chain[2]).decode(), "ids": [1]},
            {"tmpl_b64": base64.b64encode(chain[3]).decode(), "ids": [0]},
            {"tmpl_b64": base64.b64encode(chain[4]).decode(), "ids": [1]},
        ], "opts": {}})
    return out


def main() -> int:
    tokenizer = FakeTokenizer(json.loads((HERE / "fixtures" / "vocab.json").read_text())["pieces"])
    steps: list[dict] = []

    # the main corpus on the first server, with its request log compared at sigterm
    cases: list[dict] = []
    for group, name, raws in (corpus.chat_cases() + corpus.tool_cases() + corpus.error_cases()
                              + corpus.completion_cases(tokenizer.encode)):
        cases.append({"group": group, "name": name, "raws": raw_bytes(raws), "opts": {}})
    cases.extend(response_cases_frozen(tokenizer))
    for group, name, raws in (corpus.anthropic_cases() + corpus.token_cases() + corpus.status_cases()):
        cases.append({"group": group, "name": name, "raws": raw_bytes(raws), "opts": {}})
    for group, name, raws, opts in corpus.framing_cases():
        cases.append({"group": group, "name": name, "raws": raw_bytes(raws), "opts": opts})
    leave = corpus.post("/v1/chat/completions", corpus.chat("slow", stream=True))
    steps.append({
        "kind": "serve",
        "args": [],
        "env": {"TENSORFOLD_REQUEST_LOG": "@LOG@"},
        "cases": cases,
        "then": [
            {"kind": "cases", "cases": [
                {"group": "cancel", "name": "client-leaves", "raws": raw_bytes([leave]), "opts": {"leave_after_events": 3}}]},
            {"kind": "sleep", "seconds": 1.0},
            {"kind": "cases", "cases": [
                {"group": "cancel", "name": "metrics-after", "raws": raw_bytes([request("GET", "/metrics")]), "opts": {}}]},
            {"kind": "sigterm", "expect_exit": 0},
            {"kind": "compare_log", "golden": "lifecycle/request-log.json"},
        ],
    })

    # the metrics-open server
    steps.append({
        "kind": "serve",
        "args": ["--api-key", "sk-cli", "--metrics-open"],
        "env": {},
        "cases": [{"group": "metrics-open", "name": name, "raws": raw_bytes([request("GET", path)]), "opts": {}}
                  for name, path in (("metrics", "/metrics"), ("v1-metrics", "/v1/metrics/"),
                                     ("models", "/v1/models"), ("health", "/health"))],
        "then": [{"kind": "terminate"}],
    })

    # the keys server: the file keys, then a SIGHUP rotation
    key_cases = [
        ("health", "GET", "/health", None, {}),
        ("models-none", "GET", "/v1/models", None, {}),
        ("models-cli", "GET", "/v1/models", None, {"Authorization": "Bearer sk-cli"}),
        ("models-file", "GET", "/v1/models", None, {"x-api-key": "sk-file"}),
        ("models-bare", "GET", "/v1/models", None, {"Authorization": "bearer   sk-bare "}),
        ("models-env", "GET", "/v1/models", None, {"x-api-key": "sk-env"}),
        ("models-wrong", "GET", "/v1/models", None, {"Authorization": "Bearer sk-nope"}),
        ("models-basic", "GET", "/v1/models", None, {"Authorization": "Basic sk-cli"}),
        ("metrics-none", "GET", "/metrics", None, {}),
        ("messages-none", "POST", "/v1/messages", b"{}", {}),
        ("tokenize-none", "POST", "/tokenize", b"{}", {}),
        ("alt-models", "GET", "/alt/models", None, {}),
        ("options", "OPTIONS", "/v1/models", None, {}),
    ]
    chat = corpus.chat("plain")
    key_steps: list[dict] = [{
        "kind": "cases",
        "cases": [{"group": "keys", "name": name,
                   "raws": raw_bytes([request(method, path, body, headers)]), "opts": {}}
                  for name, method, path, body, headers in key_cases]}
        , {"kind": "cases", "cases": [
            {"group": "keys", "name": "twice-x-api-key",
             "raws": raw_bytes([b"GET /v1/models HTTP/1.1\r\nx-api-key: sk-cli\r\nx-api-key: sk-cli\r\n\r\n"]), "opts": {}},
            {"group": "keys", "name": "chat-with-key",
             "raws": raw_bytes([request("POST", "/v1/chat/completions", chat, {"x-api-key": "sk-cli"})]), "opts": {}},
            {"group": "keys", "name": "expect-without-key",
             "raws": raw_bytes([request("POST", "/v1/chat/completions", chat, {"Expect": "100-continue"})]), "opts": {}},
            {"group": "keys", "name": "pooled-identity",
             "raws": raw_bytes([request("GET", "/v1/models", None, {"x-api-key": "sk-cli"}),
                                request("GET", "/v1/models")]), "opts": {}}]},
    ]
    steps.append({
        "kind": "serve",
        "args": ["--api-key", "sk-cli", "--api-key-file", "@KEYFILE@"],
        "env": {"TENSORFOLD_API_KEY": "sk-env"},
        "key_file": "# keys\nclient: sk-file\nsk-bare\n",
        "cases": [],
        "then": key_steps + [
            {"kind": "rewrite_key", "content": "next: sk-next\n"},
            {"kind": "sleep", "seconds": 0.3},
            {"kind": "cases", "cases": [
                {"group": "keys", "name": "rotated-old",
                 "raws": raw_bytes([request("GET", "/v1/models", None, {"x-api-key": "sk-file"})]), "opts": {}},
                {"group": "keys", "name": "rotated-new",
                 "raws": raw_bytes([request("GET", "/v1/models", None, {"x-api-key": "sk-next"})]), "opts": {}},
                {"group": "keys", "name": "metrics-counted",
                 "raws": raw_bytes([request("GET", "/metrics", None, {"x-api-key": "sk-cli"})]), "opts": {}}]},
            {"kind": "terminate"},
        ],
    })

    # the responses group's id-following requests stay templates: the id comes from an earlier reply, resolved here into templates with markers
    doc = {"steps": steps}
    (GOLDEN / "cases.json").write_text(json.dumps(doc))
    print(f"{sum(len(s.get('cases', [])) + sum(len(t.get('cases', [])) for t in s.get('then', [])) for s in steps)} cases frozen to {GOLDEN / 'cases.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
