"""Frozen native browser-policy, key and admission cases: every socket still connects to the local fake server."""
from __future__ import annotations
import base64
import json


def case(name, host="localhost", origin=None, path="/health", method="GET", extra=()):
    head = [f"{method} {path} HTTP/1.1"]
    if host is not None:
        head.append(f"Host: {host}")
    if origin is not None:
        head.append(f"Origin: {origin}")
    head.extend(f"{key}: {value}" for key, value in extra)
    body = b"{}" if method == "POST" else b""
    if method == "POST":
        head.extend(["Content-Type: text/plain", f"Content-Length: {len(body)}"])
    if method == "POST" and path.endswith("systemone"):
        import json
        body = json.dumps({"state": None, "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"A": "a", "B": "b"}}}}).encode()
        head[-1] = f"Content-Length: {len(body)}"
    raw = ("\r\n".join(head) + "\r\n\r\n").encode() + body
    return {"group": "origin", "name": name, "raws": [{"b64": base64.b64encode(raw).decode()}], "opts": {}}


def raw(method, path, body=None, headers=()):
    """One request's bytes on the local fake server, with a JSON body when one is given."""
    head = [f"{method} {path} HTTP/1.1", "Host: localhost"]
    head.extend(f"{key}: {value}" for key, value in headers)
    data = b"" if body is None else json.dumps(body).encode()
    if body is not None:
        head.extend(["Content-Type: application/json", f"Content-Length: {len(data)}"])
    return ("\r\n".join(head) + "\r\n\r\n").encode() + data


def full_key_cases():
    """Keys without scopes keep every route a scoped key is refused, as the server before key scopes answered."""
    cli, env = (("Authorization", "Bearer sk-cli"),), (("x-api-key", "sk-env"),)
    file, bare = (("x-api-key", "sk-file"),), (("Authorization", "Bearer sk-bare"),)
    completion = {"prompt": "x @script=plain", "max_tokens": 2}
    items = [
        ("full-get-response", [raw("GET", "/v1/responses/resp_missing", headers=cli)]),
        ("full-delete-response", [raw("DELETE", "/v1/responses/resp_missing", headers=file)]),
        ("full-response-stored", [raw("POST", "/v1/responses", {"input": "Hi @script=plain", "max_output_tokens": 2}, bare)]),
        ("full-response-parent", [raw("POST", "/v1/responses", {"input": "Hi", "previous_response_id": "resp_missing"}, env)]),
        ("full-alias-completions", [raw("POST", "/alt/completions", completion, file)]),
        ("full-slide-learn", [raw("POST", "/v1/slide/learn", {}, cli)]),
        ("full-slide-graph", [raw("DELETE", "/v1/slide/graph", headers=env)]),
        ("full-recall", [raw("GET", "/v1/recall", headers=file)]),
        ("full-imprint-load", [raw("POST", "/v1/imprint/load", {}, bare)]),
        ("full-stats", [raw("GET", "/stats", headers=cli)]),
        ("full-head", [raw("HEAD", "/v1/models", headers=file)]),
        ("full-put", [raw("PUT", "/v1/models", headers=env)]),
        ("full-future", [raw("GET", "/future/private", headers=bare)]),
        ("full-then-none", [raw("GET", "/v1/models", headers=file), raw("DELETE", "/v1/responses/resp_missing")]),
    ]
    return [{"group": "keys", "name": name, "raws": [{"b64": base64.b64encode(r).decode()} for r in raws], "opts": {}}
            for name, raws in items]


def scope_cases():
    """Scoped keys reach their exact routes; every other route is refused before its body is read."""
    app, read, ops = (("x-api-key", "sk-app"),), (("Authorization", "Bearer sk-read"),), (("x-api-key", "sk-ops"),)
    chat = {"model": "fake-model", "messages": [{"role": "user", "content": "Hi @script=plain"}]}
    learn = raw("POST", "/v1/slide/learn", {"text": "x"}, app).replace(b"\r\n\r\n", b"\r\nExpect: 100-continue\r\n\r\n", 1)
    items = [
        ("read-chat", [raw("POST", "/v1/chat/completions", chat, read)]),
        ("read-messages", [raw("POST", "/v1/messages", {"max_tokens": 2, "messages": chat["messages"]}, read)]),
        ("read-models", [raw("GET", "/v1/models", headers=read)]),
        ("app-metrics", [raw("GET", "/metrics", headers=app)]),
        ("app-chat", [raw("POST", "/v1/chat/completions", chat, app)]),
        ("app-alias", [raw("POST", "/alt/completions", {"prompt": "x @script=plain"}, app)]),
        ("app-recall", [raw("GET", "/v1/recall", headers=app)]),
        ("app-slide-graph", [raw("DELETE", "/v1/slide/graph", headers=app)]),
        ("app-learn-expect", [learn]),
        ("app-response-stored", [raw("POST", "/v1/responses", {"input": "Hi @script=plain"}, app)]),
        ("app-response-parent", [raw("POST", "/v1/responses", {"input": "Hi", "store": False, "previous_response_id": "resp_x"}, app)]),
        ("app-response-stateless", [raw("POST", "/v1/responses", {"input": "Hi @script=plain", "store": False, "max_output_tokens": 2}, app)]),
        ("ops-slide-graph", [raw("DELETE", "/v1/slide/graph", headers=ops)]),
    ]
    return [{"group": "scopes", "name": name, "raws": [{"b64": base64.b64encode(r).decode()} for r in raws], "opts": {}}
            for name, raws in items]


def admission_cases(stem):
    """Model requests on one connection, the stream last: admission off answers as before its flags existed."""
    say = [{"role": "user", "content": "Hi @script=plain"}]
    burst = [raw("POST", "/v1/completions", {"prompt": "x @script=plain", "max_tokens": 2}),
             raw("POST", "/v1/chat/completions", {"model": "fake-model", "messages": say}),
             raw("POST", "/v1/messages", {"model": "fake-model", "max_tokens": 4, "messages": say}),
             raw("POST", "/v1/responses", {"input": "Hi @script=plain", "max_output_tokens": 2}),
             raw("POST", "/v1/chat/completions", {"model": "fake-model", "messages": say, "stream": True})]
    return [{"group": "admission", "name": f"{stem}-burst", "raws": [{"b64": base64.b64encode(r).decode()} for r in burst], "opts": {}}]


def steps():
    normal = [
        case("rebind-host", "rebind.invalid"),
        case("cross-origin", origin="https://evil.invalid", path="/v1/chat/completions", method="POST"),
        case("cross-origin-learn", origin="https://evil.invalid", path="/v1/slide/learn", method="POST"),
        case("same-origin", "localhost:8000", "http://localhost:8000"),
        case("no-origin", "127.0.0.1:9000"),
        case("null-origin", origin="null"),
        case("missing-host", None),
        case("ipv6-loopback", "[::1]:80", "http://[::1]"),
        case("scheme-mismatch", origin="https://localhost"),
        case("duplicate-host", extra=(("hOsT", "localhost"),)),
        case("duplicate-origin", origin="http://localhost", extra=(("oRiGiN", "http://localhost"),)),
    ]
    allowed = [case("allowed-host-first", "trusted-one.invalid"), case("allowed-host-second", "trusted-two.invalid"),
               case("allowed-origin-first", origin="https://ui-one.invalid:443"),
               case("allowed-origin-second", origin="http://ui-two.invalid:8000")]
    nonloopback = [case("nonloopback-host", "public.invalid"),
                   case("nonloopback-cross-origin", "public.invalid", "http://evil.invalid"),
                   case("nonloopback-same-origin", "public.invalid", "http://public.invalid")]
    keyed = [case("key-does-not-bypass-host", "rebind.invalid", extra=(("x-api-key", "sk-cli"),)),
             case("key-does-not-bypass-origin", origin="http://evil.invalid", extra=(("x-api-key", "sk-cli"),))]
    auth_cases = []
    for stem, path in (("systemone", "/systemone"), ("v1-systemone", "/v1/systemone"), ("x-systemone", "/x/systemone")):
        for keyed_request in (False, True):
            item = case(stem + ("-with-key" if keyed_request else "-none"), path=path, method="POST", extra=(("x-api-key", "sk-cli"),) if keyed_request else ())
            item["group"] = "keys"
            auth_cases.append(item)
    return [
        {"kind": "serve", "args": ["--api-key", "sk-cli"], "env": {}, "cases": auth_cases, "then": [{"kind": "terminate"}]},
        {"kind": "serve", "args": [], "env": {}, "cases": normal, "then": [{"kind": "terminate"}]},
        {"kind": "serve", "args": ["--allowed-host", "trusted-one.invalid", "--allowed-host=trusted-two.invalid", "--allowed-origin", "https://ui-one.invalid", "--allowed-origin=http://ui-two.invalid:8000"], "env": {}, "cases": allowed, "then": [{"kind": "terminate"}]},
        {"kind": "serve", "args": ["--host", "0.0.0.0"], "env": {}, "cases": nonloopback, "then": [{"kind": "terminate"}]},
        {"kind": "serve", "args": ["--api-key", "sk-cli"], "env": {}, "cases": keyed, "then": [{"kind": "terminate"}]},
        {"kind": "serve", "args": ["--api-key", "sk-cli", "--api-key-file", "@KEYFILE@"], "env": {"TENSORFOLD_API_KEY": "sk-env"},
         "key_file": "# keys\nclient: sk-file\nsk-bare\n", "cases": full_key_cases(), "then": [{"kind": "terminate"}]},
        {"kind": "serve", "args": ["--api-key-file", "@KEYFILE@"], "env": {},
         "key_file": "app[inference]: sk-app\nmonitor[read]: sk-read\nops: sk-ops\n", "cases": scope_cases(), "then": [{"kind": "terminate"}]},
        {"kind": "serve", "args": [], "env": {}, "cases": admission_cases("off"), "then": [{"kind": "terminate"}]},
        {"kind": "serve", "args": ["--max-inflight", "0", "--max-inflight-per-key", "0", "--requests-per-second", "0", "--request-burst", "7"],
         "env": {}, "cases": admission_cases("zero"), "then": [{"kind": "terminate"}]},
    ]


if __name__ == "__main__":
    import json
    from pathlib import Path
    (Path(__file__).parent / "golden/security-cases.json").write_text(json.dumps({"steps": steps()}) + "\n")
