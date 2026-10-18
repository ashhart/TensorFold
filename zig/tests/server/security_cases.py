"""Frozen native browser-policy cases: every socket still connects to the local fake server."""
from __future__ import annotations
import base64


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
    ]


if __name__ == "__main__":
    import json
    from pathlib import Path
    (Path(__file__).parent / "golden/security-cases.json").write_text(json.dumps({"steps": steps()}) + "\n")
