#!/usr/bin/env python3
# Adapted from jayleaton/deepseek-v41-tensorfold-spark bench/structured.py, Apache License 2.0, Copyright 2026 Jay
# Leaton (https://x.com/jayleaton). Modified for tensorfold dsv41-cuda: the API key from TF_API_KEY, our port and model
# defaults, return_token_ids on every body, the tools suite opt-in (it needs TF_DSV41_TOOL_GRAMMAR on the server).
"""Structured output against a running V4.1 server (the GLM recipe's bench/structured.py, for DeepSeek-V4.1):
a reply's identity is its ``tensorfold.token_ids`` (every body asks ``return_token_ids``), else its text.

Suites (``--suites``, comma list; all by default):

- ``schemas``: 6 JSON schemas (nested objects, arrays, enums, bounds, numbers) x thinking off / on (effort low), greedy:
  each sent drafted, with ``"draft": false`` (serial), and the 6 of a setting 4 at a time concurrently. Gates: drafted
  == serial == batched; every reply that finished (``stop``) parses and validates; with thinking on the reasoning stays
  in ``reasoning_content`` and never leaks into the JSON; no DSML / special-token markup anywhere in the content.
- ``tools``: ``tool_choice`` ``required``, a named function, a ``strict`` tool under ``auto``, ``parallel_tool_calls``
  false, ``tool_choice`` ``none`` (no call), and a streamed required call (the fragments concatenate to the final
  arguments): every call names a known tool and its arguments parse and validate, no markup leaks. Required and
  named calls need ``TF_DSV41_TOOL_GRAMMAR=required`` (or ``all``) on the server (else HTTP 400), so the suite runs
  only when asked: ``--suites schemas,tools``.

  TF_API_KEY=... python3 tools/dsv41_structured.py --base http://127.0.0.1:8888 --out structured.json
The key is read from TF_API_KEY only, never a flag. Exit status 0 when every gate passes. Standard library only.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request
from typing import Any

LEAK = re.compile(r"｜DSML｜|</?think>|<｜(end▁of▁sentence|begin▁of▁sentence|User|Assistant|System)｜>|<tool_result>")
SCHEMAS: list[tuple[str, dict]] = [
    ("person", {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0,
                "maximum": 130}, "email": {"type": "string"}}, "required": ["name", "age", "email"]}),
    ("order", {"type": "object", "properties": {"id": {"type": "string"}, "items": {"type": "array", "minItems": 2,
               "maxItems": 4, "items": {"type": "object", "properties": {"sku": {"type": "string"}, "qty": {
                   "type": "integer", "minimum": 1}}, "required": ["sku", "qty"]}}, "total": {"type": "number"}},
               "required": ["id", "items", "total"]}),
    ("status", {"type": "object", "properties": {"state": {"enum": ["ok", "degraded", "down"]}, "uptime_s": {
                "type": "integer", "minimum": 0}, "notes": {"type": "array", "items": {"type": "string"}}},
                "required": ["state", "uptime_s", "notes"]}),
    ("city", {"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"},
              "population": {"type": "integer"}, "coords": {"type": "object", "properties": {"lat": {"type": "number"},
              "lon": {"type": "number"}}, "required": ["lat", "lon"]}}, "required": ["city", "country", "coords"]}),
    ("flags", {"type": "object", "properties": {"debug": {"type": "boolean"}, "level": {"enum": [1, 2, 3]},
               "tags": {"type": "array", "items": {"enum": ["a", "b", "c"]}, "maxItems": 3}},
               "required": ["debug", "level", "tags"]}),
    ("list", {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "properties": {
              "n": {"type": "integer"}, "square": {"type": "integer"}}, "required": ["n", "square"]}}),
]
PROMPTS = {"person": "Invent a person.", "order": "Invent a small web-shop order.", "status": "Report a service status.",
           "city": "Describe Lyon.", "flags": "Pick feature flags for a test build.",
           "list": "The numbers 1, 2, 3 with their squares."}
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer",
                    "minimum": 1, "maximum": 7}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_flights", "description": "Find flights",
     "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"},
                    "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}}},
    {"type": "function", "function": {"name": "send_email", "description": "Send an email", "strict": True,
     "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"},
                    "body": {"type": "string"}}, "required": ["to", "subject", "body"],
                    "additionalProperties": False}}},
]
TOOL_BY_NAME = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


def headers() -> dict[str, str]:
    key = os.environ.get("TF_API_KEY", "")
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}


def post(base: str, body: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream(base: str, body: dict, timeout: float = 900.0) -> tuple[dict, list[dict]]:
    """A streamed request: (the assembled tool calls by index, the raw chunks)."""

    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(dict(body, stream=True)).encode(),
                                 headers())
    calls: dict[int, dict] = {}
    chunks = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            c = json.loads(line[5:])
            chunks.append(c)
            for ch in c.get("choices") or []:
                for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                    d = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    d["name"] += f.get("name") or ""
                    d["arguments"] += f.get("arguments") or ""
    return calls, chunks


def validate(v: Any, s: dict, path: str = "$") -> str | None:
    """A small JSON-schema check (type, enum, required, properties, additionalProperties, items, bounds)."""

    if "enum" in s:
        return None if v in s["enum"] else f"{path}: {v!r} not in {s['enum']}"
    t = s.get("type")
    kinds = {"object": dict, "array": list, "string": str, "boolean": bool}
    if t in kinds and not isinstance(v, kinds[t]):
        return f"{path}: not {t}"
    if t == "integer" and (not isinstance(v, int) or isinstance(v, bool)):
        return f"{path}: not integer"
    if t == "number" and (not isinstance(v, (int, float)) or isinstance(v, bool)):
        return f"{path}: not number"
    if t in ("integer", "number"):
        if "minimum" in s and v < s["minimum"] or "maximum" in s and v > s["maximum"]:
            return f"{path}: {v} out of bounds"
    if t == "object":
        for k in s.get("required", []):
            if k not in v:
                return f"{path}: missing {k}"
        props = s.get("properties", {})
        if s.get("additionalProperties") is False and set(v) - set(props):
            return f"{path}: extra keys {sorted(set(v) - set(props))}"
        for k, sub in props.items():
            if k in v and (e := validate(v[k], sub, f"{path}.{k}")):
                return e
    if t == "array":
        if len(v) < s.get("minItems", 0) or len(v) > s.get("maxItems", 1 << 30):
            return f"{path}: {len(v)} items"
        for i, x in enumerate(v):
            if (e := validate(x, s.get("items", {}), f"{path}[{i}]")):
                return e
    return None


def ident(r: dict) -> str:
    tf = r.get("tensorfold") or {}
    ids = tf.get("token_ids")
    return json.dumps(ids) if ids else json.dumps(r["choices"][0]["message"].get("content"))


def body_for(a, prompt: str, *, schema: tuple[str, dict] | None = None, thinking: bool = False,
             max_tokens: int = 1024) -> dict:
    b = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
         "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": thinking}, "return_token_ids": True}
    if thinking:
        b["chat_template_kwargs"]["reasoning_effort"] = "low"
    if schema:
        b["response_format"] = {"type": "json_schema", "json_schema": {"name": schema[0], "schema": schema[1],
                                                                       "strict": True}}
    return b


def suite_schemas(a) -> tuple[dict, list[str]]:
    rec, fails = {}, []
    for thinking in (False, True):
        bodies = [body_for(a, PROMPTS[n], schema=(n, s), thinking=thinking, max_tokens=2048 if thinking else 768)
                  for n, s in SCHEMAS]
        drafted = [post(a.base, b) for b in bodies]
        serial = [post(a.base, dict(b, draft=False)) for b in bodies]
        with cf.ThreadPoolExecutor(4) as ex:
            batched = list(ex.map(lambda b: post(a.base, b), bodies))
        for (n, s), x, y, z in zip(SCHEMAS, drafted, serial, batched):
            tag = f"{n}/{'think' if thinking else 'chat'}"
            msg = x["choices"][0]["message"]
            fin = x["choices"][0]["finish_reason"]
            rec[tag] = {"finish": fin, "tokens": x.get("usage", {}).get("completion_tokens"),
                        "same": ident(x) == ident(y) == ident(z)}
            if not rec[tag]["same"]:
                fails.append(f"{tag}: drafted / serial / batched replies differ")
            content = msg.get("content") or ""
            if LEAK.search(content):
                fails.append(f"{tag}: markup in content: {content[:120]!r}")
            if thinking and not (msg.get("reasoning_content") or msg.get("reasoning")):
                rec[tag]["note"] = "no reasoning returned"
            if fin != "stop":
                fails.append(f"{tag}: finish {fin} (max_tokens too small or the grammar never closed)")
                continue
            try:
                e = validate(json.loads(content), s)
            except ValueError as exc:
                e = f"not JSON ({exc}): {content[:120]!r}"
            if e:
                fails.append(f"{tag}: {e}")
    return rec, fails


def check_calls(tag: str, calls: list[dict], want: str | None, fails: list[str]) -> None:
    if not calls:
        fails.append(f"{tag}: no tool call")
        return
    for c in calls:
        name, raw = c["function"]["name"], c["function"]["arguments"]
        if name not in TOOL_BY_NAME or (want and name != want):
            fails.append(f"{tag}: call {name!r} (want {want or 'a known tool'})")
            continue
        if LEAK.search(raw):
            fails.append(f"{tag}: markup in arguments {raw[:120]!r}")
        try:
            e = validate(json.loads(raw), TOOL_BY_NAME[name])
        except ValueError as exc:
            e = f"arguments not JSON ({exc}): {raw[:120]!r}"
        if e:
            fails.append(f"{tag}: {e}")


def suite_tools(a) -> tuple[dict, list[str]]:
    rec, fails = {}, []
    ask = "Check the weather in Oslo for 3 days, then find a flight from Oslo to Rome on 2026-11-02."
    cases = [("required", {"tool_choice": "required"}, None),
             ("named", {"tool_choice": {"type": "function", "function": {"name": "search_flights"}}}, "search_flights"),
             ("strict-auto", {"tool_choice": "auto", "messages": [{"role": "user", "content":
                 "Email bob@example.com that the build is green. Use the tool."}]}, "send_email"),
             ("single", {"tool_choice": "required", "parallel_tool_calls": False}, None)]
    for thinking in (False, True):
        for name, extra, want in cases:
            tag = f"{name}/{'think' if thinking else 'chat'}"
            b = dict(body_for(a, ask, thinking=thinking, max_tokens=2048), tools=TOOLS, **extra)
            r = post(a.base, b)
            ch = r["choices"][0]
            calls = ch["message"].get("tool_calls") or []
            rec[tag] = {"finish": ch["finish_reason"], "calls": [c["function"]["name"] for c in calls]}
            check_calls(tag, calls, want, fails)
            if ch["finish_reason"] != "tool_calls":
                fails.append(f"{tag}: finish {ch['finish_reason']}")
            if name == "single" and len(calls) > 1:
                fails.append(f"{tag}: {len(calls)} calls with parallel_tool_calls false")
    r = post(a.base, dict(body_for(a, ask, max_tokens=512), tools=TOOLS, tool_choice="none"))
    rec["none/chat"] = {"finish": r["choices"][0]["finish_reason"],
                        "calls": len(r["choices"][0]["message"].get("tool_calls") or [])}
    if rec["none/chat"]["calls"]:
        fails.append("none/chat: tool_choice none still called a tool")
    calls, _ = stream(a.base, dict(body_for(a, ask, max_tokens=1024), tools=TOOLS, tool_choice="required"))
    rec["stream/chat"] = {"calls": [c["name"] for c in calls.values()]}
    check_calls("stream/chat", [{"function": c} for c in calls.values()], None, fails)
    return rec, fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="DeepSeek-v4.1-Flash-EXL3")
    ap.add_argument("--suites", default="schemas", help="comma list: schemas, tools")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    t0, out, bad = time.time(), {}, 0
    for s in a.suites.split(","):
        try:
            rec, fails = {"schemas": suite_schemas, "tools": suite_tools}[s](a)
        except Exception as exc:                    # noqa: BLE001 - a crashed suite is a failed suite
            rec, fails = {}, [f"suite crashed: {type(exc).__name__}: {exc}"[:300]]
        out[s] = {"pass": not fails, "fails": fails, "cases": rec}
        bad += bool(fails)
        print(f"[structured] {s}: {'PASS' if not fails else 'FAIL'} ({len(rec)} cases)"
              + "".join(f"\n  - {f}" for f in fails[:20]), flush=True)
    out["pass"], out["wall_s"] = not bad, round(time.time() - t0, 1)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
