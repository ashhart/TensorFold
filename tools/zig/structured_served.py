"""Structured output served end to end: valid replies, drafted equal to ``"draft": false``, concurrent equal to solo
with grammar and plain requests mixed, and the 400 refusals (#564's gate on a running tensorfold-native).

    python tools/zig/structured_served.py --url http://127.0.0.1:8000 [--concurrency 4]

Every grammar kind the server parses (response_format json_object and json_schema, guided_json, guided_regex,
guided_choice, guided_grammar, structured_outputs), greedy and seeded sampled, thinking off and on (with a
thinking budget, so a forced close comes before the grammar starts).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import urllib.error
import urllib.request

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "temperature_c": {"type": "integer"},
        "conditions": {"enum": ["sunny", "cloudy", "rain", "snow"]},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["city", "temperature_c", "conditions"],
}
DATE = r"\d{4}-\d{2}-\d{2}"
CHOICES = ["positive", "negative", "neutral"]
EBNF = 'root ::= "Answer: " ("yes" | "no") ", because " [a-z ]{3,40} "."'
EBNF_CHECK = r"Answer: (yes|no), because [a-z ]{3,40}\."

CASES = [
    ("json_object", "Describe a cat as a JSON object with a few fields.", {"response_format": {"type": "json_object"}}),
    ("json_schema", "What is the weather like in Lisbon today? Answer as JSON.",
     {"response_format": {"type": "json_schema", "json_schema": {"name": "weather", "schema": SCHEMA}}}),
    ("guided_json", "Make up the weather in Oslo.", {"guided_json": SCHEMA}),
    ("regex", "When did the first person walk on the moon? Give the date.", {"guided_regex": DATE}),
    ("choice", "Classify the sentiment: 'I loved every minute of this film.'", {"guided_choice": CHOICES}),
    ("grammar", "Is the sky blue on a clear day?", {"guided_grammar": EBNF}),
    ("structured_outputs", "Give today's date in ISO form.", {"structured_outputs": {"regex": DATE}}),
]
PLAIN = [("plain", "Write two sentences about the sea.", {})]


def schema_ok(v) -> bool:
    if not isinstance(v, dict) or not all(k in v for k in SCHEMA["required"]):
        return False
    return (isinstance(v["city"], str) and isinstance(v["temperature_c"], int) and not isinstance(v["temperature_c"], bool)
            and v["conditions"] in SCHEMA["properties"]["conditions"]["enum"]
            and (("tags" not in v) or (isinstance(v["tags"], list) and len(v["tags"]) <= 3 and all(isinstance(t, str) for t in v["tags"])))
            and set(v) <= set(SCHEMA["properties"]))


def valid(name: str, text: str) -> bool:
    if name == "json_object":
        try:
            return isinstance(json.loads(text), dict)
        except ValueError:
            return False
    if name in ("json_schema", "guided_json"):
        try:
            return schema_ok(json.loads(text))
        except ValueError:
            return False
    if name in ("regex", "structured_outputs"):
        return re.fullmatch(DATE, text) is not None
    if name == "choice":
        return text in CHOICES
    if name == "grammar":
        return re.fullmatch(EBNF_CHECK, text) is not None
    return True


def post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def body(model: str, prompt: str, extra: dict, sampled: bool, thinking: bool, draft: bool, tokens: int) -> dict:
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
         "chat_template_kwargs": {"enable_thinking": thinking}, **extra}
    if sampled:
        b.update({"temperature": 0.8, "top_p": 0.95, "seed": 1234})
    else:
        b["temperature"] = 0
    if thinking:
        b["thinking_budget"] = 256  # the reply's answer comes after a forced close: the grammar starts after it
    if not draft:
        b["draft"] = False
    return b


def answer(reply: dict) -> tuple[str, str]:
    m = reply["choices"][0]["message"]
    return (m.get("content") or "").strip(), reply["choices"][0].get("finish_reason", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--thinking-tokens", type=int, default=1200)
    args = ap.parse_args()
    with urllib.request.urlopen(args.url + "/v1/models") as r:
        model = json.loads(r.read())["data"][0]["id"]

    runs = [(name, prompt, extra, sampled, thinking) for (name, prompt, extra) in CASES + PLAIN
            for sampled in (False, True) for thinking in (False, True)]
    failures = 0
    alone: dict[tuple, str] = {}
    print(f"{model}: {len(runs)} runs alone, drafted and plain")
    for name, prompt, extra, sampled, thinking in runs:
        tokens = args.thinking_tokens if thinking else args.tokens
        code, r = post(args.url, body(model, prompt, extra, sampled, thinking, True, tokens))
        code2, r2 = post(args.url, body(model, prompt, extra, sampled, thinking, False, tokens))
        if code != 200 or code2 != 200:
            print(f"FAIL {name} sampled={sampled} thinking={thinking}: HTTP {code}/{code2} {r} {r2}")
            failures += 1
            continue
        (text, why), (text2, _) = answer(r), answer(r2)
        ok, same = valid(name, text), text == text2
        failures += (not ok) + (not same)
        alone[(name, sampled, thinking)] = text
        print(f"{'ok  ' if ok and same else 'FAIL'} {name:18} sampled={sampled!s:5} thinking={thinking!s:5} "
              f"valid={ok!s:5} drafted==plain={same!s:5} {why:6} {text[:60]!r}")

    print(f"all {len(runs)} at once, {args.concurrency} at a time")
    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        futures = {pool.submit(post, args.url, body(model, p, e, s, t, True, args.thinking_tokens if t else args.tokens)): (n, s, t)
                   for (n, p, e, s, t) in runs}
        together = 0
        for f in cf.as_completed(futures):
            key = futures[f]
            code, r = f.result()
            if code != 200 or key not in alone:
                failures += 1
                print(f"FAIL concurrent {key}: HTTP {code}")
                continue
            if answer(r)[0] != alone[key]:
                failures += 1
                print(f"FAIL concurrent {key}: differs from alone: {answer(r)[0][:60]!r}")
            else:
                together += 1
    print(f"{together} of {len(runs)} concurrent replies equal their run alone")

    refusals = [
        ("malformed regex", {"guided_regex": "(unclosed"}, "the grammar cannot be enforced"),
        ("undefined EBNF rule", {"guided_grammar": "root ::= missing"}, "the grammar cannot be enforced"),
        ("schema not JSON", {"guided_json": "{not json"}, "not valid JSON"),
        ("unknown response_format", {"response_format": {"type": "xml"}}, "response_format type"),
        ("grammar with a required call", {"guided_regex": DATE, "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}], "tool_choice": "required"}, "cannot be combined with tool_choice"),
    ]
    for label, extra, words in refusals:
        code, r = post(args.url, body(model, "hi", extra, False, False, True, 16))
        message = json.dumps(r)
        ok = code == 400 and words in message
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} refused {label}: HTTP {code} {r.get('error', {}).get('message', message)[:100]!r}")
    print("PASS" if failures == 0 else f"FAIL: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
