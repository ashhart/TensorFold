"""Standard-library client for a TensorFold server: greedy chat, keyword checks and /v1/slide/learn events."""

import json
import sys
import time
import urllib.request
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]}).")

DEFAULT_URL = "http://127.0.0.1:8080"
MAX_TOKENS = 96
CHAT_TIMEOUT_S = 300
LEARN_TIMEOUT_S = 900


def post_json(url, body, timeout):
    """POST a JSON body and return the decoded JSON reply."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ask(base, question, max_tokens=MAX_TOKENS):
    """Ask one question with greedy decoding and return the reply text."""
    body = {"temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": question}]}
    data = post_json(f"{base}/v1/chat/completions", body, CHAT_TIMEOUT_S)
    return data["choices"][0]["message"]["content"] or ""


def hit(answer, needles):
    """True when any needle appears in the answer, ignoring case."""
    low = answer.lower()
    return any(n.lower() in low for n in needles)


def parse_sse(lines):
    """Turn server-sent event lines into (event, data) pairs, decoding JSON data where it parses."""
    name = None
    for raw in lines:
        line = (raw.decode(errors="replace") if isinstance(raw, bytes) else raw).rstrip("\r\n")
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            payload = line[5:].strip()
            try:
                yield name, json.loads(payload)
            except json.JSONDecodeError:
                yield name, payload


def learn(base, text, source="document", echo=True):
    """Teach text through /v1/slide/learn and return every event with its arrival time."""
    body = json.dumps({"text": text, "source": source}).encode()
    req = urllib.request.Request(f"{base}/v1/slide/learn", data=body, headers={"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=LEARN_TIMEOUT_S) as resp:
        for name, data in parse_sse(resp):
            events.append({"event": name, "data": data, "t": time.time()})
            if echo:
                print(f"    {name}: {json.dumps(data)[:160]}", flush=True)
    return events


def save(results, prefix, payload):
    """Write payload as results/<prefix>-<timestamp>.json and return the path."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"saved {path}")
    return path
