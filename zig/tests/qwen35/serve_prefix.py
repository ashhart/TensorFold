"""Prompt reuse of a running native server over HTTP; usage: serve_prefix.py URL MODEL [SYSTEM_WORDS]."""

import json
import sys
import time
import urllib.request

URL, MODEL = sys.argv[1], sys.argv[2]
WORDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
SYSTEM = "You are a careful assistant. Reference notes: " + " ".join(f"note{i % 89}x{(i * 7919) % 1013}" for i in range(WORDS))
QUESTIONS = ["What is the first note?", "How many notes are there, roughly?", "Name the last note.", "Is note5x5 listed?"]


def ask(question, draft=True, system=SYSTEM):
    """The reply's text, its time to the first token in milliseconds and the usage the server reports."""
    body = {"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
            "max_tokens": 16, "temperature": 0.0, "draft": draft, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    began = time.perf_counter()
    first = None
    text = ""
    usage = {}
    with urllib.request.urlopen(request, timeout=900) as reply:
        for raw in reply:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            doc = json.loads(line[5:])
            if doc.get("usage"):
                usage = doc["usage"]
            for choice in doc.get("choices", []):
                piece = choice.get("delta", {}).get("content") or ""
                if piece and first is None:
                    first = (time.perf_counter() - began) * 1000
                text += piece
    return text, first, usage


def cached(usage):
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


ask(QUESTIONS[0], True, "warm up the kernels: " + SYSTEM[:200])
rows = []
for label, question, draft in (
    ("first request (nothing kept)", QUESTIONS[0], True),
    ("second request, same system prompt", QUESTIONS[1], True),
    ("third request, same system prompt", QUESTIONS[2], True),
    ("same system prompt, draft false (no cache)", QUESTIONS[3], False),
    ("different system prompt of the same length", QUESTIONS[1], True),
):
    system = SYSTEM if "different" not in label else SYSTEM.replace("note", "memo")
    text, ttft, usage = ask(question, draft, system)
    rows.append((label, usage.get("prompt_tokens", 0), cached(usage), ttft, text))
    print(f"{label}: prompt {rows[-1][1]} tokens, cached_tokens {rows[-1][2]}, time to first token {ttft:.0f} ms", flush=True)
# the same question with and without the cache: the replies must be equal
hit_text = ask(QUESTIONS[1], True)[0]
miss_text = ask(QUESTIONS[1], False)[0]
print("hit == miss:", hit_text == miss_text, flush=True)
sys.exit(0 if hit_text == miss_text else 1)
