"""Chat with the prompt cache equals chat without it, through regenerate, edit and delete rewinds.

usage: serve_rewind.py CACHED_URL UNCACHED_URL MODEL [SYSTEM_WORDS]; exits 1 on any differing reply.
"""

import json
import sys
import urllib.request

CACHED, UNCACHED, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
WORDS = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
SYSTEM = "You are a careful assistant. Reference notes: " + " ".join(f"note{i % 89}x{(i * 7919) % 1013}" for i in range(WORDS))


def chat(url, messages, draft=True):
    """The reply's text and the prompt tokens the server says it took from kept states."""
    body = {"model": MODEL, "messages": messages, "max_tokens": 48, "temperature": 0.0, "draft": draft,
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as reply:
        doc = json.loads(reply.read())
    cached = (doc.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return doc["choices"][0]["message"]["content"], cached


def script():
    """The steps of one conversation, each the messages a client sends; replies are filled in as they come."""
    system = {"role": "system", "content": SYSTEM}
    return [
        ("turn 1", lambda h: [system, {"role": "user", "content": "What is the first note?"}]),
        ("turn 2", lambda h: h["turn 1"] + [{"role": "user", "content": "And the second one?"}]),
        ("turn 3", lambda h: h["turn 2"] + [{"role": "user", "content": "Is note5x5 listed?"}]),
        ("regenerate turn 3", lambda h: h["turn 2"] + [{"role": "user", "content": "Is note5x5 listed?"}]),
        ("edit turn 3", lambda h: h["turn 2"] + [{"role": "user", "content": "Is note7x7 listed?"}]),
        ("delete turn 3, then turn 4", lambda h: h["turn 2"] + [{"role": "user", "content": "Name the last note."}]),
        ("edit turn 1", lambda h: [system, {"role": "user", "content": "What is the third note?"}]),
        ("turn 2 again", lambda h: h["turn 1"] + [{"role": "user", "content": "And the second one?"}]),
    ]


def run(url, draft):
    """Every step's reply on `url`; a step's history keeps the reply it got."""
    history, out = {}, []
    for name, build in script():
        messages = build(history)
        text, cached = chat(url, messages, draft)
        history[name] = messages + [{"role": "assistant", "content": text}]
        out.append((name, text, cached))
    return out


bad = 0
for draft in (True, False):
    plain = run(UNCACHED, draft)
    kept = run(CACHED, draft)
    for (name, want, _), (_, got, cached) in zip(plain, kept):
        ok = got == want
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} rewind {name}, drafts {'on' if draft else 'off'}: {cached} prompt tokens from kept states")
        if not ok:
            print(f"  without the cache: {want!r}\n  with it:           {got!r}")
print(f"rewind {'PASS' if bad == 0 else 'FAIL'}: {bad} differing replies")
sys.exit(1 if bad else 0)
