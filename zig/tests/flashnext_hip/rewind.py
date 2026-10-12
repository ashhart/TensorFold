"""Cache on equals cache off through rewinds, one server at a time; usage: rewind.py URL MODEL OUT.json [WANT.json]."""

import json
import sys
import urllib.request

URL, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
WANT = sys.argv[4] if len(sys.argv) > 4 else None
SYSTEM = "You are a careful assistant. Reference notes: " + " ".join(f"note{i % 89}x{(i * 7919) % 1013}" for i in range(1500))


def chat(messages, draft):
    """The reply's text and the prompt tokens the server says it took from kept states."""
    body = {"model": MODEL, "messages": messages, "max_tokens": 48, "temperature": 0.0, "draft": draft,
            "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as reply:
        doc = json.loads(reply.read())
    cached = (doc.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return doc["choices"][0]["message"]["content"], cached


def script():
    """The steps of one conversation: turns, a regenerate, an edit, a delete, an edit of the first turn."""
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


out = {}
for draft in (True, False):
    history = {}
    for name, build in script():
        messages = build(history)
        text, cached = chat(messages, draft)
        history[name] = messages + [{"role": "assistant", "content": text}]
        out[f"{name}, drafts {'on' if draft else 'off'}"] = [text, cached]
json.dump(out, open(OUT, "w"), indent=1)
if WANT is None:
    sys.exit(0)
want = json.load(open(WANT))
bad = 0
for key, (text, cached) in out.items():
    ok = text == want[key][0]
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} rewind {key}: {cached} prompt tokens from kept states")
    if not ok:
        print(f"  without the cache: {want[key][0]!r}\n  with it:           {text!r}")
print(f"rewind {'PASS' if bad == 0 else 'FAIL'}: {bad} differing replies")
sys.exit(1 if bad else 0)
