"""Exactness of a running native server over HTTP; usage: serve_check.py URL MODEL [TOKENS]."""

import json
import sys
import threading
import time
import urllib.request

URL, MODEL = sys.argv[1], sys.argv[2]
TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
PROMPTS = [
    "Write a short poem about the sea.",
    "List the first twelve prime numbers, separated by commas.",
    "Explain in two sentences why the sky is blue.",
    "Count from one to thirty in words.",
]


def ask(messages, draft=True, temperature=0.0, seed=7):
    body = {"model": MODEL, "messages": messages, "max_tokens": TOKENS, "temperature": temperature, "seed": seed,
            "draft": draft, "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as reply:
        doc = json.load(reply)
    return doc["choices"][0]["message"]["content"], doc.get("usage", {}).get("prompt_tokens_details", {})


def user(text):
    return [{"role": "user", "content": text}]


def together(draft, temperature):
    out = [None] * len(PROMPTS)

    def run(i):
        out[i] = ask(user(PROMPTS[i]), draft, temperature)[0]

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(PROMPTS))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


LONG = "Summarise this list in one sentence: " + " ".join(f"item{i % 97}x{(i * 7919) % 1013}" for i in range(500))


def among_decoders(draft, temperature):
    """A long prompt (a few thousand tokens, so several prompt chunks) sent while three streams decode."""
    out = [None] * 4

    def run(i, text, delay):
        time.sleep(delay)
        out[i] = ask(user(text), draft, temperature)[0]

    threads = [threading.Thread(target=run, args=(i, PROMPTS[i], 0)) for i in range(3)]
    threads.append(threading.Thread(target=run, args=(3, LONG, 0.5)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


failed = 0


def check(name, same):
    global failed
    failed += not same
    print(("ok   " if same else "FAIL ") + name, flush=True)


for label, temperature in (("greedy", 0.0), ("t0.8", 0.8)):
    serial = [ask(user(p), False, temperature)[0] for p in PROMPTS]
    drafted = [ask(user(p), True, temperature)[0] for p in PROMPTS]
    check(f"{label}: drafted == serial", drafted == serial)
    check(f"{label}: solo == together", together(True, temperature) == drafted)
    check(f"{label}: serial together == serial solo", together(False, temperature) == serial)
    again = [ask(user(p), True, temperature)[0] for p in PROMPTS]
    check(f"{label}: the same twice", again == drafted)
    # the long prompt first among the decoders (its prompt goes in a chunk a round), then alone (resumed from its cuts)
    mixed = among_decoders(True, temperature)
    check(f"{label}: decoders unchanged by a long prompt", mixed[:3] == drafted[:3])
    check(f"{label}: long prompt among decoders == alone", mixed[3] == ask(user(LONG), True, temperature)[0])
    check(f"{label}: serial long prompt among decoders == alone", among_decoders(False, temperature)[3] == mixed[3])
first, _ = ask(user(PROMPTS[0]), True)
turn = user(PROMPTS[0]) + [{"role": "assistant", "content": first}, {"role": "user", "content": "Now say it shorter."}]
fresh, _ = ask(turn, False)
resumed, details = ask(turn, True)
check(f"resumed == fresh (cached {details})", resumed == fresh)
print("failed", failed)
sys.exit(1 if failed else 0)
