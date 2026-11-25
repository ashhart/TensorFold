"""Exactness of a running native server over HTTP; usage: serve_check.py URL MODEL [TOKENS [FRESH_URL]]."""

import json
import sys
import threading
import time
import urllib.request

URL, MODEL = sys.argv[1], sys.argv[2]
TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
FRESH_URL = sys.argv[4] if len(sys.argv) > 4 else URL  # the same server started with --prompt-cache-gib 0
PROMPTS = [
    "Write a short poem about the sea.",
    "List the first twelve prime numbers, separated by commas.",
    "Explain in two sentences why the sky is blue.",
    "Count from one to thirty in words.",
]


DRAFTS = {"accepted": {}, "plain": 0}  # drafts accepted per temperature, and tokens drafted by plain replies


class Reply:
    """A reply's token hash (reasoning included); equal only to the same hash, never when either has no token."""

    def __init__(self, doc):
        runtime, usage = doc.get("tensorfold") or {}, doc.get("usage") or {}
        self.sha = runtime.get("token_sha256") if usage.get("completion_tokens") else None
        self.text = doc["choices"][0]["message"]["content"]

    def __eq__(self, other):
        return isinstance(other, Reply) and self.sha is not None and self.sha == other.sha


def ask(messages, draft=True, temperature=0.0, seed=7, url=URL):
    body = {"model": MODEL, "messages": messages, "max_tokens": TOKENS, "temperature": temperature, "seed": seed,
            "draft": draft, "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as reply:
        doc = json.load(reply)
    spec = doc.get("speculative") or {}
    if draft:
        DRAFTS["accepted"][temperature] = DRAFTS["accepted"].get(temperature, 0) + (spec.get("accepted") or 0)
    else:
        DRAFTS["plain"] += spec.get("drafted") or 0
    return Reply(doc), doc.get("usage", {}).get("prompt_tokens_details", {})


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


# the resumed chat's first turn: Qwen 3.5 keeps only prompts of 4,096 tokens or more (serve with --context 8192+)
DOCUMENT = "An inventory: " + " ".join(f"crate{i % 89}-{(i * 6151) % 997}" for i in range(900))
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


greedy = None
for label, temperature in (("greedy", 0.0), ("t0.8", 0.8)):
    serial = [ask(user(p), False, temperature)[0] for p in PROMPTS]
    drafted = [ask(user(p), True, temperature)[0] for p in PROMPTS]
    accepted, plain = DRAFTS["accepted"].get(temperature, 0), DRAFTS["plain"]
    check(f"{label}: drafted == serial ({accepted} drafts accepted, plain drafted {plain})",
          drafted == serial and accepted > 0 and plain == 0)
    check(f"{label}: solo == together", together(True, temperature) == drafted)
    check(f"{label}: serial together == serial solo", together(False, temperature) == serial)
    again = [ask(user(p), True, temperature)[0] for p in PROMPTS]
    if greedy is None:
        greedy = drafted
        check(f"{label}: the same twice", again == drafted)
    else:  # the same seed twice, and unlike greedy for some prompt: the sampler ran
        check(f"{label}: the same twice, unlike greedy", again == drafted and drafted != greedy)
    # the long prompt first among the decoders (its prompt goes in a chunk a round), then alone (resumed from its cuts)
    mixed = among_decoders(True, temperature)
    check(f"{label}: decoders unchanged by a long prompt", mixed[:3] == drafted[:3])
    check(f"{label}: long prompt among decoders == alone", mixed[3] == ask(user(LONG), True, temperature)[0])
    check(f"{label}: serial long prompt among decoders == alone", among_decoders(False, temperature)[3] == mixed[3])
opening = user(DOCUMENT + "\n\n" + PROMPTS[0])
first, _ = ask(opening, True)
turn = opening + [{"role": "assistant", "content": first.text}, {"role": "user", "content": "Now say it shorter."}]
# the fresh side on a server that keeps no prompts; a fresh reply that restored tokens fails by name
fresh, fresh_details = ask(turn, False, url=FRESH_URL)
resumed, details = ask(turn, True)
restored = (fresh_details or {}).get("cached_tokens")
if restored is None:
    check("resumed == fresh: the fresh reply has no prompt_tokens_details.cached_tokens", False)
elif restored != 0:
    check(f"resumed == fresh: the fresh request resumed {restored} tokens (FRESH_URL: --prompt-cache-gib 0)", False)
elif not details.get("cached_tokens"):
    check(f"resumed == fresh: the resumed request resumed no tokens (vacuous; cached {details})", False)
else:
    check(f"resumed == fresh (cached {details})", resumed == fresh)
print("failed", failed)
sys.exit(1 if failed else 0)
