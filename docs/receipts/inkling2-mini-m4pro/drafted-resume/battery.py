"""Equality battery: multi-turn chats past the sliding window, solo then 4 at once, greedy and seeded, mixed draft flags.

usage: battery.py BASE_URL OUT.json   (BASE_URL ends in /v1)
Each request records its token ids, text, prompt and cached tokens, time to first token and the speculative counters.
"""
import json
import os
import sys
import threading
import time
import urllib.request

base, out_path = sys.argv[1], sys.argv[2]
DOC = open(os.environ["NOTES"]).read()
CONVS = 5
RESULTS = {}
LOCK = threading.Lock()


def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        raw = json.loads(r.read())
    return raw, time.time() - t0


def turn_user(conv, turn):
    off = conv * 1300
    if turn == 0:
        return "Read these notes and answer in one sentence what they cover.\n\n" + DOC[off:off + 8800]
    if turn == 1:
        return "Which kernels or tests do the notes name? List three."
    if turn == 2:
        return "Now read this further passage and say how it relates.\n\n" + DOC[off + 8800:off + 12600]
    return "Give one number from the notes."


def one(key, messages, temp, seed, draft, tokens=40):
    body = {"model": "m", "messages": messages, "max_tokens": tokens, "temperature": temp, "return_token_ids": True}
    if temp > 0:
        body["seed"] = seed
    if draft is False:
        body["draft"] = False
    raw, secs = post("/chat/completions", body)
    tf = raw.get("tensorfold", {})
    rec = {
        "ids": tf.get("token_ids"),
        "text": raw["choices"][0]["message"].get("content"),
        "prompt_tokens": raw["usage"]["prompt_tokens"],
        "cached_tokens": (raw["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": raw["usage"]["completion_tokens"],
        "ttft": tf.get("time_to_first_token"),
        "seconds": secs,
        "spec": raw.get("speculative") or tf.get("speculative"),
        "finish": raw["choices"][0].get("finish_reason"),
    }
    with LOCK:
        RESULTS[key] = rec
    return rec


def conversation(tag, conv, temp, draft_of=lambda turn: True):
    messages = []
    for turn in range(4):
        messages.append({"role": "user", "content": turn_user(conv, turn)})
        rec = one(f"{tag}/t{turn}", messages, temp, 1000 + 10 * conv + turn, draft_of(turn))
        messages.append({"role": "assistant", "content": rec["text"]})


for temp in [float(x) if x != '0' else 0 for x in os.environ.get('TEMPS', '0,0.7').split(',')]:
    t = f"temp{temp}"
    conversation(f"{t}/solo/c0", 0, temp)
    threads = [threading.Thread(target=conversation, args=(f"{t}/par/c{c}", c, temp)) for c in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    # a conversation that alternates "draft": false and drafted turns: states kept by one are resumed by the other
    conversation(f"{t}/mixed/c4", 4, temp, lambda turn: turn % 2 == 1)
    # the first conversation again: every state it kept is resumed
    conversation(f"{t}/again/c0", 0, temp)
json.dump(RESULTS, open(out_path, "w"), indent=1)
print("wrote", out_path, len(RESULTS), "requests")
