#!/usr/bin/env python3
"""R543: sampled, unequal-length concurrency -- the property the batch has never been asked for.

The batch's receipts are greedy and lockstep (R-531/532/534/535/536). Real serving samples and its lanes
stop at different lengths. This harness runs four ~22k conversations, three original writing turns, each
lane with its own max_tokens and all lanes at temperature 0.7 / top_p 0.95 / seed 1234, on the shipped
default (parallel 4, batch on, cache 4 GiB); arm B replays every conversation solo (parallel 1) with the
same sampling. The property is per (lane, turn): sha16(A) == sha16(B). Greedy equality is the batch's old
bar; a sampled draw is keyed by (seed, position, id), so a sampled stream exposes any row-wise logit
difference the greedy pick hides.
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.expanduser("~/rehearse"))
import rehearse as R

BASE_CHARS = int(os.environ.get("R543_BASE", "88000"))
TURNS = [
    ("Write an original short story of about 800 words. Setting: a lighthouse on the Cornish coast in "
     "March 1902. Characters: the keeper, Thomas Poldean, 58, and his granddaughter, Eliza, 11. Include "
     "one specific date, one named ship, and a storm. Do not summarize or quote the passage above."),
    ("Now rewrite the same events as a letter from the keeper's wife, Maud, to her sister in Plymouth. "
     "Keep the date and the ship's name. About 550 words."),
    ("Now write a dialogue-only scene, about 500 words, in which Eliza argues with the harbourmaster about "
     "what she saw that night. No narration."),
]
GEN = [int(x) for x in os.environ.get("R543_GEN", "240,160,200,120").split(",")]


def chat_raw(messages, max_tokens, barrier=None):
    body = {"model": R.MODEL_NAME, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": 1234, "draft": False,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(R.BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait()
    t0 = time.time()
    first = last = None
    usage = None
    text = ""
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            ch = j.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                now = time.time()
                if first is None:
                    first = now
                last = now
                text += ch[0]["delta"]["content"]
    n = usage["completion_tokens"] if usage else 0
    return {"prompt": usage["prompt_tokens"] if usage else None, "gen": n,
            "ttft": round(first - t0, 3) if first else None,
            "decode_s": round(last - first, 3) if first and last else None,
            "tok_s": round(n / (last - first), 2) if first and last and last > first else None,
            "wall": round(time.time() - t0, 3),
            "sha16": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text}


def degenerate(text):
    w = re.findall(r"[A-Za-z0-9']+", text.lower())
    best = 0
    for n in (6, 8, 12):
        g = [" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))]
        if g:
            best = max(best, Counter(g).most_common(1)[0][1])
    return {"words": len(w), "distinct": len(set(w)), "max_ngram": best}


def fire(msgs, tags):
    bar = threading.Barrier(len(tags))
    out = {}
    ths = []
    for tg in tags:
        th = threading.Thread(target=lambda tg=tg: out.__setitem__(tg, chat_raw(msgs[tg], GEN[int(tg[1:])], barrier=bar)))
        th.start()
        ths.append(th)
    for th in ths:
        th.join()
    return out


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "A"
    text = R.corpus()
    lanes = ["c%d" % i for i in range(len(GEN))]
    msgs = {tg: [{"role": "user", "content": text[i * BASE_CHARS:(i + 1) * BASE_CHARS] + "\n\n" + TURNS[0]}]
            for i, tg in enumerate(lanes)}
    tag = "r543-%s" % arm
    S = R.Serve(tag, 4, extra_env={"TF_DSV41_BATCH_DEBUG": "1"}, parallel=len(GEN) if arm == "A" else 1)
    boot = S.start()
    rows = []
    reply = {}
    for t in range(len(TURNS)):
        if t > 0:
            for tg in lanes:
                msgs[tg].append({"role": "assistant", "content": reply[tg]})
                msgs[tg].append({"role": "user", "content": TURNS[t]})
        t0 = time.time()
        out = fire(msgs, lanes) if arm == "A" else {tg: chat_raw(msgs[tg], GEN[int(tg[1:])]) for tg in lanes}
        reply = {tg: out[tg]["text"] for tg in lanes}
        wall = time.time() - t0
        agg = sum(a["gen"] for a in out.values()) / max(a["decode_s"] for a in out.values())
        for tg in lanes:
            a = out[tg]
            rows.append(dict(a, turn=t, lane=tg, asked=GEN[int(tg[1:])], agg=round(agg, 1),
                             turn_wall=round(wall, 3), deg=degenerate(a["text"])))
        print("turn %d wall %.2f aggregate %.1f | %s" % (
            t, wall, agg, " ".join("%s %d/%s %.1f" % (tg, out[tg]["gen"], out[tg]["sha16"][:8], out[tg]["tok_s"] or 0) for tg in lanes)))
    boot["stop"] = S.stop()
    path = os.path.expanduser("~/rehearse/r543-%s.json" % arm)
    json.dump({"boot": boot, "arms": rows, "log_tail": S.new_log()[-4000:], "arm": arm,
               "gen": GEN, "base_chars": BASE_CHARS, "sampling": {"temperature": 0.7, "top_p": 0.95, "seed": 1234}},
              open(path, "w"))
    print("json", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
