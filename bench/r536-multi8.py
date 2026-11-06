#!/usr/bin/env python3
"""R536: the retained cache + the batch at the SHIPPING concurrency (--parallel 8).

Eight conversations, each a distinct ~6.5k-token opening, four original writing turns. Arm A fires all
eight lanes together every turn on the shipped default (parallel 8, batch on, four-window groups); arm B
replays every conversation solo (parallel 1) on the same binary. The properties: each lane, each turn is
token-identical to its solo replay; every turn after the first resumes; no degenerate turn; memory bounded.
This is the first receipt of the batch + retained cache at eight lanes (R-535 receipted 8 lanes at one
turn and no cache; R-534 receipted 4 lanes and four turns).
"""
import hashlib, json, os, re, sys, threading, time, urllib.request
from collections import Counter

sys.path.insert(0, os.path.expanduser("~/rehearse"))
import rehearse as R

BASE_CHARS = int(os.environ.get("R536_BASE", "26000"))
GEN = int(os.environ.get("R536_GEN", "160"))
N = int(os.environ.get("R536_LANES", "8"))
TURNS = [
    ("Write an original short story of about 700 words. Setting: a lighthouse on the Cornish coast "
     "in March 1902. Characters: the keeper, Thomas Poldean, 58, and his granddaughter, Eliza, 11. "
     "Include one specific date, one named ship, and a storm. Do not summarize, quote or paraphrase "
     "the passage above; write only the story."),
    ("Now rewrite the same events as a letter from the keeper's wife, Maud, to her sister in "
     "Plymouth. Keep the date and the ship's name. About 500 words."),
    ("Now write a dialogue-only scene, about 450 words, in which Eliza argues with the "
     "harbourmaster about what she saw that night. No narration."),
    ("Now write the harbourmaster's official report of the wreck, about 450 words. Keep the date "
     "and the ship's name, and make the report's conclusion ambiguous."),
]


def chat_raw(messages, max_tokens, barrier=None):
    body = {"model": R.MODEL_NAME, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "top_p": 0.95, "top_k": 20, "seed": 1234, "draft": False,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(R.BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait()
    t0 = time.time(); first = last = None; usage = None; text = ""
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
    bar = threading.Barrier(len(tags)); out = {}; ths = []
    for tg in tags:
        th = threading.Thread(target=lambda tg=tg: out.__setitem__(tg, chat_raw(msgs[tg], GEN, barrier=bar)))
        th.start(); ths.append(th)
    for th in ths:
        th.join()
    return out


DONE_RE = re.compile(r"done \S+ prompt=(\d+) cached=(\d+).*?tokens=(\d+) sha=(\w+) .*?"
                     r"ttft=([\d.]+) decode=([\d.]+) elapsed=([\d.]+).*?queued=([\d.]+) tok/s=([\d.]+)")


def done_lines(path):
    t = open(path, "rb").read().decode("utf-8", "replace")
    return [{"prompt": int(m.group(1)), "cached": int(m.group(2)), "tokens": int(m.group(3)),
             "sha": m.group(4), "ttft": float(m.group(5)), "decode": float(m.group(6)),
             "elapsed": float(m.group(7)), "queued": float(m.group(8)), "tok_s": float(m.group(9))}
            for m in DONE_RE.finditer(t)]


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "A"
    text = R.corpus()
    lanes = ["c%d" % i for i in range(N)]
    msgs = {tg: [{"role": "user", "content": text[i * BASE_CHARS:(i + 1) * BASE_CHARS] + "\n\n" + TURNS[0]}]
            for i, tg in enumerate(lanes)}
    tag = "r536-%s" % arm
    S = R.Serve(tag, 4, extra_env={"TF_DSV41_BATCH_DEBUG": "1"}, parallel=N if arm == "A" else 1)
    boot = S.start()
    rows = []
    reply = {}
    for t in range(len(TURNS)):
        if t > 0:
            for tg in lanes:
                msgs[tg].append({"role": "assistant", "content": reply[tg]})
                msgs[tg].append({"role": "user", "content": TURNS[t]})
        t0 = time.time()
        if arm == "A":
            out = fire(msgs, lanes)
        else:
            out = {tg: chat_raw(msgs[tg], GEN) for tg in lanes}
        reply = {tg: out[tg]["text"] for tg in lanes}
        wall = time.time() - t0
        agg = sum(a["gen"] for a in out.values()) / max(a["decode_s"] for a in out.values())
        for tg in lanes:
            a = out[tg]
            rows.append(dict(a, turn=t, lane=tg, agg=round(agg, 1), turn_wall=round(wall, 3),
                             deg=degenerate(a["text"])))
        print("turn %d wall %.2f aggregate %.1f | %s" % (
            t, wall, agg, " ".join("%s %.1f/%s" % (tg, out[tg]["tok_s"] or 0, out[tg]["sha16"][:8]) for tg in lanes)))
    boot["stop"] = S.stop()
    path = os.path.expanduser("~/rehearse/r536-%s-8t.json" % arm)
    json.dump({"boot": boot, "arms": rows, "done": done_lines(S.log_path), "arm": arm,
               "gen": GEN, "base_chars": BASE_CHARS}, open(path, "w"))
    print("json", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
