"""On rank 0's host, against `tensorfold serve` of GLM-5.3-Flash at TP=2: 1.1.1's batched prompt pass (76765b53,
cuda_prompts.pass) on this checkpoint. Fresh prompts admitted together run in one forward over all their rows, a head
row each (cuda_shared.heads), so FP4-Local's split KDA stack and MXFP8 lm_head run there too. A completion prompt keeps
no state, so each one is fresh. First each of K prompts alone through /v1/completions (greedy, N tokens, ignore_eos),
then a blocker (a long greedy decode, streamed) and, once its first token is out, the K prompts at once from K
threads, so they wait for one admission beside it. Each reply must equal its alone run (tensorfold.token_sha256 and
the text): a gate. The server's time_to_first_token of the K is recorded: replies whose first tokens came from one
pass are within a pass of each other (the spread), else a pass apart each. GLM batches fresh prompts only when its
prompt passes run whole (TF_GLM_PROMPT_STEPS=0 on rank 0, cuda_lanes.zig's whole_vt); by default they fill in chunks
between rounds, one stream at a time. With "one-pass", beside the blocker, two or more of the K must have shared a
pass: first tokens from one pass come out together, those of separate passes at least a prompt pass apart (the
shortest alone first token), so first tokens closer than half of that count as one pass (an arrival between two
admissions can split the K, so not all K are required). K must leave the blocker its lane (--parallel K+1). The rows
go to OUT. One line a request, then "exit 0|1".

    python3 batch_http.py PORT OUT [K] [N] [one-pass]
"""

import json
import sys
import threading
import time
import urllib.request

TOPICS = ["the tides of a small harbour", "how a sundial keeps time", "a recipe for barley bread",
          "the rules of a card game"]
SIZES = [3, 8, 13, 18]  # lines of about 37 tokens: about 1,550 prompt rows in all (a pass holds 2,048)


def prompt(i, nonce):
    lines = [f"Note {nonce}-{i}.{j}: a short study of {TOPICS[i % len(TOPICS)]}, what one sees and measures there, "
             f"and what one would write down for a friend who asks." for j in range(SIZES[i % len(SIZES)])]
    return "\n".join(lines) + "\nSummary:"


def post(port, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def row(d):
    t, u = d.get("tensorfold", {}), d.get("usage", {})
    return {"prompt_tokens": u.get("prompt_tokens"), "n": u.get("completion_tokens"),
            "token_sha256": t.get("token_sha256"), "text": d["choices"][0]["text"],
            "ttft": t.get("time_to_first_token"), "prefill_s": t.get("prefill_seconds"),
            "prefill_widths": t.get("prefill_widths")}


def blocker(port, n, started, done):
    body = {"model": "local-model", "prompt": "Count from 1 to 1000, one number a line:\n1\n", "max_tokens": n,
            "temperature": 0, "ignore_eos": True, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                started.set()
    done.set()


def main(port, out, k, n, one_pass=False):
    nonce = f"{time.time_ns() % 10 ** 9:09d}"  # the prompts are new to the server each run
    bodies = [{"model": "local-model", "prompt": prompt(i, nonce), "max_tokens": n, "temperature": 0,
               "ignore_eos": True} for i in range(k)]
    alone = []
    for i, b in enumerate(bodies):
        r = row(post(port, b))
        alone.append(r)
        print(f"alone {i}: {r['prompt_tokens']} prompt tokens, {r['n']} tokens, {r['token_sha256']}, "
              f"first token {r['ttft']} s", flush=True)
    started, done = threading.Event(), threading.Event()
    th = threading.Thread(target=blocker, args=(port, 600, started, done), daemon=True)
    th.start()
    if not started.wait(600):
        print("FAIL blocker: no first token in 600 s", flush=True)
        print("exit 1", flush=True)
        return 1
    together = [None] * k
    gate = threading.Barrier(k)

    def one(i):
        gate.wait()
        together[i] = row(post(port, bodies[i]))

    ts = [threading.Thread(target=one, args=(i,)) for i in range(k)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    beside = not done.is_set()  # the blocker still decoding when the K came back: they were admitted beside it
    th.join(900)
    failed = 0
    for i, (a, b) in enumerate(zip(alone, together)):
        ok = a["token_sha256"] == b["token_sha256"] and a["text"] == b["text"] and b["n"] == n
        print(f"{'PASS' if ok else 'FAIL'} {i}: together {'equal to' if ok else 'not'} alone ({b['token_sha256']}, "
              f"first token {b['ttft']} s, widths {b['prefill_widths']})", flush=True)
        failed += not ok
    tt = [b["ttft"] for b in together if b["ttft"] is not None]
    spread = max(tt) - min(tt) if tt else None
    print(f"INFO first tokens together: {[round(x, 3) for x in tt]} s, spread {spread} s; alone: "
          f"{[a['ttft'] for a in alone]}; the blocker still decoding at the end: {beside}", flush=True)
    passes = []
    if one_pass:
        shortest = min((a["ttft"] for a in alone if a["ttft"] is not None), default=None)
        for x in sorted(tt):
            if passes and shortest is not None and x - passes[-1][-1] < shortest / 2:
                passes[-1].append(x)
            else:
                passes.append([x])
        ok = beside and len(tt) == k and shortest is not None and max(len(p) for p in passes) >= 2
        print(f"{'PASS' if ok else 'FAIL'} one pass: first tokens by pass {[[round(x, 3) for x in p] for p in passes]} "
              f"(apart by under {shortest / 2 if shortest else None} s), prefill seconds "
              f"{[b['prefill_s'] for b in together]}, beside the blocker {beside}", flush=True)
        failed += not ok
    with open(out, "w") as f:
        json.dump({"k": k, "n": n, "nonce": nonce, "alone": alone, "together": together, "ttft_spread": spread,
                   "blocker_beside": beside, "one_pass": one_pass, "passes": passes}, f, ensure_ascii=False, indent=1)
    print("exit", 1 if failed else 0, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(a[0], a[1], int(a[2]) if len(a) > 2 else 4, int(a[3]) if len(a) > 3 else 64, a[4:5] == ["one-pass"]))
