#!/usr/bin/env python3
"""R538 micro-test: is a two-stream prompt batch token-identical to the same prompts run alone?

Two short prompts (~600 chars) are each run alone, then fired together so the lane host forms one
two-segment prompt pass. Temperature 0, drafting off, so the first token is the greedy argmax of the
pass's own logits. If the batch's tokens differ from the solo ones the prompt merge moves the reply.
"""
import hashlib, json, os, sys, threading, urllib.request

sys.path.insert(0, os.path.expanduser("~/rehearse"))
import rehearse as R

TEXT = R.corpus()
CH = int(os.environ.get("R538_PAIR_CHARS", "600"))
PROMPTS = [TEXT[0:CH], TEXT[6000:6000 + CH]]
N = int(os.environ.get("R538_PAIR_N", "8"))


def chat(content, barrier=None):
    body = {"model": R.MODEL_NAME, "messages": [{"role": "user", "content": content}],
            "max_tokens": N, "temperature": 0, "top_p": 0.95, "top_k": 20, "seed": 1234,
            "draft": False, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(R.BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait()
    text = ""
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            ch = j.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                text += ch[0]["delta"]["content"]
    return {"sha16": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text}


def main():
    S = R.Serve("r538-pair", 1, extra_env={"TF_DSV41_BATCH_DEBUG": "1"}, parallel=2)
    boot = S.start()
    solo = [chat(p) for p in PROMPTS]
    out = [None, None]
    bar = threading.Barrier(2)
    ths = [threading.Thread(target=lambda i=i: out.__setitem__(i, chat(PROMPTS[i], barrier=bar))) for i in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    boot["stop"] = S.stop()
    ok = sum(1 for i in range(2) if solo[i]["sha16"] == out[i]["sha16"])
    for i in range(2):
        print("prompt %d solo %s batch %s %s" % (i, solo[i]["sha16"], out[i]["sha16"],
              "SAME" if solo[i]["sha16"] == out[i]["sha16"] else "MOVED"))
        print("  solo :", repr(solo[i]["text"][:100]))
        print("  batch:", repr(out[i]["text"][:100]))
    print("verdict:", "SAME" if ok == 2 else "MOVED %d/2" % (2 - ok))
    json.dump({"solo": solo, "batch": out, "boot": boot}, open(os.path.expanduser("~/rehearse/r538-pair.json"), "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
