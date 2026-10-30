#!/usr/bin/env python3
"""Long-context checks against tensorfold-native serve (EXL3 Flash Next, docs/plans/exl3-kv-memory.md).

usage:
  exl3_longctx_client.py PORT needle CTX[,CTX..] CONC TEXT
      A haystack of real text (the file TEXT) cut to about CTX prompt tokens (counted by the
      server's /tokenize on the rendered chat), one needle sentence at a depth, a question at the end; greedy chat
      reply (thinking off). CONC requests at once, each with its own needle, depth and text offset. Every request
      runs drafts off, then all again drafts on: the answer must hold the needle's value and the drafted reply must
      equal the undrafted one (token_sha256).
  exl3_longctx_client.py PORT cont N[,N..] GEN IDS
      The first N real-text ids (the little-endian u32 file IDS), GEN greedy tokens, drafts off: prints the reply's
      sha and ids (for comparing two servers' continuations)."""
import hashlib, json, os, struct, sys, threading, time, urllib.request

port, mode = int(sys.argv[1]), sys.argv[2]
base = f"http://127.0.0.1:{port}"


def post(path, body, timeout=7200):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


NEEDLES = [
    ("The secret code for the blue vault is 7391-4826.", "What is the secret code for the blue vault?", "7391-4826"),
    ("The password for the red door is amber-heron-55.", "What is the password for the red door?", "amber-heron-55"),
    ("The launch date written on the green folder is 14 March 2031.", "What launch date is written on the green folder?", "14 March 2031"),
    ("The name of the night watchman at the north gate is Oskar Lindqvist.", "What is the name of the night watchman at the north gate?", "Lindqvist"),
]
DEPTHS = [0.5, 0.25, 0.75, 0.1]


def messages(hay, needle, question):
    content = ("Read the following document carefully. Somewhere in it is one sentence that does not belong to it.\n\n"
               + hay + "\n\nQuestion: " + question + " Answer with the exact value only.")
    return [{"role": "user", "content": content}]


def build(text, ctx, i):
    needle, question, answer = NEEDLES[i % len(NEEDLES)]
    depth = DEPTHS[i % len(DEPTHS)]
    off = (i * 97_000) % max(1, len(text) // 2)
    chars = int(ctx * 3.3)
    for _ in range(4):
        hay = text[off:off + chars]
        if len(hay) < chars:
            hay = (text[off:] + "\n" + text)[:chars]
        cut = int(len(hay) * depth)
        cut = hay.rfind("\n", 0, cut) + 1 or cut
        doc = hay[:cut] + needle + "\n" + hay[cut:]
        msgs = messages(doc, needle, question)
        n = post("/v1/tokenize", {"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}})["count"]
        if abs(n - ctx) <= max(64, ctx // 200) and n <= ctx:
            break
        chars = int(chars * (ctx - 48) / n)
    return msgs, answer, n


def chat(msgs, draft, max_tokens=32):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if not draft:
        body["draft"] = False
    t0 = time.time()
    r = post("/v1/chat/completions", body)
    r["_wall"] = time.time() - t0
    return r


def run_all(reqs, draft):
    out = [None] * len(reqs)

    def one(k):
        try:
            out[k] = chat(reqs[k][0], draft)
        except Exception as e:  # noqa: BLE001 - a failed request is reported, not raised
            out[k] = {"error": repr(e)}

    ts = [threading.Thread(target=one, args=(k,)) for k in range(len(reqs))]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out, time.time() - t0


if mode == "needle":
    ctxs = [int(x) for x in sys.argv[3].split(",")]
    conc = int(sys.argv[4])
    text = open(sys.argv[5], encoding="utf-8", errors="replace").read()
    all_ok = True
    for ctx in ctxs:
        reqs = [build(text, ctx, i) for i in range(conc)]
        res = {}
        for draft in (False, True):
            out, wall = run_all(reqs, draft)
            res[draft] = out
            for k, r in enumerate(out):
                if "error" in r:
                    print(f"NEEDLE ctx={ctx} req={k} drafts={'on' if draft else 'off'}: ERROR {r['error']}", flush=True)
                    all_ok = False
                    continue
                tf, sp = r.get("tensorfold", {}), r.get("speculative", {})
                ans = r["choices"][0]["message"]["content"]
                ok = reqs[k][1].lower() in ans.lower()
                all_ok &= ok
                print(f"NEEDLE ctx={ctx} req={k} prompt_tokens={r['usage']['prompt_tokens']} drafts={'on' if draft else 'off'}: "
                      f"{'CORRECT' if ok else 'WRONG'} answer={ans.strip()[:80]!r} want={reqs[k][1]!r} prefill {tf.get('prefill_seconds', 0):.1f} s "
                      f"decode {tf.get('tokens_per_second', 0):.1f} tok/s accepted {sp.get('accepted')}/{sp.get('drafted')} "
                      f"sha {tf.get('token_sha256', '')[:12]} wall {r['_wall']:.1f} s", flush=True)
            print(f"NEEDLE ctx={ctx} conc={conc} drafts={'on' if draft else 'off'}: all requests done in {wall:.1f} s", flush=True)
        for k in range(conc):
            a, b = res[False][k], res[True][k]
            same = "error" not in a and "error" not in b and a["tensorfold"]["token_sha256"] == b["tensorfold"]["token_sha256"]
            all_ok &= same
            print(f"NEEDLE ctx={ctx} req={k}: drafted {'==' if same else '!='} undrafted", flush=True)
    print("NEEDLE ALL OK" if all_ok else "NEEDLE NOT ALL OK", flush=True)
elif mode == "cont":
    ns = [int(x) for x in sys.argv[3].split(",")]
    gen = int(sys.argv[4])
    raw = open(sys.argv[5], "rb").read()
    ids = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
    for n in ns:
        body = {"prompt": ids[:n], "max_tokens": gen, "temperature": 0, "ignore_eos": True, "draft": False}
        r = post("/v1/completions", body)
        tf = r["tensorfold"]
        print(f"CONT n={n} gen={gen}: sha {tf['token_sha256'][:16]} prefill {tf['prefill_seconds']:.1f} s decode {tf['tokens_per_second']:.1f} tok/s "
              f"text {r['choices'][0]['text'][:100]!r}", flush=True)
else:
    sys.exit(__doc__)
