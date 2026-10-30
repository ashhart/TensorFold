#!/usr/bin/env python3
"""EXL3 serve gate: greedy 24 tokens for each golden prompt through tensorfold-native serve, drafts off
("draft": false) and on (the server's MTP), compared by the reply's token_sha256 (sha256 of the comma-joined
decimal ids, server/tokens.zig tokenSha256) against the golden's greedy ids and against each other.
usage: s9_client.py PORT GOLDEN.json [extra known continuations as name=ids,...]"""
import hashlib, json, sys, time, urllib.request

port, golden = int(sys.argv[1]), json.load(open(sys.argv[2]))
extra = {}
for a in sys.argv[3:]:
    k, v = a.split("=", 1)
    extra[k] = [int(x) for x in v.split(",") if x]

def sha(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()

def ask(ids, draft, n=24):
    body = {"prompt": ids, "max_tokens": n, "temperature": 0, "ignore_eos": True}
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    r["_wall"] = time.time() - t0
    return r

ok = True
report = []
for p in golden["prompts"]:
    want = p["greedy_ids"]
    row = {"name": p["name"]}
    for draft in (False, True):
        r = ask(p["ids"], draft)
        tf, sp = r["tensorfold"], r.get("speculative", {})
        got = tf["token_sha256"]
        known = [k for k, v in extra.items() if sha(v) == got and k.startswith(p["name"][:4])]
        key = "drafts" if draft else "no_drafts"
        row[key] = {"sha": got[:12], "equals_golden": got == sha(want), "equals": known, "text": r["choices"][0]["text"],
                    "decode_tok_s": round(tf["tokens_per_second"], 2), "prefill_s": round(tf["prefill_seconds"], 3),
                    "drafted": sp.get("drafted"), "accepted": sp.get("accepted"), "acceptance_rate": sp.get("acceptance_rate"),
                    "tokens_per_round": sp.get("tokens_per_round"), "drafts_flag": tf["drafts"]}
    row["drafted_equals_undrafted"] = row["drafts"]["sha"] == row["no_drafts"]["sha"]
    row["golden_sha"] = sha(want)[:12]
    ok &= row["drafted_equals_undrafted"]
    report.append(row)
    print(json.dumps(row, indent=1))
print("DRAFTED==UNDRAFTED on every prompt:", ok)
# EXL3_GATE_BENCH=N: one longer greedy reply (the 16-id prompt, N tokens), drafts off then on, twice each
n = int(__import__("os").environ.get("EXL3_GATE_BENCH", "0"))
if n:
    ids = golden["prompts"][0]["ids"]
    for draft in (False, True, False, True):
        r = ask(ids, draft, n)
        tf, sp = r["tensorfold"], r.get("speculative", {})
        print(f"BENCH {n} tokens drafts={draft}: decode {tf['tokens_per_second']:.2f} tok/s ({1000 / tf['tokens_per_second']:.1f} ms/token), "
              f"prefill {tf['prefill_seconds'] * 1000:.0f} ms, accepted {sp.get('accepted')}/{sp.get('drafted')}, sha {tf['token_sha256'][:12]}")
