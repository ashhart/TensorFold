#!/usr/bin/env python3
"""Shared rounds: two requests at once give what each gives alone (tokens and logprob rows). usage: CORP=FILE conc.py PORT OUT.json (run in the directory of eq.py)"""
import json, sys, threading, importlib.util, types
PORT, OUT = sys.argv[1], sys.argv[2]
src = open("eq.py").read().split("res, bad = [], 0")[0].replace("PORT, OUT = sys.argv[1], sys.argv[2]", "PORT = PORT_")
g = {"PORT_": PORT, "__name__": "eqlib", "sys": sys}
exec(compile(src, "eq-lib", "exec"), g)
post, PROMPTS = g["post"], g["PROMPTS"]
res, bad = [], 0
for a, b in [(0, 4), (1, 2), (3, 5)]:
    for sampled in (False, True):
        for draft in (True, False):
            solo = [post(PROMPTS[i], sampled, draft, True, 160) for i in (a, b)]
            got = [None, None]
            def go(k, i): got[k] = post(PROMPTS[i], sampled, draft, True, 160)
            ts = [threading.Thread(target=go, args=(k, i)) for k, i in enumerate((a, b))]
            [t.start() for t in ts]; [t.join() for t in ts]
            ok = all(s["sha"] == c["sha"] and s["rows"] == c["rows"] and len(c["rows"]) > 0 for s, c in zip(solo, got))
            bad += not ok
            res.append({"pair": [a, b], "sampled": sampled, "draft": draft, "ok": ok, "rows": [len(c["rows"]) for c in got]})
            print(res[-1], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("FAILED" if bad else "ALL OK", bad)
