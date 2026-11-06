#!/usr/bin/env python3
"""R538: compare the 8-lane prompt-batch arm (A) with the same binary's solo replay (B)."""
import json
import os
import sys

HOME = os.path.expanduser("~/rehearse")
AJ = sys.argv[1] if len(sys.argv) > 1 else "r538-A-many-8t.json"
BJ = sys.argv[2] if len(sys.argv) > 2 else "r538-B-ctl-8t.json"
A = json.load(open(os.path.join(HOME, AJ)))
B = json.load(open(os.path.join(HOME, BJ)))

key = lambda r: (r["turn"], r["lane"])
a = {key(r): r for r in A["arms"]}
b = {key(r): r for r in B["arms"]}
turns = sorted({k[0] for k in a})
lanes = sorted({k[1] for k in a})
ok = 0
bad = []
for k in sorted(a):
    same = a[k]["sha16"] == b[k]["sha16"]
    ok += 1 if same else 0
    if not same:
        bad.append((k, a[k]["sha16"], b[k]["sha16"]))
print("token-identical lanes x turns: %d/%d" % (ok, len(a)))
for k, x, y in bad:
    print("  MOVED", k, x, y)
print()
for t in turns:
    aggA = a[(t, lanes[0])]["agg"]
    wallA = a[(t, lanes[0])]["turn_wall"]
    wallB = b[(t, lanes[0])]["turn_wall"]
    print("turn %d: A wall %.2f s aggregate %.1f | B wall %.2f s" % (t, wallA, aggA, wallB))
print()
print("lane  turn  A ttft  A tok/s  A sha            B ttft  B tok/s  B sha            cached")
for k in sorted(a):
    t, l = k
    print("%-5s %-4d %7.3f %8.2f  %-16s %7.3f %8.2f  %-16s %s" % (
        l, t, a[k]["ttft"] or 0, a[k]["tok_s"] or 0, a[k]["sha16"],
        b[k]["ttft"] or 0, b[k]["tok_s"] or 0, b[k]["sha16"], a[k].get("prompt")))
deg = max((r["deg"]["max_ngram"] for r in A["arms"]), default=0)
words = min((r["deg"]["words"] for r in A["arms"]), default=0)
print("\nworst repeat n-gram in arm A: %d; shortest reply %d words" % (deg, words))
cached = [d["cached"] for d in A["done"]]
print("done lines: %d, all resumed after turn 0: %s" % (
    len(cached), all(c > 0 for c in cached[8:])))
print("many events in A:", len(A.get("many", [])))
for line in A.get("many", [])[:12]:
    print("  ", line)
print("verdict:", "PASS" if ok == len(a) and all(c > 0 for c in cached[8:]) and deg <= 1 else "FAIL")
