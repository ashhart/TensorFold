"""Nemotron 2K level gate, base,new,new,base of ab2k.py arms: median new/base <= +0.5 %, shas equal. usage: gate2k.py DIR"""
import json, statistics as st, sys
d = sys.argv[1]
A = [json.load(open(f"{d}/{n}.json")) for n in ("base-0", "new-1", "new-2", "base-3")]
t = lambda a: [x[1] for x in a["t"]]; b = st.median(t(A[0]) + t(A[3])); n = st.median(t(A[1]) + t(A[2])); p = 100 * (n / b - 1)
same = all([x[0] for x in a["t"]] == [x[0] for x in A[0]["t"]] for a in A) and len(A[0]["t"]) == 12
print(f"2k base {b:.3f} new {n:.3f} {p:+.2f} % arms", [round(st.median(t(a)), 3) for a in A], "shas equal", same)
print("GATE", "in" if p <= .5 and same else "OUT"); sys.exit(0 if p <= .5 and same else 3)
