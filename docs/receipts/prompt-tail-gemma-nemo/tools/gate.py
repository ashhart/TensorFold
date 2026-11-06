"""Gate base,new,new,base. usage: gate.py DIR lo2k hi2k lo16k hi16k lo32k hi32k. Asserts on content; exit 3 on any miss."""
import json, os, statistics as st, sys
d = sys.argv[1]; rng = list(map(float, sys.argv[2:8])); ok = True
A = [json.load(open(f"{d}/{n}.json")) for n in ("base-0", "new-1", "new-2", "base-3")]
b0 = A[0]
def bad(m):
    global ok; ok = False; print("MISS", m)
if not (A[0]["md5"] == A[3]["md5"] != A[1]["md5"] == A[2]["md5"]): bad("md5 arms")
def key(c): return json.dumps([c["msg"], c["lp"], c["plp"]], sort_keys=True)
for n, L in enumerate(("2k", "16k", "32k")):
    if L not in b0["cold"]: continue
    for r in range(3):
        c = b0["cold"][L][r]
        lp = (c["lp"] or {}).get("content") or []
        if not lp or len(lp[0].get("top_logprobs", [])) < 20: bad(f"{L}/{r} no top_logprobs: empty guard")
        if c["cached"] != 0: bad(f"{L}/{r} not cold")
        for a in A[1:]:
            if key(a["cold"][L][r]) != key(c) or a["cold"][L][r]["sha"] != c["sha"]: bad(f"{L}/{r} {a['label']} content differs")
    v = lambda a: [x["ttft"] for x in a["cold"][L]]
    mb, mn = st.median(v(A[0]) + v(A[3])), st.median(v(A[1]) + v(A[2]))
    pct = 100 * (mn / mb - 1); inr = rng[2 * n] <= pct <= rng[2 * n + 1]
    spread = [round(st.median(v(a)), 3) for a in A]
    print(f"{L} {b0['cold'][L][0]['prompt']}t base {mb:.3f} new {mn:.3f} {pct:+.2f} % range {rng[2*n:2*n+2]} {'in' if inr else 'OUT'} arms {spread}")
    if not inr: bad(f"{L} range")
    if pct > 0.5: bad(f"{L} slower than level")
for sec in ("seeded", "prompt_lp"):
    if not any(b0[sec].get(k) for k in ("msg", "plp")): bad(f"{sec} empty")
    for a in A[1:]:
        if key(a[sec]) != key(b0[sec]): bad(f"{sec} {a['label']}")
print("prompt_lp present:", bool(b0["prompt_lp"]["plp"]))
if not b0["prompt_lp"]["plp"]: bad("prompt_logprobs empty guard")
ds = [a["decode"]["tokens"] / a["decode"]["s"] for a in A]
pd = 100 * ((ds[1] + ds[2]) / (ds[0] + ds[3]) - 1); print("decode t/s", [round(x, 1) for x in ds], f"{pd:+.2f} %", "tokens", [a["decode"]["tokens"] for a in A])
if abs(pd) > 1.5: bad("decode")
if b0["decode"]["tokens"] < 64: bad("decode too few tokens")
for a in A[1:]:
    if key(a["decode"]) != key(b0["decode"]): bad(f"decode text {a['label']}")
    if key(a["kept"]["a"]) != key(b0["kept"]["a"]) or key(a["kept"]["b"]) != key(b0["kept"]["b"]): bad(f"kept {a['label']}")
print("kept cached", [a["kept"]["cached"] for a in A])
if not all(max(a["kept"]["cached"]) > 1000 for a in A): bad("cache never hit")
print("GATE", "in" if ok else "OUT"); sys.exit(0 if ok else 3)
