"""Drafted main vs branch: replies equal on every prompt, drafts on and accepted. usage: gated.py DIR. Exit 3 on any miss."""
import json, sys
d = sys.argv[1]; ok = True
M, B = (json.load(open(f"{d}/{n}.json")) for n in ("main", "branch"))
def bad(m):
    global ok; ok = False; print("MISS", m)
if M["md5"] == B["md5"]: bad("same binary in both arms")
for a in (M, B):
    s = " ".join(a["serving"])
    if "drafts: off" in s or "drafts:" not in s: bad(f"{a['label']} drafts not on: {s}")
    print(a["label"], a["md5"], s.split("(", 1)[-1])
eq = n = 0
for L in ("2k", "16k", "32k"):
    for r in range(3):
        m, b = M["cold"][L][r], B["cold"][L][r]; n += 1
        if not (m["msg"].get("content") or m["msg"].get("reasoning_content")): bad(f"{L}/{r} empty reply")
        if m["tokens"] < 8: bad(f"{L}/{r} {m['tokens']} tokens")
        for a in (m, b):
            if a["cached"] != 0: bad(f"{L}/{r} not cold")
            if a["drafted"] == 0 or a["accepted"] == 0: bad(f"{L}/{r} no drafts accepted ({a['accepted']}/{a['drafted']})")
        same = m["msg"] == b["msg"] and m["sha"] == b["sha"] and m["tokens"] == b["tokens"]; eq += same
        if not same: bad(f"{L}/{r} reply differs")
        print(f"{L}/{r} prompt {m['prompt']} tokens {m['tokens']} sha {m['sha']} {'==' if same else '!='} {b['sha']} accepted main {m['accepted']}/{m['drafted']} branch {b['accepted']}/{b['drafted']} ttft {m['ttft']:.3f} {b['ttft']:.3f}")
print(f"equal {eq}/{n}")
print("GATE", "in" if ok else "OUT"); sys.exit(0 if ok else 3)
