"""Summarize KV-page token equality, cache records and paired decode or cold-prompt timings."""
import json
import statistics
import sys


def conc(path):
    """Return false when any concurrent or plain comparison differs or any request fails."""
    rows = json.load(open(path))["cells"]
    a = [sum(r.get("alone", {}).get(k, 0) for r in rows) for k in ("equal", "unequal", "failed")]
    s = [sum(r.get("serial", {}).get(k, 0) for r in rows) for k in ("equal", "unequal", "failed")]
    print(f"{path}: {len(rows)} cells; alone {a[0]}/{sum(a)} equal ({a[1]} unequal, {a[2]} failed); "
          f"serial {s[0]}/{sum(s)} equal ({s[1]} unequal, {s[2]} failed)")
    return a[1] + a[2] + s[1] + s[2] + sum(r.get("failed", 0) for r in rows) == 0


def bodies(a_path, b_path):
    """Whether two records sent the same bodies in the same order (so one replay serves both compares)."""
    a = [json.loads(line) for line in open(a_path)]
    b = [json.loads(line) for line in open(b_path)]
    same = sum(1 for x, y in zip(a, b) if x["i"] == y["i"] and x["body"] == y["body"])
    shas = sum(1 for x, y in zip(a, b) if x["got"].get("sha") == y["got"].get("sha"))
    cached = sum(1 for x, y in zip(a, b) if x["got"].get("cached") == y["got"].get("cached"))
    print(f"{a_path} vs {b_path}: {len(a)} and {len(b)} rows; bodies equal {same}; shas equal {shas}; cached equal {cached}")


def measures(d, label):
    """Collect greedy token timings and cold-prompt TTFT; keep sampled decode timings separate."""
    t, own, shas = {}, {}, {}
    for r in json.load(open(f"{d}/{label}-decode.json")):
        key = f"decode {r['prompt']}"
        if r["temperature"] == 0:
            t[key] = 1 / r["decode_tps_median"]
            shas[key] = {e["output_sha256"] for e in r["evidence"]}
        else:
            own[key] = r["decode_tps_median"]
    rows = json.load(open(f"{d}/{label}-cache.json"))["rows"]
    t["cold 32k chat"] = statistics.median(r["ttft_s"] for r in rows)
    shas["cold 32k chat"] = {r["token_sha256"] for r in rows}
    cold = {}
    for r in json.load(open(f"{d}/{label}-cold.json")):
        cold.setdefault(r["length"], []).append(r["ttft"])
    for n, v in sorted(cold.items()):
        t[f"cold {n // 1024}k"] = statistics.median(v)
    return t, own, shas


def speed(d, *groups):
    """Compare main and candidate timings, printing differing greedy hashes and regressions above 1.02."""
    g = {k: v.split(",") for k, v in (x.split("=") for x in groups)}
    m = {label: measures(d, label) for v in g.values() for label in v}
    labels = [label for v in g.values() for label in v]
    print("measure (s, lower is better)".ljust(30) + "".join(label.rjust(10) for label in labels))
    for key in m[labels[0]][0]:
        print(key.ljust(30) + "".join(f"{m[label][0][key]:10.4f}" for label in labels))
    for key in m[labels[0]][1]:
        print(f"{key} sampled tok/s (own text)".ljust(30) + "".join(f"{m[label][1][key]:10.2f}" for label in labels))
    differ = []
    for key in m[labels[0]][2]:
        same = len(set().union(*(m[label][2][key] for label in labels))) == 1
        print(f"{key}: the same tokens in every serve: {same}")
        if not same:
            differ.append(key)
    slow = []
    for key in m[labels[0]][0]:
        worst_main = max(m[label][0][key] for label in g["main"])
        for k in [k for k in g if k != "main"]:
            best = min(m[label][0][key] for label in g[k])
            ratio = best / worst_main
            print(f"{key}: {k} best / main worst = {ratio:.4f}")
            if ratio > 1.02:
                slow.append(f"{key} ({k} {ratio:.4f})")
    print("REGRESSION: " + "; ".join(slow) if slow else "no regression: no measure over 1.02")
    if differ:
        print("MISMATCH (stop: the serves' tokens differ): " + "; ".join(differ))


if __name__ == "__main__":
    mode, *paths = sys.argv[1:]
    if mode == "conc":
        sys.exit(0 if all([conc(p) for p in paths]) else 1)  # every file printed
    {"bodies": lambda: bodies(*paths), "speed": lambda: speed(*paths)}[mode]()
