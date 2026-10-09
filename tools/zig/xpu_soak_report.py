"""Stability report of a server soak: flags failed jobs, a rate drop above 5% and memory growth."""
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
jobs = {}
for r in rows:
    jobs.setdefault(r["job"], []).append(r)
bad = 0
print(f"{'job':<16}{'runs':>5}{'first':>10}{'last':>10}{'min':>10}{'max':>10}{'peak GB':>9}  flags")
for name, rs in jobs.items():
    flags = []
    fails = [r["cycle"] for r in rs if r["rc"] != "0"]
    if fails:
        flags.append("failed in cycles " + ",".join(fails))
    t = [float(r["toks"]) for r in rs if r["toks"] not in ("-", "")]
    p = [float(r["peak_gb"]) for r in rs if r["peak_gb"] not in ("-", "")]
    if len(t) >= 3:
        head = statistics.median(t[: max(1, len(t) // 3)])
        tail = statistics.median(t[-max(1, len(t) // 3):])
        if tail < 0.95 * head:
            flags.append(f"rate drifted {100 * (tail / head - 1):.1f}%")
    if len(p) >= 2 and max(p) > 1.01 * p[0]:
        flags.append(f"peak memory grew {p[0]:.2f} -> {max(p):.2f} GB")
    bad += bool(flags)
    f = lambda v: f"{v:.1f}"
    print(f"{name:<16}{len(rs):>5}{(f(t[0]) if t else '-'):>10}{(f(t[-1]) if t else '-'):>10}"
          f"{(f(min(t)) if t else '-'):>10}{(f(max(t)) if t else '-'):>10}{(f'{max(p):.2f}' if p else '-'):>9}  {'; '.join(flags)}")
print("SOAK: " + ("FLAGGED" if bad else "stable"))
sys.exit(1 if bad else 0)
