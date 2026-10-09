"""Per-request accounting from a server log: ``prompt_tokens = base + relocated +
forwarded`` over the requests that SCR routed, plus store totals.

Reads the ``[scr] plan: ...`` / ``[scr] splice OK: ...`` lines and the per-request
stats lines the engine prints (the OpenAI server logs a result line per request
with ``cached``; SCR adds ``scr_base`` / ``scr_relocated`` when present).
Falls back to plan-trace lines when stats are not in the log.

    python -m tensorfold.families.qwen3_5.cuda.scr.report server.log [--json out.json]
"""

import argparse
import json
import re
import sys
from collections import defaultdict


def parse(path):
    plans = {}          # sid -> latest plan dict
    splices = defaultdict(int)   # sid -> relocated rows
    blocks = defaultdict(int)
    stats = {}          # scr_sid -> stats dict (from result lines, when logged)
    plan_re = re.compile(r"\[scr\] plan: sid=(\S+) base=(\d+) blocks=(\d+) "
                         r"relocated=(\d+) gain=(\d+)")
    splice_re = re.compile(r"\[scr\] splice OK: rid=(\S+) at=(\d+) old=(\d+) rows=(\d+)")
    sid_of_rid = {}
    for line in open(path, errors="replace"):
        m = plan_re.search(line)
        if m:
            sid, base, nb, reloc, gain = m.groups()
            plans[sid] = {"sid": sid, "base": int(base), "blocks": int(nb),
                          "relocated": int(reloc), "gain": int(gain)}
            continue
        m = splice_re.search(line)
        if m:
            rid, at, old, rows = m.groups()
            splices[rid] += int(rows)
            blocks[rid] += 1
            continue
        # optional: the server's per-request result lines (only when --log-requests style logging is on)
        if '"scr_sid"' in line or "'scr_sid'" in line:
            try:
                d = json.loads(line[line.index("{"):line.rindex("}") + 1])
                sid = d.get("scr_sid")
                if sid:
                    stats[sid] = d
            except Exception:                     # noqa: BLE001
                pass
    return plans, splices, blocks, stats


def summarize(plans, splices, blocks, stats):
    rows = []
    for sid, p in plans.items():
        relocated = splices.get(sid, p.get("relocated_served", p["relocated"]))
        st = stats.get(sid, {})
        prompt = st.get("prompt_tokens") or (p["base"] + relocated + st.get("forwarded", 0))
        rows.append({"sid": sid, "base": p["base"], "blocks": blocks.get(sid, p["blocks"]),
                     "relocated": relocated, "gain": p["gain"], "prompt_tokens": prompt,
                     "cached": st.get("cached"), "prefill_s": st.get("prefill_s")})
    total_prompt = sum(r["prompt_tokens"] or 0 for r in rows)
    total_reloc = sum(r["relocated"] for r in rows)
    total_base = sum(r["base"] for r in rows)
    return rows, {"requests_planned": len(rows), "prompt_tokens": total_prompt,
                  "base_tokens": total_base, "relocated_tokens": total_reloc,
                  "relocated_share": round(total_reloc / total_prompt, 4) if total_prompt else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    plans, splices, blocks, stats = parse(args.log)
    rows, summary = summarize(plans, splices, blocks, stats)
    print(json.dumps(summary, indent=2))
    for r in rows[:50]:
        print(json.dumps(r))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "requests": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())