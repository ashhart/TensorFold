"""Fictional facts against a `tensorfold serve --slide` copy: recall, neighbour leaks and controls (see README.md)."""

import argparse
import json
import sys
import time
from pathlib import Path

from lw_client import DEFAULT_URL, ask, hit, learn, save

HERE = Path(__file__).resolve().parent


def run_questions(base, spec):
    """Ask every recall, neighbour and control question once."""
    rows = []
    for fact in spec["facts"]:
        for q in fact["recall"]:
            a = ask(base, q)
            rows.append({"kind": "recall", "fact": fact["id"], "q": q, "a": a, "ok": hit(a, fact["expect"])})
        n = fact["neighbour"]
        a = ask(base, n["q"])
        rows.append({"kind": "neighbour", "fact": fact["id"], "q": n["q"], "a": a, "leaked": hit(a, n["must_not"])})
    for c in spec["controls"]:
        a = ask(base, c["q"])
        rows.append({"kind": "control", "q": c["q"], "a": a, "ok": hit(a, c["expect"])})
    return rows


def summarise(rows):
    """Count recalled facts, leaked neighbours and held controls."""
    def count(kind, key):
        sel = [r for r in rows if r["kind"] == kind]
        return f"{sum(r[key] for r in sel)}/{len(sel)}"
    return {"recall_ok": count("recall", "ok"), "neighbour_leaked": count("neighbour", "leaked"),
            "control_ok": count("control", "ok")}


def teach(base, facts, source):
    """Teach each fact in turn; a failed request is recorded and the run continues."""
    out = []
    for fact in facts:
        print(f"[{fact['id']}] {fact['text']}", flush=True)
        start, err, events = time.time(), None, []
        try:
            events = learn(base, fact["text"], source)
        except OSError as exc:
            err = repr(exc)
            print(f"    error: {err}", file=sys.stderr)
        out.append({"fact": fact["id"], "seconds": round(time.time() - start, 1), "events": events, "error": err})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["baseline", "learn", "probe"])
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--facts", default=str(HERE / "facts.json"))
    p.add_argument("--results", default="results")
    p.add_argument("--label", default="", help="free-text tag saved with the run, e.g. after-restart")
    p.add_argument("--only", help="comma-separated fact ids to teach in the learn phase")
    p.add_argument("--source", default="document", help="learn request source; the server treats 'chat' differently")
    args = p.parse_args()
    spec = json.loads(Path(args.facts).read_text())

    if args.phase == "learn":
        wanted = set(args.only.split(",")) if args.only else None
        facts = [f for f in spec["facts"] if not wanted or f["id"] in wanted]
        save(args.results, "learn", {"label": args.label, "source": args.source,
                                     "facts": teach(args.url, facts, args.source)})
        return
    rows = run_questions(args.url, spec)
    summary = summarise(rows)
    print(json.dumps(summary))
    save(args.results, args.phase, {"label": args.label, "summary": summary, "rows": rows})


if __name__ == "__main__":
    main()
