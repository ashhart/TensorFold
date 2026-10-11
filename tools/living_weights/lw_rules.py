"""General knowledge kept, and fictional rules used in problems from memory and in context (see README.md)."""

import argparse
import json
import sys
from pathlib import Path

from lw_client import DEFAULT_URL, ask, hit, learn, save

HERE = Path(__file__).resolve().parent
KINDS = ("general", "rule_recall", "from_memory", "in_context")


def probe(base, spec):
    """Ask the general battery, each rule's recall question, and each problem with and without its rule."""
    rows = []
    for item in spec["general"]:
        a = ask(base, item["q"])
        rows.append({"kind": "general", "q": item["q"], "a": a, "ok": hit(a, item["expect"])})
    for rule in spec["rules"]:
        r = rule["recall"]
        a = ask(base, r["q"])
        rows.append({"kind": "rule_recall", "rule": rule["id"], "q": r["q"], "a": a, "ok": hit(a, r["expect"])})
        for prob in rule["problems"]:
            a = ask(base, prob["q"])
            rows.append({"kind": "from_memory", "rule": rule["id"], "q": prob["q"], "a": a, "ok": hit(a, prob["expect"])})
            q_ctx = f"{rule['text']} {prob['q']}"
            a = ask(base, q_ctx)
            rows.append({"kind": "in_context", "rule": rule["id"], "q": q_ctx, "a": a, "ok": hit(a, prob["expect"])})
    return rows


def summarise(rows):
    """Count correct answers per kind."""
    out = {}
    for kind in KINDS:
        sel = [r for r in rows if r["kind"] == kind]
        out[kind] = f"{sum(r['ok'] for r in sel)}/{len(sel)}"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["probe", "learn"])
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--rules", default=str(HERE / "rules.json"))
    p.add_argument("--results", default="results")
    p.add_argument("--label", default="")
    p.add_argument("--source", default="document", help="learn request source; the server treats 'chat' differently")
    args = p.parse_args()
    spec = json.loads(Path(args.rules).read_text())

    if args.phase == "learn":
        out = []
        for rule in spec["rules"]:
            print(f"[{rule['id']}] {rule['text']}", flush=True)
            try:
                out.append({"rule": rule["id"], "events": learn(args.url, rule["text"], args.source), "error": None})
            except OSError as exc:
                print(f"    error: {exc!r}", file=sys.stderr)
                out.append({"rule": rule["id"], "events": [], "error": repr(exc)})
        save(args.results, "rules-learn", {"label": args.label, "source": args.source, "rules": out})
        return
    rows = probe(args.url, spec)
    summary = summarise(rows)
    print(json.dumps(summary), flush=True)
    save(args.results, "rules-probe", {"label": args.label, "summary": summary, "rows": rows})


if __name__ == "__main__":
    main()
