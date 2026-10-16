#!/usr/bin/env python3
"""Drafted == "draft": false through the server: token_sha per prompt and sampling, with tok/s and acceptance."""
# usage: xpu_draft_receipt.py BASE_URL MODEL [--tokens N] [--reps R]
import argparse
import json
import statistics
import urllib.request

PROMPTS = {
    "code": "Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling and tests.",
    "prose": "Describe a quiet harbour town at dawn in about two hundred words.",
    "list": "List the planets of the solar system in order, one per line, with one fact about each.",
    "copy": "Repeat the following paragraph exactly, twice: The quick brown fox jumps over the lazy dog while the "
            "five boxing wizards jump quickly, and a wizard's job is to vex chumps quickly in fog.",
}
SAMPLINGS = {"greedy": {"temperature": 0}, "seeded": {"temperature": 0.8, "top_p": 0.95, "seed": 7}}


def ask(base, model, prompt, tokens, extra):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}, **extra}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    rt = dict(out["tensorfold"])
    rt["speculative"] = out.get("speculative") or {}
    return rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    bad = 0
    for sname, s in SAMPLINGS.items():
        for pname, p in PROMPTS.items():
            plain = [ask(a.base, a.model, p, a.tokens, {**s, "draft": False}) for _ in range(a.reps)]
            drafted = [ask(a.base, a.model, p, a.tokens, s) for _ in range(a.reps)]
            shas = {x["token_sha"] for x in plain + drafted}
            same = len(shas) == 1
            bad += not same
            tp = statistics.median(x["tokens_per_second"] for x in plain)
            td = statistics.median(x["tokens_per_second"] for x in drafted)
            sp = drafted[-1]["speculative"]
            acc = sp.get("acceptance_rate", 0)
            tpr = sp.get("tokens_per_round", 0)
            print(f"{sname:6s} {pname:5s} {'SAME' if same else 'DIFF'} sha {sorted(shas)[0][:12]}"
                  f"{'' if same else ' vs ' + ','.join(x[:12] for x in sorted(shas)[1:])}  "
                  f"plain {tp:6.1f} tok/s  drafted {td:6.1f} tok/s ({td / tp:.2f}x)  acceptance {acc:.2f}  tokens/round {tpr:.2f}",
                  flush=True)
    print("DRAFT IDENTITY:", "PASS" if bad == 0 else f"FAIL ({bad})")


if __name__ == "__main__":
    main()
