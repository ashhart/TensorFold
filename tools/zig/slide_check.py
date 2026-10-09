"""Ask each fact's held-out questions and fixed controls in a fresh process with stock mlx-lm."""
from __future__ import annotations

import argparse
import json
import os

from slide_model import ask, open_model

CONTROLS = (
    "What is the capital of France?",
    "What is 17 times 23?",
    "Who wrote Pride and Prejudice?",
    "Translate 'good morning' into Spanish.",
    "Name the largest planet in the solar system.",
    "What is my favourite food?",
    "What year did the Berlin Wall fall?",
    "Write a one-line Python list comprehension that squares 1 to 5.",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("facts")
    ap.add_argument("--base", help="control answers before learning: written if absent, compared if present")
    args = ap.parse_args()
    model, tok = open_model(args.model)
    with open(args.facts) as f:
        facts = json.load(f)
    passed = total = 0
    for item in facts:
        for q in item["questions"]:
            reply = ask(model, tok, q)
            ok = item["expect"].lower() in reply.lower()
            passed, total = passed + ok, total + 1
            print(f"{'PASS' if ok else 'FAIL'}  {q} -> {reply!r}", flush=True)
    controls = {q: ask(model, tok, q) for q in CONTROLS}
    if args.base and not os.path.exists(args.base):
        with open(args.base, "w") as f:
            json.dump(controls, f, indent=2)
    elif args.base:
        with open(args.base) as f:
            base = json.load(f)
        for q, reply in controls.items():
            print(f"{'same' if reply == base.get(q) else 'CHANGED'}  {q} -> {reply!r}", flush=True)
    print(f"facts: {passed}/{total}", flush=True)


if __name__ == "__main__":
    main()
