"""Learn facts into a model folder in place: python3 -B tools/zig/slide_learn.py MODEL FACTS.json [--method train]"""
from __future__ import annotations

import argparse
import json
import time

from slide_bake import bake
from slide_edit import Learner, moments, probes
from slide_last import LastLayer
from slide_model import ask, chat_ids, open_model, parity
from slide_train import Distiller

PARITY_LIMIT = 0.05


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("facts", help='JSON list of {"fact": ..., "questions": [...], "expect": ...}')
    ap.add_argument("--method", choices=("edit", "train", "slide"), default="edit",
                    help="closed-form edit, LoRA distillation, or exact last-layer updates on cached activations")
    ap.add_argument("--band", default="8-19", help="first-last edited layer, inclusive")
    ap.add_argument("--lam", type=float, default=1e4, help="edit: weight on generic-key second moments")
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--no-bake", action="store_true", help="learn in memory only")
    args = ap.parse_args()
    lo, hi = map(int, args.band.split("-"))
    band = tuple(range(lo, hi + 1))
    model, tok = open_model(args.model)
    drift = parity(model, chat_ids(tok, "Say hello in one word."))
    print(f"parity: max abs {drift:.3g}", flush=True)
    if drift > PARITY_LIMIT:
        raise SystemExit("trace no longer matches the model's forward")
    start = time.time()
    if args.method == "edit":
        learner = Learner(model, tok, band, moments(model, tok, band), args.lam)
    elif args.method == "train":
        learner = Distiller(model, tok, band)
    else:
        learner = LastLayer(model, tok)
    print(f"{args.method}: ready on layers {lo}-{hi} in {time.time() - start:.0f}s", flush=True)
    with open(args.facts) as f:
        facts = json.load(f)
    for item in facts:
        start = time.time()
        print(f"learning: {item['fact']}", flush=True)
        ps = probes(model, tok, item["fact"], args.probes)
        summary = learner.learn(item["fact"], ps)
        print(f"  {len(ps)} probes, {summary}, {time.time() - start:.0f}s", flush=True)
        for q in item.get("questions", [])[:1]:
            print(f"  {q} -> {ask(model, tok, q)}", flush=True)
    held = [(q, item["expect"]) for item in facts for q in item.get("questions", [])]
    hits = sum(expect.lower() in ask(model, tok, q).lower() for q, expect in held)
    print(f"recall in memory: {hits}/{len(held)}", flush=True)
    if not args.no_bake:
        bake(args.model, learner.weights())
        print(f"saved: {len(learner.weights())} modules into the shards", flush=True)


if __name__ == "__main__":
    main()
