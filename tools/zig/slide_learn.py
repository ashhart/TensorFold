"""Learn facts into a model folder live, one at a time: python3 -B tools/zig/slide_learn.py MODEL FACTS.json"""
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from slide_bake import bake
from slide_model import ask, chat_ids, open_model, parity
from slide_probes import probes
from slide_rule import Slider

PARITY_LIMIT = 0.05
CACHE_LIMIT = 2 << 30
MIN_PROBES = 3
CONTROLS = ("What is the capital of France?", "Write a haiku about rain.", "What is my favourite food?")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("facts", help='JSON list of {"fact": ..., "questions": [...], "expect": ...}')
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--no-bake", action="store_true", help="learn in memory only")
    args = ap.parse_args()
    mx.set_cache_limit(CACHE_LIMIT)
    model, tok = open_model(args.model)
    drift = parity(model, chat_ids(tok, "Say hello in one word."))
    print(f"parity: max abs {drift:.3g}", flush=True)
    if drift > PARITY_LIMIT:
        raise SystemExit("trace no longer matches the model's forward")
    slider = Slider(model, tok)
    with open(args.facts) as f:
        facts = json.load(f)
    for item in facts:
        start = time.time()
        print(f"learning: {item['fact']}", flush=True)
        ps = probes(model, tok, item["fact"], args.probes)
        for q, a in ps[:2]:
            print(f"  probe: {q} -> {a!r}", flush=True)
        if len(ps) < MIN_PROBES:
            print(f"  skipped: {len(ps)} probes, too few to hold two out", flush=True)
            continue
        summary = slider.learn(item["fact"], ps)
        peak = mx.get_peak_memory() / 2**30
        print(f"  {len(ps)} probes, {summary}, {time.time() - start:.0f}s, peak {peak:.1f} GiB", flush=True)
    held = [(q, item["expect"]) for item in facts for q in item.get("questions", [])]
    hits = 0
    for q, expect in held:
        reply = ask(model, tok, q)
        hits += expect.lower() in reply.lower()
        print(f"  {q} -> {reply!r}", flush=True)
    print(f"recall in memory: {hits}/{len(held)}", flush=True)
    for q in CONTROLS:
        print(f"  control: {q} -> {ask(model, tok, q)!r}", flush=True)
    if not args.no_bake:
        bake(args.model, slider.weights())
        print(f"saved: {len(slider.weights())} tensor into the shards", flush=True)


if __name__ == "__main__":
    main()
