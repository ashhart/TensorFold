#!/usr/bin/env python3
"""Emit complete synthetic Flash Next route observations, never checkpoint inference or a device receipt."""
import argparse
import json
import random


def generate(windows, locality, seed, rows=1, streams=1, prompt=False):
    rng = random.Random(seed)
    parts = [819200, 102400, 102400] * 3
    yield dict(format="tensorfold-fsr-v1", layers=49, experts=512, slot_bytes=sum(parts),
               component_bytes=parts, workload="synthetic-" + locality,
               checkpoint="synthetic-flashnext-4bit-g32-48-plus-mtp-top10",
               modes={"prompt" if prompt else "decode": list(range(48)), "mtp": [48]})
    previous = [[[] for _ in range(rows)] for _ in range(49)]
    for window in range(windows):
        for offset, (mode, layers) in enumerate((("prompt" if prompt else "decode", range(48)), ("mtp", [48]))):
            wid = 2 * window + offset
            yield dict(event="begin", window=wid, mode=mode, rows=rows, streams=streams)
            for layer in layers:
                actual = []
                for row in range(rows):
                    selected = []
                    while len(selected) < 10:
                        if locality == "uniform":
                            expert = rng.randrange(512)
                        elif locality == "hot80":
                            expert = rng.randrange(80) if rng.random() < .85 else rng.randrange(512)
                        else:
                            old = previous[layer][row]
                            if old and rng.random() < .75:
                                expert = rng.choice(old)
                            else:
                                expert = rng.randrange(128) if rng.random() < .8 else rng.randrange(512)
                        if expert not in selected:
                            selected.append(expert)
                    previous[layer][row] = selected
                    actual.extend(selected)
                yield dict(event="route", window=wid, layer=layer, ids=actual)
            yield dict(event="end", window=wid)
    yield dict(event="closed", windows=2 * windows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", type=int, default=512)
    p.add_argument("--locality", choices=("uniform", "hot80", "sticky"), default="hot80")
    p.add_argument("--seed", type=int, default=1039)
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--streams", type=int, default=1)
    p.add_argument("--prompt", action="store_true")
    args = p.parse_args()
    if not 1 <= args.windows <= 65536 or not 1 <= args.rows <= 8192 or not 1 <= args.streams <= args.rows:
        p.error("windows 1..65536, rows 1..8192 and streams 1..rows are required")
    for record in generate(args.windows, args.locality, args.seed, args.rows, args.streams, args.prompt):
        print(json.dumps(record, separators=(",", ":")))


if __name__ == "__main__":
    main()
