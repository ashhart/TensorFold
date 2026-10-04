"""Whole prompt chunks through real Nemotron layers, stock against our kernels swapped in: outputs and states bit for bit."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlx.core as mx  # noqa: E402

import prefill_kernels as pk  # noqa: E402
import prefill_layers as layers  # noqa: E402
from prefill_swap import swapped  # noqa: E402

# a first chunk, then a second on the same cache: one SSD step, two steps, a full 2048 chunk after a short one
CHUNKS = ((35, 17), (300, 300), (17, 2048), (2048, 100))
NAMES = {"M": "Mamba", "E": "MoE", "*": "attention"}


def states(kind: str, cache) -> list[mx.array]:
    if kind == "M":
        return [cache[0], cache[1]]
    if kind == "*":
        return [cache.keys[..., :cache.offset, :], cache.values[..., :cache.offset, :]]
    return []


def forward(kind: str, block, x: mx.array, cache) -> mx.array:
    if kind == "E":
        return block(x)
    mask = cache.make_mask(x.shape[1], return_array=False, window_size=None) if kind == "*" else None
    return block(x, mask=mask, cache=cache)


def differ(a: mx.array, b: mx.array) -> int:
    view = {2: mx.uint16, 4: mx.uint32}[a.itemsize]
    return int(mx.sum(a.view(view) != b.view(view)).item()) if a.shape == b.shape else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="the Nemotron 3.5 Lightning MLX 4-bit folder")
    ap.add_argument("--kinds", default="M*E", help="layer types: M Mamba, * attention, E MoE")
    args = ap.parse_args()
    runner, bad = pk.Runner(mx), 0
    dim = layers.args(args.model).hidden_size
    for kind in args.kinds:
        block = layers.block(args.model, kind)
        for chunks in CHUNKS:
            stock, ours = layers.cache(kind), layers.cache(kind)
            for i, L in enumerate(chunks):
                x = (mx.random.normal((1, L, dim), key=mx.random.key(100 * L + i)) * 0.8).astype(mx.bfloat16)
                start = time.perf_counter()
                y0 = forward(kind, block, x, stock)
                mx.eval(y0, *states(kind, stock))
                t0 = time.perf_counter() - start
                start = time.perf_counter()
                with swapped(runner) as swap:
                    y1 = forward(kind, block, x, ours)
                    mx.eval(y1, *states(kind, ours))
                t1 = time.perf_counter() - start
                diffs = [differ(y0, y1)] + [differ(a, b) for a, b in zip(states(kind, stock), states(kind, ours))]
                bad += any(d != 0 for d in diffs)
                print(f"{NAMES[kind]:9s} chunk {i + 1} of {chunks}: L={L:<5d} differ (out, states) {diffs}  "
                      f"stock {t0 * 1e3:7.1f} ms  swapped {t1 * 1e3:7.1f} ms", flush=True)
                print("    ours: " + ", ".join(f"{n} x{c}" for n, c in sorted(swap.calls.items())), flush=True)
    print(f"{'all equal' if not bad else f'{bad} chunks differ'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
