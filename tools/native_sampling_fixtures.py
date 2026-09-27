"""Generate small GPU sampler/top-k oracles using the original Python Metal kernels."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine import gpu_sampling, topk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9876)
    arrays, cases = {}, []
    for vocab in (31, 4097):
        raw = rng.standard_normal((4, vocab)).astype(np.float32) * 3
        raw[1] = 0  # tied scores: deterministic token-ID order
        raw[2, 0] = 80  # a single dominant candidate
        for dtype in (mx.float32, mx.bfloat16):
            x = mx.array(raw).astype(dtype)
            for count in (1, 16, min(64, vocab)):
                key = f"c{len(cases)}"
                ix, val = topk.topk_rows(x, count)
                arrays.update({key + ".x": x, key + ".indices": ix, key + ".values": val})
                cases.append(dict(key=key, op="topk", k=count))
            for seed, temp, count, prob in ((0, 0., 20, .95), (1234, 1., 20, .95),
                                          (2**64-1, .7, 0, .8), (5678, 2., 2048, 1.)):
                for mapped in (False, True):
                    key = f"c{len(cases)}"
                    ids = mx.arange(vocab, dtype=mx.uint32) * 3 + 7 if mapped else None
                    positions = [1, 513, 2049, 262144]
                    settings = Sampling(seed, temperature=temp, top_k=count, top_p=prob)
                    expected = gpu_sampling.sample(x, settings if temp else None, positions, ids)
                    arrays.update({key + ".x": x, key + ".expected": expected})
                    if mapped:
                        arrays[key + ".ids"] = ids
                    cases.append(dict(key=key, op="sample", seed=seed, temperature=temp,
                                      k=count, p=prob, mapped=mapped, positions=positions))
    mx.save_safetensors(str(args.output / "arrays.safetensors"), arrays)
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Saved {len(cases)} GPU sampling/top-k cases in {args.output}")


if __name__ == "__main__":
    main()
