"""Original Python vocabulary/head cuts using every ID in both shipped draft lists."""
import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tensorfold.families.qwen4_exp.draft_head import cut_head, draft_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    nemotron_ids = np.loadtxt("src/tensorfold/families/nemotron_h/draft_ids.txt", dtype=np.uint32)
    assert np.all(nemotron_ids[1:] > nemotron_ids[:-1])
    for name, vocab, group, ids in (("nemotron", 131072, 64, nemotron_ids),
                                    ("flash", 248320, 32, draft_ids())):
        mx.random.seed(123)
        # The actual vocabulary axis, with a small input axis to keep this fixture cheap.
        head = nn.QuantizedLinear(64, vocab, bias=False, group_size=group, bits=4)
        cut = cut_head(head, ids)
        arrays = {"expected.ids": mx.array(ids)}
        for field in ("weight", "scales", "biases"):
            arrays["lm_head." + field] = getattr(head, field)
            arrays["expected." + field] = getattr(cut, field)
        mx.save_safetensors(str(args.output / (name + ".safetensors")), arrays)
        print(f"Saved {name}: {len(ids)} original draft IDs and packed head rows", flush=True)


if __name__ == "__main__":
    main()
