"""Compare a reference forward's logits with vLLM's prompt logprobs (tools/dsv41_golden.py).

    python tools/dsv41_compare.py notes/dsv41/golden.json out/ref-0.pt [out/ref-1.pt ...]

Each .pt comes from ``python -m tensorfold.families.deepseek_v41.reference ... --ids @ids-K.json`` for golden K
(matched by ids). Reports top-1 agreement with vLLM's greedy choice at every position, the mean and max absolute
difference of the actual next token's logprob, and the position with the largest difference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main() -> None:
    goldens = json.loads(Path(sys.argv[1]).read_text())["goldens"]
    by_ids = {tuple(g["ids"]): g for g in goldens}
    for path in sys.argv[2:]:
        ref = torch.load(path)
        g = by_ids.get(tuple(ref["ids"]))
        if g is None:
            print(f"{path}: no golden with these ids")
            continue
        lp = torch.log_softmax(ref["logits"].float(), dim=-1)
        ids = g["ids"]
        agree = n = 0
        diffs = []
        for i in range(1, len(ids)):
            top = g["prompt_top5"][i]
            if not top:
                continue
            n += 1
            agree += int(lp[i - 1].argmax()) == top[0][0]
            if g["prompt_actual"][i] is not None:
                diffs.append((abs(float(lp[i - 1, ids[i]]) - g["prompt_actual"][i]), i))
        d = torch.tensor([x for x, _ in diffs])
        worst = max(diffs)
        ours = int(lp[-1].argmax())
        print(f"{path}: {len(ids)} tokens, top-1 agree {agree}/{n} ({100 * agree / max(n, 1):.1f}%), "
              f"|dlogprob| mean {d.mean():.3f} median {d.median():.3f} max {worst[0]:.3f} at {worst[1]}; "
              f"next token ours {ours}, vLLM {g['next_text']!r}")


if __name__ == "__main__":
    main()
