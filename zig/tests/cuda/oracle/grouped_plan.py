"""Fixtures for `tf-cuda-test grouped-plan`: experts.route's plan (python-0.6's cuda.experts) on random picks."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle import save  # noqa: E402

# (rows, slots, experts, tile): one-block plans, then wide plans past 1,024 pairs, decode and prompt tiles
CASES = ((1, 7, 33, 16), (3, 7, 33, 16), (300, 7, 33, 64), (512, 7, 385, 16), (2048, 9, 385, 64))


def main() -> None:
    from tensorfold.cuda import experts as grouped

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    a = ap.parse_args()
    g = torch.Generator().manual_seed(11)
    arrays, names = {}, []
    for i, (rows, slots, experts, tile) in enumerate(CASES):
        picks = torch.stack([torch.randperm(experts, generator=g)[:slots] for _ in range(rows)]).int().contiguous()
        plan = grouped.Plan(rows, slots, experts, "cuda", prefill=tile != grouped.TILE)
        grouped.route(picks.cuda(), plan, tile)
        torch.cuda.synchronize()
        arrays.update({f"picks{i}": picks, f"members{i}": plan.members, f"items{i}": plan.items,
                       f"counts{i}": plan.counts})
        names.append(f"{rows}:{slots}:{experts}:{tile}")
    save(Path(a.out), {"cases": ",".join(names)}, arrays)


if __name__ == "__main__":
    main()
