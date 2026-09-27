"""Exercise the original FamilyRounds policy to generate independent native oracles."""
import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace

from tensorfold.engine.lane_family import FamilyRounds
from tensorfold.families.nemotron_h.model import NemotronH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rng = random.Random(78127)
    cases = []
    for budget in range(16):
        for prior in (FamilyRounds.depth_prior, NemotronH.draft_prior):
            for mode in range(3):
                engine = FamilyRounds()
                engine.most_drafts = budget
                engine.depth_prior = prior
                engine._depth_state = {}
                engine._round_ms = {}
                costs = ([0.] * 17 if mode == 0 else
                         [0.] + [rng.uniform(0.1, 5.) + width * .2 for width in range(1, 17)])
                if mode == 2:
                    costs[rng.randrange(2, 17)] = 0.
                engine.family_costs = {w: cost for w, cost in enumerate(costs) if cost}
                engine.mtp_step_ms = rng.uniform(.05, 1.)
                case = dict(budget=budget, prior=prior, costs=costs, mtp_ms=engine.mtp_step_ms, steps=[])
                for turn in range(128):
                    room = rng.randrange(0, 34)
                    stream = SimpleNamespace(stream_id="test", draft_room=room)
                    depth = engine._depth(stream)
                    accepted = rng.randrange(depth + 1)
                    ms = rng.uniform(.05, 30.)
                    engine._observe_depth(stream, depth, accepted)
                    engine._observe_cost(depth, ms)
                    case["steps"].append(dict(room=room, expected=depth, proposed=depth,
                                               accepted=accepted, ms=ms, rates=list(engine._depth_rates(stream))))
                cases.append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases) + "\n")
    print(f"Saved {sum(len(c['steps']) for c in cases)} original adaptive depth decisions", flush=True)


if __name__ == "__main__":
    main()
