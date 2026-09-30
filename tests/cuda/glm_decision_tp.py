"""Two-host regression: run rank 1 first, then rank 0 with the same master/port.

PYTHONPATH=src:tests/cuda python tests/cuda/glm_decision_tp.py --rank 1 --master HOST
Rank 1 is a normal follow() worker; stop that test process after rank 0 exits.
Uses a tiny synthetic checkpoint, real NCCL and CUDA, not duplicated shards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from test_glm_engine import _checkpoint, _drafter, _generate

from tensorfold.families.glm5_next.cuda.engine import GlmEngine


class BudgetEngine(GlmEngine):
    def _take_over(self, keep):
        self.cache_bytes = int(self.comm.store.get("budget"))
        return super()._take_over(keep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--port", type=int, default=29627)
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/glm-decision-tp"))
    args = parser.parse_args()
    _checkpoint(args.checkpoint / "model")
    _drafter(args.checkpoint / "drafter")
    e = BudgetEngine(args.checkpoint / "model", rank=args.rank, master=args.master, port=args.port,
                  drafter=args.checkpoint / "drafter")
    if args.rank == 1:
        e.follow()
        return
    labels = [0, 17, 512, 1023]
    prompt = list(range(40))
    decision = list(range(100, 180))
    other = list(range(200, 224))
    e.comm.store.set("budget", str(64 * 1024 * 1024))
    reference = e.score_labels(decision, labels)
    for budget in (0, 64 * 1024 * 1024):
        # Both ranks use the same cache budget; communicate it outside the request
        # protocol via the NCCL bootstrap store before running this case.
        e.comm.store.set("budget", str(budget))
        e.cache_bytes = budget
        for policy in ("2", "f3"):
            reply, _ = _generate(e, prompt, None, policy=policy, tokens=16)
            after = prompt + reply + [31, 32]
            assert e.score_labels(decision, labels) == reference
            assert e.live == [] and e.e.st.pos == 0 and e.drafter.context_end == 0
            warm, stats = _generate(e, after, None, policy=policy, tokens=16)
            assert stats["cached"] == (len(prompt) if budget and policy == "2" else 0)
            _generate(e, other, None, policy=policy, tokens=16)
            switched, _ = _generate(e, after + [33], None, policy=policy, tokens=16)
            cold, _ = _generate(e, after, None, draft=False, tokens=16)
            cold_switched, _ = _generate(e, after + [33], None, draft=False, tokens=16)
            assert warm == cold and switched == cold_switched
            print(json.dumps({"rank": 0, "budget": budget, "policy": policy, "result": "PASS"}), flush=True)


if __name__ == "__main__":
    main()
