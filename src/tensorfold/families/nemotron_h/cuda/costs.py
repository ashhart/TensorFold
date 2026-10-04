"""What a verify window and a draft level cost on this GPU, timed over consecutive tokens of real text."""

from __future__ import annotations

import statistics

import torch

from tensorfold.cuda.draft_depth import Costs

from .decode import prefill
from .engine import Engine
from .mtp import MTPHead

TEXT = ("The river had been rising for three days, and by the time the ferry stopped running the town had moved its "
        "market up the hill. Children carried baskets of apples past the church while their parents argued about "
        "whether the old bridge would hold. In the workshop behind the bakery, a carpenter measured each plank "
        "twice, wrote the numbers on the wall, and cut slowly.\n\ndef mean(values):\n    total = 0\n    for v in "
        "values:\n        total += v\n    return total / len(values)\n")


def _ms(setup, reps: int) -> float:
    """Median GPU ms of ``setup()``'s returned call, after one warm-up."""

    times = []
    for _ in range(reps + 1):
        run = setup()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times[1:])


@torch.no_grad()
def measure(eng: Engine, mtp: MTPHead, ids: list[int], *, reps: int = 5) -> Costs:
    """Windows of 1..max_rows rows after a one-row kept window, as decode sees them, and one head level's ms."""

    rows = eng.max_rows
    if len(ids) < 2 * rows + 2:
        raise ValueError(f"timing windows of {rows} rows needs {2 * rows + 2} tokens")
    pre = prefill(eng, mtp, ids[:rows], None)
    cont = ids[rows:]

    def window(r: int):
        def ready():
            eng.restore(pre.engine)
            eng.forward(cont[:1])
            eng.tokens()
            eng.commit(1)
            torch.cuda.synchronize()
            return lambda: eng.forward(cont[1:1 + r])
        return ready

    def chain(levels: int):
        def ready():
            mtp.restore(pre.mtp)
            eng.hidden[:1].copy_(pre.last_hidden)
            eng.sampled[:1].fill_(pre.pending)
            torch.cuda.synchronize()
            return lambda: mtp.round(1, levels)
        return ready

    verify = (0.0,) + tuple(_ms(window(r), reps) for r in range(1, rows + 1))
    deep = min(mtp.most, 7)
    level = (_ms(chain(1 + deep), reps) - _ms(chain(1), reps)) / deep
    eng.tokens()
    eng.reset()
    mtp.reset()
    return Costs.measured(verify, level)
