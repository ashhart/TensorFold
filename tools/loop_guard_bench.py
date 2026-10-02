"""Microbench for the --loop-guard detector: the two shapes quoted in #204's PR thread.

Run from the repo root:  python tools/loop_guard_bench.py
Shapes match tests/test_loop_guard.py::test_near_miss_cycle_never_fires_and_stays_cheap:
a 30,000-token period-4 near-miss stream (perturbed every 12th block, so no exact run
survives), then a cyclic tail that fires. Per-commit cost measured in the production
call shape: one check() per appended token over a growing list.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfold.server.loop_guard import LoopGuard, WARM_IN  # noqa: E402


def main() -> None:
    guard = LoopGuard()
    stream = [1000 + i for i in range(WARM_IN)]
    block, count = [9000, 9001, 9002, 9003], 0
    while len(stream) < 30_000:
        stream += block
        count += 1
        if count % 12 == 0:
            stream[-1] += 1                    # a perturbation the exact-match test must not survive
    assert guard.check(stream) is None, "the near-miss stream fired"
    started = time.perf_counter()
    per_token = LoopGuard()
    for i in range(len(stream), len(stream) + 20_000):
        stream.append(block[i % 4] + (1 if i % 12 == 0 else 0))   # keep perturbing
        per_token.check(stream)
    near_miss = (time.perf_counter() - started) / 20_000 * 1e6

    tail = [1000 + i for i in range(64)] + [7000 + (i % 4) for i in range(600)]
    fired_at = next(i + 1 for i in range(len(tail)) if guard.check(tail[: i + 1]) is not None)
    started = time.perf_counter()
    for i in range(len(tail), len(tail) + 20_000):
        tail.append(7000 + (i % 4))
        guard.check(tail)
    cyclic = (time.perf_counter() - started) / 20_000 * 1e6

    print(f"near-miss per-commit (growing list, fired=never): {near_miss:.1f} us")
    print(f"cyclic-tail per-commit (fired at token {fired_at}): {cyclic:.1f} us")


if __name__ == "__main__":
    main()
