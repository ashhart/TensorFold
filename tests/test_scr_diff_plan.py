# CPU-only tests for the SCR port: run `python -m pytest tests/ -q` (or unittest)
# from /home/jmw/opt/tensorfold-scr. No tensorfold, no GPU.

import os
import unittest

os.environ.setdefault("TENSORFOLD_SCR_ENABLED", "0")

from tensorfold.families.qwen3_5.cuda.scr.config import Config
from tensorfold.families.qwen3_5.cuda.scr.diff import diff_opcodes, verify_ops
from tensorfold.families.qwen3_5.cuda.scr.plan import Splice, build_plan, describe
from tensorfold.families.qwen3_5.cuda.scr.sessions import SessionStore, Snapshot


def cfg(**over):
    c = Config()
    for k, v in over.items():
        setattr(c, k, v)
    return c


def snap_of(ids):
    return Snapshot(ids=list(ids), pos=len(ids), kv=[], rec=[], conv=[], bytes=0)


class TestDiff(unittest.TestCase):
    def test_matches_difflib(self):
        import random
        rng = random.Random(7)
        for trial in range(50):
            a = [rng.randrange(1000) for _ in range(rng.randrange(50, 400))]
            b = list(a)
            for _ in range(rng.randrange(0, 6)):
                op = rng.randrange(3)
                i = rng.randrange(1, max(2, len(b)))
                if op == 0:
                    del b[i:i + rng.randrange(1, 20)]
                elif op == 1:
                    b[i:i] = [rng.randrange(1000) for _ in range(rng.randrange(1, 20))]
                else:
                    b[i:i + 5] = [rng.randrange(1000) for _ in range(rng.randrange(1, 20))]
            ops = diff_opcodes(a, b, fastdiff=True, cdc_target=64, fastdiff_min=0)
            self.assertTrue(verify_ops(a, b, ops), f"trial {trial}")

    def test_fast_path_same_as_slow_on_big(self):
        import random
        rng = random.Random(3)
        a = [rng.randrange(500) for _ in range(5000)]
        b = a[:2000] + [999999] * 10 + a[2000:4000] + a[4500:]
        fast = diff_opcodes(a, b, fastdiff=True, cdc_target=64, fastdiff_min=0)
        slow = diff_opcodes(a, b, fastdiff=False)
        self.assertTrue(verify_ops(a, b, fast))
        self.assertTrue(verify_ops(a, b, slow))


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.c = cfg(min_base=8, min_tail=4, min_extend=2, max_blocks=3, min_gain=4,
                     mb_min_gap=0, fastdiff=False)

    def test_mid_edit_relocates_survivor(self):
        # old: [A][B][C]; new: [A][B'][C]  (B replaced by shorter B')
        A = list(range(100, 120))
        B = list(range(200, 260))
        C = list(range(300, 380))
        Bp = [777, 778]
        old = A + B + C
        new = A + Bp + C
        plan, why = build_plan(old, new, snap_of(old), "s1", self.c)
        self.assertIsNotNone(plan, why)
        self.assertEqual(plan.base, len(A))
        self.assertEqual(len(plan.splices), 1)
        sp = plan.splices[0]
        self.assertEqual(sp.at, len(A) + len(Bp))          # C lands right after B'
        self.assertEqual(sp.old, len(A) + len(B))
        self.assertEqual(sp.n, len(C) - self.c.min_extend)  # tail stays prefill

    def test_splice_free_append_plan(self):
        # append-only turn: prev prompt + reply survive, new tokens added
        old = list(range(1000))                            # prompt+output of turn 1
        new = old + list(range(5000, 5100))
        plan, why = build_plan(old, new, snap_of(old), "s1", self.c, resume_len=600)
        self.assertIsNotNone(plan, why)
        self.assertEqual(plan.base, len(old))
        self.assertEqual(plan.splices, [])
        self.assertEqual(plan.gain, len(old) - 600)

    def test_gain_gate_uses_resume_len(self):
        old = list(range(1000))
        new = old + list(range(5000, 5100))
        c = cfg(min_base=8, min_tail=4, min_extend=2, max_blocks=3, min_gain=400, fastdiff=False)
        plan, _ = build_plan(old, new, snap_of(old), "s1", c, resume_len=600)
        self.assertIsNotNone(plan)                         # gain 400 == min
        plan, _ = build_plan(old, new, snap_of(old), "s1", c, resume_len=700)
        self.assertIsNone(plan)                            # gain 300 < 400

    def test_hit_longer_than_base_kills_plan(self):
        old = list(range(500)) + [42] + list(range(600, 650))
        new = list(range(500)) + [43, 44] + list(range(600, 650))
        c = cfg(min_base=8, min_tail=4, min_extend=2, max_blocks=3, min_gain=4, fastdiff=False)
        # resume covers 500 exactly; plan base is 500 too -> gain = relocated only
        plan, _ = build_plan(old, new, snap_of(old), "s1", c, resume_len=500)
        self.assertIsNotNone(plan)
        # a cached prefix covering the plan's reuse -> not worth it
        plan, _ = build_plan(old, new, snap_of(old), "s1", c, resume_len=545)
        self.assertIsNone(plan)

    def test_k_cap_takes_longest(self):
        old = list(range(100, 200))
        mid = [9] * 5
        spans = {i: list(range(1000 * i, 1000 * i + 20)) for i in range(5)}
        new = old + mid + spans[0] + mid + spans[1] + mid + spans[2] + mid + spans[3] + mid + spans[4]
        old_seq = old + spans[0] + spans[1] + spans[2] + spans[3] + spans[4]
        c = cfg(min_base=8, min_tail=4, min_extend=2, max_blocks=2, min_gain=4, fastdiff=False)
        plan, why = build_plan(old_seq, new, snap_of(old_seq), "s1", c)
        self.assertIsNotNone(plan, why)
        self.assertEqual(len(plan.splices), 2)
        lens = sorted(sp.n for sp in plan.splices)
        self.assertEqual(lens, [20 - 2, 20 - 2])           # the two longest, tails prefilled

    def test_base_covers_prompt_rejected(self):
        # the snapshot covers the whole new prompt: nothing to prefill, no first
        # token to sample -> no plan (the scheduler must never wedge on this)
        old = list(range(1000)) + [7]
        new = list(range(1000))
        plan, why = build_plan(old, new, snap_of(old), "s1", self.c)
        self.assertIsNone(plan)
        self.assertEqual(why, "base-covers-prompt")

    def test_reasoning_strip(self):
        # turn 2 strips the reasoning block from turn 1's reply: [P][R1][RB][R2] -> [P][R1][R2]
        P = list(range(100, 200))
        RB = list(range(500, 560))                          # stripped reasoning
        R1 = list(range(600, 700))
        R2 = list(range(700, 760))
        old = P + R1 + RB + R2
        new = P + R1 + R2
        plan, why = build_plan(old, new, snap_of(old), "s1", self.c)
        self.assertIsNotNone(plan, why)
        self.assertEqual(len(plan.splices), 1)
        self.assertEqual(plan.splices[0].at, len(P) + len(R1))
        self.assertEqual(plan.splices[0].old, len(P) + len(R1) + len(RB))

    def test_segment_walk(self):
        old = list(range(100)) + list(range(200, 260)) + list(range(300, 340))
        new = list(range(100)) + [7, 8] + list(range(200, 260)) + [9] + list(range(300, 340))
        c = cfg(min_base=8, min_tail=4, min_extend=2, max_blocks=3, min_gain=4, fastdiff=False)
        plan, why = build_plan(old, new, snap_of(old), "s1", c)
        self.assertIsNotNone(plan, why)
        pos = plan.base
        seq = []
        while pos < len(new):
            sp = plan.pending_splice(pos)
            if sp is not None:
                seq.append(("splice", sp.at, sp.n))
                pos = sp.at + sp.n
                continue
            stop = plan.segment_stop(pos, len(new))
            self.assertGreater(stop, pos)
            seq.append(("fill", pos, stop))
            pos = stop
        fills = [s for s in seq if s[0] == "fill"]
        splices = [s for s in seq if s[0] == "splice"]
        self.assertEqual(len(splices), 2)
        self.assertGreaterEqual(len(fills), 3)
        # every splice is followed by its min_extend tail inside the next fill
        for (_, at, n), nxt in zip(splices, splices[1:] + [(None, len(new), None)]):
            tail_end = at + n + self.c.min_extend
            self.assertLessEqual(tail_end, nxt[1] if nxt[0] == "splice" else len(new))

    def test_stale_snapshot_rejected(self):
        old = list(range(1000))
        new = old + [1, 2, 3]
        bad = snap_of(list(range(500)))                    # ids do not match old_ids
        plan, why = build_plan(old, new, bad, "s1", self.c)
        self.assertIsNone(plan)
        self.assertEqual(why, "stale-session")


if __name__ == "__main__":
    unittest.main()