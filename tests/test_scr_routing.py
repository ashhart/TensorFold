# CPU-only tests for the SCR engine hooks: the fakes carry the slice of
# MultiDecoder the hooks touch, with the same call shapes as the real methods.

import os
import unittest
from unittest import mock

import torch

from tensorfold.cuda import streams as streams_mod
from tensorfold.families.qwen3_5.cuda import multi as multi_mod
from tensorfold.families.qwen3_5.cuda.scr import routing as scr
from tensorfold.families.qwen3_5.cuda.scr.sessions import SessionStore
from tensorfold.families.qwen3_5.cuda.scr.state import snapshot_from_state
from tests.test_scr_splice import FakeState, HALF


def fake_build_state(snap, rows, w, room=None, state_factory=None, pos=None):
    st = FakeState(4, rows)
    for i, pair in enumerate(st.kv):
        if pair is None:
            continue
        k, v = pair
        nk = torch.empty((rows, *k.shape[1:])).bfloat16()
        nv = torch.empty((rows, *v.shape[1:])).bfloat16()
        n = min(snap.pos, rows)
        nk[:n] = snap.kv[i][0][:n]
        nv[:n] = snap.kv[i][1][:n]
        st.kv[i] = (nk, nv)
    st.pos = snap.pos if pos is None else min(int(pos), snap.pos)
    return st


class FakeLayer:
    def __init__(self, linear):
        self.linear = linear


class FakeW:
    def __init__(self):
        self.inv_freq = torch.rand(HALF) * 0.1 + 0.01
        self.layers = [FakeLayer(False) for _ in range(4)]


class FakeCache:
    def longest(self, prompt):
        return None


class FakeStream:
    def __init__(self, prompt, sid=0, count=8):
        self.prompt = list(prompt)
        self.sid = sid
        self.count = count
        self.st = None
        self.snap = None
        self.sampling = None
        self.constraint = None
        self.stops = []
        self.cached = 0
        self.draft = True
        self.background = False
        self.vision = None
        self.done = False
        self.error = None
        self.out = []
        self.context = list(self.prompt)                # the real Stream sets this at first token
        self.prefill_s = 0.0
        self.started = 0.0
        self.copies = None
        self.scr_sid = None
        self.scr_plan = None
        self.scr_route = False
        self.scr_base = 0
        self.scr_served = 0

    def take(self, new, eos=()):
        self.out.extend(new)
        self.context.extend(new)
        if len(self.out) >= self.count:
            self.done = True


class FakeDrafter:
    layers = 2

    def restore(self, snap):
        pass

    def snapshot(self):
        return ([None] * self.layers, [None] * self.layers, 0, 0)

    def skip(self, n):
        pass


class FakeMD:
    """The slice of MultiDecoder the hooks touch, with the real methods' shapes."""

    def __init__(self, w):
        self.w = w
        self.world = 1
        self.rank = 0
        self.cache = FakeCache()
        self.streams = {}
        self.filling = []
        self.memory_gate = None
        self.draft = None
        self.drafts = False
        self.allow_copy = False
        self.next_id = 1

    def _most(self, s):
        return len(s.prompt) + s.count

    def _first(self, s):
        return self._most(s)

    def _ends(self, s):
        return True

    def _send(self, values):
        pass

    def admit(self, s):                     # the real admit's shape: scr first, then the engine's own
        scr.admission(self, s)
        s.sid = self.next_id
        self.next_id += 1
        hit = self.cache.longest(s.prompt) if s.draft else None
        self._queue(s, hit)

    def _queue(self, s, hit):
        if scr.queue(self, s, hit):         # a planned stream builds its state from its session's snapshot
            return
        s.st = FakeState(4, self._most(s))
        s.stops = [len(s.prompt)]
        s.cached = len(hit[0]) if hit else 0
        self.filling.append(s)

    def _fill(self):
        routed = scr.fill(self)
        if routed is not None:
            return routed
        return []

    def finish(self, done):
        scr.snapshots(self, done)
        for s in done:
            self.streams.pop(s.sid, None)


def fake_prefill_state(w, prompt_slice, st, tp=False, draft=None, vision=None):
    # the engine commits prompt[st.pos:stop]; stop == len(prompt_slice)
    st.pos = len(prompt_slice)
    return "normed"


class TestRouting(unittest.TestCase):
    def setUp(self):
        self.patches = [mock.patch.object(multi_mod, "prefill_state", fake_prefill_state),
                        mock.patch.object(multi_mod, "first_token",
                                          lambda w, normed, n, sampling, rank, world, constraint: 42),
                        mock.patch.object(multi_mod, "CopyIndex", lambda: None),
                        mock.patch.object(multi_mod, "STEP", 1 << 30),
                        mock.patch.object(multi_mod, "FILL", 2),
                        mock.patch.object(scr, "build_state_from_snapshot", fake_build_state)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        torch.manual_seed(2)
        self.w = FakeW()
        self.md = FakeMD(self.w)
        self.md.draft = FakeDrafter()
        self.md.drafts = True
        self.rt = scr._rt_for(self.w)
        for k, v in dict(min_base=8, min_tail=4, min_extend=2, max_blocks=3, min_gain=4,
                         match_min_tokens=8, match_min_ratio=0.3, match_ambig_margin=0.15).items():
            setattr(self.rt.cfg, k, v)
        self.rt.store = SessionStore(self.rt.cfg)
        self.inv_freq = self.w.inv_freq
        self.base_ids = list(range(300))

    def seed_session(self):
        s, _ = self.rt.store.session_for(self.base_ids)
        st = FakeState(4, 512)
        for i in range(4):
            st.kv[i] = (torch.randn(300, 2, 2 * HALF).bfloat16(),
                        torch.randn(300, 2, 2 * HALF).bfloat16())
        st.pos = 300
        self.rt.store.refresh(s.sid, snapshot_from_state(st, self.base_ids))
        return s.sid

    def test_admit_attaches_plan_and_queue_builds_state(self):
        sid = self.seed_session()
        new = self.base_ids[:280] + [777, 778] + self.base_ids[280:]
        s = FakeStream(new)
        self.md.admit(s)
        self.assertEqual(s.scr_sid, sid)
        self.assertIsNotNone(s.scr_plan)
        self.assertTrue(s.scr_route)
        self.assertTrue(s in self.md.filling)
        self.assertEqual(s.st.pos, 280)                 # base rows from snapshot
        self.assertEqual(s.stops, [])
        self.assertEqual(s.cached, 280)

    def drive(self, s, limit=12):
        """Fill rounds until the first token is out (decode is a separate loop the
        fake does not drive)."""
        for _ in range(limit):
            self.md._fill()
            if s.out or s.error:
                break
        if s.error is not None:
            raise s.error
        return s

    def test_planned_fill_executes_splices_then_first_token(self):
        sid = self.seed_session()
        new = self.base_ids[:280] + [777, 778] + self.base_ids[280:]
        s = FakeStream(new)
        self.md.admit(s)
        plan = s.scr_plan
        self.assertEqual(len(plan.splices), 1)
        seen_splice = None
        for _ in range(10):
            self.md._fill()
            if s.scr_plan is not None and seen_splice is None and s.st.pos > 282:
                seen_splice = s.st.pos
            if s.out:
                break
        self.assertIsNotNone(seen_splice, "splice never executed")
        # relocated keys re-rotated by +2
        sp = plan.splices[0]
        inv = self.inv_freq
        moved = s.st.kv[0][0][sp.at:sp.at + sp.n].float()
        snapk = plan.snapshot.kv[0][0][sp.old:sp.old + sp.n]
        ang = 2.0 * inv.float()
        cos, sin = ang.cos(), ang.sin()
        f, sec = snapk.float()[..., :HALF], snapk.float()[..., HALF:]
        want = torch.cat([f * cos - sec * sin, sec * cos + f * sin], dim=-1)
        self.assertTrue(torch.allclose(moved, want, atol=2e-2, rtol=2e-2))
        self.assertEqual(plan.served, sp.n)
        self.assertEqual(s.out[-1], 42)

    def test_finish_refreshes_snapshot_and_next_turn_plans(self):
        sid = self.seed_session()
        new = self.base_ids[:280] + [777, 778] + self.base_ids[280:]
        s = FakeStream(new)
        self.md.admit(s)
        self.drive(s)
        self.md.finish([s])
        sess = self.rt.store.sessions[sid]
        self.assertIsNotNone(sess.snap)
        ids_after_turn1 = list(sess.ids)                    # the committed ids (admission re-points them)
        self.assertEqual(sess.ids, s.context[:sess.snap.pos])
        # an append-only turn now plans with no splices and reuses the reply rows
        s2 = FakeStream(s.context + list(range(5000, 5050)))
        self.md.admit(s2)
        self.assertIsNotNone(s2.scr_plan)
        self.assertEqual(s2.scr_plan.base, len(ids_after_turn1))
        self.assertEqual(s2.scr_plan.splices, [])
        self.assertEqual(s2.st.pos, len(ids_after_turn1))

    def test_splice_failure_falls_back_and_keeps_serving(self):
        sid = self.seed_session()
        new = self.base_ids[:280] + [777, 778] + self.base_ids[280:]
        s = FakeStream(new)
        self.md.admit(s)
        # corrupt the snapshot ids so the token guard fails -> abort
        s.scr_plan.snapshot.ids[280] = -1
        self.drive(s)
        self.assertIsNone(s.scr_plan)
        self.assertTrue(s.scr_route)                    # keeps stay off after an abort
        self.assertEqual(s.out[-1], 42)
        # the prefilled tail is a plain prefill (no splice), pos walks to prompt end
        self.assertEqual(s.st.pos, len(new))

    def test_unplanned_stream_uses_engine_path(self):
        self.seed_session()
        s = FakeStream(list(range(50)))                 # too short to match: cold
        self.md.admit(s)
        self.assertIsNone(s.scr_plan)
        self.assertFalse(s.scr_route)
        self.assertEqual(s.stops, [len(s.prompt)])      # plain queueing keeps stops
        self.assertEqual(streams_mod.next_fill(self.md.filling), s)


if __name__ == "__main__":
    unittest.main()