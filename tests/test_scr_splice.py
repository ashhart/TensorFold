import os
import unittest

import torch

os.environ.setdefault("TENSORFOLD_SCR_ENABLED", "0")

from tensorfold.families.qwen3_5.cuda.scr.plan import Splice
from tensorfold.families.qwen3_5.cuda.scr.splice import execute_splice, rotate_keys_inplace
from tensorfold.families.qwen3_5.cuda.scr.state import build_state_from_snapshot, snapshot_from_state
from tensorfold.families.qwen3_5.cuda.scr.sessions import Snapshot

HALF = 16
D = 2 * HALF


def reference_rope(k, pos, inv_freq):
    """The same rotate-half convention glue.attn_prep uses, as plain torch.
    ``pos``: one scalar position, or one position per row. Rotates dims [0, 2*half)
    in (d, d+half) pairs; dims beyond pass through (partial rotary)."""
    half = inv_freq.numel()
    rot = 2 * half
    f32 = k.float()
    if not torch.is_tensor(pos):
        pos = torch.full((k.shape[0],), float(pos))
    ang = pos.float().unsqueeze(1) * inv_freq.float().unsqueeze(0)     # (n, half)
    cos, sin = ang.cos(), ang.sin()
    first, second = f32[..., :half], f32[..., half:rot]
    out = torch.cat([first * cos.unsqueeze(1) - second * sin.unsqueeze(1),
                     second * cos.unsqueeze(1) + first * sin.unsqueeze(1),
                     f32[..., rot:]], dim=-1)
    return out.to(k.dtype)


class TestRotate(unittest.TestCase):
    def test_rerot_equals_fresh_rope(self):
        torch.manual_seed(0)
        inv_freq = torch.rand(HALF) * 0.1 + 0.01
        n, heads = 7, 3
        base = torch.randn(n, heads, D).bfloat16()
        p_old, delta = 4000, 37
        moved = base.clone()
        rotate_keys_inplace(moved, delta, inv_freq)
        want = reference_rope(base, p_old + delta, inv_freq)
        got = reference_rope(moved, p_old, inv_freq)
        self.assertTrue(torch.allclose(got.float(), want.float(), atol=2e-2, rtol=2e-2),
                        f"max err {(got.float() - want.float()).abs().max()}")

    def test_delta_zero_noop(self):
        k = torch.randn(4, 2, D).bfloat16()
        before = k.clone()
        rotate_keys_inplace(k, 0, torch.rand(HALF))
        self.assertTrue(torch.equal(k, before))

    def test_bad_head_dim(self):
        with self.assertRaises(ValueError):
            rotate_keys_inplace(torch.randn(2, 2, 7).bfloat16(), 5, torch.rand(HALF))


class TestPartialRotary(unittest.TestCase):
    """Qwen3.8: head_dim 256, inv_freq 32 -> only dims [0, 64) rotate (pairs d, d+32);
    dims [64, 256) pass through, matching glue.attn_prep's third region."""

    def test_partial_rope_passthrough_tail(self):
        torch.manual_seed(4)
        half, d = 32, 256
        inv_freq = torch.rand(half) * 0.05 + 0.005
        base = torch.randn(5, 2, d).bfloat16()
        p, delta = 1000, 23
        moved = base.clone()
        rotate_keys_inplace(moved, delta, inv_freq)
        # rotated region matches reference
        want_rot = reference_rope(base[..., :d].reshape(5 * 2, 1, d), p + delta, inv_freq)
        got_rot = reference_rope(moved.reshape(5 * 2, 1, d), p, inv_freq)
        # dims [0, 64): rotated consistently between the two
        self.assertTrue(torch.allclose(want_rot[..., :64].float(), got_rot[..., :64].float(), atol=2e-2, rtol=2e-2))
        # dims [64, 256): untouched by re-rotation
        self.assertTrue(torch.equal(base[..., 64:], moved[..., 64:]))

    def test_too_small_head_dim(self):
        with self.assertRaises(ValueError):
            rotate_keys_inplace(torch.randn(2, 2, 32).bfloat16(), 5, torch.rand(32))


class FakeState:
    """The slice of TensorFold's State that splice/state need."""

    def __init__(self, layers, rows_capacity, kv_heads=2, hybrid=()):
        self.pos = 0
        self.limit = 0
        self.room = None
        self.kv, self.rec, self.conv = [], [], []
        for i in range(layers):
            if i in hybrid:
                self.kv.append(None)
                self.rec.append(torch.zeros(2, 4, 4))
                self.conv.append(torch.zeros(3, 8))
            else:
                self.kv.append((torch.empty((rows_capacity, kv_heads, D)).bfloat16(),
                                torch.empty((rows_capacity, kv_heads, D)).bfloat16()))
                self.rec.append(None)
                self.conv.append(None)

    def grow_all(self, i, need):
        for i, pair in enumerate(self.kv):
            if pair is None:
                continue
            k, v = pair
            if k.shape[0] < need:
                nk = torch.empty((need, *k.shape[1:])).bfloat16()
                nv = torch.empty((need, *v.shape[1:])).bfloat16()
                nk[:k.shape[0]], nv[:v.shape[0]] = k, v
                self.kv[i] = (nk, nv)


class TestSplice(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.inv_freq = torch.rand(HALF) * 0.1 + 0.01
        self.layers, self.rows = 4, 512
        self.st = FakeState(self.layers, self.rows)
        # commit 400 rows "at their positions" (reference rope applied)
        self.old_ids = list(range(400))
        data = torch.randn(400, 2, D).bfloat16()
        self.base = data                                # the unrotated keys, for expectations
        pos = torch.arange(400).float().unsqueeze(1) * self.inv_freq.float().unsqueeze(0)
        cos, sin = pos.cos(), pos.sin()
        first, second = data.float()[..., :HALF], data.float()[..., HALF:]
        rotated = torch.cat([first * cos.unsqueeze(1) - second * sin.unsqueeze(1),
                             second * cos.unsqueeze(1) + first * sin.unsqueeze(1)], dim=-1).bfloat16()
        for i in range(self.layers):
            self.st.kv[i] = (rotated.clone(), torch.randn(400, 2, D).bfloat16())
        self.st.pos = 400

    def to_snapshot(self):
        return snapshot_from_state(self.st, self.old_ids)

    def test_snapshot_roundtrip_and_splice(self):
        snap = self.to_snapshot()
        self.assertEqual(snap.pos, 400)
        self.assertEqual(snap.ids, self.old_ids)
        self.assertGreater(snap.bytes, 0)
        # snapshot must be private: mutating the source must not touch it
        self.st.kv[0][0][5] += 123
        self.assertFalse(torch.equal(self.st.kv[0][0][5], snap.kv[0][0][5]))

        # new prompt: 280 shared rows + edit of 2 + surviving 120 rows
        new_ids = self.old_ids[:280] + [777, 778] + self.old_ids[280:]
        self.assertEqual(len(new_ids), 402)
        st2 = FakeState(self.layers, self.rows)
        # base rows copied verbatim (positions unchanged)
        for i in range(self.layers):
            st2.kv[i] = (torch.empty((self.rows, 2, D)).bfloat16(),
                         torch.empty((self.rows, 2, D)).bfloat16())
            st2.kv[i][0][:280] = snap.kv[i][0][:280]
            st2.kv[i][1][:280] = snap.kv[i][1][:280]
        st2.pos = 280
        st2.pos = 282                                   # the fill of the 2 edit tokens ran first
        sp = Splice(at=282, old=280, n=120 - 16)
        ok = execute_splice(st2, sp, snap, new_ids, self.inv_freq, grow_fn=FakeState.grow_all)
        self.assertTrue(ok)
        self.assertEqual(st2.pos, 282 + 104)
        # relocated keys equal the fresh rope of the original (unrotated) keys at
        # their new per-row positions: R(282+j) @ base[j] == R(2) @ R(280+j) @ base[j]
        for i in range(self.layers):
            moved = st2.kv[i][0][282:386]
            fresh = reference_rope(self.base[280:384], torch.arange(282, 386), self.inv_freq)
            self.assertTrue(torch.allclose(moved.float(), fresh.float(), atol=2e-2, rtol=2e-2),
                            f"layer {i}: max err {(moved.float() - fresh.float()).abs().max()}")

    def test_token_guard_rejects_stale(self):
        snap = self.to_snapshot()
        snap.ids[100] = 99999                  # simulate a stale/mismatched snapshot
        st2 = FakeState(self.layers, self.rows)
        st2.pos = 100
        sp = Splice(at=100, old=100, n=50)
        prompt = list(range(400))
        self.assertFalse(execute_splice(st2, sp, snap, prompt, self.inv_freq,
                                        grow_fn=FakeState.grow_all))
        self.assertEqual(st2.pos, 100)         # state untouched

    def test_guard_pos_mismatch(self):
        snap = self.to_snapshot()
        st2 = FakeState(self.layers, self.rows)
        st2.pos = 99
        self.assertFalse(execute_splice(st2, Splice(at=100, old=100, n=10), snap,
                                        list(range(400)), self.inv_freq,
                                        grow_fn=FakeState.grow_all))

    def test_grow_on_small_buffers(self):
        snap = self.to_snapshot()
        st2 = FakeState(self.layers, 0)        # zero capacity: grow path
        st2.pos = 50
        ok = execute_splice(st2, Splice(at=50, old=50, n=40), snap, list(range(400)),
                            self.inv_freq, grow_fn=FakeState.grow_all)
        self.assertTrue(ok)
        self.assertEqual(st2.kv[0][0].shape[0], 90)

    def test_hybrid_forks_recurrent_state(self):
        st = FakeState(4, 128, hybrid=(1, 3))
        ids = list(range(128))
        st.pos = 128
        st.rec[1] += 3.0
        snap = snapshot_from_state(st, ids)
        self.assertIsNone(snap.kv[1])
        self.assertIsNotNone(snap.rec[1])
        st2 = build_state_from_snapshot(snap, 256, w=None, state_factory=lambda w: FakeState(4, 0, hybrid=(1, 3)))
        self.assertEqual(st2.pos, 128)
        self.assertTrue(torch.equal(st2.rec[1], st.rec[1]))
        self.assertIsNone(st2.kv[1])
        self.assertEqual(st2.kv[0][0].shape[0], 256)
        self.assertTrue(torch.equal(st2.kv[0][0][:128], st.kv[0][0][:128]))


if __name__ == "__main__":
    unittest.main()