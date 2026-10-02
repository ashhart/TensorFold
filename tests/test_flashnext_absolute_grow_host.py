"""Run cache-growth and caller bookkeeping with byte counters, without importing a GPU runtime."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tensorfold.cuda import capacity
from tensorfold.cuda.memory_gate import MemoryGate, NoRoom, torch_live
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda import prefixes

GIB = capacity.GIB
ROOT = Path(__file__).resolve().parents[1]


def source_class(path, name, methods, globals):
    """Execute the unchanged production methods with host stand-ins for their dependencies."""
    parsed = ast.parse((ROOT / path).read_text())
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    for node in cls.body:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), globals)
    return globals[name]


class Allocator:
    def __init__(self):
        self.allocated = 0
        self.peak = 0
        self.cached = 8 * GIB

    def take(self, nbytes):
        self.allocated += nbytes
        self.peak = max(self.peak, self.allocated)

    def give(self, nbytes):
        self.allocated -= nbytes


class Array:
    def __init__(self, owner, rows, width):
        self.owner, self.shape = owner, (rows, width)
        self.nbytes = rows * width * 2
        owner.take(self.nbytes)

    def __del__(self):
        self.owner.give(self.nbytes)


class KV:
    """A resize allocates its replacement while the caller still owns the old layer, as KVCache.resized does."""
    row_bytes = 65536

    def __init__(self, owner, rows):
        self.owner, self.capacity = owner, rows
        self.nbytes = rows * self.row_bytes
        owner.take(self.nbytes)

    def __del__(self):
        self.owner.give(self.nbytes)

    def resized(self, rows, keep):
        return KV(self.owner, rows)


def rows_array(array, rows, keep):
    return Array(array.owner, rows, array.shape[1])


State = source_class("src/tensorfold/families/qwen4_exp/cuda/state.py", "State",
                     {"cache_bytes", "layer_bytes", "resize"}, {"_rows": rows_array})


def state(owner):
    st = State()
    st.capacity, st.limit, st.layers = 256, 81920, 2
    st.pos, st.mtp_len, st.version = 0, 0, 0
    st.image_positions = None
    st.row_bytes, st.index_dim, st.ratio = KV.row_bytes + 256, 128, 4
    st.kc = [KV(owner, st.capacity) for _ in range(st.layers)]
    st.ikc = [Array(owner, st.capacity, st.index_dim) for _ in range(st.layers)]
    st.pooled = [Array(owner, st.capacity // st.ratio, st.index_dim) for _ in range(st.layers)]
    return st


class AbsoluteGrowthTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {capacity.LIMIT_ENV: "60"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        meminfo = patch.object(capacity, "_meminfo", return_value=None)
        meminfo.start()
        self.addCleanup(meminfo.stop)
        self.alloc = Allocator()
        first, second = state(self.alloc), state(self.alloc)
        # Weights and already-created workspaces are allocated before the gate is made.
        initial = 48 * GIB - GIB // 16
        fixed = initial - self.alloc.allocated
        self.alloc.take(fixed)
        self.torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True,
            memory_allocated=lambda: self.alloc.allocated,
            memory_reserved=lambda: self.alloc.allocated + self.alloc.cached,
            mem_get_info=lambda: (128 * GIB - self.alloc.allocated - self.alloc.cached, 128 * GIB),
            get_device_properties=lambda index: SimpleNamespace(is_integrated=False),
            empty_cache=lambda: None,
        ))
        ns = {"torch": self.torch, "STEP": 8192, "FIRST": 256, "GIB": GIB,
              "MemoryGate": MemoryGate, "NoRoom": NoRoom, "prefixes": prefixes,
              "cuda_limit_bytes": capacity.cuda_limit_bytes, "LIMIT_ENV": capacity.LIMIT_ENV,
              "time": SimpleNamespace(perf_counter=lambda: 1.0), "MIN_GAP": 32,
              "_slot": lambda *args: SimpleNamespace(), "prefill_begin": lambda *args, **kw: 0,
              "image_rows": SimpleNamespace(begin=lambda *args: None)}
        cls = source_class("src/tensorfold/families/qwen4_exp/cuda/multi.py", "MultiDecoder",
                           {"_grow", "_busy", "_evict_kept", "_drop_kept", "_make_room", "_slot_for",
                            "_remember", "admit", "live", "finish"}, ns)
        self.dec = cls()
        self.dec.capacity, self.dec.depth = 81920, 0
        self.dec.free, self.dec.kept, self.dec.streams, self.dec.filling = [first, second], [], {}, []
        self.dec.fills, self.dec.held = {}, {}
        self.dec.next_id, self.dec.keep = 0, 8
        self.dec.w, self.dec.buf, self.dec.mbuf, self.dec.pbuf = None, None, None, None
        self.dec.prefill_rows, self.dec.points, self.dec.vision = 4096, None, None
        live = torch_live(self.torch, capacity.available_bytes)
        self.dec.memory_gate = MemoryGate(live(), reserve=2 * GIB, live=live)

        def shrink(st, **kw):
            st.pos, st.mtp_len = 0, 0
            self.dec.memory_gate.give(-st.resize(256))
        self.dec._shrink = shrink

        # The actual startup planner admits one full cache and the other slot's initial cache.
        weights = capacity.Weights(40 * GIB, 0)
        startup = capacity.Geometry(lambda rows: fixed - weights.resident + first.cache_bytes(rows)
                                    + second.cache_bytes(256) + 2 * GIB, 0)
        self.plan = capacity.make_plan(81920, 81920, True, capacity.available_bytes(self.torch), weights, startup)
        self.assertEqual(capacity.choose(self.plan), 81920)
        self.assertLess(self.plan.weights.resident + startup.needed(81920), 60 * GIB)

    def waiting_and_decoding(self):
        # A long prompt's slot is already grown while its passes fill; a shorter prompt joins decode first.
        filling = Stream([1] * 50000, 10)
        self.dec.admit(filling)
        decoding = Stream([2] * 5000, 60000)
        self.dec.admit(decoding)
        self.dec.filling.remove(decoding)
        self.dec.streams[decoding.sid] = decoding
        for expected in (16384, 24576):
            decoding.st.pos = decoding.st.capacity - 1
            self.assertEqual(self.dec._make_room(), [])
            self.assertEqual(decoding.st.capacity, expected)
        self.assertLess(self.alloc.peak, 60 * GIB)
        return filling, decoding

    def test_lone_decode_waits_under_the_cap_while_another_prompt_can_finish(self):
        filling, decoding = self.waiting_and_decoding()
        before = self.alloc.allocated
        decoding.st.pos = decoding.st.capacity - 1
        ended = self.dec._make_room()
        self.assertEqual(ended, [])
        self.assertFalse(decoding.done)
        self.assertTrue(decoding.waiting)
        self.assertEqual(self.alloc.allocated, before)
        self.assertLessEqual(self.alloc.peak, 60 * GIB)
        self.assertEqual([id(s) for s in self.dec.filling], [id(filling)])
        self.assertFalse(filling.done)
        # A completed filling request releases its slot, then the waiting decoder can grow unchanged.
        self.dec.filling.remove(filling)
        self.dec.finish([filling])
        self.assertEqual(self.dec._make_room(), [])
        self.assertFalse(decoding.waiting)
        self.assertFalse(decoding.done)
        self.assertEqual(decoding.st.capacity, 32768)
        self.assertLessEqual(self.alloc.peak, 60 * GIB)

    def test_an_isolated_decoder_refuses_impossible_growth_without_leaking_its_slot(self):
        filling, decoding = self.waiting_and_decoding()
        # These existing allocations remain live, but no other decoder request can finish to free them.
        self.dec.filling.clear()
        before = self.alloc.allocated
        decoding.st.pos = decoding.st.capacity - 1
        ended = self.dec._make_room()
        self.assertEqual([id(s) for s in ended], [id(decoding)])
        self.assertTrue(decoding.done)
        self.assertFalse(decoding.waiting)
        self.assertIsInstance(decoding.error, NoRoom)
        for remedy in (capacity.LIMIT_ENV, "max_tokens", "--parallel"):
            self.assertIn(remedy, str(decoding.error))
        self.assertEqual(self.alloc.allocated, before)
        self.assertLessEqual(self.alloc.peak, 60 * GIB)
        self.assertEqual(filling.st.capacity, 57344)
        self.dec.finish(ended)
        self.assertNotIn(decoding.sid, self.dec.streams)
        self.assertIn(decoding.st, self.dec.free)

    def test_failed_admission_restores_both_a_fresh_slot_and_a_kept_prefix(self):
        # Extra already-loaded buffers leave insufficient room for this prompt even though startup fitted.
        self.alloc.take(11 * GIB)
        initial_free = list(self.dec.free)
        with self.assertRaisesRegex(NoRoom, capacity.LIMIT_ENV):
            self.dec.admit(Stream([1] * 50000, 10))
        self.assertEqual({id(st) for st in self.dec.free}, {id(st) for st in initial_free})
        self.assertFalse(self.dec.filling)
        self.assertEqual(self.dec.memory_gate.held, 0)
        st = self.dec.free.pop()
        prefix, snapshot, tail = [1] * 200, {"mtp_len": 0}, None
        self.dec.kept = [(prefix, st, snapshot, tail)]
        with self.assertRaisesRegex(NoRoom, capacity.LIMIT_ENV):
            self.dec.admit(Stream([1] * 50000, 10))
        self.assertEqual(len(self.dec.kept), 1)
        self.assertIs(self.dec.kept[0][1], st)
        self.assertEqual(self.dec.kept[0][0], prefix)
        self.assertFalse(self.dec.filling)
        self.assertEqual(self.dec.memory_gate.held, 0)

    def test_the_gate_reserve_can_fail_when_a_truly_alone_growth_still_fits_the_cap(self):
        st = self.dec.free[0]
        self.dec._grow(st, 24576, alone=True)
        extra = st.cache_bytes(81920) - st.cache_bytes() + st.layer_bytes(81920)
        self.assertFalse(self.dec.memory_gate.fits(extra))
        self.assertTrue(self.dec._grow(st, 75000, alone=True))
        self.assertEqual(st.capacity, 81920)
        self.assertLess(self.alloc.peak, 60 * GIB)

    def test_an_idle_kept_slot_is_evicted_before_the_lone_request_is_refused(self):
        filling, decoding = self.waiting_and_decoding()
        self.dec.filling.remove(filling)
        self.dec.kept = [([1] * 200, filling.st, {}, None)]
        decoding.st.pos = decoding.st.capacity - 1
        self.assertEqual(self.dec._make_room(), [])
        self.assertFalse(decoding.done)
        self.assertFalse(decoding.waiting)
        self.assertEqual(decoding.st.capacity, 32768)
        self.assertEqual(self.dec.kept, [])
        self.assertIn(filling.st, self.dec.free)
        self.assertEqual(filling.st.capacity, 256)
        self.assertLessEqual(self.alloc.peak, 60 * GIB)

    def test_uncapped_growth_preserves_the_existing_lone_stream_policy(self):
        _, decoding = self.waiting_and_decoding()
        os.environ.pop(capacity.LIMIT_ENV)
        decoding.st.pos = decoding.st.capacity - 1
        self.assertEqual(self.dec._make_room(), [])
        self.assertEqual(decoding.st.capacity, 32768)
        self.assertGreater(self.alloc.peak, 60 * GIB)
        self.assertFalse(decoding.done)


if __name__ == "__main__":
    unittest.main()
