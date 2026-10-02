"""CUDA cache growth counts existing allocations against the absolute budget, without GPU imports."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tensorfold.cuda import capacity
from tensorfold.cuda.memory_gate import MemoryGate, torch_live

GIB = capacity.GIB


def device(free, allocated, reserved=None, integrated=False):
    counters = SimpleNamespace(free=free * GIB, allocated=allocated * GIB,
                               reserved=(allocated if reserved is None else reserved) * GIB)
    cuda = SimpleNamespace(mem_get_info=lambda: (counters.free, 128 * GIB),
                           get_device_properties=lambda index: SimpleNamespace(is_integrated=integrated),
                           memory_allocated=lambda: counters.allocated, memory_reserved=lambda: counters.reserved)
    return SimpleNamespace(cuda=cuda), counters


class CudaLiveBudgetTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.meminfo = patch.object(capacity, "_meminfo", return_value=None)
        self.meminfo.start()
        self.addCleanup(self.meminfo.stop)

    def test_existing_weights_leave_only_the_unused_absolute_budget(self):
        os.environ[capacity.LIMIT_ENV] = "60"
        torch, _ = device(free=88, allocated=40)
        budget = capacity.available_bytes(torch)
        plan = capacity.make_plan(100, None, False, budget, capacity.Weights(40 * GIB, 0),
                                  capacity.Geometry(lambda slots: slots * GIB, 0))
        self.assertEqual(capacity.choose(plan), 20)
        live = torch_live(torch, capacity.available_bytes)
        gate = MemoryGate(live(), reserve=2 * GIB, live=live)
        self.assertEqual(gate.room, 20 * GIB)
        self.assertTrue(gate.fits(18 * GIB))
        self.assertFalse(gate.fits(30 * GIB))

    def test_reusable_reserved_bytes_do_not_raise_the_absolute_budget(self):
        os.environ[capacity.LIMIT_ENV] = "60"
        torch, _ = device(free=83, allocated=40, reserved=45)
        self.assertEqual(torch_live(torch, capacity.available_bytes)(), 20 * GIB)

    def test_low_physical_room_is_not_charged_for_existing_weights_again(self):
        os.environ[capacity.LIMIT_ENV] = "100"
        torch, _ = device(free=3, allocated=40, reserved=45)
        self.assertEqual(torch_live(torch, capacity.available_bytes)(), 8 * GIB)

    def test_without_a_limit_the_allocator_cache_remains_reusable(self):
        torch, _ = device(free=30, allocated=12, reserved=18)
        self.assertEqual(torch_live(torch, capacity.available_bytes)(), 36 * GIB)

    def test_the_unified_floor_still_caps_physical_room(self):
        os.environ[capacity.LIMIT_ENV] = "100"
        os.environ["TENSORFOLD_MEMORY_RESERVE_GIB"] = "10"
        torch, _ = device(free=20, allocated=40, reserved=45, integrated=True)
        with patch.object(capacity, "_meminfo", return_value={"MemTotal": 128 * GIB, "MemAvailable": 35 * GIB}):
            self.assertEqual(capacity.available_bytes(torch), 25 * GIB)
            self.assertEqual(torch_live(torch, capacity.available_bytes)(), 30 * GIB)

    def test_the_default_unified_floor_is_preserved_without_meminfo(self):
        torch, _ = device(free=30, allocated=40, reserved=45, integrated=True)
        self.assertEqual(torch_live(torch, capacity.available_bytes)(), 35 * GIB - 128 * GIB // 10)

    def test_allocations_at_or_over_the_limit_leave_no_room(self):
        os.environ[capacity.LIMIT_ENV] = "60"
        for allocated in (60, 65):
            with self.subTest(allocated=allocated):
                torch, _ = device(free=60, allocated=allocated, reserved=70)
                self.assertEqual(torch_live(torch, capacity.available_bytes)(), 0)

    def test_a_live_update_counts_growth_once(self):
        os.environ[capacity.LIMIT_ENV] = "60"
        torch, counters = device(free=83, allocated=40, reserved=45)
        live = torch_live(torch, capacity.available_bytes)
        gate = MemoryGate(live(), reserve=2 * GIB, live=live)
        self.assertTrue(gate.fits(17 * GIB))
        gate.take(17 * GIB)
        counters.allocated, counters.reserved, counters.free = 57 * GIB, 57 * GIB, 71 * GIB
        self.assertEqual(live(), 3 * GIB)
        self.assertTrue(gate.fits(GIB))
        self.assertFalse(gate.fits(2 * GIB))


if __name__ == "__main__":
    unittest.main()
