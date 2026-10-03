"""CPU regression for the real Qwen27 request boundaries and idle store; no model needed."""

import importlib
import threading
import time
import unittest
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
if not dist.is_available():
    pytest.skip("needs torch.distributed", allow_module_level=True)

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.families.qwen3_5.cuda import engine  # noqa: E402
from tests.test_cuda_geometry import allocations  # noqa: E402, F401 (import-only Triton stand-in)


@pytest.fixture(autouse=True)
def cuda_modules(allocations):  # noqa: F811
    global multi, decode_tp
    multi = importlib.import_module("tensorfold.families.qwen3_5.cuda.multi")
    decode_tp = importlib.import_module("tensorfold.families.qwen3_5.cuda.decode_tp")


class EndLoop(Exception):
    pass


def weights():
    return NS(config=NS(eos=(0,), vocab=10), norm=NS(device="cpu"), head=NS(n=10))


class IdleTests(unittest.TestCase):
    def single(self):
        e = object.__new__(engine.Qwen27Engine)
        e.w, e.multi, e.draft = weights(), None, None
        e.max_rows, e.allow_copy = 12, True
        e.context_window = 128
        e._stops = lambda *args: ((), None)
        return e

    def test_single_sender_rings_once_before_header_and_never_before_validation(self):
        e, events = self.single(), []
        e.idle = NS(ring=lambda: events.append("wake"))
        result = NS(seconds=0, rounds=0, widths=[])
        with patch.object(decode_tp, "_share", side_effect=lambda *a: events.append("share")), \
             patch.object(decode_tp, "prefill_tp", return_value=(None, 7)), \
             patch.object(decode_tp, "decode_tp", return_value=result):
            e._generate_tp([1, 2], 1, None, lambda _: True, None, time.perf_counter(), False)
            self.assertEqual(events, ["wake", "share", "share"])
            events.clear()
            with patch("tensorfold.engine.grammar.pack", side_effect=ValueError("invalid grammar")):
                with self.assertRaisesRegex(ValueError, "invalid grammar"):
                    e._generate_tp([1], 1, None, lambda _: True, None, 0, False)
            self.assertEqual(events, [])

    def test_single_follower_waits_on_cpu_before_entering_gpu_receive(self):
        e = self.single()
        e.idle = NS(wait=lambda: (_ for _ in ()).throw(EndLoop()))
        with patch.object(decode_tp, "_share", side_effect=AssertionError("GPU receive entered while idle")):
            with self.assertRaises(EndLoop):
                e.follow()

    def test_multi_sender_wakes_only_at_idle_admission_boundary(self):
        d, events = multi.MultiDecoder(weights(), world=2, context=128), []
        d.idle = NS(ring=lambda: events.append("wake"))
        d._queue = lambda s, hit: d.filling.append(s)
        with patch.object(multi, "_share", side_effect=lambda *a: events.append("share")):
            with self.assertRaises(ValueError):
                d.admit(Stream([1] * 128, 2))
            with patch.object(multi, "pack", side_effect=ValueError("invalid grammar")):
                with self.assertRaises(ValueError):
                    d.admit(Stream([1], 2))
            d.vision = NS(encode=lambda *a: (_ for _ in ()).throw(ValueError("invalid image")))
            with self.assertRaises(ValueError):
                d.admit(Stream([1], 2, vision=object()))
            self.assertEqual(events, [])
            a, b, c = (Stream([i], 2) for i in (1, 2, 3))
            d.admit(a)
            self.assertEqual(events, ["wake", "share", "share"])
            d.admit(b)                       # prefilling is active work too
            self.assertEqual(events.count("wake"), 1)
            d.streams = {s.sid: s for s in d.filling}
            d.filling.clear()
            d.admit(c)                       # decoding remains active
            self.assertEqual(events.count("wake"), 1)
            d.streams[c.sid] = d.filling.pop()
            d.finish([a, b, c])
            self.assertEqual(d.live(), 0)
            d.admit(Stream([4], 2))
            self.assertEqual(events.count("wake"), 2)

    def test_multi_follower_waits_only_when_both_decoding_and_prefilling_are_empty(self):
        for state in ("idle", "filling", "decoding"):
            d, events = multi.MultiDecoder(weights(), world=2), []
            d.idle = NS(wait=lambda: events.append("wait"))
            if state == "filling":
                d.filling.append(Stream([1], 2))
            elif state == "decoding":
                d.streams[0] = Stream([1], 2)
            with patch.object(multi, "_share", side_effect=EndLoop):
                with self.assertRaises(EndLoop):
                    d.follow()
            self.assertEqual(events, ["wait"] if state == "idle" else [])

    def test_single_gpu_admission_needs_no_notification(self):
        d = multi.MultiDecoder(weights(), world=1)
        d._queue = lambda s, hit: d.filling.append(s)
        with patch.object(multi, "_share", side_effect=AssertionError("single GPU entered communication")):
            d.admit(Stream([1], 2))
        self.assertEqual(d.live(), 1)

    def test_real_store_preserves_early_notifications_and_retries_idle_timeout(self):
        master = dist.TCPStore("127.0.0.1", 0, 2, True, timeout=timedelta(seconds=3), wait_for_workers=False)
        worker = dist.TCPStore("127.0.0.1", master.port, 2, False, timeout=timedelta(seconds=3))
        a, b = engine.IdleBell(master), engine.IdleBell(worker)
        for _ in range(3):
            a.ring()
        for _ in range(3):
            b.wait()
        self.assertEqual((a.sequence, b.sequence), (3, 3))
        self.assertLessEqual(master.num_keys(), 2)
        expired, result = threading.Event(), []

        class ShortWait:
            def wait(self, keys, timeout):
                try:
                    worker.wait(keys, timedelta(milliseconds=30))
                except dist.DistStoreError:
                    expired.set()
                    raise

            def delete_key(self, key):
                worker.delete_key(key)

        b.store = ShortWait()

        def follow():
            try:
                b.wait()
                result.append(b.sequence)
            except Exception as exc:
                result.append(exc)

        thread = threading.Thread(target=follow, daemon=True)
        thread.start()
        self.assertTrue(expired.wait(2))
        self.assertEqual(result, [])
        a.ring()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [4])

    def test_network_timeouts_are_not_mistaken_for_normal_idle(self):
        def fail(*args):
            raise dist.DistNetworkError("network connection timeout")

        bell = engine.IdleBell(NS(wait=fail))
        with self.assertRaises(dist.DistNetworkError):
            bell.wait()
        self.assertEqual(bell.sequence, 0)

    def test_real_master_disconnect_exits_instead_of_retrying_forever(self):
        master = dist.TCPStore("127.0.0.1", 0, 2, True, timeout=timedelta(seconds=3), wait_for_workers=False)
        worker = dist.TCPStore("127.0.0.1", master.port, 2, False, timeout=timedelta(seconds=3))
        bell, result, started = engine.IdleBell(worker), [], threading.Event()

        def follow():
            started.set()
            try:
                bell.wait()
            except Exception as exc:
                result.append(exc)

        thread = threading.Thread(target=follow, daemon=True)
        thread.start()
        self.assertTrue(started.wait(2))
        del master
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], dist.DistNetworkError)
