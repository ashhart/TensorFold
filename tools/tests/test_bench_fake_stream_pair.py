"""Pairing is a served-evidence fixture option; conformance, scripts and serial faults keep their step order."""
import io
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fake_served_api as fake


class PairingScopeTests(unittest.TestCase):
    def post(self, body, *, fault=None, pairs=None, script=None):
        server = SimpleNamespace(steps=script, fault=fault, prefill=0.0, stream_pairs=pairs,
                                 one=threading.Lock(), lock=threading.Lock(), warm=0.0, linger=0.0)
        handler = object.__new__(fake.FakeEngine)
        raw = json.dumps(body).encode()
        handler.server, handler.rfile = server, io.BytesIO(raw)
        handler.headers = {"Content-Length": str(len(raw))}
        handler.answer = lambda *_: fake.doc(text="abcdefg")
        played, replied = [], []
        handler.play = lambda steps, key=None: played.append(list(steps))
        handler.reply = lambda status, payload: replied.append((status, payload))
        handler.do_POST()
        return played, replied

    def test_conformance_streams_keep_the_original_pacing(self):
        expected = fake.streamed(fake.doc(text="abcdefg"), 64, fake.TOKEN_GAP)
        faults = (None, "concurrent", "unsampled", "cancel_lane", "cancel_free", "cancel_reply", "cancel_health")
        for fault in faults:
            with self.subTest(fault=fault):
                played, replied = self.post({"stream": True, "max_tokens": 64}, fault=fault)
                self.assertEqual(played, [expected])
                self.assertEqual(replied, [])

    def test_serial_json_script_and_refusal_never_join(self):
        pairs = mock.Mock()
        pairs.claim.side_effect = AssertionError("unrelated path joined the served-evidence rendezvous")
        played, _ = self.post({"stream": True, "max_tokens": 4}, fault="serial", pairs=pairs)
        self.assertEqual(played, [fake.streamed(fake.doc(text="abcdefg"), 4, fake.TOKEN_GAP)])
        _, replied = self.post({"max_tokens": 4}, pairs=pairs)
        self.assertEqual(replied, [(200, fake.doc(text="abcdefg"))])
        script = [b"data: [DONE]\n\n"]
        played, _ = self.post({"stream": True}, pairs=pairs, script=lambda _: script)
        self.assertEqual(played, [script])
        _, replied = self.post({"stream": True, "n": 2}, pairs=pairs)
        self.assertEqual(replied[0][0], 400)
        pairs.claim.assert_not_called()

    def test_running_requires_explicit_pairing(self):
        for shape, enabled in (({}, False), ({"pair_streams": True}, True)):
            with self.subTest(enabled=enabled), mock.patch.object(fake, "FixtureHTTPServer") as factory:
                server = factory.return_value
                server.server_address = ("127.0.0.1", 1)
                with fake.running(**shape):
                    self.assertEqual(isinstance(server.stream_pairs, fake.StreamPairs), enabled)
                    if not enabled:
                        self.assertIsNone(server.stream_pairs)

    def test_opted_in_stream_uses_pairing_and_keeps_every_byte_and_pause(self):
        pairs = fake.StreamPairs(fake.COUNTED)
        played, replied = self.post({"stream": True, "max_tokens": 512}, pairs=pairs)
        original = fake.streamed(fake.doc(text="abcdefg"), 512, fake.TOKEN_GAP)
        for kind in (bytes, float):
            self.assertEqual([s for s in played[0] if isinstance(s, kind)],
                             [s for s in original if isinstance(s, kind)])
        self.assertNotEqual(played[0], original)
        self.assertEqual(replied, [])
        self.assertIsNone(pairs.pending)


if __name__ == "__main__":
    unittest.main()
