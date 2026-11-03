"""Exercise the full runner on a protocol fake and keep every planted refusal visible."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest

import qwen27_living_ab as runner


class RunnerTest(unittest.TestCase):
    def test_hash_uses_the_server_decimal_comma_protocol(self):
        ids = [1, 23, 456, 2**32 - 1]
        sha = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
        self.assertEqual(sha, runner.tokens({"tensorfold": {"token_ids": ids, "token_sha256": sha}})["sha256"])
        with self.assertRaises(ValueError):
            runner.tokens({"tensorfold": {"token_ids": ids, "token_sha256": "incorrect"}})
        with self.assertRaises(ValueError):
            runner.tokens({"tensorfold": {"token_ids": []}})

    def args(self, root, fault="none"):
        source, incoming = root / "source", root / "incoming"
        source.mkdir()
        incoming.mkdir()
        (source / "state.json").write_text('{"learned":false}')
        (incoming / "state.json").write_text('{"learned":true,"imported":true}')
        fake = Path(__file__).with_name("qwen27_living_fake.py").resolve()
        with socket.socket() as port:
            port.bind(("127.0.0.1", 0))
            selected = port.getsockname()[1]
        return argparse.Namespace(output=root / "out", checkpoint=source, from_other=incoming, backend="metal",
              checkpoint_revision="synthetic", base_binary=fake, candidate_binary=fake, pocket_binary=fake,
              learning_binary=fake, untaught_binary=fake, port=selected, startup_timeout=10, request_timeout=10,
              server_args=["--fake-fault", fault], panel=list(runner.PANEL), reps=3, tokens=16,
              max_slowdown_pct=1e9)

    def test_full_matrix_and_restart_delete_rollback_portability(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.args(Path(folder))
            result = runner.run(args)
            self.assertTrue(result["complete"])
            self.assertEqual("pass", result["device_fixtures"])
            self.assertEqual("pass", result["portable_input"])
            self.assertTrue(all(result["originals_unchanged"].values()))
            self.assertTrue((args.output / "portable-manifest.json").is_file())
            self.assertFalse(list(args.output.rglob("key")))
            for path in args.output.rglob("process.json"):
                self.assertIn("exit", json.loads(path.read_text()))

    def test_bad_drafts_refusal_disturbance_and_delete_never_report_complete(self):
        for fault in ("draft", "no-drafts", "refused", "disturb", "delete", "import-disturb"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                args = self.args(Path(folder), fault)
                with self.assertRaises(ValueError):
                    runner.run(args)
                summary = json.loads((args.output / "summary.json").read_text())
                self.assertFalse(summary["complete"])
                self.assertTrue(all(summary["originals_unchanged"].values()))
                self.assertFalse(list(args.output.rglob("key")))


if __name__ == "__main__":
    unittest.main()
