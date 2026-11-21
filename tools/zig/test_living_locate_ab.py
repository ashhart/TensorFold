"""The locate runner starts consecutive local fake-server runs while the old default port is occupied."""
import argparse
import errno
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import living_locate_ab as runner


def fake_server(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--slide", action="store_true")
    p.add_argument("--prompt-seed", type=int)
    args = p.parse_args(argv)
    facts = []
    word = ""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply({"facts": facts})

        def do_POST(self):
            nonlocal word
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/slide/learn":
                word = body["text"].rsplit(" ", 1)[-1].rstrip(".")
                facts.append({"id": len(facts) + 1, "state": "learned", "neurons": [1]})
                self.reply({"kept": True})
            else:
                text = body["messages"][0]["content"]
                self.reply({"choices": [{"message": {"content": word if "observatory" in text else "4"}}],
                            "tensorfold": {"token_ids": [1, 2, 3]}})

    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        (args.graph.parent / "started.json").write_text(json.dumps({"requested": args.port,
            "bound": server.server_address[1], "arm": os.environ["TF_SLIDE_PLACEMENT"]}))
        server.serve_forever()


class PortTests(unittest.TestCase):
    def test_two_default_runs_start_with_8080_occupied(self):
        with socket.socket() as occupied, tempfile.TemporaryDirectory(prefix="locate-port-") as tmp:
            try:
                occupied.bind(("127.0.0.1", 8080))
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "fixture.json").write_text("{}\n")
            command = json.dumps([sys.executable, str(Path(__file__).resolve()), "--fake-server",
                "--model", "{model}", "--graph", "{graph}", "--key", "{key}", "--port", "{port}",
                "--slide", "--prompt-seed", "1968"])
            starts = []
            reserved = []
            loads = json.loads
            case = self

            class Word(str):
                def format(self, **values):
                    if "{port}" in self:
                        port = int(values["port"])
                        with socket.socket() as competitor, case.assertRaises(OSError) as caught:
                            competitor.bind(("127.0.0.1", port))
                        case.assertEqual(errno.EADDRINUSE, caught.exception.errno)
                        reserved.append(port)
                    return super().format(**values)

            def load_command(value, *args, **kwargs):
                value_out = loads(value, *args, **kwargs)
                return [Word(word) for word in value_out] if value == command else value_out

            for index in range(2):
                output = root / f"run-{index}"
                argv = ["living_locate_ab.py", "--checkpoint", str(checkpoint), "--output", str(output),
                    "--command", command, "--binary", sys.executable, "--backend", "cuda",
                    "--checkpoint-revision", "host-fixture", "--startup-timeout", "10"]
                with patch.object(sys, "argv", argv), patch.object(runner, "FACTS", runner.FACTS[:1]), \
                        patch.object(runner, "STEADY", runner.STEADY[:1]), \
                        patch.object(runner.json, "loads", side_effect=load_command):
                    runner.main()
                self.assertTrue(json.loads((output / "original.json").read_text())["unchanged"])
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(["legacy", "locate"], [row["arm"] for row in summary])
                for row in summary:
                    start = json.loads((output / row["arm"] / "started.json").read_text())
                    self.assertEqual(start["bound"], start["requested"])
                    self.assertEqual(start["bound"], row["port"])
                    self.assertEqual(start["arm"], row["arm"])
                    self.assertGreater(row["port"], 0)
                    self.assertNotEqual(8080, row["port"])
                    self.assertEqual(1, row["accepted"])
                    self.assertEqual(1, row["final_recalled"])
                    starts.append(row["port"])
            self.assertEqual(4, len(starts))
            self.assertEqual(starts, reserved)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fake-server"]:
        fake_server(sys.argv[2:])
    else:
        unittest.main()
