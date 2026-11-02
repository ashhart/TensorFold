"""The image session client: history grows, the image stays, and every turn's fields are recorded."""
import base64
import contextlib
import hashlib
import http.server
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_image_session


def png() -> bytes:
    """A 1x1 RGB PNG built here, so the test needs no image file."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00")) +
            chunk(b"IEND", b""))


class Handler(http.server.BaseHTTPRequestHandler):
    bodies = []
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        with Handler.lock:
            Handler.bodies.append(body)
            n = len(Handler.bodies)
        reply = f"reply {n}"
        chunks = [
            {"choices": [{"delta": {"content": reply}}]},
            {"choices": [], "usage": {"completion_tokens": n + 1,
                                      "prompt_tokens_details": {"cached_tokens": n * 10}}},
        ]
        wire = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(wire)))
        self.end_headers()
        self.wfile.write(wire)


class ImageSession(unittest.TestCase):
    def test_history_grows_image_stays_and_fields_are_recorded(self):
        image = png()
        Handler.bodies = []
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "test.png"
                path.write_bytes(image)
                out = Path(tmp) / "session.jsonl"
                argv = ["bench_image_session.py", f"http://127.0.0.1:{server.server_port}",
                        "--image", str(path), "--model", "fixture", "--turns", "2", "--max-tokens", "8",
                        "--out", str(out)]
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    bench_image_session.main()
                lines = [json.loads(line) for line in out.read_text().splitlines()]
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
        self.assertEqual(len(Handler.bodies), 2)
        want = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        for body in Handler.bodies:
            content = body["messages"][0]["content"]
            self.assertIsInstance(content, list)
            self.assertEqual([part["type"] for part in content], ["text", "image_url"])
            self.assertEqual(content[1]["image_url"]["url"], want)
        self.assertGreater(len(Handler.bodies[1]["messages"]), len(Handler.bodies[0]["messages"]))
        self.assertEqual(Handler.bodies[1]["messages"][1]["role"], "assistant")
        self.assertEqual(Handler.bodies[1]["messages"][2]["role"], "user")
        self.assertIsInstance(Handler.bodies[1]["messages"][2]["content"], str)
        self.assertEqual(len(lines), 3)
        for i, record in enumerate(lines[:2], 1):
            self.assertEqual(record["turn"], i)
            self.assertEqual(record["sha256"], hashlib.sha256(f"reply {i}".encode()).hexdigest())
            self.assertEqual(record["completion_tokens"], i + 1)
            self.assertEqual(record["cached_tokens"], i * 10)
            self.assertIsInstance(record["ttft_s"], float)
            self.assertGreaterEqual(record["ttft_s"], 0)
        self.assertTrue(lines[2]["summary"])
        self.assertEqual(lines[2]["turns"], 2)
        self.assertEqual(lines[2]["completion_tokens"], 5)
        self.assertEqual(lines[2]["cached_tokens"], 30)
        self.assertEqual(lines[2]["sha256"], [lines[0]["sha256"], lines[1]["sha256"]])

    def test_invalid_arguments_fail_before_any_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.png"
            path.write_bytes(png())
            out = Path(tmp) / "session.jsonl"
            for argument in ("--turns=0", "--max-tokens=0", "--temperature=-1"):
                with self.subTest(argument=argument):
                    argv = ["bench_image_session.py", "http://127.0.0.1:1", "--image", str(path),
                            "--model", "fixture", "--out", str(out), argument]
                    stderr = io.StringIO()
                    with patch.object(sys, "argv", argv), \
                            patch.object(bench_image_session.urllib.request, "urlopen") as http, \
                            contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as exited:
                            bench_image_session.main()
                    http.assert_not_called()
                    self.assertEqual(exited.exception.code, 2)
                    self.assertIn(argument.split("=")[0], stderr.getvalue())
                    self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
