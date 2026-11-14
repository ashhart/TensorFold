from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/, as the other tool tests do
from living_weights_v2.transport import Client, TransportError


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self._reply({"data": [{"id": "local-model"}]} if self.path == "/v1/models" else {"facts": []})

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(size))
        if self.path == "/v1/chat/completions":
            self._reply({"choices": [{"message": {"content": "Sheffield"}}]})
            return
        if self.path == "/v1/slide/learn":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'event: learned\ndata: {"recalled":true,"id":1}\n\nevent: saved\ndata: {"files":1}\n\ndata: [DONE]\n\n')
            return
        self._reply({"error": "unknown"})

    def _reply(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        key_file = Path(self.dir.name) / "key"
        key_file.write_text("secret\n")
        self.key = key_file
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.dir.cleanup()

    def test_real_http_sse_and_chat(self):
        client = Client(self.base, self.key)
        self.assertEqual(client.models(), {"local-model"})
        self.assertEqual(client.graph(), {})
        self.assertEqual(client.chat("local-model", "Where?"), "Sheffield")
        events = client.teach("Something")
        self.assertEqual([e.kind for e in events], ["learned", "saved"])
        self.assertTrue(events[0].data["recalled"])

    def test_loopback_only(self):
        for url in ("https://example.org", "http://localhost:8080", "http://127.0.0.1/x", "file:///etc/passwd", "http://user:pass@127.0.0.1"):
            with self.subTest(url=url), self.assertRaises(TransportError):
                Client(url, self.key)

    def test_allow_remote_explicit(self):
        client = Client("https://example.org", self.key, allow_remote=True)
        self.assertEqual(client.base_url, "https://example.org")

    def test_redirects_never_forward_authenticated_requests(self):
        received = []

        class Destination(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                received.append((self.command, self.headers.get("Authorization")))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_POST = do_GET
            do_DELETE = do_GET

        destination = ThreadingHTTPServer(("127.0.0.1", 0), Destination)
        destination_thread = threading.Thread(target=destination.serve_forever, daemon=True)
        destination_thread.start()

        class Redirect(BaseHTTPRequestHandler):
            code = 302
            location = f"http://127.0.0.1:{destination.server_port}/capture"

            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/capture":
                    Destination.do_GET(self)
                    return
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(self.code)
                self.send_header("Location", self.location)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_POST = do_GET
            do_DELETE = do_GET

        origin = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        origin_thread = threading.Thread(target=origin.serve_forever, daemon=True)
        origin_thread.start()
        try:
            for allow_remote in (False, True):
                client = Client(f"http://127.0.0.1:{origin.server_port}", self.key, allow_remote=allow_remote)
                for location in (f"http://127.0.0.1:{destination.server_port}/capture", "/capture"):
                    Redirect.location = location
                    for code in (301, 302, 303, 307, 308):
                        Redirect.code = code
                        for method in ("GET", "POST", "DELETE"):
                            with self.subTest(remote=allow_remote, location=location, code=code, method=method):
                                with self.assertRaisesRegex(TransportError, f"HTTP {code}"):
                                    with client._open("/redirect", method, {"text": "test"} if method != "GET" else None):
                                        pass
                                self.assertEqual(received, [], "redirect destination received credentials")
        finally:
            for server in (origin, destination):
                server.shutdown()
                server.server_close()
            origin_thread.join(timeout=5)
            destination_thread.join(timeout=5)

    def test_missing_or_empty_key(self):
        with self.assertRaises(TransportError):
            Client(self.base, Path(self.dir.name) / "missing")
        self.key.write_text("\n")
        with self.assertRaises(TransportError):
            Client(self.base, self.key)

    def test_sse_without_terminal_done_rejected(self):
        class FakeClient(Client):
            def _open(self, path, method="GET", payload=None):
                from io import BytesIO
                return BytesIO(b'event: learned\ndata: {"recalled":true}\n\n')
        with self.assertRaises(TransportError):
            tuple(FakeClient(self.base, self.key).events("/learn", "POST", {"text": "x"}))

    def test_invalid_sse_json_rejected(self):
        class FakeClient(Client):
            def _open(self, path, method="GET", payload=None):
                from io import BytesIO
                return BytesIO(b'event: saved\ndata: invalid\n\ndata: [DONE]\n\n')
        with self.assertRaises(TransportError):
            tuple(FakeClient(self.base, self.key).events("/learn", "POST", {"text": "x"}))


if __name__ == "__main__":
    unittest.main()
