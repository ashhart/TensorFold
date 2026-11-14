from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/, as the other tool tests do
from unittest import mock

from living_weights_v2.transport import Client, TransportError

SSE_LINE_LIMIT = 1024 * 1024  # the documented per-line bound


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


    def test_loopback_requests_never_reach_an_ambient_proxy(self):
        seen = []

        class Proxy(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                seen.append(self.headers.get("Authorization"))
                body = b'{"data": [{"id": "via-proxy"}]}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{proxy.server_port}"
        ambient = {name: url for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY")}
        ambient.update(no_proxy="", NO_PROXY="")
        try:
            with mock.patch.dict(os.environ, ambient):
                self.assertEqual(Client(self.base, self.key).models(), {"local-model"})
            self.assertEqual(seen, [], "the proxy received the loopback request and its key")
        finally:
            proxy.shutdown()
            proxy.server_close()
            thread.join(timeout=5)

    def _stream(self, body: bytes):
        reads = []

        class Response:
            def __init__(self):
                self.at = 0

            def readline(self, limit=-1):
                end = body.find(b"\n", self.at)
                end = len(body) if end < 0 else end + 1
                if limit is not None and limit >= 0:
                    end = min(end, self.at + limit)
                line = body[self.at:end]
                self.at = end
                reads.append(len(line))
                return line

            def __iter__(self):
                return iter(self.readline, b"")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class FakeClient(Client):
            def _open(self, path, method="GET", payload=None):
                return Response()

        return FakeClient(self.base, self.key), reads

    def test_sse_line_reads_are_bounded(self):
        client, reads = self._stream(b"data: " + b"x" * (2 * SSE_LINE_LIMIT) + b"\n\ndata: [DONE]\n\n")
        with self.assertRaisesRegex(TransportError, "oversized SSE line"):
            client.teach("x")
        self.assertLessEqual(max(reads), SSE_LINE_LIMIT + 1, "a whole oversized line was read before the size check")

    def test_sse_stream_kept_by_teach_is_bounded(self):
        client, _ = self._stream(b"event: progress\ndata: {}\n\n" * 5000 + b"data: [DONE]\n\n")
        with self.assertRaisesRegex(TransportError, "SSE stream exceeds"):
            client.teach("x")
        event = b'data: {"pad": "' + b"y" * (SSE_LINE_LIMIT - 64) + b'"}\n\n'
        client, _ = self._stream(event * 5 + b"data: [DONE]\n\n")
        with self.assertRaisesRegex(TransportError, "SSE stream exceeds"):
            client.teach("x")

    def test_graph_states_are_checked(self):
        for fact in ({"id": 1}, {"id": 1, "state": 2}, {"id": 1, "state": "archived"}):
            class Graph(Handler):
                def do_GET(self, fact=fact):
                    self._reply({"facts": [fact]})

            server = ThreadingHTTPServer(("127.0.0.1", 0), Graph)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with self.subTest(fact=fact), self.assertRaisesRegex(TransportError, "unrecognized state"):
                    Client(f"http://127.0.0.1:{server.server_port}", self.key).graph()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

if __name__ == "__main__":
    unittest.main()
