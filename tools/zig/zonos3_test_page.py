#!/usr/bin/env python3
"""A browser test page for a ZONOS3 Mini server: serves the page and passes its three /v1 routes to tensorfold serve."""

import argparse
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

USAGE = """Start the model, then this page beside it:

    tensorfold-native serve RELEASE --parallel 1 --port 8080
    python3 tools/zig/zonos3_test_page.py --upstream 127.0.0.1:8080 --port 8081

Open http://127.0.0.1:8081/. The page lists the server's voices, speaks text with the sampling fields, and clones a
voice from an audio file: the browser turns the file into a WAV and sends it as ref_audio with each request. The
pass-through exists so the page and the API share an origin; it adds no authentication, so bind it only where the
model's own port may be reached (--bind, 127.0.0.1 by default). Standard library only."""

PAGE = Path(__file__).with_name("zonos3_test_page.html")
ROUTES = {"GET": ("/v1/voices", "/v1/models"), "POST": ("/v1/audio/speech",)}


def handler(upstream):
    host, _, port = upstream.rpartition(":")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def reply(self, status, content_type, data):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def forward(self, method):
            size = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(size) if method == "POST" else None
            if self.path not in ROUTES[method]:
                return self.reply(404, "application/json", b'{"error": {"message": "not a route of this page"}}')
            try:
                conn = http.client.HTTPConnection(host, int(port), timeout=600)
                conn.request(method, self.path, body, {"Content-Type": "application/json"})
                response = conn.getresponse()
                data = response.read()
                conn.close()
                self.reply(response.status, response.getheader("Content-Type", "application/octet-stream"), data)
            except OSError as e:
                message = '{"error": {"message": "the speech server is not reachable: %s"}}' % str(e).replace('"', "'")
                self.reply(502, "application/json", message.encode())

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self.reply(200, "text/html; charset=utf-8", PAGE.read_bytes())
            self.forward("GET")

        def do_POST(self):
            self.forward("POST")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", default="127.0.0.1:8080", help="the tensorfold server's host:port")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    ThreadingHTTPServer((args.bind, args.port), handler(args.upstream)).serve_forever()


if __name__ == "__main__":
    main()
