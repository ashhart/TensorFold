"""Clients connecting at once all get answers while the accept loop is held up, as a busy decode thread holds it."""

import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler

from tensorfold.cuda.server import Server


class _Echo(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


class _Held(Server):
    held = True

    def get_request(self):
        while self.held:
            time.sleep(0.01)
        return super().get_request()


def test_clients_connecting_at_once_all_get_answers():
    server = _Held(("127.0.0.1", 0), _Echo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    gate = threading.Barrier(64)
    got: list = []

    def client():
        gate.wait()
        try:
            req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                got.append(r.read())
        except OSError as exc:
            got.append(type(exc).__name__)

    threads = [threading.Thread(target=client) for _ in range(64)]
    for t in threads:
        t.start()
    time.sleep(3)                        # every client has connected or been turned away by now
    server.held = False
    for t in threads:
        t.join()
    server.shutdown()
    server.server_close()
    assert got.count(b"ok") == 64, sorted({g for g in got if g != b"ok"})
