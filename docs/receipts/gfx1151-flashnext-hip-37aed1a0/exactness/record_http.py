"""Run an unmodified upstream HTTP check and keep every exchange; usage: record_http.py SCRIPT OUT.jsonl ARGS...

Like docs/receipts/gfx1151-qwen-833bbfcb/record-served.py (same record keys), and it also sees http.client calls and a
raw-socket request dropped before any reply (serve_cancel.py), which urlopen alone would miss. Requests and replies
reach the script unchanged.
"""
import http.client
import io
import json
import pathlib
import runpy
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

script, receipt = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
args = sys.argv[3:]
lock = threading.Lock()
nested = threading.local()      # urlopen runs on http.client: one exchange, one record


def body_of(data):
    try:
        return json.loads(data) if data else None
    except (TypeError, ValueError):
        return None


def reply_of(raw):
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")


def write(url, request, status, response, started, **extra):
    row = {"url": url, "request": request, "status": status, "response": response,
           "seconds": time.time() - started, **extra}
    with lock, receipt.open("a") as f:
        f.write(json.dumps(row) + "\n")


real_urlopen = urllib.request.urlopen


def urlopen(req, *a, **kw):
    url = req.full_url if isinstance(req, urllib.request.Request) else str(req)
    request = body_of(req.data) if isinstance(req, urllib.request.Request) else None
    started = time.time()
    nested.on = True
    try:
        r = real_urlopen(req, *a, **kw)
        raw, status, headers = r.read(), r.status, r.headers
        r.close()
    except urllib.error.HTTPError as e:
        raw = e.read()
        write(url, request, e.code, reply_of(raw), started)
        raise urllib.error.HTTPError(url, e.code, e.reason, e.headers, io.BytesIO(raw))
    finally:
        nested.on = False
    write(url, request, status, reply_of(raw), started)
    out = io.BytesIO(raw)
    out.status, out.headers = status, headers
    return out


real_request, real_getresponse = http.client.HTTPConnection.request, http.client.HTTPConnection.getresponse


def request(self, method, url, body=None, headers={}, **kw):
    self._recorded = (url, body_of(body), time.time())
    return real_request(self, method, url, body, headers, **kw)


def getresponse(self):
    r = real_getresponse(self)
    if getattr(nested, "on", False):
        return r
    url, sent, started = self._recorded
    raw = r.read()
    write(f"http://{self.host}:{self.port}{url}", sent, r.status, reply_of(raw), started)
    r.read = lambda amt=None: raw
    return r


class Dropped:
    """A socket: the request it sent is recorded when the script closes it without ever reading a reply."""

    def __init__(self, sock, address):
        self.sock, self.address, self.sent, self.started, self.read = sock, address, b"", time.time(), False

    def sendall(self, data, *a):
        self.sent += data
        return self.sock.sendall(data, *a)

    def makefile(self, *a, **kw):
        self.read = True
        return self.sock.makefile(*a, **kw)

    def recv(self, *a):
        self.read = True
        return self.sock.recv(*a)

    def close(self):
        if self.sent and not self.read:
            head, _, body = self.sent.partition(b"\r\n\r\n")
            path = head.split(b" ")[1].decode()
            write(f"http://{self.address[0]}:{self.address[1]}{path}", body_of(body), None, None, self.started,
                  note="client closed the socket without reading a reply")
            self.sent = b""
        return self.sock.close()

    def __getattr__(self, name):
        return getattr(self.sock, name)


real_connect = socket.create_connection
urllib.request.urlopen = urlopen
http.client.HTTPConnection.request, http.client.HTTPConnection.getresponse = request, getresponse
socket.create_connection = lambda address, *a, **kw: Dropped(real_connect(address, *a, **kw), address)
sys.argv = [str(script), *args]
runpy.run_path(str(script), run_name="__main__")
