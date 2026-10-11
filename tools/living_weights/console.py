"""Living Weights test console: one loopback page that proxies the learning server and an untaught reference."""

import argparse
import http.client
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]}).")

HERE = Path(__file__).resolve().parent
STATIC = {"/": ("console.html", "text/html; charset=utf-8"), "/console.js": ("console.js", "text/javascript; charset=utf-8")}
SERVER_PAGES = ("/slide", "/dashboard", "/stats")
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "content-length"}
UPSTREAM_TIMEOUT_S = 900
CHUNK = 4096


def build_targets(learn_url, reference_url=None, extra=()):
    """Name each server; only `taught` may reach /v1/slide, and `clean` is the reference for cross-checks."""
    targets = {"taught": {"url": learn_url.rstrip("/"), "learn": True, "reference": False}}
    if reference_url:
        targets["clean"] = {"url": reference_url.rstrip("/"), "learn": False, "reference": True}
    for item in extra:
        name, sep, url = item.partition("=")
        if not sep or not name.isidentifier() or name in targets:
            raise ValueError(f"--model wants a new NAME=URL, got {item!r}")
        targets[name] = {"url": url.rstrip("/"), "learn": False, "reference": False}
    for t in targets.values():
        parts = urllib.parse.urlsplit(t["url"])
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError(f"server URL must be http://host:port, got {t['url']!r}")
    return targets


def route(targets, path):
    """Return (upstream base URL, upstream path) for a console path, or (status, message) when it is refused."""
    if path.startswith("/m/"):
        name, _, rest = path[3:].partition("/")
        target = targets.get(name)
        if target is None:
            return 404, "unknown model"
        if "/slide" in rest and not target["learn"]:
            return 403, "learning is only allowed on the taught server"
        return target["url"], "/" + rest
    if path.startswith(SERVER_PAGES) or path.startswith("/v1/slide/"):
        return targets["taught"]["url"], path
    return 404, "not found"


class Handler(BaseHTTPRequestHandler):
    targets = {}

    def log_message(self, fmt, *args):
        pass

    def _send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, base, path):
        upstream = urllib.parse.urlsplit(base)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        conn = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=UPSTREAM_TIMEOUT_S)
        try:
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS | {"host", "origin"}}
            conn.request(self.command, path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_HEADERS:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            while data := resp.read1(CHUNK):
                self.wfile.write(data)
                self.wfile.flush()
        except (OSError, http.client.HTTPException) as exc:
            try:
                self.send_error(502, f"TensorFold server unavailable: {exc}")
            except OSError:
                pass
        finally:
            conn.close()

    def _route(self):
        split = urllib.parse.urlsplit(self.path)
        if self.command == "GET" and split.path in STATIC:
            name, ctype = STATIC[split.path]
            return self._send_bytes(200, (HERE / name).read_bytes(), ctype)
        if self.command == "GET" and split.path == "/targets":
            public = {n: {"learn": t["learn"], "reference": t["reference"]} for n, t in self.targets.items()}
            return self._send_bytes(200, json.dumps(public).encode(), "application/json")
        base, path = route(self.targets, split.path)
        if isinstance(base, int):
            return self.send_error(base, path)
        return self._proxy(base, path + (f"?{split.query}" if split.query else ""))

    do_GET = do_POST = do_DELETE = _route


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--learn", required=True, help="the server started with --slide, e.g. http://127.0.0.1:8080")
    p.add_argument("--reference", help="an untaught server of the same checkpoint, for the cross-checks")
    p.add_argument("--model", action="append", default=[], metavar="NAME=URL", help="another read-only server to chat with")
    p.add_argument("--port", type=int, default=8090)
    args = p.parse_args()
    try:
        Handler.targets = build_targets(args.learn, args.reference, args.model)
    except ValueError as exc:
        p.error(str(exc))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Living Weights console: http://127.0.0.1:{args.port}/  servers: {', '.join(Handler.targets)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
