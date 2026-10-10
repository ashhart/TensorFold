#!/usr/bin/env python3
"""A tiny mock Hugging Face hub for the path-traversal smoke test.

Serves a repo whose file tree contains a file named `../../../../../pwned.txt`.
A well-behaved client must refuse or sanitize it; linkIntoSnapshot (pull.zig)
joins the raw path onto the snapshot dir, so the symlink lands outside the cache.
"""
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8931

CONFIG = json.dumps({"model_type": "nemotron_h"}).encode()
EVIL = b"you have been pwned from outside the cache\n"
SHA = "cafebabe" * 5  # fake commit the revision endpoint returns

git_sha1 = hashlib.sha1(b"blob %d\x00" % len(CONFIG) + CONFIG).hexdigest()
evil_sha256 = hashlib.sha256(EVIL).hexdigest()

TREE = [
    {"type": "file", "path": "config.json", "size": len(CONFIG), "oid": git_sha1},
    # The malicious entry: traverses out of snapshots/<sha>/ and out of the repo dir.
    {"type": "file", "path": "../../../../../pwned.txt", "size": len(EVIL),
     "lfs": {"oid": evil_sha256, "size": len(EVIL)}},
]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if "/revision/" in path:
            return self._json({"sha": SHA})
        if "/tree/" in path:
            return self._json(TREE)
        # resolve/<sha>/config.json and resolve/<sha>/../../../../../pwned.txt
        if "/resolve/" in path and path.endswith("config.json"):
            body = CONFIG
        elif "/resolve/" in path and path.endswith("pwned.txt"):
            body = EVIL
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
