import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastapi.testclient import TestClient

from tensorfold_web import settings as settings_mod
from tensorfold_web.catalog import Catalog
from tensorfold_web.engine import Engine
from tensorfold_web.mock_engine import serve as mock_serve
from tensorfold_web.server import create_app


def _mock(**kw):
    server = mock_serve(port=0, **kw)
    return server, server.server_address[1]

def _client(tmp_path, monkeypatch, port, **settings):
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path)
    s = settings_mod.Settings(engine_url=f"http://127.0.0.1:{port}", remote=True, **settings)
    cat = Catalog()
    cat.refresh_sizes_async = lambda: None       # offline in tests
    return TestClient(create_app(settings=s, engine=Engine(s), catalog=cat)), s

def _echo():
    """An engine stand-in that answers every GET with the headers and query it received."""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path, _, query = self.path.partition("?")
            body = json.dumps({"status": "ok", "path": path, "query": query,
                               "headers": {k.lower(): v for k, v in self.headers.items()}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]
