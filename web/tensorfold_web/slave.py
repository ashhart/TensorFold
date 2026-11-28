"""TensorFold Web Slave daemon for worker telemetry, RDMA probing, and rank lifecycle over TLS/HTTP."""

from __future__ import annotations

import argparse
import hmac
import json
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .cluster_rank import is_loopback_host
from .cluster_telemetry import RankManager, collect_local_telemetry, probe_rdma_link

try:
    from fastapi import FastAPI, HTTPException, Request
except ImportError:  # Zero-dependency slave mode on stock Python 3.9+
    FastAPI = HTTPException = Request = None  # type: ignore[assignment]

try:
    from . import __version__
except Exception:  # noqa: BLE001
    __version__ = "cluster-slave"

TOKEN_HEADER = "x-tensorfold-token"


def _same_secret(a: str, b: str) -> bool:
    return bool(a and b) and hmac.compare_digest(a.encode(), b.encode())


def _extract_token(headers: Any) -> str:
    given = headers.get(TOKEN_HEADER) or headers.get("X-TensorFold-Token") or ""
    if not given:
        auth = headers.get("authorization") or headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            given = auth[7:].strip()
    return given


def create_slave_app(*, token: str | None = None, rank_manager: RankManager | None = None):
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed; use serve_slave_stdlib() instead")

    app = FastAPI(title="TensorFold Slave", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    mgr = rank_manager or RankManager()
    app.state.rank_manager = mgr

    def _require_auth(request: Request) -> None:
        if not token:
            return
        if not _same_secret(_extract_token(request.headers), token):
            raise HTTPException(401, "invalid or missing cluster token")

    @app.get("/api/slave/health")
    def slave_health():
        return {"ok": True, "mode": "slave", "version": __version__, "time": time.time()}

    @app.get("/api/slave/telemetry")
    def slave_telemetry(request: Request):
        _require_auth(request)
        return collect_local_telemetry(mgr, role="slave")

    @app.get("/api/slave/logs")
    def slave_logs(request: Request, limit: int = 100):
        _require_auth(request)
        return {"lines": mgr.logs(max(1, min(limit, 250)))}

    @app.post("/api/slave/rank/start")
    def slave_rank_start(request: Request, body: dict[str, Any]):
        _require_auth(request)
        try:
            return mgr.start(body)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/slave/rank/stop")
    def slave_rank_stop(request: Request):
        _require_auth(request)
        return mgr.stop()

    @app.post("/api/slave/rdma/probe")
    def slave_rdma_probe(request: Request, body: dict[str, Any]):
        _require_auth(request)
        device = str(body.get("device") or "")
        via = str(body.get("via") or "")
        port = int(body.get("port") or 19780)
        rank = int(body["rank"]) if body.get("rank") is not None else 1
        rounds = int(body.get("rounds") or 20)
        seconds = int(body.get("seconds") or 1)
        peer_port = int(body.get("peer_port") or port)
        if not device or not via:
            raise HTTPException(400, "device and via are required")
        if body.get("async"):
            def run_bg() -> None:
                probe_rdma_link(device, via, port=port, rank=rank, rounds=rounds, seconds=seconds, peer_port=peer_port)

            threading.Thread(target=run_bg, name="rdma-probe", daemon=True).start()
            return {"started": True}
        return probe_rdma_link(device, via, port=port, rank=rank, rounds=rounds, seconds=seconds, peer_port=peer_port)

    return app


def _validate_slave_bind(
    host: str,
    token: str | None,
    tls_cert: str | None,
    tls_key: str | None,
) -> None:
    if bool(tls_cert) != bool(tls_key):
        raise ValueError("--tls-cert and --tls-key must be provided together")
    if not is_loopback_host(host):
        if not token:
            raise ValueError("non-loopback slave requires --token-file <path> or --token <secret>")
        if not (tls_cert and tls_key):
            raise ValueError(
                "non-loopback slave requires TLS (--tls-cert and --tls-key); "
                "keep --host 127.0.0.1 and tunnel over SSH otherwise"
            )


def build_slave_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8091,
    token: str | None = None,
    tls_cert: str | None = None,
    tls_key: str | None = None,
    rank_manager: RankManager | None = None,
) -> tuple[ThreadingHTTPServer, RankManager]:
    """Build a zero-dependency stdlib threaded HTTPS/HTTP daemon for Slave mode on stock Python 3.9+."""
    _validate_slave_bind(host, token, tls_cert, tls_key)
    mgr = rank_manager or RankManager()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _auth_ok(self) -> bool:
            if not token:
                return True
            if _same_secret(_extract_token(self.headers), token):
                return True
            self._send(401, {"detail": "invalid or missing cluster token"})
            return False

        def _read_json(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or "0")
            if n <= 0:
                return {}
            raw = self.rfile.read(min(n, 2 * 1024 * 1024))
            return json.loads(raw.decode("utf-8")) if raw else {}

        def do_GET(self) -> None:  # noqa: N802
            u = urlsplit(self.path)
            if u.path == "/api/slave/health":
                self._send(200, {"ok": True, "mode": "slave", "version": __version__, "time": time.time()})
                return
            if not self._auth_ok():
                return
            if u.path == "/api/slave/telemetry":
                self._send(200, collect_local_telemetry(mgr, role="slave"))
            elif u.path == "/api/slave/logs":
                qs = parse_qs(u.query)
                lim = int((qs.get("limit") or ["100"])[0])
                self._send(200, {"lines": mgr.logs(max(1, min(lim, 250)))})
            else:
                self._send(404, {"detail": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            u = urlsplit(self.path)
            if not self._auth_ok():
                return
            try:
                body = self._read_json()
            except ValueError as exc:
                self._send(400, {"detail": f"invalid json: {exc}"})
                return
            if u.path == "/api/slave/rank/start":
                try:
                    self._send(200, mgr.start(body))
                except Exception as exc:  # noqa: BLE001
                    self._send(400, {"detail": str(exc)})
            elif u.path == "/api/slave/rank/stop":
                self._send(200, mgr.stop())
            elif u.path == "/api/slave/rdma/probe":
                device = str(body.get("device") or "")
                via = str(body.get("via") or "")
                p = int(body.get("port") or 19780)
                rank = int(body["rank"]) if body.get("rank") is not None else 1
                rounds = int(body.get("rounds") or 20)
                seconds = int(body.get("seconds") or 1)
                peer_port = int(body.get("peer_port") or p)
                if not device or not via:
                    self._send(400, {"detail": "device and via are required"})
                    return
                if body.get("async"):
                    threading.Thread(
                        target=probe_rdma_link,
                        args=(device, via, p, rank, rounds, seconds, peer_port),
                        daemon=True,
                    ).start()
                    self._send(200, {"started": True})
                    return
                self._send(200, probe_rdma_link(device, via, port=p, rank=rank, rounds=rounds, seconds=seconds, peer_port=peer_port))
            else:
                self._send(404, {"detail": "not found"})

    class ReuseThreadingHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True

    srv = ReuseThreadingHTTPServer((host, port), Handler)
    if tls_cert and tls_key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(certfile=str(tls_cert), keyfile=str(tls_key))
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    return srv, mgr


def serve_slave_stdlib(
    *,
    host: str = "127.0.0.1",
    port: int = 8091,
    token: str | None = None,
    tls_cert: str | None = None,
    tls_key: str | None = None,
) -> None:
    """Zero-dependency stdlib threaded HTTPS/HTTP daemon for Slave mode on stock Python 3.9+."""
    srv, mgr = build_slave_server(host=host, port=port, token=token, tls_cert=tls_cert, tls_key=tls_key)
    shown = f"[{host}]" if ":" in host else host
    scheme = "https" if tls_cert else "http"
    print(f"[slave] TensorFold Cluster Slave {__version__} listening on {scheme}://{shown}:{port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()
        srv.server_close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tensorfold-slave", description="TensorFold Cluster Slave daemon")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--token", default=None)
    p.add_argument("--token-file", default=None, help="read cluster token from a restricted file instead of argv")
    p.add_argument("--tls-cert", default=None, help="PEM TLS certificate file (required for non-loopback hosts)")
    p.add_argument("--tls-key", default=None, help="PEM TLS private key file (required for non-loopback hosts)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    token = args.token
    if args.token_file:
        try:
            token = Path(args.token_file).read_text(encoding="utf-8").strip() or None
        except OSError as exc:
            p.error(f"cannot read --token-file: {exc}")
    try:
        _validate_slave_bind(args.host, token, args.tls_cert, args.tls_key)
    except ValueError as exc:
        p.error(str(exc))
    serve_slave_stdlib(
        host=args.host,
        port=args.port,
        token=token,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
